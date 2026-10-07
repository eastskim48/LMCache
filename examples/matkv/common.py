# SPDX-License-Identifier: Apache-2.0
"""Shared data, retrieval, and vLLM helpers for the MatKV experiment."""

import argparse
import json
from pathlib import Path
import re
import string
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "LongBench-HotpotQA"
DEFAULT_MODEL = (
    "/home/dongseob/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
    "0e9e39f249a16976918f6564b8830bc894c89659"
)
DEFAULT_EMBED_MODEL = (
    "/home/dongseob/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
DEFAULT_CACHE_ROOT = Path("/mnt/nvme0/dongseob/cache/matkv_hotpotqa")
COLLECTION_NAME = "hotpotqa_chunks"


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--document-bos", choices=("none", "each"), required=True
    )


def add_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server-host", default="tcp://127.0.0.1")
    parser.add_argument("--server-port", type=int, default=6555)
    parser.add_argument("--server-http-port", type=int, default=7555)


def validate_data_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.chunk_size < 2:
        parser.error("--chunk-size must be at least 2")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    for relative in (
        "documents",
        "questions/query.jsonl",
        "answers/answer.jsonl",
    ):
        if not (args.dataset / relative).exists():
            parser.error(f"dataset entry does not exist: {args.dataset / relative}")


def artifact_dir(dataset: Path, document_bos: str, chunk_size: int) -> Path:
    return dataset / "matkv_artifacts" / f"bos_{document_bos}_c{chunk_size}"


def chroma_dir(dataset: Path, document_bos: str, chunk_size: int) -> Path:
    return artifact_dir(dataset, document_bos, chunk_size) / "chroma"


def selected_path(dataset: Path, document_bos: str, chunk_size: int) -> Path:
    return artifact_dir(dataset, document_bos, chunk_size) / "selected_chunks.json"


def retrievals_path(dataset: Path, document_bos: str, chunk_size: int) -> Path:
    return artifact_dir(dataset, document_bos, chunk_size) / "retrievals.jsonl"


def make_chunk_tokens(
    content_tokens: list[int], bos_token_id: int, document_bos: str
) -> list[int]:
    if document_bos == "each":
        return [bos_token_id, *content_tokens]
    return content_tokens


def content_chunk_size(chunk_size: int, document_bos: str) -> int:
    return chunk_size - 1 if document_bos == "each" else chunk_size


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class BgeEmbedder:
    """Small local BGE encoder used consistently for indexing and queries."""

    def __init__(self, model_path: str = DEFAULT_EMBED_MODEL) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings: list[list[float]] = []
        with torch.inference_mode():
            for text_batch in batched(texts, batch_size):
                inputs = self.tokenizer(
                    text_batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                output = self.model(**inputs).last_hidden_state[:, 0]
                output = F.normalize(output, p=2, dim=1)
                embeddings.extend(output.cpu().tolist())
        return embeddings


def open_collection(path: Path):
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    return client.get_collection(COLLECTION_NAME)


def query_collection(collection, embedder: BgeEmbedder, query: str, top_k: int):
    return collection.query(
        query_embeddings=embedder.encode([query]),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def metadata_token_ids(metadata: dict[str, Any]) -> list[int]:
    return [int(token) for token in json.loads(metadata["token_ids"])]


def build_llm(
    args: argparse.Namespace,
    use_matkv: bool,
    max_model_len: int,
) -> LLM:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }
    if use_matkv:
        kwargs["kv_transfer_config"] = {
            "kv_connector": "MatKVConnector",
            "kv_connector_module_path": (
                "lmcache.integration.vllm.matkv_connector"
            ),
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": args.server_host,
                "lmcache.mp.port": args.server_port,
            },
        }
    return LLM(**kwargs)


def generate(
    llm: LLM,
    token_ids: list[int],
    max_tokens: int,
    with_stats: bool,
    matkv_mode: str | None = None,
):
    kv_transfer_params: dict[str, Any] = {}
    if with_stats:
        kv_transfer_params["cached_token_stats"] = True
    if matkv_mode is not None:
        kv_transfer_params["matkv_mode"] = matkv_mode
    extra_args = (
        {"kv_transfer_params": kv_transfer_params}
        if kv_transfer_params
        else None
    )
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        extra_args=extra_args,
    )
    start = time.perf_counter()
    output = llm.generate(
        [{"prompt_token_ids": token_ids}], params, use_tqdm=False
    )[0]
    return output, (time.perf_counter() - start) * 1000


def normalize_answer(value: str) -> str:
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())
