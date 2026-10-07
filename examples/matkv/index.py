# SPDX-License-Identifier: Apache-2.0
"""Build and persist the HotpotQA Chroma index, without querying it."""

import argparse
import json

import chromadb
from transformers import AutoTokenizer

from common import (
    BgeEmbedder,
    COLLECTION_NAME,
    add_data_args,
    artifact_dir,
    batched,
    chroma_dir,
    content_chunk_size,
    make_chunk_tokens,
    validate_data_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args()
    validate_data_args(parser, args)
    return args


def build_chunks(args: argparse.Namespace, tokenizer):
    payload_size = content_chunk_size(args.chunk_size, args.document_bos)
    for document_path in sorted((args.dataset / "documents").glob("*.txt")):
        text = document_path.read_text(encoding="utf-8")
        document_tokens = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(document_tokens), payload_size):
            content_tokens = document_tokens[start : start + payload_size]
            if len(content_tokens) != payload_size:
                continue
            token_ids = make_chunk_tokens(
                content_tokens, tokenizer.bos_token_id, args.document_bos
            )
            yield {
                "id": f"{document_path.stem}:{start // payload_size}",
                "text": tokenizer.decode(content_tokens),
                "metadata": {
                    "document": document_path.name,
                    "chunk_index": start // payload_size,
                    "token_ids": json.dumps(token_ids, separators=(",", ":")),
                },
            }


def main() -> None:
    args = parse_args()
    target = artifact_dir(args.dataset, args.document_bos, args.chunk_size)
    target.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir(
        args.dataset, args.document_bos, args.chunk_size
    )))
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "hnsw:num_threads": 1},
    )
    if collection.count():
        raise RuntimeError(f"collection is not empty: {target}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    embedder = BgeEmbedder(args.embedding_model) if args.embedding_model else BgeEmbedder()
    chunks = list(build_chunks(args, tokenizer))
    for chunk_batch in batched(chunks, 128):
        documents = [chunk["text"] for chunk in chunk_batch]
        collection.add(
            ids=[chunk["id"] for chunk in chunk_batch],
            documents=documents,
            metadatas=[chunk["metadata"] for chunk in chunk_batch],
            embeddings=embedder.encode(documents, args.embedding_batch_size),
        )
        print(f"indexed {collection.count()}/{len(chunks)} chunks", flush=True)
    print(json.dumps({"indexed_chunks": collection.count()}, indent=2))


if __name__ == "__main__":
    main()
