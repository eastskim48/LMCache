# SPDX-License-Identifier: Apache-2.0
"""Run sequential HotpotQA evaluation with baseline or MatKV retrieval."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string

from transformers import AutoTokenizer

from common import (
    BgeEmbedder,
    add_data_args,
    add_server_args,
    build_llm,
    chroma_dir,
    generate,
    load_jsonl,
    metadata_token_ids,
    normalize_answer,
    open_collection,
    query_collection,
    validate_data_args,
)


INSTRUCTION = (
    "\n\nAnswer the question based on the given passages. "
    "Answer within 5 words. Do not repeat the question.\n\nQuestion: "
)
ANSWER_PREFIX_RE = re.compile(r"answer\s*:\s*", re.IGNORECASE)
QUESTION_SPLIT_RE = re.compile(r"\bquestion\s*:\s*", re.IGNORECASE)


def clean_prediction(text: str) -> str:
    cleaned = text.strip()
    answer_match = ANSWER_PREFIX_RE.search(cleaned)
    if answer_match:
        cleaned = cleaned[answer_match.end() :]
    cleaned = QUESTION_SPLIT_RE.split(cleaned, maxsplit=1)[0].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[0] if lines else ""


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("vanilla", "baseline", "matkv"), required=True
    )
    add_data_args(parser)
    add_server_args(parser)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--check-prepared-retrievals",
        action="store_true",
        help="check live top-k against prepared retrievals and exit before LLM startup",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_data_args(parser, args)
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be positive")
    return args


def main() -> None:
    args = parse_args()
    questions = load_jsonl(args.dataset / "questions" / "query.jsonl")
    answers = {
        int(item["id"]): item["answer"]
        for item in load_jsonl(args.dataset / "answers" / "answer.jsonl")
    }
    if args.max_queries is not None:
        questions = questions[: args.max_queries]

    collection = open_collection(
        chroma_dir(args.dataset, args.document_bos, args.chunk_size)
    )
    embedder = (
        BgeEmbedder(args.embedding_model)
        if args.embedding_model
        else BgeEmbedder()
    )
    if args.check_prepared_retrievals:
        from common import retrievals_path

        prepared = {
            int(row["id"]): row["chunk_ids"]
            for row in load_jsonl(
                retrievals_path(args.dataset, args.document_bos, args.chunk_size)
            )
        }
        failures = []
        for item in questions:
            live = query_collection(
                collection, embedder, item["query"], args.top_k
            )["ids"][0]
            missing = [chunk_id for chunk_id in live if chunk_id not in prepared[int(item["id"])]]
            if missing:
                failures.append({"id": item["id"], "missing": missing})
        print(json.dumps({
            "queries": len(questions),
            "prepared_contains_live_top_k": len(questions) - len(failures),
            "failures": failures,
        }, indent=2))
        if failures:
            raise RuntimeError(f"{len(failures)} retrieval subset checks failed")
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    max_model_len = args.top_k * args.chunk_size + 512
    use_matkv = args.mode == "matkv"
    llm = build_llm(args, use_matkv=use_matkv, max_model_len=max_model_len)
    expected_hits = args.top_k * args.chunk_size
    results = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for index, item in enumerate(questions, start=1):
            retrieval = query_collection(
                collection, embedder, item["query"], args.top_k
            )
            chunk_ids = retrieval["ids"][0]
            metadatas = retrieval["metadatas"][0]
            document_tokens = [
                token
                for metadata in metadatas
                for token in metadata_token_ids(metadata)
            ]
            suffix = tokenizer.encode(
                INSTRUCTION + item["query"], add_special_tokens=False
            )
            output, elapsed_ms = generate(
                llm,
                document_tokens + suffix,
                max_tokens=args.max_new_tokens,
                with_stats=use_matkv,
                matkv_mode="read_only" if use_matkv else None,
            )
            prediction = output.outputs[0].text.strip()
            cleaned_prediction = clean_prediction(prediction)
            stats = (output.kv_transfer_params or {}).get("cached_token_stats", {})
            cached = int(stats.get("num_lmcache_cached_tokens", 0))
            if use_matkv and cached != expected_hits:
                raise RuntimeError(
                    f"query {item['id']}: expected {expected_hits} hits, got {cached}"
                )
            gold = answers.get(int(item["id"]), "")
            em = exact_match(cleaned_prediction, gold)
            f1 = token_f1(cleaned_prediction, gold)
            result = {
                "id": item["id"],
                "query": item["query"],
                "answer": gold,
                "prediction": prediction,
                "cleaned_prediction": cleaned_prediction,
                "exact_match": em,
                "f1": f1,
                "chunk_ids": chunk_ids,
                "elapsed_ms": elapsed_ms,
                "vllm_num_cached_tokens": output.num_cached_tokens,
                "cached_token_stats": stats,
            }
            results.append(result)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            print(f"evaluated {index}/{len(questions)} queries", flush=True)

    exact = sum(result["exact_match"] for result in results)
    f1 = sum(result["f1"] for result in results)
    print(
        json.dumps(
            {
                "queries": len(results),
                "exact_match": exact / len(results),
                "f1": f1 / len(results),
                "average_elapsed_ms": sum(r["elapsed_ms"] for r in results)
                / len(results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
