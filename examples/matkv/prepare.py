# SPDX-License-Identifier: Apache-2.0
"""Query an already-persisted HotpotQA index and save the top-k union."""

import argparse
import json

from common import (
    BgeEmbedder,
    add_data_args,
    chroma_dir,
    load_jsonl,
    open_collection,
    query_collection,
    retrievals_path,
    selected_path,
    validate_data_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()
    validate_data_args(parser, args)
    return args


def main() -> None:
    args = parse_args()
    collection = open_collection(
        chroma_dir(args.dataset, args.document_bos, args.chunk_size)
    )
    if collection.count() == 0:
        raise RuntimeError("collection is empty; run index.py first")
    embedder = BgeEmbedder(args.embedding_model) if args.embedding_model else BgeEmbedder()
    questions = load_jsonl(args.dataset / "questions" / "query.jsonl")
    selected: set[str] = set()
    retrieval_file = retrievals_path(
        args.dataset, args.document_bos, args.chunk_size
    )
    with retrieval_file.open("w", encoding="utf-8") as destination:
        for index, item in enumerate(questions, start=1):
            ids = query_collection(
                collection, embedder, item["query"], args.top_k
            )["ids"][0]
            selected.update(ids)
            destination.write(json.dumps({
                "id": item["id"], "query": item["query"], "chunk_ids": ids
            }, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(questions):
                print(f"retrieved {index}/{len(questions)} queries", flush=True)

    selection = {
        "document_bos": args.document_bos,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "indexed_chunks": collection.count(),
        "queries": len(questions),
        "chunk_ids": sorted(selected),
    }
    selected_path(args.dataset, args.document_bos, args.chunk_size).write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**selection, "chunk_ids": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
