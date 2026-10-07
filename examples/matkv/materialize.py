# SPDX-License-Identifier: Apache-2.0
"""Materialize only HotpotQA chunks selected during retrieval preparation."""

import argparse
import json
import time
from urllib.request import urlopen

from common import (
    add_data_args,
    add_server_args,
    build_llm,
    generate,
    metadata_token_ids,
    open_collection,
    selected_path,
    validate_data_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_server_args(parser)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--registration-timeout", type=float, default=60.0)
    parser.add_argument("--max-chunks", type=int)
    args = parser.parse_args()
    validate_data_args(parser, args)
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error("--max-chunks must be positive")
    return args


def main() -> None:
    args = parse_args()
    selection_file = selected_path(args.dataset, args.document_bos, args.chunk_size)
    if not selection_file.exists():
        raise FileNotFoundError(
            f"missing retrieval selection: {selection_file}; run prepare first"
        )

    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection["top_k"] != args.top_k:
        raise ValueError(
            f"prepared top_k={selection['top_k']}, requested top_k={args.top_k}"
        )
    selected = selection["chunk_ids"]
    if args.max_chunks is not None:
        selected = selected[: args.max_chunks]
    collection = open_collection(selection_file.parent / "chroma")
    records = collection.get(ids=selected, include=["metadatas"])
    by_id = dict(zip(records["ids"], records["metadatas"]))
    llm = build_llm(args, use_matkv=True, max_model_len=args.chunk_size + 16)
    suffix = [50000 + index for index in range(8)]

    unique_token_chunks: set[tuple[int, ...]] = set()
    for index, chunk_id in enumerate(selected, start=1):
        token_ids = metadata_token_ids(by_id[chunk_id])
        if len(token_ids) != args.chunk_size:
            raise ValueError(f"{chunk_id} has {len(token_ids)} tokens")
        unique_token_chunks.add(tuple(token_ids))
        generate(
            llm,
            token_ids + suffix,
            max_tokens=1,
            with_stats=False,
            matkv_mode="store_only",
        )
        if index % 25 == 0 or index == len(selected):
            print(f"materialized {index}/{len(selected)} chunks", flush=True)

    deadline = time.monotonic() + args.registration_timeout
    status_url = f"http://127.0.0.1:{args.server_http_port}/status"
    expected_fingerprints = len(unique_token_chunks)
    while True:
        with urlopen(status_url, timeout=5) as response:
            status = json.load(response)
        registered = int(status.get("registered_fingerprints", 0))
        pending = int(status.get("pending_fingerprints", 0))
        queued = int(status.get("fingerprint_queue_size", 0))
        if registered >= expected_fingerprints and pending == 0 and queued == 0:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "fingerprint registration did not finish: "
                f"registered={registered}/{expected_fingerprints}, "
                f"pending={pending}, queued={queued}"
            )
        time.sleep(0.1)
    print(
        json.dumps(
            {
                "operation": "materialize",
                "document_bos": args.document_bos,
                "chunk_size": args.chunk_size,
                "top_k": args.top_k,
                "materialized_chunks": len(selected),
                "materialized_tokens": len(selected) * args.chunk_size,
                "registered_fingerprints": registered,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
