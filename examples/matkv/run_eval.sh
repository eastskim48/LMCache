#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DOCUMENT_BOS="${DOCUMENT_BOS:-none}"
MODE="${MODE:-matkv}"
OUTPUT="${OUTPUT:-outputs/matkv_hotpotqa_${MODE}_${DOCUMENT_BOS}.jsonl}"

cd "${REPO_ROOT}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  conda run --no-capture-output -n lmcache \
  python examples/matkv/eval.py \
  --mode "${MODE}" \
  --document-bos "${DOCUMENT_BOS}" \
  --chunk-size 1024 \
  --top-k 5 \
  --output "${OUTPUT}" \
  "$@"
