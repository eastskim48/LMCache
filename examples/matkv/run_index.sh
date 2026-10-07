#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DOCUMENT_BOS="${DOCUMENT_BOS:-none}"

cd "${REPO_ROOT}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run --no-capture-output -n lmcache \
  python examples/matkv/index.py \
  --document-bos "${DOCUMENT_BOS}" \
  --chunk-size 1024 \
  --top-k 10 \
  "$@"
