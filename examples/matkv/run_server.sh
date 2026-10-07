#!/usr/bin/env bash
set -euo pipefail

DOCUMENT_BOS="${DOCUMENT_BOS:-none}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/nvme0/dongseob/cache/matkv_hotpotqa}"
CACHE_PATH="${CACHE_ROOT}/bos_${DOCUMENT_BOS}_c1024"

mkdir -p "${CACHE_PATH}"
exec conda run --no-capture-output -n lmcache \
  lmcache server \
  --host 127.0.0.1 --port 6555 \
  --http-host 127.0.0.1 --http-port 7555 \
  --chunk-size 1024 --l1-size-gb 4 --eviction-policy LRU \
  --engine-type blend --supported-transfer-mode lmcache_driven \
  --l2-store-policy skip_l1 \
  --l2-adapter "{\"type\":\"fs\",\"base_path\":\"${CACHE_PATH}\"}" \
  --disable-observability
