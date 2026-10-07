# SPDX-License-Identifier: Apache-2.0
"""Blend fingerprint matcher: token-level probe over chunk poly-hashes."""

# Standard
import threading

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.token_hasher import (
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
)

logger = init_logger(__name__)


class BlendTokenRangeMatcher:
    """Fingerprint matcher: token-level probe at any offset."""

    _MAX_CHUNKS: int = 1 << 20
    _BASE: np.uint64 = np.uint64(0x9E3779B97F4A7C15)  # Fibonacci-hashing const

    def __init__(self, chunk_size: int = 256, dedup_content: bool = False):
        """Initialize with the chunk size (tokens per fingerprint chunk) and
        whether to skip registering already-indexed poly hashes."""
        self.chunk_size = chunk_size
        self._dedup_content = dedup_content
        # Full poly hash -> compact chunk ID. A dictionary is required here:
        # a single-slot direct-address table loses the older entry whenever
        # two full hashes share their low bits, causing deterministic misses.
        self._poly_hash_to_compact_id: dict[int, int] = {}
        # compact_chunk_id -> caller token_hash (full bytes); None once evicted
        self._chunk_token_hash: list[bytes | None] = []
        # token_hash -> start position in its registered sequence
        self._token_hash_to_start: dict[bytes, int] = {}
        # token_hash -> compact_chunk_id (for eviction lookup)
        self._token_hash_to_compact_id: dict[bytes, int] = {}
        self._lock = threading.Lock()
        # compact_chunk_id -> full poly hash, for collision reject.
        self._chunk_poly_hash: list[int] = []

    def on_new_token_hashes(
        self,
        token_ids: list[int],
        token_hashes: list[bytes],
        start_chunk_idx: int = 0,
        position_offset: int = 0,
    ) -> int:
        """Index a stored sequence's non-overlapping chunks. Thread-safe.

        Already-indexed token hashes are skipped; under ``dedup_content``,
        already-indexed poly hashes are skipped too (same text behind
        different prefixes is indexed once).

        Args:
            token_ids: The stored sequence's token IDs.
            token_hashes: Per-chunk content hashes (dedup/eviction key).
            start_chunk_idx: First chunk to index.
            position_offset: Added to each recorded start position.

        Returns:
            Number of chunks newly indexed (0 if all registered, no full
            chunk, or the compact-ID table is full).
        """
        arr = np.array(token_ids, dtype=np.uint64)
        chunk_hashes = chunk_hash_windows_numba(arr, self.chunk_size, self._BASE)
        n = int(chunk_hashes.shape[0])
        if n == 0 or start_chunk_idx >= n:
            return 0

        with self._lock:
            new_idxs: list[int] = []
            batch_poly: set[int] = set()
            for i in range(start_chunk_idx, n):
                if token_hashes[i] in self._token_hash_to_compact_id:
                    continue
                if self._dedup_content:
                    poly_hash = int(chunk_hashes[i])
                    if poly_hash in batch_poly or self._poly_hash_registered(poly_hash):
                        continue
                    batch_poly.add(poly_hash)
                new_idxs.append(i)
            if not new_idxs:
                return 0
            n_new = len(new_idxs)
            new_chunk_hashes = chunk_hashes[new_idxs]

            base_id = len(self._chunk_token_hash)
            if base_id + n_new > self._MAX_CHUNKS:
                logger.error(
                    "BlendTokenRangeMatcher compact-ID overflow: %d chunks "
                    "registered, cannot add %d more (limit %d). Skipping.",
                    base_id,
                    n_new,
                    self._MAX_CHUNKS,
                )
                return 0
            if base_id + n_new > int(self._MAX_CHUNKS * 0.8):
                logger.warning(
                    "BlendTokenRangeMatcher nearing capacity: %d/%d "
                    "compact IDs used.",
                    base_id + n_new,
                    self._MAX_CHUNKS,
                )
            compact_ids = np.arange(base_id, base_id + n_new, dtype=np.int64)

            for k, orig_i in enumerate(new_idxs):
                th = token_hashes[orig_i]
                cid = int(compact_ids[k])
                poly_hash = int(new_chunk_hashes[k])
                self._chunk_token_hash.append(th)
                self._chunk_poly_hash.append(poly_hash)
                self._poly_hash_to_compact_id[poly_hash] = cid
                self._token_hash_to_start[th] = (
                    position_offset + orig_i * self.chunk_size
                )
                self._token_hash_to_compact_id[th] = cid
        return n_new

    def _poly_hash_registered(self, poly_hash: int) -> bool:
        """Whether a live chunk with this poly hash is indexed.

        Caller must hold ``self._lock``.
        """
        cid = self._poly_hash_to_compact_id.get(poly_hash)
        if cid is None:
            return False
        return (
            self._chunk_poly_hash[cid] == poly_hash
            and self._chunk_token_hash[cid] is not None
        )

    def registered_count(self) -> int:
        """Return the number of live token-hash fingerprints."""
        with self._lock:
            return len(self._token_hash_to_compact_id)

    def match_sub_sequence(
        self,
        token_ids: list[int],
    ) -> list[CBMatchResult]:
        """Find every registered chunk reused anywhere in a query sequence.

        Compute rolling hashes for all token positions, then probe the
        full-hash index. Thread-safe.

        Returns:
            One result per matching query position (the same stored chunk may
            appear more than once); empty if the query is shorter than one
            chunk or nothing matched.
        """
        if len(token_ids) < self.chunk_size:
            return []

        arr = np.array(token_ids, dtype=np.uint64)
        rolling = rolling_hash_windows_numba(arr, self.chunk_size, self._BASE)

        with self._lock:
            if not self._chunk_token_hash:
                return []

            results: list[CBMatchResult] = []
            table_hits = 0
            for pos, poly_hash_value in enumerate(rolling):
                cid = self._poly_hash_to_compact_id.get(int(poly_hash_value))
                if cid is None:
                    continue
                table_hits += 1
                th = self._chunk_token_hash[cid]
                if th is None:
                    continue  # evicted
                old_st = self._token_hash_to_start.get(th)
                if old_st is None:
                    continue
                results.append(
                    CBMatchResult(
                        old_st=old_st,
                        old_ed=old_st + self.chunk_size,
                        cur_st=pos,
                        cur_ed=pos + self.chunk_size,
                        hash=th,
                    )
                )
            logger.info(
                "[match_probe] n_tok=%d table_hits=%d matches=%d",
                len(token_ids),
                table_hits,
                len(results),
            )
            return results

    def remove_chunks(self, token_hashes: list[bytes]) -> None:
        """Evict the given chunks so later probes cannot match them.
        Thread-safe."""
        with self._lock:
            for th in token_hashes:
                cid = self._token_hash_to_compact_id.get(th)
                if cid is None:
                    continue
                poly_hash = self._chunk_poly_hash[cid]
                if self._poly_hash_to_compact_id.get(poly_hash) == cid:
                    del self._poly_hash_to_compact_id[poly_hash]
                self._chunk_token_hash[cid] = None
                self._chunk_poly_hash[cid] = 0
                self._token_hash_to_start.pop(th, None)
                del self._token_hash_to_compact_id[th]


def _unique_token_coverage(results: list[CBMatchResult]) -> int:
    """Total token coverage, merging overlapping ranges (sliding-window probe
    can return overlaps; naive sum would double-count)."""
    if not results:
        return 0
    intervals = sorted((r.cur_st, r.cur_ed) for r in results)
    coverage = 0
    cur_end = -1
    for st, ed in intervals:
        if st >= cur_end:
            coverage += ed - st
        elif ed > cur_end:
            coverage += ed - cur_end
        cur_end = max(cur_end, ed)
    return coverage
