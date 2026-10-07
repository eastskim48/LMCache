# SPDX-License-Identifier: Apache-2.0
"""Contiguous full-chunk lookup planning for MatKV."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
)


@dataclass(frozen=True)
class MatKVLookupPlan:
    """The contiguous cached prompt selected for one online request."""

    prefix_tokens: int
    non_prefix_segments: tuple[CBMatchResult, ...]
    matched_tokens: int


def build_matkv_lookup_plan(
    result: CBUnifiedLookupResult,
    prompt_tokens: int,
    chunk_size: int,
) -> MatKVLookupPlan:
    """Select full cached chunks contiguous from token zero.

    Prefix coverage is rounded down to a full chunk. Fingerprint matches may
    extend that prefix only one aligned chunk at a time. Selection stops at
    the first missing chunk, so ``matched_tokens`` is always a multiple of
    ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if prompt_tokens < 0:
        raise ValueError("prompt_tokens must be non-negative")

    reusable_end = prompt_tokens // chunk_size * chunk_size
    prefix_tokens = min(result.prefix_coverage_tokens, reusable_end)
    prefix_tokens = prefix_tokens // chunk_size * chunk_size

    segments_by_start = {
        segment.cur_st: segment
        for segment in result.non_prefix_segments
        if _is_full_aligned_chunk(segment, chunk_size)
    }

    matched_tokens = prefix_tokens
    selected: list[CBMatchResult] = []
    while matched_tokens < reusable_end:
        segment = segments_by_start.get(matched_tokens)
        if segment is None:
            break
        selected.append(segment)
        matched_tokens += chunk_size

    return MatKVLookupPlan(
        prefix_tokens=prefix_tokens,
        non_prefix_segments=tuple(selected),
        matched_tokens=matched_tokens,
    )


def _is_full_aligned_chunk(segment: CBMatchResult, chunk_size: int) -> bool:
    """Return whether a match represents one aligned source and target chunk."""
    return (
        segment.old_st % chunk_size == 0
        and segment.cur_st % chunk_size == 0
        and segment.old_ed - segment.old_st == chunk_size
        and segment.cur_ed - segment.cur_st == chunk_size
    )
