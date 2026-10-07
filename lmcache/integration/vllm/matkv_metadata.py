# SPDX-License-Identifier: Apache-2.0
"""Connector metadata for MatKV non-prefix retrieval."""

# Standard
from dataclasses import dataclass
from typing import Any

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
from lmcache.v1.multiprocess.custom_types import CBMatchResult


@dataclass
class MatKVRetrieveMetadata:
    """One request's non-prefix matches and destination block mapping."""

    request_id: str
    op: LoadStoreOp
    matches: tuple[CBMatchResult, ...]
    cache_salt: str = ""
    request_configs: dict[str, Any] | None = None


class MatKVConnectorMetadata(LMCacheMPConnectorMetadata):
    """Standard LMCache metadata plus MatKV non-prefix retrieves."""

    def __init__(self) -> None:
        super().__init__()
        self.matkv_retrieves: list[MatKVRetrieveMetadata] = []

    def add_matkv_retrieve(self, metadata: MatKVRetrieveMetadata) -> None:
        """Append one non-prefix retrieve operation."""
        self.matkv_retrieves.append(metadata)
