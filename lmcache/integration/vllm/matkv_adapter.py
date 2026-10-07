# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side unified lookup adapter for MatKV."""

# Standard
from typing import Any

# Third Party
import zmq

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.custom_types import (
    CBUnifiedLookupResult,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.mq import MessagingFuture
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform import resolve_kv_wrapper_factory


class MatKVWorkerAdapter(LMCacheMPWorkerAdapter):
    """Register the model's RoPE table for server-side key relocation."""

    def register_matkv_rope(
        self,
        cos_sin_cache: Any,
        head_size: int,
        is_neox_style: bool,
    ) -> None:
        """Keep and export the RoPE table after paged KV registration."""
        self._matkv_rope_cache = cos_sin_cache
        wrap = resolve_kv_wrapper_factory(cos_sin_cache.device.type)
        future = self.req_client.cb_register_rope(
            self.instance_id,
            [wrap(cos_sin_cache)],
            head_size,
            is_neox_style,
            [],
            [],
        )
        future.result(timeout=self._mq_timeout)


class MatKVSchedulerAdapter(LMCacheMPSchedulerAdapter):
    """Issue and poll CacheBlend unified lookups for one LMCache server."""

    def __init__(
        self,
        server_urls: list[str],
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy | int,
        legacy_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if len(server_urls) != 1:
            raise ValueError("MatKV PoC requires exactly one LMCache server")
        super().__init__(
            server_urls,
            context,
            model_name,
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            **kwargs,
        )
        self._matkv_futures: dict[
            str, MessagingFuture[CBUnifiedLookupResult | None]
        ] = {}
        self._matkv_keys: dict[str, IPCCacheServerKey] = {}
        self._matkv_results: dict[str, CBUnifiedLookupResult] = {}

    def maybe_submit_matkv_lookup(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> None:
        """Submit one unified lookup over the prompt's full chunks."""
        if request_id in self._matkv_futures or request_id in self._matkv_results:
            return

        aligned_end = len(token_ids) // self.lmcache_tokens_per_chunk
        aligned_end *= self.lmcache_tokens_per_chunk
        if aligned_end == 0:
            self._matkv_results[request_id] = CBUnifiedLookupResult(0, [])
            return
        key = self._create_key(
            token_ids[:aligned_end],
            start=0,
            end=aligned_end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        ).no_worker_id_version()
        self._matkv_keys[request_id] = key
        self._matkv_futures[request_id] = self._matkv_client.cb_unified_lookup(
            key, self.tp_size
        )

    def check_matkv_lookup(
        self, request_id: str
    ) -> CBUnifiedLookupResult | None:
        """Poll a unified lookup, resubmitting while server work is pending."""
        if request_id in self._matkv_results:
            return self._matkv_results[request_id]

        future = self._matkv_futures.get(request_id)
        if future is None or not future.query():
            return None

        result = future.result(timeout=0)
        if result is None:
            key = self._matkv_keys[request_id]
            self._matkv_futures[request_id] = self._matkv_client.cb_unified_lookup(
                key, self.tp_size
            )
            return None

        self._matkv_futures.pop(request_id)
        self._matkv_results[request_id] = result
        return result

    def cleanup_lookup_result(self, request_id: str) -> None:
        """Drop both standard and MatKV lookup state for a request."""
        super().cleanup_lookup_result(request_id)
        self._matkv_futures.pop(request_id, None)
        self._matkv_keys.pop(request_id, None)
        self._matkv_results.pop(request_id, None)

    @property
    def _matkv_client(self) -> RequestClient:
        """Return the single request client used by this PoC."""
        return self.req_clients[self._server_urls[0]]
