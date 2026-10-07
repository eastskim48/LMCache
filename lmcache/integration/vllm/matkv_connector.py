# SPDX-License-Identifier: Apache-2.0
"""vLLM connector entry point for MatKV document reuse."""

# Standard
from typing import TYPE_CHECKING, Any

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from lmcache.integration.vllm.matkv_adapter import (
    MatKVSchedulerAdapter,
    MatKVWorkerAdapter,
)
from lmcache.integration.vllm.matkv_lookup import (
    MatKVLookupPlan,
    build_matkv_lookup_plan,
)
from lmcache.integration.vllm.matkv_metadata import (
    MatKVConnectorMetadata,
    MatKVRetrieveMetadata,
)
from lmcache.v1.multiprocess.custom_types import CBUnifiedLookupResult
from lmcache.v1.multiprocess.group_view import slice_block_ids_per_group
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
from lmcache.utils import init_logger


logger = init_logger(__name__)

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class MatKVConnector(LMCacheMPConnector):
    """LMCache MP connector specialized for position-independent documents.

    The initial implementation inherits the standard multiprocess lifecycle.
    Retrieval is added in subsequent steps.
    """

    scheduler_adapter_cls = MatKVSchedulerAdapter
    worker_adapter_cls = MatKVWorkerAdapter
    metadata_cls = MatKVConnectorMetadata

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.lookup_plans: dict[str, MatKVLookupPlan] = {}

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Register paged KV first, then the exact vLLM Llama RoPE table."""
        super().register_kv_caches(kv_caches)

        # Third Party
        from vllm.model_executor.layers.rotary_embedding import get_rope

        model_config = self._vllm_config.model_config
        hf_config = model_config.hf_config
        if getattr(hf_config, "model_type", None) != "llama":
            raise ValueError("MatKV PoC currently supports Llama models only")

        head_size = model_config.get_head_size()
        rope = get_rope(
            head_size=head_size,
            max_position=model_config.max_model_len,
            is_neox_style=True,
            rope_parameters=hf_config.rope_parameters,
            dtype=model_config.dtype,
        )
        cache = rope.cos_sin_cache.to(next(iter(kv_caches.values())).device)
        self.worker_adapter.register_matkv_rope(cache, head_size, True)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Return the chunk-aligned MatKV hit beyond local KV coverage."""
        tracker = self._get_or_create_request_tracker(request)
        mode = self._request_mode(request)
        if mode == "store_only":
            tracker.num_vllm_hit_tokens = 0
            tracker.num_lmcache_hit_tokens = 0
            return 0, False

        plan = self.lookup_plans.get(request.request_id)
        if plan is None:
            self.scheduler_adapter.maybe_submit_matkv_lookup(
                request.request_id,
                token_ids=tracker.get_token_ids(),
                cache_salt=tracker.cache_salt,
                request_configs=tracker.request_configs,
            )
            result = self.scheduler_adapter.check_matkv_lookup(request.request_id)
            if result is None:
                return None, True
            plan = self.build_lookup_plan(
                result,
                prompt_tokens=len(tracker.get_token_ids()),
                chunk_size=self.scheduler_adapter.lmcache_tokens_per_chunk,
            )
            logger.info(
                "MatKV lookup for %s: prefix=%d shifted=%d matched=%d",
                request.request_id,
                plan.prefix_tokens,
                len(plan.non_prefix_segments),
                plan.matched_tokens,
            )
            self.lookup_plans[request.request_id] = plan
            if mode == "read_only":
                aligned_prompt = len(tracker.get_token_ids()) // (
                    self.scheduler_adapter.lmcache_tokens_per_chunk
                )
                aligned_prompt *= self.scheduler_adapter.lmcache_tokens_per_chunk
                tracker.increase_num_stored_tokens(aligned_prompt)
            else:
                tracker.increase_num_stored_tokens(plan.matched_tokens)
            tracker.num_lmcache_hit_tokens = plan.matched_tokens

        tracker.num_vllm_hit_tokens = (
            num_computed_tokens
            // self._hit_alignment_tokens
            * self._hit_alignment_tokens
        )
        new_matched_tokens = max(0, plan.matched_tokens - num_computed_tokens)
        return new_matched_tokens, new_matched_tokens > 0

    def on_new_request(self, request: "Request") -> None:
        """Do not prefetch requests that explicitly materialize fresh KV."""
        if self._request_mode(request) == "store_only":
            return
        super().on_new_request(request)

    @staticmethod
    def _request_mode(request: "Request") -> str | None:
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict):
            return None
        mode = params.get("matkv_mode")
        if mode not in (None, "store_only", "read_only"):
            raise ValueError(f"unsupported MatKV request mode: {mode}")
        return mode

    def _process_retrieve_requests(
        self,
        metadata: MatKVConnectorMetadata,
    ) -> None:
        chunk_size = self.scheduler_adapter.lmcache_tokens_per_chunk
        for tracker in self.request_trackers.values():
            if tracker.state != LMCacheMPRequestState.WAITING_FOR_LOAD:
                continue
            plan = self.lookup_plans[tracker.request_id]
            start = tracker.num_vllm_hit_tokens // chunk_size * chunk_size

            if start < plan.prefix_tokens:
                prefix_op = self._make_retrieve_op(
                    tracker, start, plan.prefix_tokens
                )
                metadata.add_request_metadata(
                    LMCacheMPRequestMetadata(
                        request_id=tracker.request_id,
                        direction="RETRIEVE",
                        op=prefix_op,
                        cache_salt=tracker.cache_salt,
                        request_configs=tracker.request_configs,
                    )
                )

            matches = tuple(
                match
                for match in plan.non_prefix_segments
                if match.cur_ed > tracker.num_vllm_hit_tokens
            )
            if matches:
                metadata.add_matkv_retrieve(
                    MatKVRetrieveMetadata(
                        request_id=tracker.request_id,
                        op=self._make_retrieve_op(tracker, 0, plan.matched_tokens),
                        matches=matches,
                        cache_salt=tracker.cache_salt,
                        request_configs=tracker.request_configs,
                    )
                )
            tracker.state = LMCacheMPRequestState.READY

    def _make_retrieve_op(
        self,
        tracker: LMCacheMPRequestTracker,
        start: int,
        end: int,
    ) -> LoadStoreOp:
        block_ids = slice_block_ids_per_group(
            tracker.allocated_block_ids,
            self._group_tokens_per_block,
            start,
            end,
        )
        return LoadStoreOp(
            token_ids=tracker.get_token_ids(),
            block_ids=block_ids,
            start=start,
            end=end,
        )

    def start_load_kv(
        self,
        forward_context: Any,
        **kwargs: Any,
    ) -> None:
        """Load prefix KV, then re-RoPE and scatter non-prefix chunks."""
        super().start_load_kv(forward_context, **kwargs)
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MatKVConnectorMetadata)
        if not metadata.matkv_retrieves:
            return

        event = self.worker_adapter.create_recorded_event()
        for retrieve in metadata.matkv_retrieves:
            self.worker_adapter.submit_matkv_retrieve_request(
                retrieve.request_id,
                retrieve.op,
                list(retrieve.matches),
                event,
                cache_salt=retrieve.cache_salt,
                request_configs=retrieve.request_configs,
            )

    @staticmethod
    def build_lookup_plan(
        result: CBUnifiedLookupResult,
        prompt_tokens: int,
        chunk_size: int,
    ) -> MatKVLookupPlan:
        """Build the reusable contiguous prefix from a blend lookup result."""
        return build_matkv_lookup_plan(result, prompt_tokens, chunk_size)
