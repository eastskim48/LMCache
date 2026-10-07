# SPDX-License-Identifier: Apache-2.0
"""Tests for the MatKV vLLM connector entry point."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
)


CHUNK_SIZE = 4


def _match(old_st: int, cur_st: int, size: int = CHUNK_SIZE) -> CBMatchResult:
    return CBMatchResult(
        old_st=old_st,
        old_ed=old_st + size,
        cur_st=cur_st,
        cur_ed=cur_st + size,
        hash=bytes([cur_st]),
    )


def test_vllm_factory_loads_matkv_connector() -> None:
    """The external connector path resolves without a vLLM registry change."""
    pytest.importorskip("vllm", reason="requires vLLM")

    # Third Party
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from lmcache.integration.vllm.matkv_connector import MatKVConnector

    config = KVTransferConfig(
        kv_connector="MatKVConnector",
        kv_connector_module_path="lmcache.integration.vllm.matkv_connector",
        kv_role="kv_both",
    )

    assert KVConnectorFactory.get_connector_class(config) is MatKVConnector
    assert issubclass(MatKVConnector, LMCacheMPConnector)
    assert KVConnectorFactory.supports_hma_config(config)


def test_lookup_plan_reuses_prefix_and_shifted_document() -> None:
    """Canonical [D][A][Q] reuses D by prefix and A by fingerprint."""
    # First Party
    from lmcache.integration.vllm.matkv_connector import MatKVConnector

    result = CBUnifiedLookupResult(
        prefix_coverage_tokens=CHUNK_SIZE,
        non_prefix_segments=[_match(old_st=0, cur_st=CHUNK_SIZE)],
    )

    plan = MatKVConnector.build_lookup_plan(result, prompt_tokens=10, chunk_size=4)

    assert plan.prefix_tokens == CHUNK_SIZE
    assert plan.non_prefix_segments == tuple(result.non_prefix_segments)
    assert plan.matched_tokens == 2 * CHUNK_SIZE


def test_lookup_plan_stops_at_first_gap() -> None:
    """A later hit cannot extend P across a missing full chunk."""
    # First Party
    from lmcache.integration.vllm.matkv_connector import MatKVConnector

    result = CBUnifiedLookupResult(
        prefix_coverage_tokens=CHUNK_SIZE,
        non_prefix_segments=[_match(old_st=0, cur_st=2 * CHUNK_SIZE)],
    )

    plan = MatKVConnector.build_lookup_plan(result, prompt_tokens=12, chunk_size=4)

    assert plan.non_prefix_segments == ()
    assert plan.matched_tokens == CHUNK_SIZE


@pytest.mark.parametrize(
    "segment",
    [
        _match(old_st=0, cur_st=CHUNK_SIZE, size=CHUNK_SIZE - 1),
        _match(old_st=1, cur_st=CHUNK_SIZE),
        _match(old_st=0, cur_st=CHUNK_SIZE + 1),
    ],
)
def test_lookup_plan_rejects_non_full_or_misaligned_chunks(
    segment: CBMatchResult,
) -> None:
    """Only complete source and target chunks can extend P."""
    # First Party
    from lmcache.integration.vllm.matkv_connector import MatKVConnector

    result = CBUnifiedLookupResult(
        prefix_coverage_tokens=CHUNK_SIZE,
        non_prefix_segments=[segment],
    )

    plan = MatKVConnector.build_lookup_plan(result, prompt_tokens=12, chunk_size=4)

    assert plan.matched_tokens == CHUNK_SIZE


def test_lookup_plan_never_reuses_partial_prompt_tail() -> None:
    """P remains chunk-aligned when the prompt ends with a partial chunk."""
    # First Party
    from lmcache.integration.vllm.matkv_connector import MatKVConnector

    result = CBUnifiedLookupResult(
        prefix_coverage_tokens=9,
        non_prefix_segments=[],
    )

    plan = MatKVConnector.build_lookup_plan(result, prompt_tokens=10, chunk_size=4)

    assert plan.prefix_tokens == 2 * CHUNK_SIZE
    assert plan.matched_tokens == 2 * CHUNK_SIZE


def _make_scheduler_adapter() -> Any:
    """Build an adapter shell around a mocked request client."""
    # First Party
    from lmcache.integration.vllm.matkv_adapter import MatKVSchedulerAdapter

    adapter = MatKVSchedulerAdapter.__new__(MatKVSchedulerAdapter)
    adapter._server_urls = ["tcp://server"]
    adapter.req_clients = {"tcp://server": MagicMock()}
    adapter.model_name = "model"
    adapter.lmcache_tokens_per_chunk = CHUNK_SIZE
    adapter.parallel_strategy = SimpleNamespace(
        kv_world_size=1,
        kv_tp_size=1,
        num_kv_readers=1,
    )
    adapter._matkv_futures = {}
    adapter._matkv_keys = {}
    adapter._matkv_results = {}
    return adapter


def test_scheduler_adapter_submits_only_full_prompt_chunks() -> None:
    """Unified lookup excludes the prompt's partial tail."""
    adapter = _make_scheduler_adapter()
    future = MagicMock()
    adapter._matkv_client.cb_unified_lookup.return_value = future

    adapter.maybe_submit_matkv_lookup("req", list(range(10)))

    key, tp_size = adapter._matkv_client.cb_unified_lookup.call_args.args
    assert key.token_ids == tuple(range(8))
    assert key.start == 0
    assert key.end == 8
    assert tp_size == 1


def test_scheduler_adapter_skips_lookup_without_a_full_chunk() -> None:
    """A short prompt resolves to zero reusable tokens without an RPC."""
    adapter = _make_scheduler_adapter()

    adapter.maybe_submit_matkv_lookup("req", list(range(CHUNK_SIZE - 1)))

    result = adapter.check_matkv_lookup("req")
    assert result is not None
    assert result.prefix_coverage_tokens == 0
    assert result.non_prefix_segments == []
    adapter._matkv_client.cb_unified_lookup.assert_not_called()


def test_scheduler_adapter_polls_unified_lookup_until_ready() -> None:
    """A pending server job is recalled and its resolved result is retained."""
    adapter = _make_scheduler_adapter()
    pending = MagicMock()
    pending.query.return_value = True
    pending.result.return_value = None
    ready = MagicMock()
    ready.query.return_value = True
    expected = CBUnifiedLookupResult(
        prefix_coverage_tokens=CHUNK_SIZE,
        non_prefix_segments=[_match(old_st=0, cur_st=CHUNK_SIZE)],
    )
    ready.result.return_value = expected
    adapter._matkv_client.cb_unified_lookup.side_effect = [pending, ready]

    adapter.maybe_submit_matkv_lookup("req", list(range(10)))

    assert adapter.check_matkv_lookup("req") is None
    assert adapter.check_matkv_lookup("req") is expected
    assert adapter.check_matkv_lookup("req") is expected
    assert adapter._matkv_client.cb_unified_lookup.call_count == 2


def _make_request(
    prompt_tokens: int = 10, matkv_mode: str | None = None
) -> SimpleNamespace:
    # Third Party
    from vllm.v1.request import RequestStatus

    token_ids = list(range(prompt_tokens))
    return SimpleNamespace(
        request_id="req",
        status=RequestStatus.WAITING,
        cache_salt="",
        all_token_ids=token_ids,
        prompt_token_ids=token_ids,
        mm_features=[],
        sampling_params=SimpleNamespace(extra_args=None),
        kv_transfer_params=(
            {"matkv_mode": matkv_mode} if matkv_mode is not None else None
        ),
    )


def _make_scheduler_connector(
    result: CBUnifiedLookupResult | None,
) -> tuple[Any, MagicMock]:
    # First Party
    from lmcache.integration.vllm.matkv_connector import MatKVConnector

    connector = MatKVConnector.__new__(MatKVConnector)
    connector.request_trackers = {}
    connector.lookup_plans = {}
    connector._hit_alignment_tokens = CHUNK_SIZE
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = CHUNK_SIZE
    adapter.check_matkv_lookup.return_value = result
    connector.scheduler_adapter = adapter
    return connector, adapter


def test_connector_reports_only_tokens_after_local_coverage() -> None:
    """vLLM receives P minus the tokens already represented by local KV."""
    result = CBUnifiedLookupResult(
        prefix_coverage_tokens=CHUNK_SIZE,
        non_prefix_segments=[_match(old_st=0, cur_st=CHUNK_SIZE)],
    )
    connector, adapter = _make_scheduler_connector(result)
    request = _make_request()

    assert connector.get_num_new_matched_tokens(request, 0) == (8, True)
    assert connector.get_num_new_matched_tokens(request, 4) == (4, True)
    assert connector.lookup_plans[request.request_id].matched_tokens == 8
    adapter.maybe_submit_matkv_lookup.assert_called_once()


def test_connector_defers_while_unified_lookup_is_pending() -> None:
    connector, _ = _make_scheduler_connector(None)

    assert connector.get_num_new_matched_tokens(_make_request(), 0) == (None, True)


def test_connector_reports_zero_when_first_chunk_misses() -> None:
    result = CBUnifiedLookupResult(
        prefix_coverage_tokens=0,
        non_prefix_segments=[_match(old_st=0, cur_st=CHUNK_SIZE)],
    )
    connector, _ = _make_scheduler_connector(result)

    assert connector.get_num_new_matched_tokens(_make_request(), 0) == (0, False)


def test_store_only_bypasses_lookup_and_keeps_prompt_storeable() -> None:
    """Materialization recomputes even when the same prefix exists in L2."""
    connector, adapter = _make_scheduler_connector(None)
    request = _make_request(matkv_mode="store_only")

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    tracker = connector.request_trackers[request.request_id]
    assert tracker.num_stored_tokens == 0
    adapter.maybe_submit_matkv_lookup.assert_not_called()


def test_read_only_marks_every_full_prompt_chunk_as_already_stored() -> None:
    """Evaluation reads MatKV but never writes its online prompt back."""
    result = CBUnifiedLookupResult(
        prefix_coverage_tokens=CHUNK_SIZE,
        non_prefix_segments=[],
    )
    connector, _ = _make_scheduler_connector(result)
    request = _make_request(prompt_tokens=10, matkv_mode="read_only")

    assert connector.get_num_new_matched_tokens(request, 0) == (4, True)
    tracker = connector.request_trackers[request.request_id]
    assert tracker.num_stored_tokens == 2 * CHUNK_SIZE
    assert tracker.num_lmcache_hit_tokens == CHUNK_SIZE


def test_metadata_separates_prefix_and_non_prefix_retrieves() -> None:
    """D uses dense retrieve while shifted A uses CacheBlend retrieve."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestState
    from lmcache.integration.vllm.matkv_connector import MatKVConnector
    from lmcache.integration.vllm.matkv_lookup import MatKVLookupPlan
    from lmcache.integration.vllm.matkv_metadata import MatKVConnectorMetadata

    connector = MatKVConnector.__new__(MatKVConnector)
    connector.request_trackers = {}
    connector.lookup_plans = {}
    connector._group_tokens_per_block = [CHUNK_SIZE]
    connector.scheduler_adapter = SimpleNamespace(
        lmcache_tokens_per_chunk=CHUNK_SIZE
    )
    tracker = connector._get_or_create_request_tracker(_make_request())
    tracker.allocated_block_ids = {0: [10, 11, 12]}
    tracker.num_vllm_hit_tokens = 0
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    shifted = _match(old_st=0, cur_st=CHUNK_SIZE)
    connector.lookup_plans[tracker.request_id] = MatKVLookupPlan(
        prefix_tokens=CHUNK_SIZE,
        non_prefix_segments=(shifted,),
        matched_tokens=2 * CHUNK_SIZE,
    )
    metadata = MatKVConnectorMetadata()

    connector._process_retrieve_requests(metadata)

    assert len(metadata.requests) == 1
    assert metadata.requests[0].op.start == 0
    assert metadata.requests[0].op.end == CHUNK_SIZE
    assert metadata.requests[0].op.block_ids == [[10]]
    assert len(metadata.matkv_retrieves) == 1
    assert metadata.matkv_retrieves[0].matches == (shifted,)
    assert metadata.matkv_retrieves[0].op.block_ids == [[10, 11]]
    assert tracker.state == LMCacheMPRequestState.READY


def test_worker_dispatches_dense_and_matkv_retrieves() -> None:
    """Worker sends prefix and shifted chunks through their respective paths."""
    # First Party
    from lmcache.integration.vllm.matkv_connector import MatKVConnector
    from lmcache.integration.vllm.matkv_metadata import (
        MatKVConnectorMetadata,
        MatKVRetrieveMetadata,
    )
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    prefix_op = LoadStoreOp(list(range(8)), [[10]], 0, 4)
    shifted_op = LoadStoreOp(list(range(8)), [[10, 11]], 0, 8)
    shifted = _match(old_st=0, cur_st=4)
    metadata = MatKVConnectorMetadata()
    metadata.add_request_metadata(
        LMCacheMPRequestMetadata("req", "RETRIEVE", prefix_op)
    )
    metadata.add_matkv_retrieve(
        MatKVRetrieveMetadata("req", shifted_op, (shifted,))
    )
    connector = MatKVConnector.__new__(MatKVConnector)
    connector._connector_metadata = metadata
    connector.worker_adapter = MagicMock()
    event = MagicMock()
    connector.worker_adapter.create_recorded_event.return_value = event

    connector.start_load_kv(None)

    connector.worker_adapter.batched_submit_retrieve_requests.assert_called_once()
    connector.worker_adapter.submit_matkv_retrieve_request.assert_called_once_with(
        "req",
        shifted_op,
        [shifted],
        event,
        cache_salt="",
        request_configs=None,
    )
