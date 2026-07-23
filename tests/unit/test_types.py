import pytest

import llm_exec_core
import llm_exec_core.types as types_module
from llm_exec_core.types import ExecutionMetadata, LLMResult, TokenUsage


def test_package_imports_with_version():
    import llm_exec_core

    assert llm_exec_core.__version__ == "0.4.1"


def test_llm_result_to_legacy_tuple_preserves_existing_usage_shape():
    result = LLMResult(
        text="hello",
        usage=TokenUsage(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            input_cost=0.1,
            output_cost=0.2,
            total_cost=0.3,
            currency="$",
        ),
        metadata=ExecutionMetadata(
            request_id="req-1",
            run_id="run-1",
            request_name="brief",
            model_name="test-model",
            model_id="provider-model-id",
            provider_name="test-provider",
            started_at="2026-06-18T00:00:00Z",
            finished_at="2026-06-18T00:00:01Z",
            duration_seconds=1.0,
            trace_context={"source": "unit"},
        ),
    )

    text, usage = result.to_legacy_tuple()

    assert text == "hello"
    assert usage["total_input_tokens"] == 1
    assert usage["total_output_tokens"] == 2
    assert usage["cost"]["total_cost"] == 0.3
    assert usage["process_times"]["total_time"] == 1.0
    assert usage["metadata"]["request_id"] == "req-1"


def test_token_usage_to_legacy_dict_includes_legacy_shape():
    usage = TokenUsage(
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
        input_cost=0.1,
        output_cost=0.2,
        total_cost=0.3,
        currency="$",
    )

    legacy = usage.to_legacy_dict(duration_seconds=1.25)

    assert legacy["total_input_tokens"] == 1
    assert legacy["total_output_tokens"] == 2
    assert legacy["cost"]["input_cost"] == 0.1
    assert legacy["cost"]["output_cost"] == 0.2
    assert legacy["cost"]["total_cost"] == 0.3
    assert legacy["process_times"]["total_time"] == 1.25
    assert legacy["process_times"]["request_times"] == []


def test_token_usage_to_legacy_dict_preserves_default_process_times_shape():
    result = LLMResult(
        text="hello",
        usage=TokenUsage(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            input_cost=0.1,
            output_cost=0.2,
            total_cost=0.3,
            currency="$",
        ),
        metadata=ExecutionMetadata(
            request_id="req-1",
            run_id="run-1",
            request_name="brief",
            model_name="test-model",
            model_id="provider-model-id",
            provider_name="test-provider",
            started_at="2026-06-18T00:00:00Z",
            finished_at="2026-06-18T00:00:01Z",
            duration_seconds=1.0,
            trace_context={"source": "unit"},
        ),
    )

    _, usage = result.to_legacy_tuple()

    assert usage["process_times"]["request_times"] == []


def test_llm_result_to_legacy_tuple_preserves_none_metadata_values():
    result = LLMResult(
        text="hello",
        usage=TokenUsage(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            input_cost=0.1,
            output_cost=0.2,
            total_cost=0.3,
            currency="$",
        ),
        metadata=ExecutionMetadata(
            request_id=None,
            run_id=None,
            request_name="brief",
            model_name="test-model",
            model_id="provider-model-id",
            provider_name="test-provider",
            started_at="2026-06-18T00:00:00Z",
            finished_at="2026-06-18T00:00:01Z",
            duration_seconds=1.0,
            trace_context={"source": "unit"},
        ),
    )

    _, usage = result.to_legacy_tuple()

    assert usage["metadata"]["request_id"] is None
    assert usage["metadata"]["run_id"] is None


def test_accounting_status_is_public_and_defaults_to_available():
    status = types_module.AccountingStatus()
    usage = TokenUsage(
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
        input_cost=0.1,
        output_cost=0.2,
        total_cost=0.3,
        currency="USD",
    )

    assert llm_exec_core.AccountingStatus is types_module.AccountingStatus
    assert status.tokens_available is True
    assert status.cost_available is True
    assert status.reason is None
    assert usage.accounting == status
    assert "accounting" not in usage.to_legacy_dict()


def test_unavailable_accounting_serializes_none_fields_and_finite_status():
    status = types_module.AccountingStatus(
        tokens_available=False,
        cost_available=False,
        reason="missing_usage",
    )
    usage = TokenUsage(
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        input_cost=None,
        output_cost=None,
        total_cost=None,
        currency=None,
        accounting=status,
    )

    assert usage.to_legacy_dict() == {
        "total_input_tokens": None,
        "total_output_tokens": None,
        "requests": [],
        "process_times": {"request_times": [], "total_time": 0.0},
        "cost": {
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
        },
        "accounting": {
            "tokens_available": False,
            "cost_available": False,
            "reason": "missing_usage",
        },
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "tokens_available": False,
            "cost_available": False,
            "reason": None,
        },
        {
            "tokens_available": True,
            "cost_available": True,
            "reason": "missing_usage",
        },
        {
            "tokens_available": False,
            "cost_available": False,
            "reason": "raw provider error text",
        },
    ],
)
def test_accounting_status_rejects_inconsistent_or_unbounded_reasons(
    arguments,
):
    with pytest.raises(ValueError):
        types_module.AccountingStatus(**arguments)
