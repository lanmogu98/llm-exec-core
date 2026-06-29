from llm_exec_core.types import ExecutionMetadata, LLMResult, TokenUsage


def test_package_imports_with_version():
    import llm_exec_core

    assert llm_exec_core.__version__ == "0.1.0"


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
