from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_exec_core.client import LLMClient


def _config(
    request_overrides=None,
    max_tokens=128,
    model_capabilities=None,
    provider_name="test-provider",
    api_base_url="https://example.invalid/chat/completions",
):
    model_config = {
        "id": "provider-model-id",
        "pricing": {"input": 1.0, "output": 2.0},
    }
    if model_capabilities is not None:
        model_config["capabilities"] = model_capabilities

    provider_config = {
        "api_key_env_var": "TEST_API_KEY",
        "api_base_url": api_base_url,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "context_window": 4096,
        "pricing_currency": "$",
        "models": {
            "test-model": model_config,
        },
    }
    if request_overrides is not None:
        provider_config["request_overrides"] = request_overrides
    return {provider_name: provider_config}


def _success_response(content="ok"):
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    response.raise_for_status = MagicMock()
    return response


def _disable_rate_limit(client):
    client._min_interval = 0


def _strict_capabilities():
    return {
        "version": "unit-strict-2026-07-02",
        "source": "unit",
        "source_date": "2026-07-02",
        "strict_response_schema": True,
        "json_object_response": True,
        "tools": True,
        "tool_streaming": True,
        "tool_choice": True,
        "parallel_tool_calls": True,
        "reasoning_controls": [
            "reasoning",
            "include_reasoning",
            "reasoning_effort",
        ],
        "openrouter_supported_parameters": [
            "response_format",
            "structured_outputs",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "include_reasoning",
            "reasoning_effort",
        ],
    }


def _json_only_capabilities():
    return {
        "version": "unit-json-only-2026-07-02",
        "source": "unit",
        "source_date": "2026-07-02",
        "strict_response_schema": False,
        "json_object_response": True,
        "tools": True,
        "tool_streaming": True,
        "tool_choice": True,
        "reasoning_controls": ["thinking", "reasoning_effort"],
    }


def _qwen_capabilities():
    return {
        "version": "unit-qwen-2026-07-02",
        "source": "unit",
        "source_date": "2026-07-02",
        "strict_response_schema": False,
        "json_object_response": False,
        "tools": True,
        "tool_streaming": False,
        "tool_choice": True,
        "reasoning_controls": [],
    }


def _qwen_no_tools_capabilities():
    capabilities = _qwen_capabilities()
    capabilities["tools"] = False
    capabilities["tool_choice"] = False
    return capabilities


@pytest.mark.asyncio
async def test_generate_merges_request_options_into_payload_without_mutation(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    provider_overrides = {
        "top_p": 0.7,
        "provider": {"only": ["z-ai"], "allow_fallbacks": False},
        "extra_body": {
            "enable_search": False,
            "provider_flag": "from-config",
        },
    }
    request_options = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {"type": "object"},
            },
        },
        "top_p": 0.2,
        "extra_body": {
            "enable_search": True,
            "top_p": 0.3,
            "extra_body": {
                "google": {"thinking_config": {"thinking_budget": 1024}}
            },
        },
    }
    original_provider_overrides = deepcopy(provider_overrides)
    original_request_options = deepcopy(request_options)

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(request_overrides=provider_overrides),
        )
        await client.generate("Hello", request_options=request_options)

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload == {
        "model": "provider-model-id",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.1,
        "max_tokens": 128,
        "stream": False,
        "provider": {"only": ["z-ai"], "allow_fallbacks": False},
        "provider_flag": "from-config",
        "enable_search": True,
        "response_format": request_options["response_format"],
        "top_p": 0.2,
        "extra_body": {
            "google": {"thinking_config": {"thinking_budget": 1024}}
        },
    }
    assert provider_overrides == original_provider_overrides
    assert request_options == original_request_options


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_options",
    [
        {"model": "other-model"},
        {"messages": [{"role": "user", "content": "override"}]},
        {"stream": True},
        {"extra_body": {"model": "other-model"}},
        {"extra_body": {"messages": []}},
        {"extra_body": {"stream": True}},
    ],
)
async def test_generate_rejects_protected_request_option_fields(
    monkeypatch,
    request_options,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient("test-model", config_source=_config())

    with pytest.raises(ValueError, match="protected request option"):
        await client.generate("Hello", request_options=request_options)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_overrides",
    [
        {"model": "other-model"},
        {"messages": [{"role": "user", "content": "override"}]},
        {"stream": True},
        {"extra_body": {"model": "other-model"}},
        {"extra_body": {"messages": []}},
        {"extra_body": {"stream": True}},
    ],
)
async def test_generate_rejects_protected_provider_request_overrides(
    monkeypatch,
    request_overrides,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_config(request_overrides=request_overrides),
    )

    with pytest.raises(
        ValueError, match="protected provider request override"
    ):
        await client.generate("Hello")


@pytest.mark.asyncio
async def test_max_completion_tokens_suppresses_core_default_max_tokens(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config(max_tokens=128))
        await client.generate(
            "Hello",
            request_options={"max_completion_tokens": 512},
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["max_completion_tokens"] == 512
    assert "max_tokens" not in payload


@pytest.mark.asyncio
async def test_explicit_max_tokens_survives_with_max_completion_tokens(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                request_overrides={"max_completion_tokens": 1024}
            ),
        )
        await client.generate("Hello", request_options={"max_tokens": 64})

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["max_completion_tokens"] == 1024
    assert payload["max_tokens"] == 64


@pytest.mark.asyncio
async def test_generate_response_accepts_request_options(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        response, _ = await client.generate_response(
            "Hello",
            request_options={"top_p": 0.4},
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert response == "ok"
    assert payload["top_p"] == 0.4


@pytest.mark.asyncio
async def test_non_streaming_without_stream_options_omits_stream_options(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        await client.generate("Hello")

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert "stream_options" not in payload


@pytest.mark.asyncio
async def test_generate_passes_reasoning_and_tool_controls(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        await client.generate(
            "Hello",
            request_options={
                "reasoning_effort": "high",
                "tools": [tool],
                "tool_choice": "auto",
            },
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["reasoning_effort"] == "high"
    assert payload["tools"] == [tool]
    assert payload["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_capability_metadata_validates_reasoning_controls(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_json_only_capabilities()
            ),
        )
        await client.generate(
            "Hello",
            request_options={"reasoning_effort": "high"},
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_capability_metadata_rejects_unknown_reasoning_controls(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(model_capabilities=_qwen_capabilities()),
        )
        with pytest.raises(
            ValueError,
            match="does not support reasoning_effort",
        ):
            await client.generate(
                "Hello",
                request_options={"reasoning_effort": "high"},
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_capability_metadata_rejects_unsupported_verbosity(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(model_capabilities=_strict_capabilities()),
        )
        with pytest.raises(ValueError, match="does not support verbosity"):
            await client.generate(
                "Hello",
                request_options={"verbosity": "high"},
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_capability_metadata_allows_supported_verbosity(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    capabilities = _strict_capabilities()
    capabilities["reasoning_controls"].append("verbosity")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(model_capabilities=capabilities),
        )
        await client.generate(
            "Hello",
            request_options={"verbosity": "high"},
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["verbosity"] == "high"


@pytest.mark.asyncio
async def test_structured_output_require_uses_strict_schema_and_metadata(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response(
            '{"title":"A"}'
        )
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_strict_capabilities(),
                provider_name="openrouter-test",
                api_base_url="https://openrouter.ai/api/v1/chat/completions",
            ),
        )
        result = await client.generate(
            "Hello",
            structured_output={
                "schema": schema,
                "name": "title_result",
                "mode": "require",
            },
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "title_result",
            "strict": True,
            "schema": schema,
        },
    }
    assert payload["provider"]["require_parameters"] is True
    assert result.metadata.planning == {
        "strategy": "strict_schema",
        "fallback_reason": None,
        "validation_status": "provider_enforced",
        "capability_version": "unit-strict-2026-07-02",
    }


@pytest.mark.asyncio
async def test_structured_output_require_fails_on_json_only_route(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_json_only_capabilities()
            ),
        )
        with pytest.raises(ValueError, match="does not support strict schema"):
            await client.generate(
                "Hello",
                structured_output={
                    "schema": {"type": "object"},
                    "mode": "require",
                },
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_structured_output_prefer_falls_back_to_json_object(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response(
            '{"title":"A"}'
        )
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_json_only_capabilities()
            ),
        )
        result = await client.generate(
            "Extract the title.",
            structured_output={"schema": schema, "mode": "prefer"},
            structured_output_hook=lambda text: {"parsed": text},
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["response_format"] == {"type": "json_object"}
    assert "Respond with valid JSON" in payload["messages"][0]["content"]
    assert '"title"' in payload["messages"][0]["content"]
    assert result.structured == {"parsed": '{"title":"A"}'}
    assert result.metadata.planning == {
        "strategy": "json_object",
        "fallback_reason": "strict_schema_not_supported",
        "validation_status": "hook_validated",
        "capability_version": "unit-json-only-2026-07-02",
    }


@pytest.mark.asyncio
async def test_generate_response_structured_output_prefer_uses_prompt_only(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response(
            '{"title":"A"}'
        )
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(model_capabilities=_qwen_capabilities()),
        )
        response, usage = await client.generate_response(
            "Extract the title.",
            structured_output={"schema": schema, "mode": "prefer"},
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert response == '{"title":"A"}'
    assert "response_format" not in payload
    assert "Respond with valid JSON" in payload["messages"][0]["content"]
    assert usage["metadata"]["planning"] == {
        "strategy": "prompt_json",
        "fallback_reason": "response_format_not_supported",
        "validation_status": "not_validated",
        "capability_version": "unit-qwen-2026-07-02",
    }


@pytest.mark.asyncio
async def test_structured_output_off_leaves_raw_payload_unchanged(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        await client.generate(
            "Hello",
            request_options={"top_p": 0.2},
            structured_output={"mode": "off"},
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["top_p"] == 0.2
    assert "structured_output" not in payload


@pytest.mark.asyncio
async def test_structured_output_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient("test-model", config_source=_config())

    with pytest.raises(ValueError, match="structured_output.mode"):
        await client.generate(
            "Hello",
            structured_output={"mode": "silent"},
        )


@pytest.mark.asyncio
async def test_qwen_tools_with_streaming_is_rejected(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(model_capabilities=_qwen_capabilities()),
        )
        with pytest.raises(ValueError, match="tools with stream=True"):
            await client.generate_response(
                "Hello",
                stream=True,
                request_options={"tools": [tool]},
            )

    mock_httpx_client.stream.assert_not_called()


@pytest.mark.asyncio
async def test_qwen_flash_tools_without_documented_support_is_rejected(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_qwen_no_tools_capabilities()
            ),
        )
        with pytest.raises(ValueError, match="does not support tools"):
            await client.generate(
                "Hello",
                request_options={"tools": [tool]},
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_strict_response_format_rejected_on_json_only_route(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_json_only_capabilities()
            ),
        )
        with pytest.raises(ValueError, match="strict schema response_format"):
            await client.generate(
                "Hello",
                request_options={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "schema": {"type": "object"},
                        },
                    }
                },
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_response_format_rejected_when_not_documented(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(model_capabilities=_qwen_capabilities()),
        )
        with pytest.raises(ValueError, match="json_object response_format"):
            await client.generate(
                "Hello",
                request_options={"response_format": {"type": "json_object"}},
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_openrouter_raw_structured_output_requires_parameters(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    schema = {"type": "object"}

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                request_overrides={"provider": {"only": ["openai"]}},
                model_capabilities=_strict_capabilities(),
                provider_name="openrouter-test",
                api_base_url="https://openrouter.ai/api/v1/chat/completions",
            ),
        )
        await client.generate(
            "Hello",
            request_options={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": schema},
                }
            },
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["provider"] == {
        "only": ["openai"],
        "require_parameters": True,
    }


@pytest.mark.asyncio
async def test_openrouter_tools_require_parameters(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                request_overrides={"provider": {"only": ["openai"]}},
                model_capabilities=_strict_capabilities(),
                provider_name="openrouter-test",
                api_base_url="https://openrouter.ai/api/v1/chat/completions",
            ),
        )
        await client.generate(
            "Hello",
            request_options={"tools": [tool]},
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["provider"] == {
        "only": ["openai"],
        "require_parameters": True,
    }


@pytest.mark.asyncio
async def test_openrouter_tools_fail_without_supported_parameter(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    capabilities = _strict_capabilities()
    capabilities["openrouter_supported_parameters"].remove("tools")
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=capabilities,
                provider_name="openrouter-test",
                api_base_url="https://openrouter.ai/api/v1/chat/completions",
            ),
        )
        with pytest.raises(
            ValueError,
            match="does not support tools",
        ):
            await client.generate(
                "Hello",
                request_options={"tools": [tool]},
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_openrouter_strict_schema_requires_structured_outputs_metadata(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    capabilities = _strict_capabilities()
    capabilities["openrouter_supported_parameters"].remove(
        "structured_outputs"
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=capabilities,
                provider_name="openrouter-test",
                api_base_url="https://openrouter.ai/api/v1/chat/completions",
            ),
        )
        with pytest.raises(
            ValueError,
            match="does not support structured_outputs",
        ):
            await client.generate(
                "Hello",
                structured_output={
                    "schema": {"type": "object"},
                    "mode": "require",
                },
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_openrouter_strict_schema_fails_without_supported_parameters(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    capabilities = _strict_capabilities()
    capabilities["openrouter_supported_parameters"] = []

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=capabilities,
                provider_name="openrouter-test",
                api_base_url="https://openrouter.ai/api/v1/chat/completions",
            ),
        )
        with pytest.raises(
            ValueError,
            match="does not support response_format",
        ):
            await client.generate(
                "Hello",
                structured_output={
                    "schema": {"type": "object"},
                    "mode": "require",
                },
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_structured_output_strategy_separates_cache_from_raw_payload(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    schema = {"type": "object"}
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": schema,
        },
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = [
            _success_response("raw"),
            _success_response("planned"),
        ]
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_strict_capabilities(),
                provider_name="openrouter-test",
                api_base_url="https://openrouter.ai/api/v1/chat/completions",
            ),
        )
        _disable_rate_limit(client)
        client._cache_enabled = True

        raw = await client.generate(
            "Hello",
            request_options={"response_format": response_format},
        )
        planned = await client.generate(
            "Hello",
            structured_output={
                "schema": schema,
                "name": "answer",
                "mode": "require",
            },
        )

    assert raw.text == "raw"
    assert planned.text == "planned"
    assert mock_httpx_client.post.await_count == 2
    assert client.get_cache_stats()["misses"] == 2


@pytest.mark.asyncio
async def test_cache_key_includes_request_options(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = [
            _success_response("schema-a"),
            _success_response("schema-b"),
        ]
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        _disable_rate_limit(client)
        client._cache_enabled = True

        first = await client.generate(
            "Hello",
            request_options={"response_format": {"type": "json_object"}},
        )
        second = await client.generate(
            "Hello",
            request_options={"top_p": 0.2},
        )

    assert first.text == "schema-a"
    assert second.text == "schema-b"
    assert mock_httpx_client.post.await_count == 2
    assert client.get_cache_stats()["misses"] == 2


@pytest.mark.asyncio
async def test_cache_key_includes_reasoning_effort(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = [
            _success_response("low-reasoning"),
            _success_response("high-reasoning"),
        ]
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        _disable_rate_limit(client)
        client._cache_enabled = True

        first = await client.generate(
            "Hello",
            request_options={"reasoning_effort": "low"},
        )
        second = await client.generate(
            "Hello",
            request_options={"reasoning_effort": "high"},
        )

    assert first.text == "low-reasoning"
    assert second.text == "high-reasoning"
    assert mock_httpx_client.post.await_count == 2
    assert client.get_cache_stats()["misses"] == 2


@pytest.mark.asyncio
async def test_cache_key_includes_openrouter_provider_routing(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = [
            _success_response("openai-route"),
            _success_response("anthropic-route"),
        ]
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        _disable_rate_limit(client)
        client._cache_enabled = True

        first = await client.generate(
            "Hello",
            request_options={"provider": {"only": ["openai"]}},
        )
        second = await client.generate(
            "Hello",
            request_options={"provider": {"only": ["anthropic"]}},
        )

    assert first.text == "openai-route"
    assert second.text == "anthropic-route"
    assert mock_httpx_client.post.await_count == 2
    assert client.get_cache_stats()["misses"] == 2


@pytest.mark.asyncio
async def test_cache_key_uses_normalized_extra_body(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response("cached")
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        _disable_rate_limit(client)
        client._cache_enabled = True

        first = await client.generate(
            "Hello",
            request_options={"extra_body": {"top_p": 0.2}},
        )
        second = await client.generate(
            "Hello",
            request_options={"top_p": 0.2},
        )

    assert first.text == "cached"
    assert second.text == "cached"
    assert mock_httpx_client.post.await_count == 1
    assert client.get_cache_stats()["hits"] == 1


@pytest.mark.asyncio
async def test_cache_key_preserves_nested_raw_extra_body(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = [
            _success_response("budget-64"),
            _success_response("budget-128"),
        ]
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        _disable_rate_limit(client)
        client._cache_enabled = True

        first = await client.generate(
            "Hello",
            request_options={
                "extra_body": {
                    "extra_body": {
                        "google": {"thinking_config": {"thinking_budget": 64}}
                    }
                }
            },
        )
        second = await client.generate(
            "Hello",
            request_options={
                "extra_body": {
                    "extra_body": {
                        "google": {"thinking_config": {"thinking_budget": 128}}
                    }
                }
            },
        )

    assert first.text == "budget-64"
    assert second.text == "budget-128"
    assert mock_httpx_client.post.await_count == 2
    assert client.get_cache_stats()["misses"] == 2


@pytest.mark.asyncio
async def test_cache_key_excludes_non_request_metadata(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response("cached")
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        _disable_rate_limit(client)
        client._cache_enabled = True

        first = await client.generate(
            "Hello",
            request_name="first",
            request_id="req-1",
            run_id="run-1",
            trace_context={"attempt": 1},
            request_options={"top_p": 0.2},
        )
        second = await client.generate(
            "Hello",
            request_name="second",
            request_id="req-2",
            run_id="run-2",
            trace_context={"attempt": 2},
            request_options={"top_p": 0.2},
        )

    assert first.text == "cached"
    assert second.text == "cached"
    assert mock_httpx_client.post.await_count == 1
    assert client.get_cache_stats()["hits"] == 1


@pytest.mark.asyncio
async def test_streaming_without_request_options_adds_usage_stream_options(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    payloads = []

    async def mock_lines():
        yield 'data: {"choices": [{"delta": {"content": "A"}}]}'
        yield 'data: {"usage": {"prompt_tokens": 5, "completion_tokens": 6}}'
        yield "data: [DONE]"

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        payloads.append(kwargs["json"])
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_lines = mock_lines
        yield response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        _disable_rate_limit(client)
        client._cache_enabled = True

        response = await client.generate_response("Hello", stream=True)

    assert response[0] == "A"
    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert client.get_cache_stats()["hits"] == 0
    assert client.get_cache_stats()["misses"] == 0


@pytest.mark.asyncio
async def test_streaming_merges_stream_options_and_bypasses_cache(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    payloads = []

    async def mock_lines():
        yield 'data: {"choices": [{"delta": {"content": "A"}}]}'
        yield 'data: {"usage": {"prompt_tokens": 5, "completion_tokens": 6}}'
        yield "data: [DONE]"

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        payloads.append(kwargs["json"])
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_lines = mock_lines
        yield response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                request_overrides={
                    "stream_options": {
                        "provider_trace": True,
                        "include_usage": True,
                        "nested": {
                            "provider_value": "kept",
                            "override_value": "provider",
                        },
                    }
                }
            ),
        )
        _disable_rate_limit(client)
        client._cache_enabled = True

        first = await client.generate_response(
            "Hello",
            stream=True,
            request_options={
                "stream_options": {
                    "include_usage": False,
                    "chunk_size": 1,
                    "nested": {
                        "override_value": "request",
                        "request_value": "added",
                    },
                }
            },
        )
        second = await client.generate_response(
            "Hello",
            stream=True,
            request_options={
                "stream_options": {
                    "include_usage": False,
                    "chunk_size": 1,
                    "nested": {
                        "override_value": "request",
                        "request_value": "added",
                    },
                }
            },
        )

    assert first[0] == "A"
    assert second[0] == "A"
    assert len(payloads) == 2
    assert payloads[0]["stream_options"] == {
        "provider_trace": True,
        "include_usage": False,
        "chunk_size": 1,
        "nested": {
            "provider_value": "kept",
            "override_value": "request",
            "request_value": "added",
        },
    }
    assert client.get_cache_stats()["hits"] == 0
    assert client.get_cache_stats()["misses"] == 0


@pytest.mark.asyncio
async def test_streaming_adds_usage_to_mapping_stream_options_when_absent(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    payloads = []

    async def mock_lines():
        yield 'data: {"choices": [{"delta": {"content": "A"}}]}'
        yield 'data: {"usage": {"prompt_tokens": 5, "completion_tokens": 6}}'
        yield "data: [DONE]"

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        payloads.append(kwargs["json"])
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_lines = mock_lines
        yield response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=_config())
        _disable_rate_limit(client)

        await client.generate_response(
            "Hello",
            stream=True,
            request_options={"stream_options": {"chunk_size": 1}},
        )

    assert payloads[0]["stream_options"] == {
        "chunk_size": 1,
        "include_usage": True,
    }


@pytest.mark.asyncio
async def test_streaming_stream_options_none_suppresses_usage_default(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    payloads = []

    async def mock_lines():
        yield 'data: {"choices": [{"delta": {"content": "A"}}]}'
        yield "data: [DONE]"

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        payloads.append(kwargs["json"])
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_lines = mock_lines
        yield response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                request_overrides={
                    "stream_options": {
                        "include_usage": True,
                        "provider_trace": True,
                    }
                }
            ),
        )
        _disable_rate_limit(client)
        await client.generate_response(
            "Hello",
            stream=True,
            request_options={"stream_options": None},
        )

    assert payloads[0]["stream_options"] is None


@pytest.mark.asyncio
async def test_request_stream_options_none_overrides_provider_options(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                request_overrides={
                    "stream_options": {
                        "include_usage": True,
                        "provider_trace": True,
                    }
                }
            ),
        )
        await client.generate(
            "Hello", request_options={"stream_options": None}
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["stream_options"] is None
