from contextlib import asynccontextmanager
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_exec_core import StructuredOutputValidationError
from llm_exec_core.client import LLMClient

BUNDLED_CONFIG_PATH = (
    Path(__file__).parents[2] / "src" / "llm_exec_core" / "llm_config.yml"
)


def _config(
    request_overrides=None,
    max_tokens=128,
    model_capabilities=None,
    provider_name="test-provider",
    api_base_url="https://example.invalid/chat/completions",
    api_key_env_var="TEST_API_KEY",
    model_name="test-model",
    provider_settings=None,
    model_settings=None,
):
    model_config = {
        "id": "provider-model-id",
        "pricing": {"input": 1.0, "output": 2.0},
    }
    if model_capabilities is not None:
        model_config["capabilities"] = model_capabilities
    if model_settings is not None:
        model_config.update(deepcopy(model_settings))

    provider_config = {
        "api_key_env_var": api_key_env_var,
        "api_base_url": api_base_url,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "context_window": 4096,
        "pricing_currency": "$",
        "models": {
            model_name: model_config,
        },
    }
    if request_overrides is not None:
        provider_config["request_overrides"] = request_overrides
    if provider_settings is not None:
        provider_config.update(deepcopy(provider_settings))
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


def test_client_resolves_model_request_policy_scalars_over_provider_defaults(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    provider_values = {
        "temperature": 0.2,
        "max_tokens": 256,
        "context_window": 8192,
        "output_token_field": "max_completion_tokens",
    }

    fallback_client = LLMClient(
        "test-model",
        config_source=_config(
            provider_settings=provider_values,
            model_settings={
                "temperature": None,
                "max_tokens": None,
                "context_window": None,
                "output_token_field": None,
            },
        ),
    )
    override_client = LLMClient(
        "test-model",
        config_source=_config(
            provider_settings=provider_values,
            model_settings={
                "temperature": 0.7,
                "max_tokens": 512,
                "context_window": 16384,
                "output_token_field": "max_tokens",
            },
        ),
    )

    assert (
        fallback_client.temperature,
        fallback_client.max_tokens,
        fallback_client.context_window,
        fallback_client.output_token_field,
    ) == (0.2, 256, 8192, "max_completion_tokens")
    assert (
        override_client.temperature,
        override_client.max_tokens,
        override_client.context_window,
        override_client.output_token_field,
    ) == (0.7, 512, 16384, "max_tokens")


def test_effective_request_defaults_are_isolated_from_catalog_and_clients(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    provider_overrides = {
        "provider_only": {"values": ["provider"]},
        "shared": {"values": ["provider"]},
    }
    model_overrides = {
        "model_only": {"values": ["model"]},
        "shared": {"values": ["model"]},
    }
    config = _config(
        request_overrides=provider_overrides,
        model_settings={"request_overrides": model_overrides},
    )

    first = LLMClient("test-model", config_source=config)
    first.request_overrides["provider_only"]["values"].append("mutated")
    first.request_overrides["model_only"]["values"].append("mutated")
    second = LLMClient("test-model", config_source=config)

    assert second.request_overrides == {
        "provider_only": {"values": ["provider"]},
        "shared": {"values": ["model"]},
        "model_only": {"values": ["model"]},
    }
    assert provider_overrides == {
        "provider_only": {"values": ["provider"]},
        "shared": {"values": ["provider"]},
    }
    assert model_overrides == {
        "model_only": {"values": ["model"]},
        "shared": {"values": ["model"]},
    }


@pytest.mark.asyncio
async def test_request_defaults_merge_normalized_provider_model_and_per_call(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    provider_overrides = {
        "top_p": 0.9,
        "provider_only": True,
        "cross_layer": "provider-direct",
        "extra_body": {
            "promoted": "provider-extra",
            "cross_layer": "provider-extra",
        },
    }
    model_overrides = {
        "top_p": 0.6,
        "model_only": True,
        "extra_body": {
            "promoted": "model-extra",
            "cross_layer": "model-extra",
        },
    }
    request_options = {
        "top_p": 0.2,
        "call_only": True,
        "extra_body": {"promoted": "call-extra"},
    }
    originals = deepcopy(
        (provider_overrides, model_overrides, request_options)
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                request_overrides=provider_overrides,
                model_settings={"request_overrides": model_overrides},
            ),
        )
        await client.generate("Hello", request_options=request_options)

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["provider_only"] is True
    assert payload["model_only"] is True
    assert payload["call_only"] is True
    assert payload["cross_layer"] == "model-extra"
    assert payload["promoted"] == "call-extra"
    assert payload["top_p"] == 0.2
    assert (provider_overrides, model_overrides, request_options) == originals


@pytest.mark.asyncio
async def test_stream_options_deep_merge_provider_model_and_per_call(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    provider_stream_options = {
        "provider_only": True,
        "nested": {"provider": 1, "shared": "provider"},
    }
    model_stream_options = {
        "model_only": True,
        "nested": {"model": 2, "shared": "model"},
    }
    request_stream_options = {
        "call_only": True,
        "nested": {"call": 3, "shared": "call"},
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                request_overrides={"stream_options": provider_stream_options},
                model_settings={
                    "request_overrides": {
                        "stream_options": model_stream_options
                    }
                },
            ),
        )
        await client.generate(
            "Hello",
            request_options={"stream_options": request_stream_options},
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["stream_options"] == {
        "provider_only": True,
        "model_only": True,
        "call_only": True,
        "nested": {
            "provider": 1,
            "model": 2,
            "call": 3,
            "shared": "call",
        },
    }
    assert provider_stream_options == {
        "provider_only": True,
        "nested": {"provider": 1, "shared": "provider"},
    }
    assert model_stream_options == {
        "model_only": True,
        "nested": {"model": 2, "shared": "model"},
    }
    assert request_stream_options == {
        "call_only": True,
        "nested": {"call": 3, "shared": "call"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_settings", "expected_field"),
    [
        ({}, "max_tokens"),
        (
            {
                "max_tokens": 321,
                "output_token_field": "max_completion_tokens",
            },
            "max_completion_tokens",
        ),
    ],
    ids=["legacy-max-tokens", "model-selected-completion-tokens"],
)
async def test_generated_token_limit_uses_effective_output_field(
    monkeypatch,
    model_settings,
    expected_field,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(model_settings=model_settings),
        )
        await client.generate("Hello")

    payload = mock_httpx_client.post.await_args.kwargs["json"]
    other_field = (
        "max_completion_tokens"
        if expected_field == "max_tokens"
        else "max_tokens"
    )

    assert payload[expected_field] == model_settings.get("max_tokens", 128)
    assert other_field not in payload
    assert client.output_token_field == expected_field


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_field", "generated_field"),
    [
        ("max_tokens", "max_completion_tokens"),
        ("max_completion_tokens", "max_tokens"),
    ],
)
async def test_explicit_token_limit_precedence_suppresses_generated_default(
    monkeypatch,
    token_field,
    generated_field,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                request_overrides={token_field: 111},
                provider_settings={"output_token_field": generated_field},
                model_settings={
                    "request_overrides": {token_field: 222},
                },
            ),
        )
        await client.generate("Hello", request_options={token_field: 333})

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload[token_field] == 333
    assert generated_field not in payload


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
async def test_dual_token_limit_fields_fail_before_http_after_normalization(
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
        with pytest.raises(
            ValueError,
            match="both max_tokens and max_completion_tokens",
        ):
            await client.generate(
                "Hello",
                request_options={"extra_body": {"max_tokens": 64}},
            )

    mock_cls.assert_not_called()
    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_request_defaults_keep_protected_field_validation(
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
                model_settings={
                    "request_overrides": {
                        "extra_body": {"model": "other-model"}
                    }
                }
            ),
        )
        with pytest.raises(ValueError, match="protected"):
            await client.generate("Hello")

    mock_cls.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token_field", ["max_tokens", "max_completion_tokens"]
)
async def test_retry_lowers_the_present_integer_token_limit_field(
    monkeypatch,
    caplog,
    token_field,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    caplog.set_level("WARNING", logger="llm_exec_core.client")
    request = httpx.Request("POST", "https://example.invalid/chat/completions")
    error_response = httpx.Response(
        404,
        text='{"error":"No allowed providers"}',
        request=request,
    )
    payloads = []

    async def post_side_effect(*args, **kwargs):
        payloads.append(deepcopy(kwargs["json"]))
        if len(payloads) == 1:
            return error_response
        return _success_response()

    with (
        patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls,
        patch("llm_exec_core.client.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = post_side_effect
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                max_tokens=1000,
                provider_settings={
                    "output_token_field": token_field,
                    "max_tokens_retry": {
                        "status_code": 404,
                        "body_contains": "no allowed providers",
                        "max_tokens_limit": 800,
                    },
                },
            ),
        )
        _disable_rate_limit(client)
        result = await client.generate("Hello")

    other_field = (
        "max_completion_tokens"
        if token_field == "max_tokens"
        else "max_tokens"
    )
    assert result.text == "ok"
    assert [payload[token_field] for payload in payloads] == [1000, 500]
    assert all(other_field not in payload for payload in payloads)
    assert f"lowering {token_field} 1000 -> 500" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit_value", "error_body"),
    [
        (1000, '{"error":"different failure"}'),
        ("1000", '{"error":"No allowed providers"}'),
    ],
    ids=["body-does-not-match", "limit-is-not-integer"],
)
async def test_retry_does_not_lower_nonmatching_or_non_integer_limit(
    monkeypatch,
    limit_value,
    error_body,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    request = httpx.Request("POST", "https://example.invalid/chat/completions")
    error_response = httpx.Response(
        404,
        text=error_body,
        request=request,
    )
    payloads = []

    async def post_side_effect(*args, **kwargs):
        payloads.append(deepcopy(kwargs["json"]))
        if len(payloads) == 1:
            return error_response
        return _success_response()

    with (
        patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls,
        patch("llm_exec_core.client.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = post_side_effect
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                provider_settings={
                    "max_tokens_retry": {
                        "status_code": 404,
                        "body_contains": "no allowed providers",
                        "max_tokens_limit": 800,
                    }
                }
            ),
        )
        _disable_rate_limit(client)
        result = await client.generate(
            "Hello",
            request_options={"max_completion_tokens": limit_value},
        )

    assert result.text == "ok"
    assert [payload["max_completion_tokens"] for payload in payloads] == [
        limit_value,
        limit_value,
    ]
    assert all("max_tokens" not in payload for payload in payloads)


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
@pytest.mark.parametrize(
    "capabilities",
    [None, _json_only_capabilities()],
    ids=["capabilities-absent", "reasoning-effort-supported"],
)
async def test_thinking_level_maps_to_reasoning_effort_when_allowed(
    monkeypatch,
    capabilities,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            thinking_level="high",
            config_source=_config(model_capabilities=capabilities),
        )
        await client.generate("Hello")

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert payload["reasoning_effort"] == "high"


def test_thinking_level_rejected_before_request_construction_when_unsupported(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        with pytest.raises(
            ValueError,
            match="thinking_level.*reasoning_effort",
        ):
            LLMClient(
                "test-model",
                thinking_level="high",
                config_source=_config(model_capabilities=_qwen_capabilities()),
            )

    mock_cls.assert_not_called()


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
        "validation_status": "client_validated",
        "capability_version": "unit-strict-2026-07-02",
        "hook_applied": False,
    }
    assert result.structured == {"title": "A"}


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
        "validation_status": "client_validated",
        "capability_version": "unit-json-only-2026-07-02",
        "hook_applied": True,
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
        "validation_status": "client_validated",
        "capability_version": "unit-qwen-2026-07-02",
        "hook_applied": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capabilities_factory", "invalid_content", "expected_cause"),
    [
        (_json_only_capabilities, "not-json", "JSONDecodeError"),
        (
            _json_only_capabilities,
            '{"count":"wrong"}',
            "ValidationError",
        ),
        (_qwen_capabilities, "not-json", "JSONDecodeError"),
        (_qwen_capabilities, '{"count":"wrong"}', "ValidationError"),
    ],
    ids=[
        "json-object-invalid-json",
        "json-object-schema-invalid",
        "prompt-json-invalid-json",
        "prompt-json-schema-invalid",
    ],
)
async def test_structured_output_validation_failure_does_not_populate_cache(
    monkeypatch,
    capabilities_factory,
    invalid_content,
    expected_cause,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": False,
    }
    structured_output = {"schema": schema, "mode": "prefer"}

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = [
            _success_response(invalid_content),
            _success_response('{"count":1}'),
        ]
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(model_capabilities=capabilities_factory()),
        )
        _disable_rate_limit(client)
        client._cache_enabled = True

        with pytest.raises(
            StructuredOutputValidationError,
            match="Structured output validation failed",
        ) as exc_info:
            await client.generate(
                "Extract the count.",
                structured_output=structured_output,
            )

        assert client.get_cache_stats()["size"] == 0

        result = await client.generate(
            "Extract the count.",
            structured_output=structured_output,
        )

    assert type(exc_info.value.__cause__).__name__ == expected_cause
    assert result.structured == {"count": 1}
    assert mock_httpx_client.post.await_count == 2
    assert client.get_cache_stats() == {
        "hits": 0,
        "misses": 2,
        "hit_rate": "0.0%",
        "size": 1,
        "max_size": 100,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capabilities_factory",
    [_json_only_capabilities, _qwen_capabilities],
    ids=["json-object", "prompt-json"],
)
@pytest.mark.parametrize(
    "constant",
    ["NaN", "Infinity", "-Infinity"],
    ids=["nan", "infinity", "negative-infinity"],
)
async def test_structured_output_rejects_non_finite_json_numbers(
    monkeypatch,
    capabilities_factory,
    constant,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    structured_output = {
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
        },
        "mode": "prefer",
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response(
            f'{{"value":{constant}}}'
        )
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(model_capabilities=capabilities_factory()),
        )
        _disable_rate_limit(client)
        client._cache_enabled = True

        with pytest.raises(StructuredOutputValidationError) as exc_info:
            await client.generate(
                "Extract the value.",
                structured_output=structured_output,
            )

    assert type(exc_info.value.__cause__).__name__ == "ValueError"
    assert mock_httpx_client.post.await_count == 1
    assert client.get_cache_stats()["size"] == 0


@pytest.mark.asyncio
async def test_structured_output_hook_runs_only_after_schema_validation(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    hook = MagicMock()

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response(
            '{"count":"wrong"}'
        )
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_json_only_capabilities()
            ),
        )
        _disable_rate_limit(client)
        client._cache_enabled = True

        with pytest.raises(StructuredOutputValidationError):
            await client.generate(
                "Extract the count.",
                structured_output={
                    "schema": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                    },
                    "mode": "prefer",
                },
                structured_output_hook=hook,
            )

    hook.assert_not_called()
    assert client.get_cache_stats()["size"] == 0


@pytest.mark.asyncio
async def test_structured_output_hook_failure_does_not_populate_cache(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    hook_calls = 0

    def transform(text):
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            raise RuntimeError("hook failed")
        return {"transformed": json.loads(text)["title"]}

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = [
            _success_response('{"title":"A"}'),
            _success_response('{"title":"A"}'),
        ]
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_json_only_capabilities()
            ),
        )
        _disable_rate_limit(client)
        client._cache_enabled = True

        with pytest.raises(RuntimeError, match="hook failed"):
            await client.generate(
                "Extract the title.",
                structured_output={"schema": schema, "mode": "prefer"},
                structured_output_hook=transform,
            )

        result = await client.generate(
            "Extract the title.",
            structured_output={"schema": schema, "mode": "prefer"},
            structured_output_hook=transform,
        )

    assert result.structured == {"transformed": "A"}
    assert mock_httpx_client.post.await_count == 2
    assert client.get_cache_stats()["hits"] == 0
    assert client.get_cache_stats()["misses"] == 2
    assert result.metadata.planning["validation_status"] == "client_validated"
    assert result.metadata.planning["hook_applied"] is True


@pytest.mark.asyncio
async def test_structured_output_is_parsed_on_network_and_cache_hit(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    structured_output = {
        "schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
        "mode": "prefer",
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
                model_capabilities=_json_only_capabilities()
            ),
        )
        _disable_rate_limit(client)
        client._cache_enabled = True

        network_result = await client.generate(
            "Extract the title.",
            structured_output=structured_output,
        )
        cached_result = await client.generate(
            "Extract the title.",
            structured_output=structured_output,
        )

    assert network_result.structured == {"title": "A"}
    assert cached_result.structured == {"title": "A"}
    assert mock_httpx_client.post.await_count == 1
    assert client.get_cache_stats()["hits"] == 1


@pytest.mark.asyncio
async def test_cached_structured_response_is_parsed_and_validated_again(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    structured_output = {"schema": schema, "mode": "prefer"}

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response(
            '{"title":"network"}'
        )
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=_json_only_capabilities()
            ),
        )
        _disable_rate_limit(client)
        client._cache_enabled = True

        network_result = await client.generate(
            "Extract the title.",
            structured_output=structured_output,
        )
        data, planning = client._build_request_plan(
            "Extract the title.",
            False,
            None,
            structured_output,
        )
        client._cache.set(
            client._build_cache_payload(data, planning),
            "not-json",
        )

        with pytest.raises(StructuredOutputValidationError) as exc_info:
            await client.generate(
                "Extract the title.",
                structured_output=structured_output,
            )

        recovered_result = await client.generate(
            "Extract the title.",
            structured_output=structured_output,
        )

    assert network_result.structured == {"title": "network"}
    assert recovered_result.structured == {"title": "network"}
    assert type(exc_info.value.__cause__).__name__ == "JSONDecodeError"
    assert mock_httpx_client.post.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [
        "gemini-3-flash",
        "gemini-3.1-flash-lite",
    ],
)
@pytest.mark.parametrize(
    "request_options",
    [
        {
            "reasoning_effort": "high",
            "extra_body": {
                "google": {"thinking_config": {"thinking_budget": 1024}}
            },
        },
        {
            "reasoning_effort": "high",
            "extra_body": {
                "extra_body": {
                    "google": {"thinking_config": {"thinking_level": "low"}}
                }
            },
        },
    ],
    ids=["google-thinking-config", "extra-body-google-thinking-config"],
)
async def test_gemini_rejects_overlapping_thinking_controls_before_http(
    monkeypatch,
    model_name,
    request_options,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_FT_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            model_name,
            config_source=_config(
                provider_name="gemini",
                api_key_env_var="GEMINI_API_KEY",
                model_name=model_name,
            ),
        )
        with pytest.raises(
            ValueError,
            match="reasoning_effort cannot be combined",
        ):
            await client.generate(
                "Hello",
                request_options=request_options,
            )

    mock_httpx_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_gemini_include_thoughts_alone_is_not_a_conflicting_control(
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    request_options = {
        "reasoning_effort": "high",
        "extra_body": {
            "google": {"thinking_config": {"include_thoughts": True}}
        },
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "gemini-3-flash",
            config_source=_config(
                provider_name="gemini",
                api_key_env_var="GEMINI_API_KEY",
                model_name="gemini-3-flash",
            ),
        )
        _disable_rate_limit(client)
        await client.generate("Hello", request_options=request_options)

    payload = mock_httpx_client.post.await_args.kwargs["json"]
    assert payload["reasoning_effort"] == "high"
    assert payload["google"]["thinking_config"] == {"include_thoughts": True}


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
@pytest.mark.parametrize(
    ("model_name", "api_key_env_var", "supports_temperature"),
    [
        ("gpt-5.5-or", "OPENAI_API_KEY_OPENROUTER", False),
        ("claude-sonnet-5-or", "ANTHROPIC_API_KEY_OPENROUTER", False),
        ("claude-opus-4.8-or", "ANTHROPIC_API_KEY_OPENROUTER", True),
    ],
)
async def test_openrouter_target_models_plan_temperature_from_snapshot(
    monkeypatch,
    model_name,
    api_key_env_var,
    supports_temperature,
):
    monkeypatch.setenv(api_key_env_var, "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(model_name, config_source=BUNDLED_CONFIG_PATH)
        assert client.capabilities is not None
        assert (
            "temperature"
            in client.capabilities.openrouter_supported_parameters
        ) is supports_temperature
        await client.generate("Hello")

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert ("temperature" in payload) is supports_temperature


@pytest.mark.asyncio
async def test_openrouter_core_max_tokens_uses_supported_completion_tokens(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    capabilities = _strict_capabilities()
    if "max_tokens" in capabilities["openrouter_supported_parameters"]:
        capabilities["openrouter_supported_parameters"].remove("max_tokens")
    capabilities["openrouter_supported_parameters"].append(
        "max_completion_tokens"
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=capabilities,
                provider_name="openrouter-test",
                api_base_url="https://openrouter.ai/api/v1/chat/completions",
            ),
        )
        await client.generate("Hello")

    payload = mock_httpx_client.post.await_args.kwargs["json"]

    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 128


@pytest.mark.asyncio
async def test_openrouter_explicit_unsupported_temperature_fails(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    capabilities = _strict_capabilities()

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_config(
                model_capabilities=capabilities,
                provider_name="openrouter-test",
                api_base_url="https://openrouter.ai/api/v1/chat/completions",
            ),
        )
        with pytest.raises(ValueError, match="does not support temperature"):
            await client.generate(
                "Hello",
                request_options={"temperature": 0.2},
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
            _success_response("{}"),
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
    assert planned.text == "{}"
    assert planned.structured == {}
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
