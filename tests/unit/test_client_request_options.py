from contextlib import asynccontextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import llm_exec_core.config as config_module
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


def _supported_rule(
    minimum=0.0,
    maximum=1.0,
    minimum_inclusive=True,
    maximum_inclusive=True,
):
    return {
        "state": "supported",
        "range": {
            "minimum": minimum,
            "maximum": maximum,
            "minimum_inclusive": minimum_inclusive,
            "maximum_inclusive": maximum_inclusive,
        },
    }


def _policy(
    *,
    availability="optional",
    allowed_modes=None,
    default_mode=None,
    can_disable=None,
    mode_path=("thinking", "type"),
    mode_values=None,
    effort=True,
    effort_path=("reasoning", "effort"),
    effort_omission="provider-selected",
    effort_default=None,
    budget=True,
    budget_path=("reasoning", "budget_tokens"),
    budget_omission="provider-selected",
    budget_default=None,
    budget_minimum=0,
    budget_maximum=8192,
    output_limit_relation="none",
    allow_effort_with_budget=True,
    temperature=None,
    top_p=None,
):
    if allowed_modes is None:
        allowed_modes = {
            "unavailable": ("disabled",),
            "optional": ("disabled", "enabled"),
            "adaptive": ("disabled", "adaptive"),
            "always-on": ("always-on",),
        }[availability]
    if default_mode is None:
        default_mode = {
            "unavailable": "disabled",
            "optional": "disabled",
            "adaptive": "adaptive",
            "always-on": "always-on",
        }[availability]
    if can_disable is None:
        can_disable = availability in {"optional", "adaptive"} and (
            "disabled" in allowed_modes
        )
    if mode_values is None:
        mode_values = {
            selectable_mode: selectable_mode
            for selectable_mode in allowed_modes
            if selectable_mode != "always-on"
        }
    active_modes = [
        mode
        for mode in allowed_modes
        if mode in {"enabled", "adaptive", "always-on"}
    ]
    reasoning = {
        "availability": availability,
        "allowed_modes": list(allowed_modes),
        "default_mode": default_mode,
        "can_disable": can_disable,
        "mode": (
            None
            if availability in {"unavailable", "always-on"}
            else {"path": list(mode_path), "values": deepcopy(mode_values)}
        ),
        "effort": None,
        "budget_tokens": None,
        "allow_effort_with_budget_in": [],
    }
    if effort:
        reasoning["effort"] = {
            "path": list(effort_path),
            "allowed_values": ["low", "high", "max"],
            "aliases": {"medium": "high", "xhigh": "max"},
            "modes": {
                mode: {
                    "omission": effort_omission,
                    "default": effort_default,
                }
                for mode in active_modes
            },
        }
    if budget:
        reasoning["budget_tokens"] = {
            "path": list(budget_path),
            "minimum": budget_minimum,
            "maximum": budget_maximum,
            "modes": {
                mode: {
                    "omission": budget_omission,
                    "default": budget_default,
                    "output_limit_relation": output_limit_relation,
                }
                for mode in active_modes
            },
        }
    if effort and budget and allow_effort_with_budget:
        reasoning["allow_effort_with_budget_in"] = active_modes
    if temperature is None:
        temperature = {"base": _supported_rule(0.0, 2.0, True, False)}
    if top_p is None:
        top_p = {"base": _supported_rule(0.0, 1.0, False, True)}
    return {
        "reasoning": reasoning,
        "sampling": {
            "temperature": deepcopy(temperature),
            "top_p": deepcopy(top_p),
        },
    }


def _policy_capabilities(policy=None, version="unit-policy-2026-07-24"):
    return {
        "version": version,
        "source": "https://example.invalid/generation-policy",
        "source_date": "2026-07-24",
        "reasoning_controls": [],
        "generation_policy": deepcopy(
            policy if policy is not None else _policy()
        ),
    }


def _policy_config(
    *,
    policy=None,
    version="unit-policy-2026-07-24",
    request_overrides=None,
    provider_settings=None,
    model_settings=None,
    provider_name="test-provider",
    model_name="test-model",
    api_base_url="https://example.invalid/chat/completions",
):
    return _config(
        request_overrides=request_overrides,
        model_capabilities=_policy_capabilities(policy, version),
        provider_settings=provider_settings,
        model_settings=model_settings,
        provider_name=provider_name,
        model_name=model_name,
        api_base_url=api_base_url,
    )


async def _capture_policy_request(
    config_source,
    *,
    client_kwargs=None,
    generate_kwargs=None,
    response_content="ok",
):
    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response(
            response_content
        )
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=config_source,
            **(client_kwargs or {}),
        )
        _disable_rate_limit(client)
        result = await client.generate(
            "Hello",
            **(generate_kwargs or {}),
        )
    payload = mock_httpx_client.post.await_args.kwargs["json"]
    return payload, client, mock_httpx_client, result


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


@pytest.mark.asyncio
async def test_policy_absent_omits_only_core_generated_unset_temperature(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _config(provider_settings={"temperature": None})

    payload, _, _, _ = await _capture_policy_request(
        config,
        generate_kwargs={
            "request_options": {
                "vendor_null": None,
                "temperature": None,
            }
        },
    )

    assert payload["temperature"] is None
    assert payload["vendor_null"] is None

    payload, _, _, _ = await _capture_policy_request(config)
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_generation_controls_require_a_generation_policy(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient("test-model", config_source=_config())
    client._cache_enabled = True
    client._cache.get = MagicMock()
    client._wait_for_rate_limit = AsyncMock()
    client._get_client = AsyncMock()

    with pytest.raises(ValueError, match="generation_policy"):
        await client.generate(
            "Hello",
            generation_controls={"sampling": {"temperature": 0.2}},
        )

    client._cache.get.assert_not_called()
    client._wait_for_rate_limit.assert_not_awaited()
    client._get_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_typed_generation_controls_are_legacy_safe(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    payload, _, _, _ = await _capture_policy_request(
        _config(),
        generate_kwargs={
            "generation_controls": config_module.GenerationControls()
        },
    )

    assert payload["temperature"] == 0.1


@pytest.mark.asyncio
async def test_generate_response_accepts_generation_controls_and_keeps_tuple(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response("legacy")
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=_policy_config(),
        )
        _disable_rate_limit(client)
        text, usage = await client.generate_response(
            "Hello",
            generation_controls={
                "reasoning": {"mode": "enabled", "effort": "medium"},
                "sampling": {"temperature": 0.25},
            },
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]
    assert text == "legacy"
    assert isinstance(usage, dict)
    assert payload["thinking"]["type"] == "enabled"
    assert payload["reasoning"]["effort"] == "high"
    assert payload["temperature"] == 0.25


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode_values", "raw_value", "expected_mode"),
    [
        ({"disabled": False, "enabled": True}, True, "enabled"),
        ({"disabled": "off", "enabled": "on"}, "on", "enabled"),
    ],
    ids=["boolean-wire", "string-wire"],
)
async def test_reasoning_mode_raw_values_decode_and_reencode_exactly(
    monkeypatch, mode_values, raw_value, expected_mode
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(mode_values=mode_values, effort=False, budget=False)

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        generate_kwargs={"request_options": {"thinking": {"type": raw_value}}},
    )

    assert payload["thinking"]["type"] == mode_values[expected_mode]


@pytest.mark.asyncio
async def test_reasoning_mode_wire_comparison_keeps_bool_distinct_from_int(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        mode_values={"disabled": False, "enabled": True},
        effort=False,
        budget=False,
    )

    with pytest.raises(ValueError, match="reasoning mode"):
        await LLMClient(
            "test-model",
            config_source=_policy_config(policy=policy),
        ).generate(
            "Hello",
            request_options={"thinking": {"type": 1}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("availability", "controls", "expected_mode_field"),
    [
        (
            "optional",
            {"reasoning": {"mode": "enabled"}},
            "enabled",
        ),
        (
            "adaptive",
            {},
            None,
        ),
        (
            "always-on",
            {},
            None,
        ),
        (
            "unavailable",
            {},
            None,
        ),
    ],
)
async def test_reasoning_availability_modes_plan_without_provider_branches(
    monkeypatch, availability, controls, expected_mode_field
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        availability=availability,
        effort=False,
        budget=False,
    )

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        generate_kwargs=(
            {"generation_controls": controls} if controls else None
        ),
    )

    if expected_mode_field is None:
        assert "thinking" not in payload
    else:
        assert payload["thinking"]["type"] == expected_mode_field


@pytest.mark.asyncio
async def test_adaptive_policy_can_expose_evidence_backed_manual_mode(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        availability="adaptive",
        allowed_modes=("disabled", "enabled", "adaptive"),
        default_mode="adaptive",
        effort=False,
        budget=False,
    )

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        generate_kwargs={
            "generation_controls": {"reasoning": {"mode": "enabled"}}
        },
    )

    assert payload["thinking"]["type"] == "enabled"


@pytest.mark.asyncio
async def test_always_on_rejects_caller_mode_and_unavailable_rejects_effort(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    always_client = LLMClient(
        "test-model",
        config_source=_policy_config(
            policy=_policy(
                availability="always-on",
                effort=True,
                budget=False,
            )
        ),
    )
    unavailable_client = LLMClient(
        "test-model",
        config_source=_policy_config(
            policy=_policy(
                availability="unavailable",
                effort=False,
                budget=False,
            )
        ),
    )

    with pytest.raises(ValueError, match="always-on"):
        await always_client.generate(
            "Hello",
            generation_controls={"reasoning": {"mode": "enabled"}},
        )
    with pytest.raises(ValueError, match="unavailable"):
        await unavailable_client.generate(
            "Hello",
            request_options={"reasoning_effort": "high"},
        )


@pytest.mark.asyncio
async def test_effort_aliases_and_thinking_level_use_declared_wire_path(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        budget=False,
        effort_path=("controls", "depth"),
    )

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        client_kwargs={"thinking_level": "medium"},
    )
    assert payload["controls"]["depth"] == "high"
    assert "reasoning_effort" not in payload

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        client_kwargs={"thinking_level": "low"},
        generate_kwargs={
            "generation_controls": {"reasoning": {"effort": "xhigh"}}
        },
    )
    assert payload["controls"]["depth"] == "max"


@pytest.mark.asyncio
async def test_per_call_effort_tombstone_clears_thinking_level(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(default_mode="enabled", budget=False)

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        client_kwargs={"thinking_level": "high"},
        generate_kwargs={
            "generation_controls": {"reasoning": {"effort": None}}
        },
    )

    assert "reasoning" not in payload


def test_policy_thinking_level_rejects_unsupported_or_invalid_effort(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with pytest.raises(ValueError, match="does not declare effort"):
        LLMClient(
            "test-model",
            thinking_level="high",
            config_source=_policy_config(
                policy=_policy(effort=False, budget=False)
            ),
        )
    with pytest.raises(ValueError, match="allowed"):
        LLMClient(
            "test-model",
            thinking_level="ultra",
            config_source=_policy_config(
                policy=_policy(default_mode="enabled", budget=False)
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("omission", "default", "should_fail"),
    [
        ("provider-default", "high", False),
        ("provider-selected", None, False),
        ("required", None, True),
    ],
)
async def test_effort_omission_modes_are_enforced(
    monkeypatch, omission, default, should_fail
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        effort_omission=omission,
        effort_default=default,
        budget=False,
    )
    client = LLMClient(
        "test-model",
        config_source=_policy_config(policy=policy),
    )
    _disable_rate_limit(client)

    if should_fail:
        with pytest.raises(ValueError, match="required"):
            await client.generate("Hello")
    else:
        payload, _, _, _ = await _capture_policy_request(
            _policy_config(policy=policy)
        )
        assert "reasoning" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_tokens", [0, 8192])
async def test_budget_inclusive_boundaries_are_sent(
    monkeypatch, budget_tokens
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        effort=False,
        budget_minimum=0,
        budget_maximum=8192,
    )

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        generate_kwargs={
            "generation_controls": {
                "reasoning": {"budget_tokens": budget_tokens}
            }
        },
    )

    assert payload["reasoning"]["budget_tokens"] == budget_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget_tokens", "match"),
    [
        (True, "plain integer"),
        (-1, "minimum"),
        (8193, "maximum"),
    ],
)
async def test_budget_rejects_bool_and_out_of_bounds_values(
    monkeypatch, budget_tokens, match
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_policy_config(
            policy=_policy(default_mode="enabled", effort=False)
        ),
    )

    with pytest.raises(ValueError, match=match):
        await client.generate(
            "Hello",
            request_options={"reasoning": {"budget_tokens": budget_tokens}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relation", "budget", "token_options", "should_pass", "match"),
    [
        ("none", 128, {}, True, None),
        ("less-than", 127, {"max_tokens": 128}, True, None),
        ("less-than", 128, {"max_tokens": 128}, False, "less than"),
        (
            "less-than-or-equal",
            128,
            {"max_completion_tokens": 128},
            True,
            None,
        ),
        (
            "less-than-or-equal",
            129,
            {"max_completion_tokens": 128},
            False,
            "less than or equal",
        ),
        (
            "less-than",
            10,
            {"max_tokens": None},
            False,
            "plain integer",
        ),
    ],
)
async def test_budget_output_limit_relations_use_final_token_field(
    monkeypatch, relation, budget, token_options, should_pass, match
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        effort=False,
        output_limit_relation=relation,
    )
    client = LLMClient(
        "test-model",
        config_source=_policy_config(policy=policy),
    )
    _disable_rate_limit(client)
    kwargs = {
        "request_options": {
            **token_options,
            "reasoning": {"budget_tokens": budget},
        }
    }

    if should_pass:
        with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
            mock_httpx_client = AsyncMock()
            mock_httpx_client.post.return_value = _success_response()
            mock_cls.return_value = mock_httpx_client
            await client.generate("Hello", **kwargs)
        assert mock_httpx_client.post.await_count == 1
    else:
        with pytest.raises(ValueError, match=match):
            await client.generate("Hello", **kwargs)


@pytest.mark.asyncio
async def test_budget_relation_rejects_missing_and_dual_output_limits(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        effort=False,
        output_limit_relation="less-than",
    )
    missing_limit_config = _policy_config(
        policy=policy,
        provider_settings={"max_tokens": None},
    )

    with pytest.raises(ValueError, match="token-limit"):
        await LLMClient(
            "test-model",
            config_source=missing_limit_config,
        ).generate(
            "Hello",
            generation_controls={"reasoning": {"budget_tokens": 10}},
        )

    with pytest.raises(ValueError, match="both max_tokens"):
        await LLMClient(
            "test-model",
            config_source=_policy_config(policy=policy),
        ).generate(
            "Hello",
            request_options={
                "max_tokens": 100,
                "max_completion_tokens": 100,
                "reasoning": {"budget_tokens": 10},
            },
        )


@pytest.mark.asyncio
async def test_effort_and_budget_pair_requires_declared_mode_permission(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    forbidden_policy = _policy(
        default_mode="enabled",
        allow_effort_with_budget=False,
    )
    allowed_policy = _policy(
        default_mode="enabled",
        allow_effort_with_budget=True,
    )
    controls = {
        "reasoning": {
            "effort": "high",
            "budget_tokens": 1024,
        }
    }

    with pytest.raises(ValueError, match="cannot be combined"):
        await LLMClient(
            "test-model",
            config_source=_policy_config(policy=forbidden_policy),
        ).generate("Hello", generation_controls=controls)

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=allowed_policy),
        generate_kwargs={"generation_controls": controls},
    )
    assert payload["reasoning"] == {
        "effort": "high",
        "budget_tokens": 1024,
    }


@pytest.mark.asyncio
async def test_explicit_disable_suppresses_lower_effort_and_budget(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _policy_config(
        policy=_policy(default_mode="enabled"),
        request_overrides={
            "thinking": {"type": "enabled"},
            "reasoning": {"effort": "high", "budget_tokens": 1024},
        },
    )

    payload, _, _, _ = await _capture_policy_request(
        config,
        generate_kwargs={
            "generation_controls": {"reasoning": {"mode": "disabled"}}
        },
    )

    assert payload["thinking"]["type"] == "disabled"
    assert "reasoning" not in payload


@pytest.mark.asyncio
async def test_same_layer_disable_with_effort_or_budget_is_contradictory(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_policy_config(policy=_policy(default_mode="enabled")),
    )

    with pytest.raises(ValueError, match="disabled"):
        await client.generate(
            "Hello",
            generation_controls={
                "reasoning": {
                    "mode": "disabled",
                    "effort": "high",
                }
            },
        )
    with pytest.raises(ValueError, match="disabled"):
        await client.generate(
            "Hello",
            request_options={
                "thinking": {"type": "disabled"},
                "reasoning": {"budget_tokens": 1024},
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.0),
        ("temperature", 1.999),
        ("top_p", 0.001),
        ("top_p", 1.0),
    ],
)
async def test_supported_sampling_accepts_exact_range_boundaries(
    monkeypatch, field, value
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=_policy(effort=False, budget=False)),
        generate_kwargs={"generation_controls": {"sampling": {field: value}}},
    )

    assert payload[field] == value
    assert type(payload[field]) is float


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("temperature", 2.0, "range"),
        ("top_p", 0.0, "range"),
        ("temperature", True, "number"),
        ("top_p", math.inf, "finite"),
        ("temperature", math.nan, "finite"),
    ],
)
async def test_supported_sampling_rejects_invalid_type_finite_and_boundaries(
    monkeypatch, field, value, match
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_policy_config(
            policy=_policy(effort=False, budget=False)
        ),
    )

    with pytest.raises(ValueError, match=match):
        await client.generate(
            "Hello",
            request_options={field: value},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("negative_zero", [-0.0, 0.0])
async def test_sampling_canonicalizes_signed_zero(monkeypatch, negative_zero):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=_policy(effort=False, budget=False)),
        generate_kwargs={
            "generation_controls": {"sampling": {"temperature": negative_zero}}
        },
    )

    assert payload["temperature"] == 0.0
    assert math.copysign(1.0, payload["temperature"]) == 1.0


@pytest.mark.asyncio
async def test_fixed_sampling_omits_defaults_and_equal_explicit_values(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        effort=False,
        budget=False,
        temperature={"base": {"state": "fixed", "fixed_value": 1.0}},
        top_p={"base": {"state": "fixed", "fixed_value": 0.95}},
    )
    config = _policy_config(
        policy=policy,
        request_overrides={"top_p": 0.1},
        provider_settings={"temperature": 0.4},
        model_settings={
            "temperature": 0.6,
            "request_overrides": {"top_p": 0.2},
        },
    )

    omitted, _, _, _ = await _capture_policy_request(config)
    equal, _, _, _ = await _capture_policy_request(
        config,
        generate_kwargs={
            "generation_controls": {
                "sampling": {"temperature": 1, "top_p": 0.95}
            }
        },
    )

    assert "temperature" not in omitted
    assert "top_p" not in omitted
    assert "temperature" not in equal
    assert "top_p" not in equal

    with pytest.raises(ValueError, match="fixed"):
        await LLMClient(
            "test-model",
            config_source=config,
        ).generate(
            "Hello",
            generation_controls={"sampling": {"temperature": 0.9}},
        )


@pytest.mark.asyncio
async def test_ignored_sampling_omits_finite_values_and_rejects_invalid(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        effort=False,
        budget=False,
        temperature={"base": {"state": "ignored"}},
    )
    config = _policy_config(policy=policy)

    payload, _, _, _ = await _capture_policy_request(
        config,
        generate_kwargs={
            "generation_controls": {"sampling": {"temperature": 0.7}}
        },
    )
    assert "temperature" not in payload

    with pytest.raises(ValueError, match="number"):
        await LLMClient("test-model", config_source=config).generate(
            "Hello",
            request_options={"temperature": True},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["deprecated", "forbidden"])
async def test_deprecated_and_forbidden_sampling_fail_only_for_explicit_values(
    monkeypatch, state
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        effort=False,
        budget=False,
        temperature={"base": {"state": state}},
    )
    config = _policy_config(
        policy=policy,
        provider_settings={"temperature": 0.4},
    )

    payload, _, _, _ = await _capture_policy_request(config)
    assert "temperature" not in payload

    with pytest.raises(ValueError, match=state):
        await LLMClient("test-model", config_source=config).generate(
            "Hello",
            generation_controls={"sampling": {"temperature": 0.4}},
        )


@pytest.mark.asyncio
async def test_reasoning_mode_selects_conditional_sampling_rule(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    temperature = {
        "base": _supported_rule(0.0, 1.0),
        "by_reasoning_mode": {
            "enabled": {"state": "forbidden"},
        },
    }
    policy = _policy(
        effort=False,
        budget=False,
        temperature=temperature,
    )

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        generate_kwargs={
            "generation_controls": {
                "reasoning": {"mode": "disabled"},
                "sampling": {"temperature": 0.5},
            }
        },
    )
    assert payload["temperature"] == 0.5

    with pytest.raises(ValueError, match="forbidden"):
        await LLMClient(
            "test-model",
            config_source=_policy_config(policy=policy),
        ).generate(
            "Hello",
            generation_controls={
                "reasoning": {"mode": "enabled"},
                "sampling": {"temperature": 0.5},
            },
        )


@pytest.mark.asyncio
async def test_generation_control_precedence_is_layered_path_by_path(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(default_mode="enabled", budget=False)
    config = _policy_config(
        policy=policy,
        request_overrides={
            "temperature": 0.2,
            "reasoning": {"effort": "low"},
        },
        provider_settings={"temperature": 0.1},
        model_settings={
            "temperature": 0.3,
            "request_overrides": {
                "temperature": 0.4,
                "reasoning": {"effort": "medium"},
            },
        },
    )

    payload, _, _, _ = await _capture_policy_request(
        config,
        client_kwargs={"thinking_level": "xhigh"},
    )
    assert payload["temperature"] == 0.4
    assert payload["reasoning"]["effort"] == "max"

    raw_payload, _, _, _ = await _capture_policy_request(
        config,
        client_kwargs={"thinking_level": "xhigh"},
        generate_kwargs={
            "request_options": {
                "temperature": 0.5,
                "reasoning": {"effort": "low"},
            }
        },
    )
    assert raw_payload["temperature"] == 0.5
    assert raw_payload["reasoning"]["effort"] == "low"

    typed_payload, _, _, _ = await _capture_policy_request(
        config,
        client_kwargs={"thinking_level": "xhigh"},
        generate_kwargs={
            "generation_controls": {
                "reasoning": {"effort": "medium"},
                "sampling": {"temperature": 0.6},
            }
        },
    )
    assert typed_payload["temperature"] == 0.6
    assert typed_payload["reasoning"]["effort"] == "high"


@pytest.mark.asyncio
async def test_model_tombstones_clear_provider_policy_controls(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(default_mode="enabled", budget=False)
    config = _policy_config(
        policy=policy,
        request_overrides={
            "temperature": 0.2,
            "reasoning": {"effort": "high"},
        },
        model_settings={
            "request_overrides": {
                "temperature": None,
                "reasoning": {"effort": None},
            }
        },
    )

    payload, _, _, _ = await _capture_policy_request(config)

    assert "temperature" not in payload
    assert "reasoning" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_options", "generation_controls", "control"),
    [
        (
            {"temperature": None},
            {"sampling": {"temperature": None}},
            "temperature",
        ),
        (
            {"reasoning": {"effort": None}},
            {"reasoning": {"effort": None}},
            "effort",
        ),
        (
            {"thinking": {"type": "enabled"}},
            {"reasoning": {"mode": "enabled"}},
            "mode",
        ),
    ],
)
async def test_raw_and_typed_same_control_collision_is_rejected(
    monkeypatch, request_options, generation_controls, control
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_policy_config(policy=_policy(default_mode="enabled")),
    )

    with pytest.raises(ValueError, match=f"{control}.*raw.*typed"):
        await client.generate(
            "Hello",
            request_options=request_options,
            generation_controls=generation_controls,
        )


@pytest.mark.asyncio
async def test_raw_and_typed_different_controls_can_be_combined(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=_policy(default_mode="enabled", budget=False)),
        generate_kwargs={
            "request_options": {"top_p": 0.8},
            "generation_controls": {
                "reasoning": {"effort": "medium"},
                "sampling": {"temperature": 0.2},
            },
        },
    )

    assert payload["top_p"] == 0.8
    assert payload["temperature"] == 0.2
    assert payload["reasoning"]["effort"] == "high"


@pytest.mark.asyncio
async def test_extra_body_direct_key_wins_before_semantic_validation(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=_policy(effort=False, budget=False)),
        generate_kwargs={
            "request_options": {
                "temperature": 0.4,
                "extra_body": {"temperature": 0.3},
            }
        },
    )

    assert payload["temperature"] == 0.4


@pytest.mark.asyncio
async def test_nested_declared_paths_preserve_unknown_siblings(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        mode_path=("vendor", "thinking", "mode"),
        effort_path=("vendor", "reasoning", "effort"),
        budget_path=("vendor", "reasoning", "budget"),
    )
    request_options = {
        "vendor": {
            "thinking": {"mode": "enabled", "keep_mode_sibling": True},
            "reasoning": {
                "effort": "medium",
                "budget": 1024,
                "keep_reasoning_sibling": "yes",
            },
            "keep_root_sibling": [1, 2],
        },
        "unknown_null": None,
    }

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        generate_kwargs={"request_options": request_options},
    )

    assert payload["vendor"] == {
        "thinking": {"mode": "enabled", "keep_mode_sibling": True},
        "reasoning": {
            "effort": "high",
            "budget": 1024,
            "keep_reasoning_sibling": "yes",
        },
        "keep_root_sibling": [1, 2],
    }
    assert payload["unknown_null"] is None
    assert request_options["vendor"]["reasoning"]["effort"] == "medium"


@pytest.mark.asyncio
async def test_declared_path_survives_stream_options_normalization(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        effort_path=("stream_options", "reasoning_effort"),
        budget=False,
    )

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        generate_kwargs={
            "request_options": {
                "stream_options": {
                    "reasoning_effort": "medium",
                    "vendor_option": True,
                }
            },
        },
    )

    assert payload["stream_options"] == {
        "reasoning_effort": "high",
        "vendor_option": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_options",
    [
        {"vendor": "not-a-mapping"},
        {"vendor": {"reasoning": "not-a-mapping"}},
    ],
)
async def test_declared_path_mapping_scalar_collisions_fail_safely(
    monkeypatch, request_options
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        mode_path=("vendor", "thinking", "mode"),
        effort_path=("vendor", "reasoning", "effort"),
        budget=False,
    )

    with pytest.raises(ValueError, match="reasoning.*path"):
        await LLMClient(
            "test-model",
            config_source=_policy_config(policy=policy),
        ).generate("Hello", request_options=request_options)


@pytest.mark.asyncio
async def test_policy_resolution_does_not_mutate_settings_inputs_or_clients(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    provider_overrides = {
        "reasoning": {"effort": "medium"},
        "nested": {"provider": [1]},
    }
    model_overrides = {
        "temperature": 0.4,
        "nested": {"model": [2]},
    }
    request_options = {
        "thinking": {"type": "enabled"},
        "top_p": 0.8,
    }
    originals = deepcopy(
        (provider_overrides, model_overrides, request_options)
    )
    config = _policy_config(
        policy=_policy(default_mode="enabled", budget=False),
        request_overrides=provider_overrides,
        model_settings={"request_overrides": model_overrides},
    )

    first = LLMClient("test-model", config_source=config)
    second = LLMClient("test-model", config_source=config)
    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client
        _disable_rate_limit(first)
        await first.generate("Hello", request_options=request_options)

    assert (provider_overrides, model_overrides, request_options) == originals
    assert first.request_overrides == second.request_overrides
    assert first.request_overrides["reasoning"]["effort"] == "medium"
    assert config["test-provider"]["request_overrides"] == provider_overrides


@pytest.mark.asyncio
async def test_same_upstream_id_can_have_route_specific_policies(monkeypatch):
    monkeypatch.setenv("NATIVE_KEY", "test-key")
    monkeypatch.setenv("THIRD_PARTY_KEY", "test-key")
    supported = _policy(effort=False, budget=False)
    fixed = _policy(
        availability="always-on",
        effort=False,
        budget=False,
        temperature={"base": {"state": "fixed", "fixed_value": 1.0}},
    )
    config = {
        **_policy_config(
            policy=supported,
            provider_name="native-route",
            model_name="native-model",
            provider_settings={"api_key_env_var": "NATIVE_KEY"},
            model_settings={"id": "same-upstream-id"},
        ),
        **_policy_config(
            policy=fixed,
            provider_name="third-party-route",
            model_name="third-party-model",
            provider_settings={"api_key_env_var": "THIRD_PARTY_KEY"},
            model_settings={"id": "same-upstream-id"},
        ),
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client
        native = LLMClient("native-model", config_source=config)
        third_party = LLMClient("third-party-model", config_source=config)
        _disable_rate_limit(native)
        _disable_rate_limit(third_party)
        await native.generate(
            "Hello",
            generation_controls={"sampling": {"temperature": 0.5}},
        )
        await third_party.generate(
            "Hello",
            generation_controls={"sampling": {"temperature": 1.0}},
        )

    native_payload = mock_httpx_client.post.await_args_list[0].kwargs["json"]
    third_payload = mock_httpx_client.post.await_args_list[1].kwargs["json"]
    assert native_payload["model"] == third_payload["model"]
    assert native_payload["temperature"] == 0.5
    assert "temperature" not in third_payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "model_name", "api_base_url"),
    [
        ("renamed-a", "alias-a", "https://a.invalid/chat/completions"),
        ("renamed-b", "alias-b", "https://b.invalid/chat/completions"),
    ],
)
async def test_generation_policy_behavior_is_name_and_endpoint_neutral(
    monkeypatch, provider_name, model_name, api_base_url
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _policy_config(
        policy=_policy(default_mode="enabled", budget=False),
        provider_name=provider_name,
        model_name=model_name,
        api_base_url=api_base_url,
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(model_name, config_source=config)
        _disable_rate_limit(client)
        await client.generate(
            "Hello",
            generation_controls={
                "reasoning": {"effort": "medium"},
                "sampling": {"temperature": 0.2},
            },
        )

    payload = mock_httpx_client.post.await_args.kwargs["json"]
    assert payload["reasoning"]["effort"] == "high"
    assert payload["temperature"] == 0.2


@pytest.mark.asyncio
async def test_cache_identity_uses_canonical_effort_aliases(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response("cached")
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=_policy_config(
                policy=_policy(default_mode="enabled", budget=False)
            ),
        )
        client._cache_enabled = True
        _disable_rate_limit(client)
        first = await client.generate(
            "Hello",
            generation_controls={"reasoning": {"effort": "medium"}},
        )
        second = await client.generate(
            "Hello",
            generation_controls={"reasoning": {"effort": "high"}},
        )

    assert first.text == second.text == "cached"
    assert mock_httpx_client.post.await_count == 1
    assert client.get_cache_stats()["hits"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["fixed", "ignored"])
async def test_cache_identity_normalizes_no_effect_sampling_to_omission(
    monkeypatch, state
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    rule = (
        {"state": "fixed", "fixed_value": 1.0}
        if state == "fixed"
        else {"state": "ignored"}
    )
    policy = _policy(
        effort=False,
        budget=False,
        temperature={"base": rule},
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response("cached")
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=_policy_config(policy=policy),
        )
        client._cache_enabled = True
        _disable_rate_limit(client)
        await client.generate("Hello")
        await client.generate(
            "Hello",
            generation_controls={
                "sampling": {"temperature": 1.0 if state == "fixed" else 0.3}
            },
        )

    assert mock_httpx_client.post.await_count == 1
    assert client.get_cache_stats()["hits"] == 1


@pytest.mark.asyncio
async def test_cache_separates_distinct_final_generation_payloads(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response()
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=_policy_config(
                policy=_policy(default_mode="enabled", budget=False)
            ),
        )
        client._cache_enabled = True
        _disable_rate_limit(client)
        for controls in (
            {"reasoning": {"effort": "low"}},
            {"reasoning": {"effort": "high"}},
            {"sampling": {"temperature": 0.2}},
            {"sampling": {"temperature": 0.3}},
        ):
            await client.generate("Hello", generation_controls=controls)

    assert mock_httpx_client.post.await_count == 4
    assert client.get_cache_stats()["misses"] == 4


@pytest.mark.asyncio
async def test_cache_identity_includes_capability_version(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    shared_cache = None

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _success_response("cached")
        mock_cls.return_value = mock_httpx_client
        first = LLMClient(
            "test-model",
            config_source=_policy_config(version="policy-v1"),
        )
        second = LLMClient(
            "test-model",
            config_source=_policy_config(version="policy-v2"),
        )
        first._cache_enabled = True
        second._cache_enabled = True
        shared_cache = first._cache
        second._cache = shared_cache
        _disable_rate_limit(first)
        _disable_rate_limit(second)
        await first.generate("Hello")
        await second.generate("Hello")

    assert mock_httpx_client.post.await_count == 2
    assert shared_cache.get_stats()["misses"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutated_value", "match"),
    [
        (None, "temperature.*null"),
        (1, "temperature.*canonical"),
        (-0.0, "temperature.*canonical"),
    ],
)
async def test_final_payload_is_revalidated_after_later_planners(
    monkeypatch, mutated_value, match
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_policy_config(
            policy=_policy(effort=False, budget=False)
        ),
    )
    client._cache_enabled = True
    client._cache.get = MagicMock()
    client._wait_for_rate_limit = AsyncMock()
    client._get_client = AsyncMock()

    def inject_invalid_policy_value(data, structured_output):
        data["temperature"] = mutated_value
        return client._planning_metadata()

    monkeypatch.setattr(
        client, "_plan_structured_output", inject_invalid_policy_value
    )

    with pytest.raises(ValueError, match=match):
        await client.generate("Hello")

    client._cache.get.assert_not_called()
    client._wait_for_rate_limit.assert_not_awaited()
    client._get_client.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "request_options", "generation_controls", "match"),
    [
        (
            _policy(effort=False, budget=False),
            {"temperature": 2.0},
            None,
            "range",
        ),
        (
            _policy(default_mode="enabled", budget=False),
            {"reasoning": {"effort": "invalid"}},
            None,
            "allowed",
        ),
        (
            _policy(default_mode="enabled", effort=False),
            {"reasoning": {"budget_tokens": True}},
            None,
            "plain integer",
        ),
        (
            _policy(default_mode="enabled"),
            None,
            {
                "reasoning": {
                    "mode": "disabled",
                    "effort": "high",
                }
            },
            "disabled",
        ),
        (
            _policy(
                effort=False,
                budget=False,
                temperature={"base": {"state": "forbidden"}},
            ),
            None,
            {"sampling": {"temperature": 0.2}},
            "forbidden",
        ),
    ],
    ids=[
        "sampling-range",
        "effort-enum",
        "budget-type",
        "reasoning-conflict",
        "sampling-state",
    ],
)
async def test_invalid_controls_fail_before_all_execution_boundaries(
    monkeypatch, policy, request_options, generation_controls, match
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_policy_config(policy=policy),
    )
    client._cache_enabled = True
    client._cache.get = MagicMock()
    client._wait_for_rate_limit = AsyncMock()
    client._get_client = AsyncMock()

    with pytest.raises(ValueError, match=match):
        await client.generate(
            "Hello",
            request_options=request_options,
            generation_controls=generation_controls,
        )

    client._cache.get.assert_not_called()
    client._wait_for_rate_limit.assert_not_awaited()
    client._get_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_typed_controls_do_not_echo_request_values(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    sensitive_value = "do-not-echo-this-request-value"
    client = LLMClient(
        "test-model",
        config_source=_policy_config(
            policy=_policy(default_mode="enabled", budget=False)
        ),
    )
    client._cache_enabled = True
    client._cache.get = MagicMock()
    client._wait_for_rate_limit = AsyncMock()
    client._get_client = AsyncMock()

    with pytest.raises(ValueError) as exc_info:
        await client.generate(
            "Hello",
            generation_controls={"sampling": {"temperature": sensitive_value}},
        )

    assert sensitive_value not in str(exc_info.value)
    client._cache.get.assert_not_called()
    client._wait_for_rate_limit.assert_not_awaited()
    client._get_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_declared_known_paths_do_not_require_legacy_duplication(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        mode_path=("thinking", "type"),
        effort_path=("reasoning_effort",),
        budget=False,
    )

    payload, _, _, _ = await _capture_policy_request(
        _policy_config(policy=policy),
        generate_kwargs={
            "request_options": {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "medium",
            }
        },
    )

    assert payload["thinking"]["type"] == "enabled"
    assert payload["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_undeclared_known_reasoning_control_keeps_legacy_guard(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_policy_config(
            policy=_policy(effort=False, budget=False)
        ),
    )

    with pytest.raises(ValueError, match="does not support verbosity"):
        await client.generate(
            "Hello",
            request_options={"verbosity": "high"},
        )


@pytest.mark.asyncio
async def test_policy_sampling_is_authoritative_over_openrouter_temperature(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    capabilities = _policy_capabilities(_policy(effort=False, budget=False))
    capabilities["openrouter_supported_parameters"] = ["max_tokens"]
    config = _config(
        provider_name="renamed-openrouter-route",
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        model_capabilities=capabilities,
    )

    payload, _, _, _ = await _capture_policy_request(
        config,
        generate_kwargs={
            "generation_controls": {"sampling": {"temperature": 0.2}}
        },
    )

    assert payload["temperature"] == 0.2


@pytest.mark.asyncio
async def test_policy_streaming_preserves_assembly_and_bypasses_cache(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    payloads = []

    async def mock_lines():
        yield 'data: {"choices": [{"delta": {"content": "A"}}]}'
        yield 'data: {"choices": [{"delta": {"content": "B"}}]}'
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
            config_source=_policy_config(
                policy=_policy(default_mode="enabled", budget=False)
            ),
        )
        client._cache_enabled = True
        _disable_rate_limit(client)
        result = await client.generate(
            "Hello",
            stream=True,
            generation_controls={"reasoning": {"effort": "medium"}},
        )

    assert result.text == "AB"
    assert payloads[0]["reasoning"]["effort"] == "high"
    assert payloads[0]["stream_options"]["include_usage"] is True
    assert client.get_cache_stats()["hits"] == 0
    assert client.get_cache_stats()["misses"] == 0


@pytest.mark.asyncio
async def test_policy_structured_output_keeps_semantic_planner(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    capabilities = _policy_capabilities(_policy(effort=False, budget=False))
    capabilities.update(
        {
            "strict_response_schema": True,
            "json_object_response": True,
        }
    )

    payload, _, _, result = await _capture_policy_request(
        _config(model_capabilities=capabilities),
        response_content='{"answer": "ok"}',
        generate_kwargs={
            "generation_controls": {"sampling": {"temperature": 0.2}},
            "structured_output": {
                "mode": "require",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    )

    assert payload["temperature"] == 0.2
    assert payload["response_format"]["type"] == "json_schema"
    assert result.structured == {"answer": "ok"}


@pytest.mark.asyncio
async def test_token_retry_revalidates_budget_relation_before_second_http(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    policy = _policy(
        default_mode="enabled",
        effort=False,
        output_limit_relation="less-than",
    )
    config = _policy_config(
        policy=policy,
        provider_settings={
            "max_tokens_retry": {
                "status_code": 400,
                "body_contains": "reduce limit",
                "max_tokens_limit": 1000,
            }
        },
    )
    request = httpx.Request("POST", "https://example.invalid/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        text="please reduce limit",
    )
    error = httpx.HTTPStatusError(
        "bad request",
        request=request,
        response=response,
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = error
        mock_cls.return_value = mock_httpx_client
        client = LLMClient("test-model", config_source=config)
        _disable_rate_limit(client)
        with patch("llm_exec_core.client.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ValueError, match="less than"):
                await client.generate(
                    "Hello",
                    request_options={
                        "max_tokens": 2000,
                        "reasoning": {"budget_tokens": 1500},
                    },
                )

    assert mock_httpx_client.post.await_count == 1
