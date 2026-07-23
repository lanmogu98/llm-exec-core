import asyncio
import inspect
import json
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

import llm_exec_core.client as client_module
import llm_exec_core.config as config_module
from llm_exec_core.client import LLMClient

USAGE_FIELD_MISSING = object()


def _openai_usage(
    prompt_tokens=100,
    completion_tokens=10,
    prompt_tokens_details=USAGE_FIELD_MISSING,
):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if prompt_tokens_details is not USAGE_FIELD_MISSING:
        usage["prompt_tokens_details"] = prompt_tokens_details
    return usage


def _rich_client_rule(
    *,
    rule_id="rich-rule",
    cache_mode="none",
    currency="USD",
    input_tokens_gt=0,
    input_tokens_lte=None,
    request_mode="realtime",
):
    cache_rates = {
        "none": (None, None),
        "implicit": (0.5, None),
        "explicit": (0.5, 3.0),
    }
    cache_read_input, cache_write_input = cache_rates[cache_mode]
    return {
        "rule_id": rule_id,
        "billing_model_id": "provider-model-id",
        "capability_snapshot_id": None,
        "region": "us-east",
        "service_scope": "global",
        "deployment_type": "serverless",
        "output_mode": "thinking",
        "request_mode": request_mode,
        "cache_mode": cache_mode,
        "input_tokens_gt": input_tokens_gt,
        "input_tokens_lte": input_tokens_lte,
        "currency": currency,
        "unit_tokens": 1_000,
        "effective_from": None,
        "effective_until": None,
        "rate_type": "standard",
        "rates": {
            "input": 2.0,
            "output": 4.0,
            "cache_read_input": cache_read_input,
            "cache_write_input": cache_write_input,
        },
        "evidence": [
            {
                "url": "https://help.aliyun.com/zh/model-studio/model-pricing",
                "retrieved_on": "2026-07-23",
                "facts": [
                    "Synthetic offline rule backed by public semantics."
                ],
            }
        ],
    }


def _rich_client_config(rules):
    return {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://example.invalid/chat/completions",
            "temperature": 0.1,
            "max_tokens": 128,
            "context_window": 4096,
            "pricing_currency": "$",
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {
                        "schema": "pricing-rules-v1",
                        "rules": deepcopy(rules),
                    },
                }
            },
        }
    }


def _rich_client_context(cache_mode="none", **overrides):
    context = {
        "region": "us-east",
        "service_scope": "global",
        "deployment_type": "serverless",
        "output_mode": "thinking",
        "request_mode": "realtime",
        "cache_mode": cache_mode,
    }
    context.update(overrides)
    return context


def _http_response(usage=USAGE_FIELD_MISSING):
    response = MagicMock()
    payload = {"choices": [{"message": {"content": "ok"}}]}
    if usage is not USAGE_FIELD_MISSING:
        payload["usage"] = usage
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


@pytest.mark.asyncio
async def test_generate_applies_structured_output_hook(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://example.invalid/chat/completions",
            "temperature": 0.1,
            "max_tokens": 128,
            "context_window": 4096,
            "pricing_currency": "$",
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                }
            },
        }
    }
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"title": "A"}'}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = mock_response
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        result = await client.generate(
            "Hello",
            request_name="extract",
            structured_output_hook=json.loads,
            trace_context={"source": "unit"},
            run_id="run-1",
            request_id="req-1",
        )

    assert result.text == '{"title": "A"}'
    assert result.structured == {"title": "A"}
    assert result.metadata.provider_name == "test-provider"
    assert result.metadata.request_id == "req-1"
    assert result.metadata.model_name == "test-model"
    assert result.metadata.model_id == "provider-model-id"
    assert result.metadata.duration_seconds > 0
    assert result.metadata.trace_context == {"source": "unit"}
    assert result.metadata.planning["validation_status"] == "not_applicable"
    assert result.metadata.planning["hook_applied"] is True


@pytest.mark.asyncio
async def test_generate_leaves_optional_metadata_as_none_when_omitted(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://example.invalid/chat/completions",
            "temperature": 0.1,
            "max_tokens": 128,
            "context_window": 4096,
            "pricing_currency": "$",
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                }
            },
        }
    }
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = mock_response
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        result = await client.generate("Hello", request_name="extract")

    assert result.metadata.request_id is None
    assert result.metadata.run_id is None


@pytest.mark.asyncio
async def test_generate_response_legacy_tuple_preserves_duration(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://example.invalid/chat/completions",
            "temperature": 0.1,
            "max_tokens": 128,
            "context_window": 4096,
            "pricing_currency": "$",
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                }
            },
        }
    }
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    mock_response.raise_for_status = MagicMock()

    with (
        patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls,
        patch(
            "llm_exec_core.client.time.time",
            side_effect=[100.0, 100.0, 101.5],
        ),
    ):
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = mock_response
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        _, usage = await client.generate_response("Hello")

    assert usage["process_times"]["total_time"] == 1.5
    assert usage["metadata"]["provider_name"] == "test-provider"
    assert usage["metadata"]["request_id"] is None
    assert usage["metadata"]["run_id"] is None


@pytest.mark.asyncio
async def test_generate_lowers_max_tokens_from_configured_retry_policy(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://proxy.example.invalid/chat/completions",
            "temperature": 0.1,
            "max_tokens": 65536,
            "context_window": 200000,
            "pricing_currency": "$",
            "max_tokens_retry": {
                "status_code": 404,
                "body_contains": "no allowed providers",
                "max_tokens_limit": 8192,
            },
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                }
            },
        }
    }
    request = httpx.Request(
        "POST", "https://proxy.example.invalid/chat/completions"
    )
    error_response = httpx.Response(
        404,
        text='{"error":"No allowed providers"}',
        request=request,
    )
    success_response = MagicMock()
    success_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    success_response.raise_for_status = MagicMock()
    payloads = []

    async def post_side_effect(*args, **kwargs):
        payloads.append(dict(kwargs["json"]))
        if len(payloads) == 1:
            return error_response
        return success_response

    with (
        patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls,
        patch("llm_exec_core.client.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = post_side_effect
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        result = await client.generate("Hello")

    assert result.text == "ok"
    assert client.api_url == "https://proxy.example.invalid/chat/completions"
    assert [payload["max_tokens"] for payload in payloads] == [65536, 8192]


@pytest.mark.asyncio
async def test_generate_retry_never_increases_existing_max_tokens(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://proxy.example.invalid/chat/completions",
            "temperature": 0.1,
            "max_tokens": 900,
            "context_window": 4096,
            "pricing_currency": "$",
            "max_tokens_retry": {
                "status_code": 404,
                "body_contains": "no allowed providers",
                "max_tokens_limit": 800,
            },
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                }
            },
        }
    }
    request = httpx.Request(
        "POST", "https://proxy.example.invalid/chat/completions"
    )
    error_response = httpx.Response(
        404,
        text='{"error":"No allowed providers"}',
        request=request,
    )
    success_response = MagicMock()
    success_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    success_response.raise_for_status = MagicMock()
    payloads = []

    async def post_side_effect(*args, **kwargs):
        payloads.append(dict(kwargs["json"]))
        if len(payloads) == 1:
            return error_response
        return success_response

    with (
        patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls,
        patch("llm_exec_core.client.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = post_side_effect
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        result = await client.generate("Hello")

    assert result.text == "ok"
    assert [payload["max_tokens"] for payload in payloads] == [900, 450]


@pytest.mark.asyncio
async def test_generate_lowers_max_tokens_before_generic_429_retry(
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    caplog.set_level("WARNING", logger="llm_exec_core.client")
    config = {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://proxy.example.invalid/chat/completions",
            "temperature": 0.1,
            "max_tokens": 65536,
            "context_window": 200000,
            "pricing_currency": "$",
            "max_tokens_retry": {
                "status_code": 429,
                "body_contains": "too many output tokens",
                "max_tokens_limit": 8192,
            },
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                }
            },
        }
    }
    request = httpx.Request(
        "POST", "https://proxy.example.invalid/chat/completions"
    )
    error_response = httpx.Response(
        429,
        text='{"error":"Too many output tokens"}',
        request=request,
    )
    success_response = MagicMock()
    success_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    success_response.raise_for_status = MagicMock()
    payloads = []

    async def post_side_effect(*args, **kwargs):
        payloads.append(dict(kwargs["json"]))
        if len(payloads) == 1:
            return error_response
        return success_response

    with (
        patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls,
        patch("llm_exec_core.client.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = post_side_effect
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        result = await client.generate("Hello")

    assert result.text == "ok"
    assert [payload["max_tokens"] for payload in payloads] == [65536, 8192]
    assert "HTTP error 429; lowering max_tokens 65536 -> 8192" in caplog.text


@pytest.mark.parametrize(
    ("cache_mode", "usage", "expected"),
    [
        ("none", _openai_usage(), (100, 10, 100, 0, 0)),
        (
            "none",
            _openai_usage(prompt_tokens_details=None),
            (100, 10, 100, 0, 0),
        ),
        (
            "none",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "ignored": "provider-data",
                }
            ),
            (100, 10, 100, 0, 0),
        ),
        (
            "implicit",
            _openai_usage(prompt_tokens_details={"cached_tokens": 0}),
            (100, 10, 100, 0, 0),
        ),
        (
            "implicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 20,
                    "cache_creation_input_tokens": 0,
                    "ignored": ["provider-data"],
                }
            ),
            (100, 10, 80, 20, 0),
        ),
        (
            "explicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            ),
            (100, 10, 100, 0, 0),
        ),
        (
            "explicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 20,
                    "cache_creation_input_tokens": 30,
                    "ignored": {"retained": True},
                }
            ),
            (100, 10, 50, 20, 30),
        ),
    ],
)
def test_openai_chat_usage_parser_accepts_exact_cache_mode_shapes(
    cache_mode, usage, expected
):
    parts = client_module._parse_openai_chat_usage(usage, cache_mode)

    assert asdict(parts) == dict(
        zip(
            (
                "prompt_tokens",
                "completion_tokens",
                "ordinary_input_tokens",
                "cache_read_input_tokens",
                "cache_write_input_tokens",
            ),
            expected,
        )
    )


INVALID_USAGE_CASES = [
    ("none", None),
    ("none", []),
    ("none", {}),
    ("none", {"prompt_tokens": 100}),
    ("none", {"completion_tokens": 10}),
]
for field in ("prompt_tokens", "completion_tokens"):
    for invalid_value in (True, 1.5, "1", None, -1):
        invalid_usage = _openai_usage()
        invalid_usage[field] = invalid_value
        INVALID_USAGE_CASES.append(("none", invalid_usage))

INVALID_USAGE_CASES.extend(
    [
        ("none", _openai_usage(prompt_tokens_details=[])),
        ("implicit", _openai_usage()),
        ("implicit", _openai_usage(prompt_tokens_details=None)),
        ("implicit", _openai_usage(prompt_tokens_details=[])),
        ("implicit", _openai_usage(prompt_tokens_details={})),
        ("explicit", _openai_usage()),
        ("explicit", _openai_usage(prompt_tokens_details=None)),
        ("explicit", _openai_usage(prompt_tokens_details=[])),
        (
            "explicit",
            _openai_usage(
                prompt_tokens_details={"cache_creation_input_tokens": 0}
            ),
        ),
        (
            "explicit",
            _openai_usage(prompt_tokens_details={"cached_tokens": 0}),
        ),
        (
            "implicit",
            _openai_usage(prompt_tokens_details={"cached_tokens": 101}),
        ),
        (
            "explicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 60,
                    "cache_creation_input_tokens": 41,
                }
            ),
        ),
    ]
)
for key in ("cached_tokens", "cache_creation_input_tokens"):
    for invalid_value in (1, True, 0.5, "0", None, -1):
        none_details = {
            "cached_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        none_details[key] = invalid_value
        INVALID_USAGE_CASES.append(
            ("none", _openai_usage(prompt_tokens_details=none_details))
        )

for invalid_value in (True, 0.5, "0", None, -1):
    INVALID_USAGE_CASES.append(
        (
            "implicit",
            _openai_usage(
                prompt_tokens_details={"cached_tokens": invalid_value}
            ),
        )
    )
for invalid_value in (1, True, 0.5, "0", None, -1):
    INVALID_USAGE_CASES.append(
        (
            "implicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 0,
                    "cache_creation_input_tokens": invalid_value,
                }
            ),
        )
    )
for key in ("cached_tokens", "cache_creation_input_tokens"):
    for invalid_value in (True, 0.5, "0", None, -1):
        explicit_details = {
            "cached_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        explicit_details[key] = invalid_value
        INVALID_USAGE_CASES.append(
            (
                "explicit",
                _openai_usage(prompt_tokens_details=explicit_details),
            )
        )


@pytest.mark.parametrize(("cache_mode", "usage"), INVALID_USAGE_CASES)
def test_openai_chat_usage_parser_rejects_complete_malformed_matrix(
    cache_mode, usage
):
    with pytest.raises(config_module.PricingCalculationError):
        client_module._parse_openai_chat_usage(usage, cache_mode)


@pytest.mark.asyncio
async def test_rich_non_streaming_explicit_cache_cost_and_currency(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule(cache_mode="explicit")])
    usage = _openai_usage(
        prompt_tokens_details={
            "cached_tokens": 20,
            "cache_creation_input_tokens": 30,
            "ignored": "provider-data",
        }
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(usage)
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context("explicit"),
        )
        result = await client.generate("Hello", request_name="rich")

    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 10
    assert result.usage.input_cost == pytest.approx(0.2)
    assert result.usage.output_cost == pytest.approx(0.04)
    assert result.usage.total_cost == pytest.approx(0.24)
    assert result.usage.currency == "USD"
    assert result.usage.requests[0]["input_tokens"] == 100
    assert client.get_token_usage()["cost"]["total_cost"] == pytest.approx(
        0.24
    )
    assert client.pricing_currency == "USD"


@pytest.mark.asyncio
async def test_legacy_non_streaming_ignores_cache_details_and_shape(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://example.invalid/chat/completions",
            "pricing_currency": "$",
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                }
            },
        }
    }
    usage = _openai_usage(
        prompt_tokens=10,
        completion_tokens=20,
        prompt_tokens_details={
            "cached_tokens": 9,
            "cache_creation_input_tokens": 8,
        },
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(usage)
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        result = await client.generate("Hello")

    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.usage.input_cost == pytest.approx(0.00001)
    assert result.usage.output_cost == pytest.approx(0.00004)
    assert result.usage.currency == "$"


def test_rich_client_requires_context_and_rejects_batch(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    realtime_config = _rich_client_config([_rich_client_rule()])

    with pytest.raises(config_module.PricingContextRequiredError):
        LLMClient("test-model", config_source=realtime_config)

    batch_config = _rich_client_config(
        [_rich_client_rule(request_mode="batch")]
    )
    with pytest.raises(config_module.PricingCalculationError):
        LLMClient(
            "test-model",
            config_source=batch_config,
            pricing_context=_rich_client_context(request_mode="batch"),
        )


def test_pricing_context_constructor_argument_is_keyword_only():
    parameter = inspect.signature(LLMClient).parameters["pricing_context"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_rich_client_pricing_context_rejects_public_reassignment(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_rich_client_config([_rich_client_rule()]),
        pricing_context=_rich_client_context(),
    )
    replacement = config_module.PricingContext.model_validate(
        _rich_client_context(region="eu-west")
    )

    with pytest.raises(AttributeError):
        client.pricing_context = replacement


def test_rich_client_pricing_context_rejects_public_field_mutation(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    client = LLMClient(
        "test-model",
        config_source=_rich_client_config([_rich_client_rule()]),
        pricing_context=_rich_client_context(),
    )
    exposed_context = client.pricing_context

    assert exposed_context is not None
    with pytest.raises(ValidationError):
        exposed_context.region = "eu-west"
    assert client.pricing_context.region == "us-east"


@pytest.mark.asyncio
async def test_rich_client_deep_copies_pricing_context(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule()])
    context = _rich_client_context()

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(_openai_usage())
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model", config_source=config, pricing_context=context
        )
        context["region"] = "mutated-after-construction"
        result = await client.generate("Hello")

    assert result.usage.currency == "USD"
    assert client.pricing_context.region == "us-east"


@pytest.mark.asyncio
async def test_rich_client_uses_request_start_pricing_context_snapshot(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    us_rule = _rich_client_rule(rule_id="us-rule")
    eu_rule = _rich_client_rule(rule_id="eu-rule", currency="EUR")
    eu_rule["region"] = "eu-west"
    post_started = asyncio.Event()
    release_response = asyncio.Event()

    async def delayed_post(*args, **kwargs):
        post_started.set()
        await release_response.wait()
        return _http_response(_openai_usage())

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = delayed_post
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_rich_client_config([us_rule, eu_rule]),
            pricing_context=_rich_client_context(),
        )
        request = asyncio.create_task(client.generate("Hello"))
        await asyncio.wait_for(post_started.wait(), timeout=1)
        stored_context = client.__dict__.get("_pricing_context")
        if stored_context is None:
            stored_context = client.__dict__["pricing_context"]
        object.__setattr__(stored_context, "region", "eu-west")
        release_response.set()
        result = await request

    assert result.usage.currency == "USD"


@pytest.mark.asyncio
async def test_rich_client_rejects_batch_context_at_request_start(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    realtime_rule = _rich_client_rule(rule_id="realtime-rule")
    batch_rule = _rich_client_rule(rule_id="batch-rule", request_mode="batch")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(_openai_usage())
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=_rich_client_config([realtime_rule, batch_rule]),
            pricing_context=_rich_client_context(),
        )
        stored_context = client.__dict__.get("_pricing_context")
        if stored_context is None:
            stored_context = client.__dict__["pricing_context"]
        object.__setattr__(stored_context, "request_mode", "batch")

        with pytest.raises(
            config_module.PricingCalculationError,
            match="realtime pricing contexts only",
        ):
            await client.generate("Hello")

    assert mock_httpx_client.post.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        USAGE_FIELD_MISSING,
        None,
        {},
        {"prompt_tokens": 100, "completion_tokens": True},
    ],
)
async def test_rich_non_streaming_malformed_usage_fails_before_mutation(
    monkeypatch, usage
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule()])

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(usage)
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        before = deepcopy(client.get_token_usage())

        with pytest.raises(config_module.PricingCalculationError):
            await client.generate("Hello")

    assert client.get_token_usage() == before


@pytest.mark.asyncio
async def test_rich_no_match_fails_before_usage_mutation(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule(input_tokens_lte=10)])

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(_openai_usage())
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        before = deepcopy(client.get_token_usage())

        with pytest.raises(config_module.PricingNoMatchError):
            await client.generate("Hello")

    assert client.get_token_usage() == before


@pytest.mark.asyncio
async def test_rich_mixed_currency_accumulation_fails_atomically(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    first = _rich_client_rule(
        rule_id="usd-lower", currency="USD", input_tokens_lte=10
    )
    second = _rich_client_rule(
        rule_id="eur-upper",
        currency="EUR",
        input_tokens_gt=10,
        input_tokens_lte=None,
    )
    config = _rich_client_config([first, second])

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = [
            _http_response(_openai_usage(prompt_tokens=10)),
            _http_response(_openai_usage(prompt_tokens=11)),
        ]
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        first_result = await client.generate("first")
        after_first = deepcopy(client.get_token_usage())

        with pytest.raises(config_module.PricingCalculationError):
            await client.generate("second")

    assert first_result.usage.currency == "USD"
    assert client.get_token_usage() == after_first
    assert len(client.get_token_usage()["requests"]) == 1


@pytest.mark.asyncio
async def test_rich_local_response_cache_hit_has_zero_usage_and_cost(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule()])

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(_openai_usage())
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        client._cache_enabled = True
        first = await client.generate("same prompt")
        aggregate_after_first = deepcopy(client.get_token_usage())
        cached = await client.generate("same prompt")

    assert first.usage.total_tokens == 110
    assert cached.text == "ok"
    assert cached.usage.input_tokens == 0
    assert cached.usage.output_tokens == 0
    assert cached.usage.total_cost == 0.0
    assert cached.usage.currency == "USD"
    assert cached.usage.requests == []
    assert client.get_token_usage() == aggregate_after_first
    assert mock_httpx_client.post.await_count == 1
