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
    **extra,
):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        **extra,
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
    cache_read_input=USAGE_FIELD_MISSING,
    cache_write_input=USAGE_FIELD_MISSING,
):
    cache_rates = {
        "none": (None, None),
        "implicit": (0.5, None),
        "explicit": (0.5, 3.0),
    }
    default_cache_read, default_cache_write = cache_rates[cache_mode]
    if cache_read_input is USAGE_FIELD_MISSING:
        cache_read_input = default_cache_read
    if cache_write_input is USAGE_FIELD_MISSING:
        cache_write_input = default_cache_write
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


def _rich_cache_policy(rules):
    modes = []
    for rule in rules:
        cache_mode = rule["cache_mode"]
        if cache_mode in {entry["cache_mode"] for entry in modes}:
            continue
        modes.append(
            {
                "cache_mode": cache_mode,
                "activation": {
                    "none": "not-applicable",
                    "implicit": "automatic",
                    "explicit": "request",
                }[cache_mode],
            }
        )
    return {
        "support": (
            "supported"
            if any(entry["cache_mode"] != "none" for entry in modes)
            else "unsupported"
        ),
        "modes": modes,
        "evidence": [
            {
                "url": "https://example.invalid/cache-policy",
                "retrieved_on": "2026-07-23",
                "facts": ["Synthetic offline cache activation evidence."],
            }
        ],
    }


def _rich_client_profile(rules):
    if any(rule["rates"]["cache_write_input"] is not None for rule in rules):
        return "openai-chat-cache-creation-v1"
    if any(rule["cache_mode"] != "none" for rule in rules):
        return "openai-chat-cached-tokens-v1"
    return "openai-chat-standard-v1"


def _rich_client_config(
    rules,
    *,
    usage_accounting=USAGE_FIELD_MISSING,
    cache_policy=USAGE_FIELD_MISSING,
):
    if usage_accounting is USAGE_FIELD_MISSING:
        usage_accounting = _rich_client_profile(rules)
    if cache_policy is USAGE_FIELD_MISSING:
        cache_policy = _rich_cache_policy(rules)
    return {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://example.invalid/chat/completions",
            "temperature": 0.1,
            "max_tokens": 128,
            "context_window": 4096,
            "pricing_currency": "$",
            "usage_accounting": usage_accounting,
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "cache_policy": deepcopy(cache_policy),
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
    ("profile", "cache_mode", "usage", "expected"),
    [
        (
            "openai-chat-standard-v1",
            "none",
            _openai_usage(
                prompt_tokens_details={"ignored_extension": "provider-data"}
            ),
            (100, 10, 100, 0, 0),
        ),
        (
            "openai-chat-cached-tokens-v1",
            "none",
            _openai_usage(prompt_tokens_details=None),
            (100, 10, 100, 0, 0),
        ),
        (
            "openai-chat-cached-tokens-v1",
            "implicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 20,
                    "ignored": "tencent-provider-data",
                }
            ),
            (100, 10, 80, 20, 0),
        ),
        (
            "openai-chat-cached-tokens-v1",
            "implicit",
            _openai_usage(
                completion_tokens=0,
                prompt_tokens_details={"cached_tokens": 100},
            ),
            (100, 0, 0, 100, 0),
        ),
        (
            "openai-chat-cache-creation-v1",
            "implicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            ),
            (100, 10, 100, 0, 0),
        ),
        (
            "openai-chat-cache-creation-v1",
            "explicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 20,
                    "cache_creation_input_tokens": 30,
                    "ignored": ["alibaba-provider-data"],
                }
            ),
            (100, 10, 50, 20, 30),
        ),
        (
            "openai-chat-cache-creation-v1",
            "explicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 60,
                    "cache_creation_input_tokens": 40,
                }
            ),
            (100, 10, 0, 60, 40),
        ),
        (
            "openai-chat-cache-write-v1",
            "explicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 20,
                    "cache_write_tokens": 30,
                    "ignored": {"openrouter": True},
                }
            ),
            (100, 10, 50, 20, 30),
        ),
        (
            "openai-chat-cache-write-v1",
            "explicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 100,
                    "cache_write_tokens": 0,
                }
            ),
            (100, 10, 0, 100, 0),
        ),
        (
            "openai-chat-cache-hit-miss-v1",
            "implicit",
            _openai_usage(
                prompt_cache_hit_tokens=20,
                prompt_cache_miss_tokens=80,
            ),
            (100, 10, 80, 20, 0),
        ),
        (
            "openai-chat-cache-hit-miss-v1",
            "none",
            _openai_usage(
                prompt_cache_hit_tokens=0,
                prompt_cache_miss_tokens=100,
            ),
            (100, 10, 100, 0, 0),
        ),
    ],
)
def test_openai_chat_usage_parser_accepts_all_profile_shapes(
    profile, cache_mode, usage, expected
):
    parts = client_module._parse_openai_chat_usage(usage, profile, cache_mode)

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


PROFILE_VALID_USAGE = {
    "openai-chat-standard-v1": ("none", _openai_usage()),
    "openai-chat-cached-tokens-v1": (
        "implicit",
        _openai_usage(prompt_tokens_details={"cached_tokens": 20}),
    ),
    "openai-chat-cache-creation-v1": (
        "explicit",
        _openai_usage(
            prompt_tokens_details={
                "cached_tokens": 20,
                "cache_creation_input_tokens": 30,
            }
        ),
    ),
    "openai-chat-cache-write-v1": (
        "explicit",
        _openai_usage(
            prompt_tokens_details={
                "cached_tokens": 20,
                "cache_write_tokens": 30,
            }
        ),
    ),
    "openai-chat-cache-hit-miss-v1": (
        "implicit",
        _openai_usage(
            prompt_cache_hit_tokens=20,
            prompt_cache_miss_tokens=80,
        ),
    ),
}

INVALID_USAGE_CASES = [
    ("openai-chat-standard-v1", "none", None),
    ("openai-chat-standard-v1", "none", []),
    ("openai-chat-standard-v1", "none", {}),
    (
        "openai-chat-standard-v1",
        "none",
        {"prompt_tokens": 100},
    ),
    (
        "openai-chat-standard-v1",
        "none",
        {"completion_tokens": 10},
    ),
]
for profile, (cache_mode, valid_usage) in PROFILE_VALID_USAGE.items():
    for field in ("prompt_tokens", "completion_tokens"):
        for invalid_value in (True, 1.5, "1", None, -1):
            invalid_usage = deepcopy(valid_usage)
            invalid_usage[field] = invalid_value
            INVALID_USAGE_CASES.append((profile, cache_mode, invalid_usage))

for profile, write_field in (
    ("openai-chat-cache-creation-v1", "cache_creation_input_tokens"),
    ("openai-chat-cache-write-v1", "cache_write_tokens"),
):
    for missing_details in (USAGE_FIELD_MISSING, None, [], {}):
        usage = _openai_usage(prompt_tokens_details=missing_details)
        if missing_details is USAGE_FIELD_MISSING:
            usage.pop("prompt_tokens_details", None)
        INVALID_USAGE_CASES.append((profile, "explicit", usage))
    for field in ("cached_tokens", write_field):
        for invalid_value in (True, 0.5, "0", None, -1):
            details = {"cached_tokens": 0, write_field: 0}
            details[field] = invalid_value
            INVALID_USAGE_CASES.append(
                (
                    profile,
                    "explicit",
                    _openai_usage(prompt_tokens_details=details),
                )
            )
    INVALID_USAGE_CASES.append(
        (
            profile,
            "explicit",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 60,
                    write_field: 41,
                }
            ),
        )
    )

for missing_details in (USAGE_FIELD_MISSING, None, [], {}):
    usage = _openai_usage(prompt_tokens_details=missing_details)
    if missing_details is USAGE_FIELD_MISSING:
        usage.pop("prompt_tokens_details", None)
    INVALID_USAGE_CASES.append(
        (
            "openai-chat-cached-tokens-v1",
            "implicit",
            usage,
        )
    )
for invalid_value in (True, 0.5, "0", None, -1, 101):
    INVALID_USAGE_CASES.append(
        (
            "openai-chat-cached-tokens-v1",
            "implicit",
            _openai_usage(
                prompt_tokens_details={"cached_tokens": invalid_value}
            ),
        )
    )

for field in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
    for invalid_value in (USAGE_FIELD_MISSING, True, 0.5, "0", None, -1):
        usage = _openai_usage(
            prompt_cache_hit_tokens=20,
            prompt_cache_miss_tokens=80,
        )
        if invalid_value is USAGE_FIELD_MISSING:
            usage.pop(field)
        else:
            usage[field] = invalid_value
        INVALID_USAGE_CASES.append(
            ("openai-chat-cache-hit-miss-v1", "implicit", usage)
        )
for hit, miss in ((20, 79), (20, 81), (101, 0)):
    INVALID_USAGE_CASES.append(
        (
            "openai-chat-cache-hit-miss-v1",
            "implicit",
            _openai_usage(
                prompt_cache_hit_tokens=hit,
                prompt_cache_miss_tokens=miss,
            ),
        )
    )

INVALID_USAGE_CASES.extend(
    [
        (
            "openai-chat-standard-v1",
            "implicit",
            _openai_usage(),
        ),
        (
            "openai-chat-cached-tokens-v1",
            "none",
            _openai_usage(prompt_tokens_details={"cached_tokens": 1}),
        ),
        (
            "openai-chat-cache-creation-v1",
            "none",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 0,
                    "cache_creation_input_tokens": 1,
                }
            ),
        ),
        (
            "openai-chat-cache-write-v1",
            "none",
            _openai_usage(
                prompt_tokens_details={
                    "cached_tokens": 0,
                    "cache_write_tokens": 1,
                }
            ),
        ),
        (
            "openai-chat-cache-hit-miss-v1",
            "none",
            _openai_usage(
                prompt_cache_hit_tokens=1,
                prompt_cache_miss_tokens=99,
            ),
        ),
    ]
)


@pytest.mark.parametrize(
    ("profile", "cache_mode", "usage"), INVALID_USAGE_CASES
)
def test_openai_chat_usage_parser_rejects_complete_malformed_matrix(
    profile, cache_mode, usage
):
    with pytest.raises(config_module.PricingCalculationError):
        client_module._parse_openai_chat_usage(usage, profile, cache_mode)


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
    assert set(result.usage.requests[0]) == {
        "name",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "process_time",
        "input_cost",
        "output_cost",
        "total_cost",
        "timestamp",
    }
    assert client.get_token_usage()["cost"]["total_cost"] == pytest.approx(
        0.24
    )
    assert client.pricing_currency == "USD"
    assert result.usage.accounting.tokens_available is True
    assert result.usage.accounting.cost_available is True
    assert result.usage.accounting.reason is None
    assert "accounting" not in result.usage.to_legacy_dict()


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
    ("usage", "reason"),
    [
        (USAGE_FIELD_MISSING, "missing_usage"),
        (None, "malformed_usage"),
        ({}, "malformed_usage"),
        (
            {"prompt_tokens": 100, "completion_tokens": True},
            "malformed_usage",
        ),
    ],
)
async def test_rich_unknown_usage_returns_text_and_incompleteness(
    monkeypatch, usage, reason
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
        result = await client.generate("Hello", request_name="unknown")

    assert result.text == "ok"
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.input_cost is None
    assert result.usage.output_cost is None
    assert result.usage.total_cost is None
    assert result.usage.currency is None
    assert result.usage.accounting.tokens_available is False
    assert result.usage.accounting.cost_available is False
    assert result.usage.accounting.reason == reason
    aggregate = client.get_token_usage()
    assert aggregate["total_input_tokens"] is None
    assert aggregate["total_output_tokens"] is None
    assert aggregate["cost"] == {
        "input_cost": None,
        "output_cost": None,
        "total_cost": None,
    }
    assert aggregate["accounting"] == {
        "tokens_available": False,
        "cost_available": False,
        "reason": reason,
    }
    assert aggregate["requests"][0]["name"] == "unknown"
    assert aggregate["requests"][0]["process_time"] >= 0
    assert aggregate["requests"][0]["accounting"]["reason"] == reason


@pytest.mark.asyncio
async def test_rich_no_match_returns_known_tokens_and_unknown_cost(
    monkeypatch,
):
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
        result = await client.generate("Hello", request_name="no-match")

    assert result.text == "ok"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 10
    assert result.usage.total_tokens == 110
    assert result.usage.input_cost is None
    assert result.usage.output_cost is None
    assert result.usage.total_cost is None
    assert result.usage.currency is None
    assert result.usage.accounting.tokens_available is True
    assert result.usage.accounting.cost_available is False
    assert result.usage.accounting.reason == "pricing_no_match"
    aggregate = client.get_token_usage()
    assert aggregate["total_input_tokens"] == 100
    assert aggregate["total_output_tokens"] == 10
    assert aggregate["cost"]["total_cost"] is None
    assert aggregate["accounting"]["reason"] == "pricing_no_match"


@pytest.mark.asyncio
async def test_rich_mixed_currency_preserves_result_and_marks_aggregate_cost(
    monkeypatch,
):
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
        second_result = await client.generate("second")

    assert first_result.usage.currency == "USD"
    assert second_result.text == "ok"
    assert second_result.usage.currency == "EUR"
    assert second_result.usage.total_cost == pytest.approx(0.062)
    assert second_result.usage.accounting.tokens_available is True
    assert second_result.usage.accounting.cost_available is True
    assert second_result.usage.accounting.reason is None
    aggregate = client.get_token_usage()
    assert aggregate["total_input_tokens"] == 21
    assert aggregate["total_output_tokens"] == 20
    assert aggregate["cost"] == {
        "input_cost": None,
        "output_cost": None,
        "total_cost": None,
    }
    assert aggregate["accounting"] == {
        "tokens_available": True,
        "cost_available": False,
        "reason": "aggregate_currency_conflict",
    }
    assert len(aggregate["requests"]) == 2
    assert all("currency" not in request for request in aggregate["requests"])


@pytest.mark.asyncio
async def test_rich_malformed_cache_bucket_retains_authoritative_base_tokens(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    rule = _rich_client_rule(cache_mode="implicit")
    config = _rich_client_config(
        [rule],
        usage_accounting="openai-chat-cached-tokens-v1",
    )
    usage = _openai_usage(prompt_tokens_details={"cached_tokens": True})

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(usage)
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context("implicit"),
        )
        result = await client.generate("Hello")

    assert result.text == "ok"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 10
    assert result.usage.total_tokens == 110
    assert result.usage.total_cost is None
    assert result.usage.accounting.tokens_available is True
    assert result.usage.accounting.cost_available is False
    assert result.usage.accounting.reason == "malformed_usage"


@pytest.mark.asyncio
async def test_rich_profile_contradiction_retains_authoritative_base_tokens(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    rule = _rich_client_rule(cache_mode="none")
    config = _rich_client_config(
        [rule],
        usage_accounting="openai-chat-cached-tokens-v1",
    )
    usage = _openai_usage(prompt_tokens_details={"cached_tokens": 1})

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(usage)
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context("none"),
        )
        result = await client.generate("Hello")

    assert result.text == "ok"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 10
    assert result.usage.total_cost is None
    assert result.usage.accounting.reason == "profile_contradiction"


@pytest.mark.asyncio
async def test_rich_ambiguity_returns_generated_text_and_unknown_cost(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    first = _rich_client_rule(rule_id="first")
    second = _rich_client_rule(rule_id="second")
    config = _rich_client_config([first, second])

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(_openai_usage())
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        result = await client.generate("Hello")

    assert result.text == "ok"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 10
    assert result.usage.total_cost is None
    assert result.usage.accounting.reason == "pricing_ambiguity"


@pytest.mark.asyncio
async def test_rich_calculation_failure_returns_generated_text(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule()])

    with (
        patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls,
        patch(
            "llm_exec_core.client.calculate_pricing_cost",
            side_effect=config_module.PricingCalculationError(
                "sanitized direct calculation failure"
            ),
        ),
    ):
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(_openai_usage())
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        result = await client.generate("Hello")

    assert result.text == "ok"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 10
    assert result.usage.total_cost is None
    assert result.usage.accounting.reason == "calculation_failure"
    assert "sanitized direct calculation failure" not in repr(
        result.usage.accounting
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_tokens",
    [
        pytest.param(10**308, id="non-finite-float"),
        pytest.param(10**400, id="integer-conversion-overflow"),
    ],
)
async def test_rich_arithmetic_failure_preserves_result_and_request(
    monkeypatch, prompt_tokens
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule()])
    usage = _openai_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=10,
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(usage)
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        result = await client.generate(
            "Hello", request_name="arithmetic-failure"
        )

    assert mock_httpx_client.post.await_count == 1
    assert result.text == "ok"
    assert result.usage.input_tokens == prompt_tokens
    assert result.usage.output_tokens == 10
    assert result.usage.total_tokens == prompt_tokens + 10
    assert result.usage.input_cost is None
    assert result.usage.output_cost is None
    assert result.usage.total_cost is None
    assert result.usage.currency is None
    assert result.usage.accounting.tokens_available is True
    assert result.usage.accounting.cost_available is False
    assert result.usage.accounting.reason == "calculation_failure"
    aggregate = client.get_token_usage()
    assert aggregate["total_input_tokens"] == prompt_tokens
    assert aggregate["total_output_tokens"] == 10
    assert aggregate["cost"] == {
        "input_cost": None,
        "output_cost": None,
        "total_cost": None,
    }
    assert aggregate["accounting"] == {
        "tokens_available": True,
        "cost_available": False,
        "reason": "calculation_failure",
    }
    assert len(aggregate["requests"]) == 1
    request = aggregate["requests"][0]
    assert request["name"] == "arithmetic-failure"
    assert request["input_tokens"] == prompt_tokens
    assert request["output_tokens"] == 10
    assert request["input_cost"] is None
    assert request["output_cost"] is None
    assert request["total_cost"] is None
    assert request["accounting"] == aggregate["accounting"]


@pytest.mark.asyncio
async def test_generate_response_returns_text_none_fields_and_status(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule()])

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response()
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        text, usage = await client.generate_response("Hello")

    assert text == "ok"
    assert usage["total_input_tokens"] is None
    assert usage["total_output_tokens"] is None
    assert usage["cost"] == {
        "input_cost": None,
        "output_cost": None,
        "total_cost": None,
    }
    assert usage["accounting"] == {
        "tokens_available": False,
        "cost_available": False,
        "reason": "missing_usage",
    }


@pytest.mark.asyncio
async def test_valid_structured_output_survives_accounting_unavailability(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response()
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=_rich_client_config([_rich_client_rule()]),
            pricing_context=_rich_client_context(),
        )
        result = await client.generate(
            "Hello",
            structured_output_hook=lambda text: {"value": text},
        )

    assert result.text == "ok"
    assert result.structured == {"value": "ok"}
    assert result.usage.accounting.reason == "missing_usage"


@pytest.mark.asyncio
async def test_incomplete_aggregate_stays_incomplete_after_later_valid_request(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule()])

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = [
            _http_response(),
            _http_response(_openai_usage(prompt_tokens=7)),
        ]
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        first = await client.generate("first")
        second = await client.generate("second")

    assert first.usage.total_tokens is None
    assert second.usage.total_tokens == 17
    assert second.usage.total_cost == pytest.approx(0.054)
    aggregate = client.get_token_usage()
    assert aggregate["total_input_tokens"] is None
    assert aggregate["total_output_tokens"] is None
    assert aggregate["cost"]["total_cost"] is None
    assert aggregate["accounting"]["reason"] == "missing_usage"
    assert len(aggregate["requests"]) == 2
    assert aggregate["process_times"]["total_time"] >= 0
    assert len(aggregate["process_times"]["request_times"]) == 2


@pytest.mark.asyncio
async def test_unavailable_response_is_cached_and_later_local_hit_stays_zero(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule()])

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response()
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

    assert first.usage.accounting.reason == "missing_usage"
    assert cached.text == "ok"
    assert cached.usage.input_tokens == 0
    assert cached.usage.output_tokens == 0
    assert cached.usage.total_tokens == 0
    assert cached.usage.input_cost == 0.0
    assert cached.usage.output_cost == 0.0
    assert cached.usage.total_cost == 0.0
    assert cached.usage.accounting.tokens_available is True
    assert cached.usage.accounting.cost_available is True
    assert client.get_token_usage() == aggregate_after_first
    assert mock_httpx_client.post.await_count == 1


@pytest.mark.asyncio
async def test_unrelated_programming_error_in_accounting_is_not_swallowed(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config([_rich_client_rule()])

    with (
        patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls,
        patch.object(
            LLMClient,
            "_calculate_rich_pricing",
            side_effect=RuntimeError("programming bug"),
        ),
    ):
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(_openai_usage())
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_rich_client_context(),
        )
        with pytest.raises(RuntimeError, match="programming bug"):
            await client.generate("Hello")


@pytest.mark.asyncio
async def test_malformed_generation_content_is_not_accounting_unavailable(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "choices": [],
        "usage": _openai_usage(),
    }

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = response
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=_rich_client_config([_rich_client_rule()]),
            pricing_context=_rich_client_context(),
        )
        with pytest.raises(IndexError):
            await client.generate("Hello")


@pytest.mark.asyncio
async def test_structured_output_hook_failure_is_not_swallowed(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = _http_response(_openai_usage())
        mock_cls.return_value = mock_httpx_client
        client = LLMClient(
            "test-model",
            config_source=_rich_client_config([_rich_client_rule()]),
            pricing_context=_rich_client_context(),
        )

        def fail_hook(value):
            raise RuntimeError("structured output failed")

        with pytest.raises(RuntimeError, match="structured output failed"):
            await client.generate("Hello", structured_output_hook=fail_hook)


def test_invalid_rich_route_configuration_fails_before_http(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _rich_client_config(
        [_rich_client_rule()],
        usage_accounting=None,
    )

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        with pytest.raises(ValidationError):
            LLMClient(
                "test-model",
                config_source=config,
                pricing_context=_rich_client_context(),
            )

    mock_cls.assert_not_called()


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
