import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import llm_exec_core.config as config_module
from llm_exec_core.client import LLMClient

STREAM_USAGE_MISSING = object()


def _stream_usage(
    prompt_tokens=100,
    completion_tokens=10,
    prompt_tokens_details=STREAM_USAGE_MISSING,
):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if prompt_tokens_details is not STREAM_USAGE_MISSING:
        usage["prompt_tokens_details"] = prompt_tokens_details
    return usage


def _stream_rule(cache_mode="none"):
    cache_rates = {
        "none": (None, None),
        "implicit": (0.5, None),
        "explicit": (0.5, 3.0),
    }
    cache_read_input, cache_write_input = cache_rates[cache_mode]
    return {
        "rule_id": f"{cache_mode}-rule",
        "billing_model_id": "provider-model-id",
        "capability_snapshot_id": None,
        "region": "us-east",
        "service_scope": "global",
        "deployment_type": "serverless",
        "output_mode": "thinking",
        "request_mode": "realtime",
        "cache_mode": cache_mode,
        "input_tokens_gt": 0,
        "input_tokens_lte": None,
        "currency": "USD",
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
                "url": "https://help.aliyun.com/zh/model-studio/context-cache",
                "retrieved_on": "2026-07-23",
                "facts": ["Cache modes expose distinct token buckets."],
            }
        ],
    }


def _stream_config(cache_mode="none"):
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
                        "rules": [_stream_rule(cache_mode)],
                    },
                }
            },
        }
    }


def _stream_context(cache_mode="none"):
    return {
        "region": "us-east",
        "service_scope": "global",
        "deployment_type": "serverless",
        "output_mode": "thinking",
        "request_mode": "realtime",
        "cache_mode": cache_mode,
    }


def _mock_streaming_response(chunks):
    response = MagicMock()
    response.raise_for_status = MagicMock()

    async def mock_lines():
        for chunk in chunks:
            if chunk == "[DONE]":
                yield "data: [DONE]"
            else:
                yield f"data: {json.dumps(chunk)}"

    response.aiter_lines = mock_lines
    return response


def _mock_non_streaming_response(usage):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": usage,
    }
    return response


@pytest.mark.asyncio
async def test_streaming_without_callback_does_not_print(monkeypatch, capsys):
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
    mock_response.raise_for_status = MagicMock()

    async def mock_lines():
        yield 'data: {"choices": [{"delta": {"content": "A"}}]}'
        yield 'data: {"choices": [{"delta": {"content": "B"}}]}'
        yield "data: [DONE]"

    mock_response.aiter_lines = mock_lines

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield mock_response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        response, _ = await client.generate_response("Hello", stream=True)

    assert response == "AB"
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_stream_callback_receives_chunks_in_order(monkeypatch):
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
    mock_response.raise_for_status = MagicMock()

    async def mock_lines():
        yield 'data: {"choices": [{"delta": {"content": "hel"}}]}'
        yield 'data: {"choices": [{"delta": {"content": "lo"}}]}'
        yield 'data: {"choices": [{"delta": {"content": "!"}}]}'
        yield "data: [DONE]"

    mock_response.aiter_lines = mock_lines

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield mock_response

    received_chunks = []

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        response, _ = await client.generate_response(
            "Hello",
            stream=True,
            stream_callback=received_chunks.append,
        )

    assert received_chunks == ["hel", "lo", "!"]
    assert response == "hello!"


@pytest.mark.asyncio
async def test_generate_response_reraises_cancelled_error(monkeypatch):
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

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = asyncio.CancelledError()
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)

        with pytest.raises(asyncio.CancelledError):
            await client.generate_response("Hello")


@pytest.mark.asyncio
async def test_streaming_cancellation_during_iteration_propagates(monkeypatch):
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
    mock_response.raise_for_status = MagicMock()

    async def mock_lines():
        yield 'data: {"choices": [{"delta": {"content": "A"}}]}'
        raise asyncio.CancelledError()

    mock_response.aiter_lines = mock_lines

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield mock_response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)

        with pytest.raises(asyncio.CancelledError):
            await client.generate_response("Hello", stream=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cache_mode", "usage"),
    [
        ("none", _stream_usage(prompt_tokens_details=None)),
        (
            "implicit",
            _stream_usage(
                prompt_tokens_details={
                    "cached_tokens": 20,
                    "cache_creation_input_tokens": 0,
                }
            ),
        ),
        (
            "explicit",
            _stream_usage(
                prompt_tokens_details={
                    "cached_tokens": 20,
                    "cache_creation_input_tokens": 30,
                }
            ),
        ),
    ],
)
async def test_rich_streaming_and_non_streaming_have_identical_usage_costs(
    monkeypatch, cache_mode, usage
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _stream_config(cache_mode)
    stream_response = _mock_streaming_response(
        [
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [], "usage": usage},
            "[DONE]",
        ]
    )

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield stream_response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_httpx_client.post.return_value = _mock_non_streaming_response(
            usage
        )
        mock_cls.return_value = mock_httpx_client

        streaming_client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_stream_context(cache_mode),
        )
        non_streaming_client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_stream_context(cache_mode),
        )
        streaming_result = await streaming_client.generate(
            "Hello", stream=True
        )
        non_streaming_result = await non_streaming_client.generate("Hello")

    streaming_values = (
        streaming_result.usage.input_tokens,
        streaming_result.usage.output_tokens,
        streaming_result.usage.input_cost,
        streaming_result.usage.output_cost,
        streaming_result.usage.total_cost,
        streaming_result.usage.currency,
    )
    non_streaming_values = (
        non_streaming_result.usage.input_tokens,
        non_streaming_result.usage.output_tokens,
        non_streaming_result.usage.input_cost,
        non_streaming_result.usage.output_cost,
        non_streaming_result.usage.total_cost,
        non_streaming_result.usage.currency,
    )
    assert streaming_values[:-1] == pytest.approx(non_streaming_values[:-1])
    assert streaming_values[-1] == non_streaming_values[-1] == "USD"


PARITY_INVALID_USAGE_CASES = [
    (
        "none",
        _stream_usage(prompt_tokens_details={"cached_tokens": 1}),
    ),
    ("none", _stream_usage(prompt_tokens=True)),
    ("none", _stream_usage(completion_tokens="10")),
    ("implicit", _stream_usage()),
    (
        "implicit",
        _stream_usage(prompt_tokens_details={"cached_tokens": True}),
    ),
    (
        "implicit",
        _stream_usage(prompt_tokens_details={"cached_tokens": 101}),
    ),
    (
        "implicit",
        _stream_usage(
            prompt_tokens_details={
                "cached_tokens": 0,
                "cache_creation_input_tokens": 1,
            }
        ),
    ),
    (
        "explicit",
        _stream_usage(prompt_tokens_details={"cached_tokens": 0}),
    ),
    (
        "explicit",
        _stream_usage(
            prompt_tokens_details={
                "cached_tokens": 0.5,
                "cache_creation_input_tokens": 0,
            }
        ),
    ),
    (
        "explicit",
        _stream_usage(
            prompt_tokens_details={
                "cached_tokens": 0,
                "cache_creation_input_tokens": -1,
            }
        ),
    ),
    (
        "explicit",
        _stream_usage(
            prompt_tokens_details={
                "cached_tokens": 60,
                "cache_creation_input_tokens": 41,
            }
        ),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("cache_mode", "usage"), PARITY_INVALID_USAGE_CASES)
async def test_rich_streaming_and_non_streaming_have_identical_failures(
    monkeypatch, cache_mode, usage
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _stream_config(cache_mode)
    stream_response = _mock_streaming_response(
        [{"choices": [], "usage": usage}, "[DONE]"]
    )

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield stream_response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_httpx_client.post.return_value = _mock_non_streaming_response(
            usage
        )
        mock_cls.return_value = mock_httpx_client

        streaming_client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_stream_context(cache_mode),
        )
        non_streaming_client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_stream_context(cache_mode),
        )
        streaming_before = deepcopy(streaming_client.get_token_usage())
        non_streaming_before = deepcopy(non_streaming_client.get_token_usage())

        with pytest.raises(
            config_module.PricingCalculationError
        ) as stream_error:
            await streaming_client.generate("Hello", stream=True)
        with pytest.raises(
            config_module.PricingCalculationError
        ) as non_stream_error:
            await non_streaming_client.generate("Hello")

    assert type(stream_error.value) is type(non_stream_error.value)
    assert streaming_client.get_token_usage() == streaming_before
    assert non_streaming_client.get_token_usage() == non_streaming_before


@pytest.mark.asyncio
async def test_rich_streaming_uses_last_non_null_usage(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _stream_config("none")
    stream_response = _mock_streaming_response(
        [
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [], "usage": _stream_usage(prompt_tokens=5)},
            {"choices": [], "usage": None},
            {"choices": [], "usage": _stream_usage(prompt_tokens=7)},
            "[DONE]",
        ]
    )

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield stream_response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_stream_context(),
        )
        result = await client.generate("Hello", stream=True)

    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        [{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"],
        [
            {"choices": [], "usage": _stream_usage()},
            {"choices": [], "usage": []},
            "[DONE]",
        ],
    ],
)
async def test_rich_streaming_missing_or_malformed_final_usage_is_atomic(
    monkeypatch, chunks
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _stream_config("none")
    stream_response = _mock_streaming_response(chunks)

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield stream_response

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient(
            "test-model",
            config_source=config,
            pricing_context=_stream_context(),
        )
        before = deepcopy(client.get_token_usage())

        with pytest.raises(config_module.PricingCalculationError):
            await client.generate("Hello", stream=True)

    assert client.get_token_usage() == before


@pytest.mark.asyncio
async def test_legacy_streaming_still_estimates_zero_or_missing_usage(
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
    stream_response = _mock_streaming_response(
        [{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"]
    )

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield stream_response

    with (
        patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls,
        patch("llm_exec_core.client.estimate_tokens", side_effect=[3, 2]),
    ):
        mock_httpx_client = AsyncMock()
        mock_httpx_client.stream = mock_stream
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        result = await client.generate("Hello", stream=True)

    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 2
    assert result.usage.input_cost == pytest.approx(0.000003)
    assert result.usage.output_cost == pytest.approx(0.000004)
    assert result.usage.currency == "$"
