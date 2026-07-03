import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_exec_core.client import LLMClient


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
