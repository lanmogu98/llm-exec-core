import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_exec_core.client import LLMClient


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
