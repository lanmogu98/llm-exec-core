import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_exec_core.client import LLMClient


def test_get_supported_models_is_staticmethod_with_optional_config_source():
    sig = inspect.signature(LLMClient.get_supported_models)

    assert isinstance(
        inspect.getattr_static(LLMClient, "get_supported_models"), staticmethod
    )
    assert list(sig.parameters) == ["config_source"]
    assert sig.parameters["config_source"].default is None


def test_get_supported_models_forwards_config_source_unchanged():
    config_source = {"test-provider": {}}
    expected_models = ["test-model"]

    with patch(
        "llm_exec_core.client.get_supported_models",
        return_value=expected_models,
    ) as get_models:
        result = LLMClient.get_supported_models(config_source)

    assert result is expected_models
    get_models.assert_called_once_with(config_source)


@pytest.mark.asyncio
async def test_generate_response_returns_legacy_tuple(monkeypatch):
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
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = mock_response
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        response, usage = await client.generate_response("Hello")

    assert response == "Test response"
    assert usage["total_input_tokens"] == 10
    assert usage["total_output_tokens"] == 20
    assert usage["process_times"]["total_time"] >= 0
    assert "accounting" not in usage
    assert "accounting" not in client.get_token_usage()


@pytest.mark.asyncio
async def test_flat_route_with_optional_usage_profile_retains_legacy_behavior(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://example.invalid/chat/completions",
            "pricing_currency": "$",
            "usage_accounting": "openai-chat-cache-hit-miss-v1",
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                }
            },
        }
    }
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": "legacy"}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "prompt_cache_hit_tokens": 9,
            "prompt_cache_miss_tokens": 1,
        },
    }
    response.raise_for_status = MagicMock()

    with patch("llm_exec_core.client.httpx.AsyncClient") as mock_cls:
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.return_value = response
        mock_cls.return_value = mock_httpx_client

        client = LLMClient("test-model", config_source=config)
        result = await client.generate("Hello")

    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.usage.input_cost == pytest.approx(0.00001)
    assert result.usage.output_cost == pytest.approx(0.00004)
    assert "accounting" not in result.usage.to_legacy_dict()
