from llm_exec_core.client import LLMClient
from llm_exec_core.config import (
    get_model_details,
    get_supported_models,
    load_all_settings,
)
import yaml

CUSTOM_CONFIG = {
    "_shared": {"ignored": True},
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
    },
}


def test_load_all_settings_accepts_dict_and_skips_private_keys():
    settings = load_all_settings(CUSTOM_CONFIG)

    assert list(settings) == ["test-provider"]


def test_get_supported_models_accepts_dict():
    assert get_supported_models(CUSTOM_CONFIG) == ["test-model"]


def test_get_model_details_returns_provider_name_settings_and_model():
    provider_name, provider_settings, model_details = get_model_details(
        "test-model", CUSTOM_CONFIG
    )

    assert provider_name == "test-provider"
    assert provider_settings.api_key_env_var == "TEST_API_KEY"
    assert model_details.id == "provider-model-id"


def test_model_details_accepts_capability_metadata():
    config = {
        "test-provider": {
            **CUSTOM_CONFIG["test-provider"],
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                    "capabilities": {
                        "version": "unit-2026-07-02",
                        "source": "unit",
                        "source_date": "2026-07-02",
                        "strict_response_schema": True,
                        "json_object_response": True,
                        "tools": True,
                        "tool_streaming": False,
                        "tool_choice": True,
                        "parallel_tool_calls": True,
                        "reasoning_controls": ["reasoning_effort"],
                        "openrouter_supported_parameters": [
                            "response_format",
                            "tools",
                        ],
                    },
                }
            },
        }
    }

    _, _, model_details = get_model_details("test-model", config)

    assert model_details.capabilities is not None
    assert model_details.capabilities.version == "unit-2026-07-02"
    assert model_details.capabilities.strict_response_schema is True
    assert model_details.capabilities.tool_streaming is False
    assert model_details.capabilities.reasoning_controls == [
        "reasoning_effort"
    ]


def test_default_config_includes_review_target_capabilities():
    expected_models = {
        "deepseek-v3.2": {
            "id": "deepseek-v3-2-251201",
            "strict": False,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "deepseek-r1": {
            "id": "deepseek-r1-250528",
            "strict": False,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "deepseek-v4-flash": {
            "id": "deepseek-v4-flash",
            "strict": False,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "deepseek-v4-pro": {
            "id": "deepseek-v4-pro",
            "strict": False,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "glm-5.2": {
            "id": "glm-5.2",
            "strict": False,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "glm-5.2-or": {
            "id": "z-ai/glm-5.2",
            "strict": True,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": True,
        },
        "qwen3.6-flash": {
            "id": "qwen3.6-flash",
            "strict": False,
            "json": False,
            "tools": False,
            "tool_streaming": False,
            "tool_choice": False,
            "parallel": False,
        },
        "qwen-max": {
            "id": "qwen-max",
            "strict": False,
            "json": False,
            "tools": True,
            "tool_streaming": False,
            "tool_choice": True,
            "parallel": False,
        },
        "gemini-3-flash": {
            "id": "gemini-3-flash-preview",
            "strict": True,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "gemini-3.1-flash-lite": {
            "id": "gemini-3.1-flash-lite-preview",
            "strict": True,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "gpt-5.5-or": {
            "id": "openai/gpt-5.5",
            "strict": True,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "claude-sonnet-5-or": {
            "id": "anthropic/claude-sonnet-5",
            "strict": True,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "claude-opus-4.8-or": {
            "id": "anthropic/claude-opus-4.8",
            "strict": True,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "doubao-seed-2.1-pro": {
            "id": "doubao-seed-2-1-pro-260628",
            "strict": True,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
    }

    settings = load_all_settings()
    models = {
        model_name: model_details
        for provider_settings in settings.values()
        for model_name, model_details in provider_settings.models.items()
    }

    assert set(expected_models).issubset(models)
    for model_name, expected in expected_models.items():
        model_details = models[model_name]
        capabilities = models[model_name].capabilities
        assert capabilities is not None
        assert model_details.id == expected["id"]
        assert capabilities.version
        assert capabilities.source
        assert capabilities.source_date
        assert capabilities.strict_response_schema is expected["strict"]
        assert capabilities.json_object_response is expected["json"]
        assert capabilities.tools is expected["tools"]
        assert capabilities.tool_streaming is expected["tool_streaming"]
        assert capabilities.tool_choice is expected["tool_choice"]
        assert capabilities.parallel_tool_calls is expected["parallel"]


def test_default_openrouter_capabilities_match_supported_parameters():
    openrouter_models = {
        "glm-5.2-or",
        "gpt-5.5-or",
        "claude-sonnet-5-or",
        "claude-opus-4.8-or",
    }

    settings = load_all_settings()
    models = {
        model_name: model_details
        for provider_settings in settings.values()
        for model_name, model_details in provider_settings.models.items()
    }

    for model_name in openrouter_models:
        capabilities = models[model_name].capabilities
        assert capabilities is not None
        supported = capabilities.openrouter_supported_parameters
        assert "response_format" in supported
        assert "structured_outputs" in supported
        assert "provider" not in supported


def test_provider_settings_accepts_configured_max_tokens_retry_policy():
    config = {
        "test-provider": {
            **CUSTOM_CONFIG["test-provider"],
            "max_tokens_retry": {
                "status_code": 404,
                "body_contains": "no allowed providers",
                "max_tokens_limit": 8192,
            },
        }
    }

    _, provider_settings, _ = get_model_details("test-model", config)

    assert provider_settings.max_tokens_retry is not None
    assert provider_settings.max_tokens_retry.status_code == 404
    assert (
        provider_settings.max_tokens_retry.body_contains
        == "no allowed providers"
    )
    assert provider_settings.max_tokens_retry.max_tokens_limit == 8192


def test_explicit_config_source_is_not_polluted_by_default_cache():
    get_supported_models()

    first = get_supported_models(CUSTOM_CONFIG)
    second = get_supported_models(
        {
            "other-provider": {
                "api_key_env_var": "OTHER_API_KEY",
                "api_base_url": "https://example.invalid/chat/completions",
                "temperature": 0.1,
                "max_tokens": 128,
                "context_window": 4096,
                "pricing_currency": "$",
                "models": {
                    "other-model": {
                        "id": "other-model-id",
                        "pricing": {"input": 1.0, "output": 2.0},
                    }
                },
            }
        }
    )

    assert first == ["test-model"]
    assert second == ["other-model"]


def test_explicit_config_source_does_not_use_default_cache():
    default_models = set(get_supported_models())
    custom_models = get_supported_models(
        {
            "other-provider": {
                "api_key_env_var": "OTHER_API_KEY",
                "api_base_url": "https://example.invalid/chat/completions",
                "temperature": 0.1,
                "max_tokens": 128,
                "context_window": 4096,
                "pricing_currency": "$",
                "models": {
                    "other-model": {
                        "id": "other-model-id",
                        "pricing": {"input": 1.0, "output": 2.0},
                    }
                },
            }
        }
    )

    assert set(custom_models) == {"other-model"}
    assert default_models.isdisjoint(custom_models)


def test_load_all_settings_accepts_path(tmp_path):
    custom_config = {
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

    config_path = tmp_path / "llm_config.yml"
    with config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(custom_config, fp)

    settings = load_all_settings(config_path)

    assert set(settings.keys()) == {"test-provider"}
    assert list(settings["test-provider"].models.keys()) == ["test-model"]


def test_default_settings_cache_returns_isolated_objects(monkeypatch):
    first = load_all_settings()
    provider_name = next(iter(first))
    provider_settings = first[provider_name]
    model_name = next(iter(provider_settings.models))
    model_details = provider_settings.models[model_name]

    original_api_base_url = provider_settings.api_base_url
    original_model_id = model_details.id
    original_input_price = model_details.pricing.input

    monkeypatch.setenv(provider_settings.api_key_env_var, "test-key")

    provider_settings.api_base_url = "https://mutated.invalid/v1"
    model_details.id = "mutated-model-id"
    model_details.pricing.input = 999.0

    second = load_all_settings()
    second_provider = second[provider_name]
    second_model = second_provider.models[model_name]

    assert second_provider.api_base_url == original_api_base_url
    assert second_model.id == original_model_id
    assert second_model.pricing.input == original_input_price

    client = LLMClient(model_name)

    assert client.api_url == original_api_base_url
    assert client.model == original_model_id
    assert client.pricing.input == original_input_price
