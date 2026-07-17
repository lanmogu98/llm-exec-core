import warnings
from unittest.mock import patch

import pytest
import yaml

import llm_exec_core.config as config_module
from llm_exec_core.client import LLMClient
from llm_exec_core.config import (
    get_model_details,
    get_provider_settings,
    get_supported_models,
    load_all_settings,
)

DEPRECATION_MESSAGE = (
    "Loading the bundled model catalog without config_source is deprecated. "
    "Pass config_source with a caller-owned catalog; see the migration "
    "contract at https://github.com/lanmogu98/llm-exec-core/issues/5."
)

EXPECTED_DEFAULT_MODELS = [
    "deepseek-v3.2",
    "deepseek-r1",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "gemini-3-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro",
    "gemini-2.5-flash-free",
    "gemini-2.5-flash-lite-free",
    "gemini-3-flash-free",
    "gemini-3.1-flash-lite-free",
    "qwen3.6-flash",
    "qwen-max",
    "qwen-turbo",
    "qwen-plus",
    "qwen3.5-plus",
    "qwen3-max-preview",
    "qwen3-max",
    "glm-5.2",
    "glm-4.5",
    "glm-4.6",
    "glm-4.7",
    "glm-5",
    "glm-5.1",
    "glm-5.2-or",
    "glm-4.5-or",
    "glm-4.6-or",
    "glm-4.7-or",
    "glm-5-or",
    "glm-5.1-or",
    "glm-5-turbo-or",
    "doubao-seed-2.1-pro",
    "doubao-seed-1.6",
    "gpt-4o-or",
    "gpt-4.1-or",
    "gpt-5-or",
    "gpt-5.2-or",
    "gpt-5.4-or",
    "gpt-5.5-or",
    "claude-sonnet-5-or",
    "claude-opus-4.8-or",
    "claude-sonnet-4-or",
    "claude-opus-4.6-or",
    "claude-sonnet-4.6-or",
    "claude-opus-4.7-or",
    "claude-haiku-4.5-or",
]

CATALOG_API_CASES = [
    "load_all_settings",
    "get_supported_models",
    "get_model_details_valid",
    "get_model_details_unknown",
    "get_provider_settings_valid",
    "get_provider_settings_unknown",
    "LLMClient.get_supported_models",
    "LLMClient",
]

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


def _call_catalog_api(api_name, config_source=None):
    source_args = () if config_source is None else (config_source,)
    model_name = "deepseek-v3.2" if config_source is None else "test-model"
    provider_name = (
        "deepseek-volcengine" if config_source is None else "test-provider"
    )

    if api_name == "load_all_settings":
        return load_all_settings(*source_args)
    if api_name == "get_supported_models":
        return get_supported_models(*source_args)
    if api_name == "get_model_details_valid":
        return get_model_details(model_name, *source_args)
    if api_name == "get_model_details_unknown":
        with pytest.raises(ValueError):
            get_model_details("unknown-model", *source_args)
        return None
    if api_name == "get_provider_settings_valid":
        return get_provider_settings(provider_name, *source_args)
    if api_name == "get_provider_settings_unknown":
        with pytest.raises(ValueError):
            get_provider_settings("unknown-provider", *source_args)
        return None
    if api_name == "LLMClient.get_supported_models":
        return LLMClient.get_supported_models(*source_args)
    if config_source is None:
        return LLMClient(model_name)
    return LLMClient(model_name, config_source=config_source)


@pytest.mark.parametrize("cache_state", ["cold", "warm"])
@pytest.mark.parametrize("api_name", CATALOG_API_CASES)
def test_default_catalog_public_calls_emit_one_actionable_warning(
    monkeypatch, cache_state, api_name
):
    monkeypatch.setattr(config_module, "_DEFAULT_PROVIDER_SETTINGS", None)
    monkeypatch.setenv("DEEPSEEK_API_KEY_VOLC", "test-key")

    if cache_state == "warm":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            load_all_settings()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _call_catalog_api(api_name)

    deprecations = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]
    assert deprecations == [DEPRECATION_MESSAGE]


@pytest.mark.parametrize("source_kind", ["dict", "path"])
@pytest.mark.parametrize("api_name", CATALOG_API_CASES)
def test_explicit_catalog_public_calls_emit_no_deprecation_warning(
    monkeypatch, tmp_path, source_kind, api_name
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config_source = CUSTOM_CONFIG
    if source_kind == "path":
        config_source = tmp_path / "llm_config.yml"
        with config_source.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(CUSTOM_CONFIG, handle)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _call_catalog_api(api_name, config_source)

    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]


def test_implicit_catalog_warning_uses_fixed_stacklevel(monkeypatch):
    monkeypatch.setattr(config_module, "_DEFAULT_PROVIDER_SETTINGS", None)

    with patch("llm_exec_core.config.warnings.warn") as warn:
        load_all_settings()

    warn.assert_called_once_with(
        DEPRECATION_MESSAGE,
        DeprecationWarning,
        stacklevel=2,
    )


def test_zero_argument_supported_models_preserve_order(monkeypatch):
    monkeypatch.setattr(config_module, "_DEFAULT_PROVIDER_SETTINGS", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models = LLMClient.get_supported_models()

    assert models == EXPECTED_DEFAULT_MODELS
    assert [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ] == [DEPRECATION_MESSAGE]


def test_unknown_model_reuses_snapshot_and_preserves_error_order(
    monkeypatch,
):
    monkeypatch.setattr(config_module, "_DEFAULT_PROVIDER_SETTINGS", None)

    with (
        patch(
            "llm_exec_core.config.load_all_settings", wraps=load_all_settings
        ) as loader,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as error:
            get_model_details("unknown-model")

    assert loader.call_count == 1
    assert str(error.value) == (
        "Model 'unknown-model' not found. Available models: "
        + ", ".join(EXPECTED_DEFAULT_MODELS)
    )
    assert [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ] == [DEPRECATION_MESSAGE]


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
        "glm-5.2-or": {"max_tokens", "temperature", "reasoning_effort"},
        "gpt-5.5-or": {
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
        },
        "claude-sonnet-5-or": {
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
        },
        "claude-opus-4.8-or": {
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
            "temperature",
        },
    }

    settings = load_all_settings()
    models = {
        model_name: model_details
        for provider_settings in settings.values()
        for model_name, model_details in provider_settings.models.items()
    }

    for model_name, required_parameters in openrouter_models.items():
        capabilities = models[model_name].capabilities
        assert capabilities is not None
        supported = capabilities.openrouter_supported_parameters
        assert "response_format" in supported
        assert "structured_outputs" in supported
        assert "provider" not in supported
        assert required_parameters.issubset(supported)
        assert capabilities.source_date == "2026-07-15"
        assert capabilities.version.endswith("2026-07-15")

    for model_name in (
        "gpt-5.5-or",
        "claude-sonnet-5-or",
    ):
        capabilities = models[model_name].capabilities
        assert capabilities is not None
        assert "temperature" not in (
            capabilities.openrouter_supported_parameters
        )


def test_paid_and_free_gemini_aliases_share_model_capabilities():
    settings = load_all_settings()

    for paid_name, free_name in (
        ("gemini-3-flash", "gemini-3-flash-free"),
        ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite-free"),
    ):
        paid = settings["gemini"].models[paid_name]
        free = settings["gemini-free"].models[free_name]

        assert paid.id == free.id
        assert paid.capabilities is not None
        assert free.capabilities == paid.capabilities
        assert free.pricing != paid.pricing


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


def test_default_cache_does_not_pollute_explicit_config_source(monkeypatch):
    monkeypatch.setattr(config_module, "_DEFAULT_PROVIDER_SETTINGS", None)
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


def test_explicit_config_source_does_not_populate_default_cache(monkeypatch):
    monkeypatch.setattr(config_module, "_DEFAULT_PROVIDER_SETTINGS", None)
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

    assert config_module._DEFAULT_PROVIDER_SETTINGS is None

    default_models = set(get_supported_models())

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
