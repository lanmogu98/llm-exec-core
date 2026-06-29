from llm_exec_core.config import (
    get_model_details,
    get_supported_models,
    load_all_settings,
)

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


def test_explicit_config_source_is_not_polluted_by_default_cache():
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
