from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import llm_exec_core.config as config_module
from llm_exec_core.client import LLMClient
from llm_exec_core.config import (
    ModelDetails,
    Pricing,
    ProviderSettings,
    get_model_details,
    get_provider_settings,
    get_supported_models,
    load_all_settings,
)

REQUIRED_SOURCE_MESSAGE = (
    "config_source is required; pass a complete caller-owned catalog as a "
    "pathlib.Path or dict. See "
    "https://github.com/lanmogu98/llm-exec-core/issues/5."
)

ROOT = Path(__file__).parents[2]
BUNDLED_CONFIG_PATH = ROOT / "src" / "llm_exec_core" / "llm_config.yml"
README_PATH = ROOT / "README.md"

EXPECTED_DEFAULT_MODELS = [
    "deepseek-v4-flash-volc",
    "deepseek-v4-pro-volc",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "gemini-3-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro",
    "gemini-3.5-flash",
    "qwen3.6-flash",
    "glm-5.2",
    "glm-5",
    "glm-5.1",
    "glm-5.2-or",
    "glm-5-or",
    "glm-5.1-or",
    "glm-5-turbo-or",
    "doubao-seed-2.1-pro",
    "doubao-seed-1.6",
    "gpt-5.5-or",
    "claude-sonnet-5-or",
    "claude-opus-4.8-or",
]

REMOVED_DEFAULT_MODELS = [
    "deepseek-v3.2",
    "deepseek-r1",
    "gemini-2.5-flash-free",
    "gemini-2.5-flash-lite-free",
    "gemini-3-flash-free",
    "gemini-3.1-flash-lite-free",
    "qwen-max",
    "qwen-turbo",
    "qwen-plus",
    "qwen3.5-plus",
    "qwen3-max-preview",
    "qwen3-max",
    "glm-4.5",
    "glm-4.6",
    "glm-4.7",
    "glm-4.5-or",
    "glm-4.6-or",
    "glm-4.7-or",
    "gpt-4o-or",
    "gpt-4.1-or",
    "gpt-5-or",
    "gpt-5.2-or",
    "gpt-5.4-or",
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


OMITTED = object()


def _call_catalog_api(api_name, config_source=OMITTED):
    source_args = () if config_source is OMITTED else (config_source,)
    model_name = "test-model"
    provider_name = "test-provider"

    if api_name == "load_all_settings":
        return load_all_settings(*source_args)
    if api_name == "get_supported_models":
        return get_supported_models(*source_args)
    if api_name == "get_model_details_valid":
        return get_model_details(model_name, *source_args)
    if api_name == "get_model_details_unknown":
        if config_source is OMITTED or config_source is None:
            return get_model_details("unknown-model", *source_args)
        with pytest.raises(ValueError):
            return get_model_details("unknown-model", *source_args)
    if api_name == "get_provider_settings_valid":
        return get_provider_settings(provider_name, *source_args)
    if api_name == "get_provider_settings_unknown":
        if config_source is OMITTED or config_source is None:
            return get_provider_settings("unknown-provider", *source_args)
        with pytest.raises(ValueError):
            return get_provider_settings("unknown-provider", *source_args)
    if api_name == "LLMClient.get_supported_models":
        return LLMClient.get_supported_models(*source_args)
    if config_source is OMITTED:
        return LLMClient(model_name)
    return LLMClient(model_name, config_source=config_source)


@pytest.mark.parametrize("source_mode", ["omitted", "none"])
@pytest.mark.parametrize("api_name", CATALOG_API_CASES)
def test_catalog_public_calls_require_explicit_source_before_build(
    source_mode, api_name
):
    config_source = OMITTED if source_mode == "omitted" else None

    with patch.object(config_module, "_build_settings") as build_settings:
        with pytest.raises(ValueError) as error:
            _call_catalog_api(api_name, config_source)

    assert str(error.value) == REQUIRED_SOURCE_MESSAGE
    build_settings.assert_not_called()


def test_legacy_default_catalog_symbols_are_removed():
    assert not hasattr(config_module, "_DEFAULT_PROVIDER_SETTINGS")
    assert not hasattr(config_module, "_get_default_config_path")


@pytest.mark.parametrize("source_kind", ["dict", "path"])
@pytest.mark.parametrize("api_name", CATALOG_API_CASES)
def test_explicit_catalog_public_calls_preserve_behavior(
    monkeypatch, tmp_path, source_kind, api_name
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config_source = CUSTOM_CONFIG
    if source_kind == "path":
        config_source = tmp_path / "llm_config.yml"
        with config_source.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(CUSTOM_CONFIG, handle)

    _call_catalog_api(api_name, config_source)


def test_explicit_reference_supported_models_preserve_order():
    models = LLMClient.get_supported_models(BUNDLED_CONFIG_PATH)

    assert models == EXPECTED_DEFAULT_MODELS


def test_unknown_model_uses_explicit_snapshot_and_preserves_error_order():
    with patch(
        "llm_exec_core.config.load_all_settings", wraps=load_all_settings
    ) as loader:
        with pytest.raises(ValueError) as error:
            get_model_details("unknown-model", BUNDLED_CONFIG_PATH)

    assert loader.call_count == 1
    assert str(error.value) == (
        "Model 'unknown-model' not found. Available models: "
        + ", ".join(EXPECTED_DEFAULT_MODELS)
    )


@pytest.mark.parametrize("removed_model", REMOVED_DEFAULT_MODELS)
def test_removed_default_models_preserve_value_error_shape(removed_model):
    with pytest.raises(ValueError) as error:
        get_model_details(removed_model, BUNDLED_CONFIG_PATH)

    assert str(error.value) == (
        f"Model '{removed_model}' not found. Available models: "
        + ", ".join(EXPECTED_DEFAULT_MODELS)
    )


@pytest.mark.parametrize(
    (
        "alias",
        "provider_name",
        "model_id",
        "api_key_env_var",
        "api_base_url",
        "input_price",
        "output_price",
        "max_tokens",
        "context_window",
        "source",
    ),
    [
        (
            "deepseek-v4-flash-volc",
            "deepseek-volcengine",
            "deepseek-v4-flash-260425",
            "DEEPSEEK_API_KEY_VOLC",
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            1.0,
            2.0,
            384000,
            1024000,
            "https://www.volcengine.com/docs/82379/1330310?lang=zh",
        ),
        (
            "deepseek-v4-pro-volc",
            "deepseek-volcengine",
            "deepseek-v4-pro-260425",
            "DEEPSEEK_API_KEY_VOLC",
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            12.0,
            24.0,
            384000,
            1024000,
            "https://www.volcengine.com/docs/82379/1330310?lang=zh",
        ),
        (
            "deepseek-v4-flash",
            "deepseek",
            "deepseek-v4-flash",
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com/chat/completions",
            1.0,
            2.0,
            384000,
            1000000,
            "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
        ),
        (
            "deepseek-v4-pro",
            "deepseek",
            "deepseek-v4-pro",
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com/chat/completions",
            3.0,
            6.0,
            384000,
            1000000,
            "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
        ),
    ],
)
def test_deepseek_routes_have_exact_ids_prices_limits_and_sources(
    alias,
    provider_name,
    model_id,
    api_key_env_var,
    api_base_url,
    input_price,
    output_price,
    max_tokens,
    context_window,
    source,
):
    actual_provider, provider, model = get_model_details(
        alias, BUNDLED_CONFIG_PATH
    )

    assert actual_provider == provider_name
    assert provider.api_key_env_var == api_key_env_var
    assert provider.api_base_url == api_base_url
    assert provider.pricing_currency == "¥"
    assert provider.max_tokens == max_tokens
    assert provider.context_window == context_window
    assert provider.request_overrides is None
    assert model.id == model_id
    assert model.pricing.input == input_price
    assert model.pricing.output == output_price
    assert model.capabilities is not None
    assert model.capabilities.source == source
    assert model.capabilities.source_date == "2026-07-20"


def test_deepseek_native_and_ark_aliases_do_not_shadow_or_fallback():
    settings = load_all_settings(BUNDLED_CONFIG_PATH)

    assert list(settings["deepseek-volcengine"].models) == [
        "deepseek-v4-flash-volc",
        "deepseek-v4-pro-volc",
    ]
    assert list(settings["deepseek"].models) == [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]
    assert set(settings["deepseek-volcengine"].models).isdisjoint(
        settings["deepseek"].models
    )


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
        "deepseek-v4-flash-volc": {
            "id": "deepseek-v4-flash-260425",
            "strict": False,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "deepseek-v4-pro-volc": {
            "id": "deepseek-v4-pro-260425",
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
            "id": "gemini-3.1-flash-lite",
            "strict": True,
            "json": True,
            "tools": True,
            "tool_streaming": True,
            "tool_choice": True,
            "parallel": False,
        },
        "gemini-3.5-flash": {
            "id": "gemini-3.5-flash",
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

    settings = load_all_settings(BUNDLED_CONFIG_PATH)
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

    settings = load_all_settings(BUNDLED_CONFIG_PATH)
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


def test_paid_gemini_catalog_has_exact_ids_and_gemini_3_5_facts():
    settings = load_all_settings(BUNDLED_CONFIG_PATH)
    gemini = settings["gemini"]

    assert "gemini-free" not in settings
    assert list(gemini.models) == [
        "gemini-3-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro",
        "gemini-3.5-flash",
    ]
    assert {
        model_name: model.id for model_name, model in gemini.models.items()
    } == {
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
        "gemini-3.1-pro": "gemini-3.1-pro-preview",
        "gemini-3.5-flash": "gemini-3.5-flash",
    }
    assert gemini.context_window == 1048576
    assert gemini.max_tokens == 65536
    assert gemini.request_overrides is None

    gemini_3_5 = gemini.models["gemini-3.5-flash"]
    assert gemini_3_5.pricing.input == 1.5
    assert gemini_3_5.pricing.output == 9.0
    assert gemini_3_5.capabilities is not None
    assert gemini_3_5.capabilities.source == (
        "https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash"
    )
    assert gemini_3_5.capabilities.source_date == "2026-07-20"
    assert all(
        not model_name.endswith("-free")
        for provider in settings.values()
        for model_name in provider.models
    )


def test_pruned_provider_models_match_the_transitional_policy():
    settings = load_all_settings(BUNDLED_CONFIG_PATH)

    assert {
        provider_name: list(settings[provider_name].models)
        for provider_name in (
            "qwen",
            "zhipu",
            "zhipu-openrouter",
            "openai-openrouter",
            "anthropic-openrouter",
        )
    } == {
        "qwen": ["qwen3.6-flash"],
        "zhipu": ["glm-5.2", "glm-5", "glm-5.1"],
        "zhipu-openrouter": [
            "glm-5.2-or",
            "glm-5-or",
            "glm-5.1-or",
            "glm-5-turbo-or",
        ],
        "openai-openrouter": ["gpt-5.5-or"],
        "anthropic-openrouter": [
            "claude-sonnet-5-or",
            "claude-opus-4.8-or",
        ],
    }


def test_doubao_provider_settings_remain_unchanged():
    with BUNDLED_CONFIG_PATH.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    assert raw_config["doubao"] == {
        "api_key_env_var": "DOUBAO_API_KEY",
        "api_base_url": (
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        ),
        "temperature": 0.6,
        "max_tokens": 32000,
        "context_window": 256000,
        "pricing_currency": "¥",
        "models": {
            "doubao-seed-2.1-pro": {
                "id": "doubao-seed-2-1-pro-260628",
                "pricing": {"input": 0.8, "output": 2.0},
                "capabilities": {
                    "version": "volcengine-doubao-seed-2.1-pro-2026-07-02",
                    "source": (
                        "https://www.volcengine.com/docs/82379/1330310"
                    ),
                    "source_date": "2026-07-02",
                    "strict_response_schema": True,
                    "json_object_response": True,
                    "tools": True,
                    "tool_streaming": True,
                    "tool_choice": True,
                    "reasoning_controls": [
                        "thinking",
                        "reasoning_effort",
                    ],
                },
            },
            "doubao-seed-1.6": {
                "id": "doubao-seed-1-6-250615",
                "pricing": {"input": 0.8, "output": 2.0},
            },
        },
    }


def test_readme_removes_only_stale_gemini_free_alias_guidance():
    readme = README_PATH.read_text(encoding="utf-8")

    assert (
        "Gemini paid/free aliases for the reviewed models share capabilities."
        not in readme
    )
    assert (
        "On Gemini\nroutes, `reasoning_effort` cannot be combined with "
        "`thinking_level` or\n`thinking_budget` under either normalized "
        "`google.thinking_config` payload\nshape; `include_thoughts` alone is "
        "allowed.\n\nOpenRouter target capabilities remain static at runtime."
        in readme
    )
    assert readme.startswith("# llm-exec-core\n")
    assert readme.endswith("Proprietary. See [LICENSE](LICENSE).\n")


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


@pytest.mark.parametrize("source_order", [("dict", "path"), ("path", "dict")])
def test_explicit_sources_return_fresh_typed_settings_in_both_orders(
    tmp_path, source_order
):
    config_path = tmp_path / "llm_config.yml"
    config_path.write_text(
        yaml.safe_dump(CUSTOM_CONFIG, sort_keys=False), encoding="utf-8"
    )
    sources = {"dict": CUSTOM_CONFIG, "path": config_path}

    first = load_all_settings(sources[source_order[0]])
    first["test-provider"].api_base_url = "https://mutated.invalid/v1"
    first["test-provider"].models["test-model"].id = "mutated-model-id"
    first["test-provider"].models["test-model"].pricing.input = 999.0

    for settings in (
        load_all_settings(sources[source_order[1]]),
        load_all_settings(sources[source_order[0]]),
    ):
        provider = settings["test-provider"]
        model = provider.models["test-model"]
        assert list(settings) == ["test-provider"]
        assert list(provider.models) == ["test-model"]
        assert isinstance(provider, ProviderSettings)
        assert isinstance(model, ModelDetails)
        assert isinstance(model.pricing, Pricing)
        assert (
            provider.api_base_url
            == CUSTOM_CONFIG["test-provider"]["api_base_url"]
        )
        assert model.id == "provider-model-id"
        assert model.pricing.input == 1.0


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


def test_packaged_reference_can_be_copied_and_loaded_explicitly(tmp_path):
    resource = files("llm_exec_core").joinpath("llm_config.yml")
    destination = tmp_path / "llm_config.yml"

    assert resource.is_file()
    reference = resource.read_bytes()
    assert reference
    destination.write_bytes(reference)
    settings = load_all_settings(destination)

    assert list(settings) == list(load_all_settings(BUNDLED_CONFIG_PATH))
    for config_source in (OMITTED, None):
        with pytest.raises(ValueError) as error:
            _call_catalog_api("load_all_settings", config_source)
        assert str(error.value) == REQUIRED_SOURCE_MESSAGE
