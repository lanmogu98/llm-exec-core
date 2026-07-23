import math
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

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


def _rich_pricing_rule(
    *,
    rule_id="standard-us-realtime-none",
    cache_mode="none",
    input_tokens_gt=0,
    input_tokens_lte=100,
    currency="USD",
    request_mode="realtime",
    effective_from="2026-07-01T00:00:00Z",
    effective_until=None,
    rate_type="standard",
    cache_read_input=OMITTED,
    cache_write_input=OMITTED,
):
    cache_rates = {
        "none": (None, None),
        "implicit": (0.5, None),
        "explicit": (0.5, 3.0),
    }
    default_cache_read, default_cache_write = cache_rates[cache_mode]
    if cache_read_input is OMITTED:
        cache_read_input = default_cache_read
    if cache_write_input is OMITTED:
        cache_write_input = default_cache_write
    return {
        "rule_id": rule_id,
        "billing_model_id": "provider-model-id",
        "capability_snapshot_id": "provider-model-snapshot-2026-07-01",
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
        "effective_from": effective_from,
        "effective_until": effective_until,
        "rate_type": rate_type,
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
                "facts": ["Prices vary by total request input tier."],
            }
        ],
    }


def _cache_policy_for_rules(rules):
    modes = []
    for rule in rules:
        cache_mode = rule["cache_mode"]
        if cache_mode in {mode["cache_mode"] for mode in modes}:
            continue
        activation = {
            "none": "not-applicable",
            "implicit": "automatic",
            "explicit": "request",
        }[cache_mode]
        modes.append({"cache_mode": cache_mode, "activation": activation})
    support = (
        "supported"
        if any(mode["cache_mode"] != "none" for mode in modes)
        else "unsupported"
    )
    return {
        "support": support,
        "modes": modes
        or [{"cache_mode": "none", "activation": "not-applicable"}],
        "evidence": [
            {
                "url": "https://example.invalid/cache-policy",
                "retrieved_on": "2026-07-23",
                "facts": ["Synthetic offline cache-policy evidence."],
            }
        ],
    }


def _usage_profile_for_rules(rules):
    if any(rule["rates"]["cache_write_input"] is not None for rule in rules):
        return "openai-chat-cache-creation-v1"
    if any(rule["cache_mode"] != "none" for rule in rules):
        return "openai-chat-cached-tokens-v1"
    return "openai-chat-standard-v1"


def _rich_pricing_config(
    rules, *, usage_accounting=OMITTED, cache_policy=OMITTED
):
    if usage_accounting is OMITTED:
        usage_accounting = _usage_profile_for_rules(rules)
    if cache_policy is OMITTED:
        cache_policy = _cache_policy_for_rules(rules)
    return {
        "test-provider": {
            **CUSTOM_CONFIG["test-provider"],
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


def _pricing_context(**overrides):
    context = {
        "region": "us-east",
        "service_scope": "global",
        "deployment_type": "serverless",
        "output_mode": "thinking",
        "request_mode": "realtime",
        "cache_mode": "none",
    }
    context.update(overrides)
    return context


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


def test_provider_settings_connection_declarations_are_additive():
    provider = ProviderSettings(**CUSTOM_CONFIG["test-provider"])
    dumped = provider.model_dump()
    schema_properties = ProviderSettings.model_json_schema()["properties"]

    assert "api_key_env_aliases" in ProviderSettings.model_fields
    assert "api_base_url_env_var" in ProviderSettings.model_fields
    assert "api_key_env_aliases" in schema_properties
    assert "api_base_url_env_var" in schema_properties
    assert dumped["api_key_env_aliases"] == []
    assert dumped["api_base_url_env_var"] is None
    assert provider.api_key_env_var == "TEST_API_KEY"
    assert provider.api_base_url == "https://example.invalid/chat/completions"


def test_request_policy_schema_fields_have_legacy_safe_defaults():
    provider = ProviderSettings(**CUSTOM_CONFIG["test-provider"])
    model = provider.models["test-model"]
    model_schema = ModelDetails.model_json_schema()["properties"]
    provider_schema = ProviderSettings.model_json_schema()["properties"]

    for field_name in (
        "temperature",
        "max_tokens",
        "context_window",
        "request_overrides",
        "output_token_field",
    ):
        assert field_name in ModelDetails.model_fields
        assert field_name in model_schema
        assert getattr(model, field_name) is None

    assert "output_token_field" in ProviderSettings.model_fields
    assert "output_token_field" in provider_schema
    assert provider.output_token_field == "max_tokens"
    assert provider.model_dump()["output_token_field"] == "max_tokens"


@pytest.mark.parametrize("target", ["provider", "model"])
def test_request_policy_schema_rejects_unknown_output_token_field(target):
    if target == "provider":
        with pytest.raises(ValueError):
            ProviderSettings(
                **CUSTOM_CONFIG["test-provider"],
                output_token_field="completion_tokens",
            )
        return

    with pytest.raises(ValueError):
        ModelDetails(
            id="provider-model-id",
            pricing={"input": 1.0, "output": 2.0},
            output_token_field="completion_tokens",
        )


def test_provider_settings_preserves_alias_order_and_duplicates():
    aliases = [
        "SECONDARY_KEY",
        "TEST_API_KEY",
        "SECONDARY_KEY",
        "CONTROL\x07KEY",
    ]
    provider = ProviderSettings(
        **CUSTOM_CONFIG["test-provider"],
        api_key_env_aliases=aliases,
        api_base_url_env_var="SYNTHETIC_API_BASE_URL",
    )

    assert provider.api_key_env_aliases == aliases
    assert provider.api_base_url_env_var == "SYNTHETIC_API_BASE_URL"


@pytest.mark.parametrize(
    ("field_name", "declaration"),
    [
        ("api_key_env_aliases", "SINGLE_STRING"),
        ("api_key_env_aliases", ["VALID_ALIAS", 1]),
        ("api_base_url_env_var", 1),
    ],
)
def test_provider_settings_requires_strings_for_new_environment_names(
    field_name, declaration
):
    with pytest.raises(ValueError):
        ProviderSettings(
            **CUSTOM_CONFIG["test-provider"],
            **{field_name: declaration},
        )


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        " ",
        "KEY\tNAME",
        "KEY\u2003NAME",
        "KEY=VALUE",
        "KEY\x00VALUE",
    ],
)
@pytest.mark.parametrize(
    "field_name", ["api_key_env_aliases", "api_base_url_env_var"]
)
def test_provider_settings_rejects_invalid_new_environment_names(
    field_name, invalid_name
):
    declaration = (
        [invalid_name] if field_name == "api_key_env_aliases" else invalid_name
    )

    with pytest.raises(ValueError):
        ProviderSettings(
            **CUSTOM_CONFIG["test-provider"],
            **{field_name: declaration},
        )


@pytest.mark.parametrize("legacy_name", ["", "LEGACY KEY", "A=B", "A\x00B"])
def test_provider_settings_does_not_retroactively_validate_primary_key_name(
    legacy_name,
):
    provider = ProviderSettings(
        **{
            **CUSTOM_CONFIG["test-provider"],
            "api_key_env_var": legacy_name,
        }
    )

    assert provider.api_key_env_var == legacy_name


@pytest.mark.parametrize("endpoint_name", ["TEST_API_KEY", "SECONDARY_KEY"])
def test_provider_settings_rejects_endpoint_and_credential_name_overlap(
    endpoint_name,
):
    with pytest.raises(ValueError):
        ProviderSettings(
            **CUSTOM_CONFIG["test-provider"],
            api_key_env_aliases=["SECONDARY_KEY", "TEST_API_KEY"],
            api_base_url_env_var=endpoint_name,
        )


def test_provider_settings_default_alias_lists_are_isolated():
    first = ProviderSettings(**CUSTOM_CONFIG["test-provider"])
    second = ProviderSettings(**CUSTOM_CONFIG["test-provider"])

    first.api_key_env_aliases.append("MUTATED_ALIAS")

    assert second.api_key_env_aliases == []


@pytest.mark.parametrize("source_kind", ["dict", "path"])
def test_connection_declarations_are_isolated_across_explicit_loads(
    tmp_path, source_kind
):
    source_data = {
        "test-provider": {
            **CUSTOM_CONFIG["test-provider"],
            "api_key_env_aliases": ["SECONDARY_KEY"],
            "api_base_url_env_var": "SYNTHETIC_API_BASE_URL",
        }
    }
    config_source = source_data
    if source_kind == "path":
        config_source = tmp_path / "llm_config.yml"
        config_source.write_text(
            yaml.safe_dump(source_data, sort_keys=False), encoding="utf-8"
        )

    first = load_all_settings(config_source)
    first["test-provider"].api_key_env_aliases.append("MUTATED_ALIAS")
    first["test-provider"].api_base_url_env_var = "MUTATED_BASE_URL"
    second = load_all_settings(config_source)

    assert second["test-provider"].api_key_env_aliases == ["SECONDARY_KEY"]
    assert (
        second["test-provider"].api_base_url_env_var
        == "SYNTHETIC_API_BASE_URL"
    )
    assert source_data["test-provider"]["api_key_env_aliases"] == [
        "SECONDARY_KEY"
    ]


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


def test_get_model_details_returns_raw_provider_and_model_request_policy():
    config = {
        "test-provider": {
            **CUSTOM_CONFIG["test-provider"],
            "temperature": 0.2,
            "max_tokens": 256,
            "context_window": 8192,
            "request_overrides": {"shared": "provider"},
            "output_token_field": "max_tokens",
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "context_window": 16384,
                    "request_overrides": {"shared": "model"},
                    "output_token_field": "max_completion_tokens",
                }
            },
        }
    }

    provider_name, provider, model = get_model_details("test-model", config)

    assert provider_name == "test-provider"
    assert provider.temperature == 0.2
    assert provider.max_tokens == 256
    assert provider.context_window == 8192
    assert provider.request_overrides == {"shared": "provider"}
    assert provider.output_token_field == "max_tokens"
    assert model.temperature == 0.7
    assert model.max_tokens == 512
    assert model.context_window == 16384
    assert model.request_overrides == {"shared": "model"}
    assert model.output_token_field == "max_completion_tokens"


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


def test_request_policy_settings_are_deeply_isolated_across_explicit_loads():
    source = {
        "test-provider": {
            **CUSTOM_CONFIG["test-provider"],
            "request_overrides": {"routing": {"only": ["provider-route"]}},
            "models": {
                "test-model": {
                    "id": "provider-model-id",
                    "pricing": {"input": 1.0, "output": 2.0},
                    "request_overrides": {
                        "routing": {"only": ["model-route"]}
                    },
                }
            },
        }
    }

    _, first_provider, first_model = get_model_details("test-model", source)
    assert first_provider.request_overrides is not None
    assert first_model.request_overrides is not None
    first_provider.request_overrides["routing"]["only"].append("mutated")
    first_model.request_overrides["routing"]["only"].append("mutated")

    _, second_provider, second_model = get_model_details("test-model", source)

    assert second_provider.request_overrides == {
        "routing": {"only": ["provider-route"]}
    }
    assert second_model.request_overrides == {
        "routing": {"only": ["model-route"]}
    }
    assert source["test-provider"]["request_overrides"] == {
        "routing": {"only": ["provider-route"]}
    }
    assert source["test-provider"]["models"]["test-model"][
        "request_overrides"
    ] == {"routing": {"only": ["model-route"]}}


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


def test_legacy_pricing_validation_and_serialization_remain_permissive():
    pricing = Pricing(
        input=-1.0,
        output=float("inf"),
        ignored_legacy_extra="still-permitted",
    )

    assert pricing.model_dump() == {"input": -1.0, "output": float("inf")}
    assert math.isinf(pricing.output)


@pytest.mark.parametrize("rules", [[_rich_pricing_rule()], []])
def test_explicit_rich_schema_never_falls_back_to_legacy_pricing(rules):
    pricing = {
        "schema": "pricing-rules-v1",
        "rules": rules,
        "input": 1.0,
        "output": 2.0,
    }

    with pytest.raises(ValidationError):
        ModelDetails(id="provider-model-id", pricing=pricing)


def test_rich_pricing_canonical_json_serialization_and_raw_lookup():
    first_rule = _rich_pricing_rule()
    second_rule = _rich_pricing_rule(
        rule_id="explicit-upper-tier",
        cache_mode="explicit",
        input_tokens_gt=100,
        input_tokens_lte=None,
        effective_from=None,
    )
    config = _rich_pricing_config([first_rule, second_rule])

    _, _, model = get_model_details("test-model", config)
    dumped = model.pricing.model_dump(mode="json")

    assert isinstance(model.pricing, config_module.PricingSchedule)
    assert dumped["schema"] == "pricing-rules-v1"
    assert [rule["rule_id"] for rule in dumped["rules"]] == [
        "standard-us-realtime-none",
        "explicit-upper-tier",
    ]
    assert dumped["rules"][0]["effective_from"] == "2026-07-01T00:00:00Z"
    assert dumped["rules"][0]["effective_until"] is None
    assert dumped["rules"][0]["rates"]["cache_read_input"] is None
    assert dumped["rules"][0]["evidence"][0]["retrieved_on"] == "2026-07-23"
    assert dumped["rules"][0]["evidence"][0]["url"] == (
        "https://help.aliyun.com/zh/model-studio/model-pricing"
    )


def test_rich_schedule_accepts_only_public_schema_alias():
    with pytest.raises(ValidationError):
        config_module.PricingSchedule(
            schema_="pricing-rules-v1", rules=[_rich_pricing_rule()]
        )


def test_pricing_exception_hierarchy_is_exact():
    assert issubclass(
        config_module.PricingContextRequiredError,
        config_module.PricingSelectionError,
    )
    assert issubclass(
        config_module.PricingNoMatchError,
        config_module.PricingSelectionError,
    )
    assert issubclass(
        config_module.PricingAmbiguityError,
        config_module.PricingSelectionError,
    )
    assert issubclass(config_module.PricingSelectionError, ValueError)
    assert issubclass(config_module.PricingCalculationError, ValueError)
    assert not issubclass(
        config_module.PricingCalculationError,
        config_module.PricingSelectionError,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rule_id", ""),
        ("rule_id", " surrounded "),
        ("billing_model_id", " "),
        ("billing_model_id", " provider-model-id"),
        ("capability_snapshot_id", ""),
        ("region", ""),
        ("service_scope", " global"),
        ("deployment_type", " "),
        ("output_mode", "Thinking Mode"),
    ],
)
def test_rich_pricing_rejects_invalid_identifiers(field, value):
    rule = _rich_pricing_rule()
    rule[field] = value

    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[rule])


@pytest.mark.parametrize("fact", ["", " ", " leading", "trailing "])
def test_rich_pricing_rejects_blank_or_padded_evidence_facts(fact):
    rule = _rich_pricing_rule()
    rule["evidence"][0]["facts"] = [fact]

    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[rule])


@pytest.mark.parametrize(
    "url",
    [
        "http://help.aliyun.com/zh/model-studio/model-pricing",
        "not-a-url",
        "",
    ],
)
def test_rich_pricing_requires_explicit_https_evidence_url(url):
    rule = _rich_pricing_rule()
    rule["evidence"][0]["url"] = url

    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[rule])


@pytest.mark.parametrize(
    ("rate_field", "value"),
    [
        ("input", -0.1),
        ("output", float("inf")),
        ("output", float("nan")),
        ("cache_read_input", -1),
        ("cache_write_input", float("inf")),
    ],
)
def test_rich_pricing_rejects_negative_or_non_finite_rates(rate_field, value):
    rule = _rich_pricing_rule(cache_mode="explicit")
    rule["rates"][rate_field] = value

    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[rule])


@pytest.mark.parametrize("currency", ["usd", "US", "USDD", "US1", " USD"])
def test_rich_pricing_requires_uppercase_three_letter_currency(currency):
    rule = _rich_pricing_rule(currency=currency)

    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[rule])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unit_tokens", 0),
        ("unit_tokens", -1),
        ("unit_tokens", True),
        ("input_tokens_gt", -1),
        ("input_tokens_gt", True),
        ("input_tokens_lte", 0),
    ],
)
def test_rich_pricing_rejects_invalid_units_and_token_ranges(field, value):
    rule = _rich_pricing_rule()
    rule[field] = value

    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[rule])


@pytest.mark.parametrize(
    ("effective_from", "effective_until", "rate_type"),
    [
        ("2026-07-01T00:00:00", None, "standard"),
        (
            "2026-07-02T00:00:00Z",
            "2026-07-01T00:00:00Z",
            "standard",
        ),
        (
            "2026-07-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
            "standard",
        ),
        (None, "2026-07-02T00:00:00Z", "promotional"),
        ("2026-07-01T00:00:00Z", None, "promotional"),
    ],
)
def test_rich_pricing_rejects_invalid_effective_intervals(
    effective_from, effective_until, rate_type
):
    rule = _rich_pricing_rule(
        effective_from=effective_from,
        effective_until=effective_until,
        rate_type=rate_type,
    )

    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[rule])


@pytest.mark.parametrize(
    ("cache_mode", "cache_read", "cache_write"),
    [
        ("none", 0.0, None),
        ("none", None, 0.0),
        ("implicit", None, None),
        ("explicit", None, 0.0),
    ],
)
def test_rich_pricing_rejects_cache_rate_structure_mismatches(
    cache_mode, cache_read, cache_write
):
    rule = _rich_pricing_rule(cache_mode=cache_mode)
    rule["rates"]["cache_read_input"] = cache_read
    rule["rates"]["cache_write_input"] = cache_write

    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[rule])


@pytest.mark.parametrize("cache_mode", ["implicit", "explicit"])
def test_non_none_rule_allows_independent_optional_cache_write_rate(
    cache_mode,
):
    rule = _rich_pricing_rule(
        cache_mode=cache_mode,
        cache_write_input=None,
    )

    schedule = config_module.PricingSchedule(
        schema="pricing-rules-v1", rules=[rule]
    )

    assert schedule.rules[0].rates.cache_read_input == 0.5
    assert schedule.rules[0].rates.cache_write_input is None


def test_rich_pricing_allows_zero_cache_rates_when_structurally_applicable():
    implicit = _rich_pricing_rule(cache_mode="implicit")
    implicit["rates"]["cache_read_input"] = 0.0
    explicit = _rich_pricing_rule(rule_id="explicit", cache_mode="explicit")
    explicit["rates"]["cache_read_input"] = 0.0
    explicit["rates"]["cache_write_input"] = 0.0

    implicit_schedule = config_module.PricingSchedule(
        schema="pricing-rules-v1", rules=[implicit]
    )
    explicit_schedule = config_module.PricingSchedule(
        schema="pricing-rules-v1", rules=[explicit]
    )

    assert implicit_schedule.rules[0].rates.cache_read_input == 0.0
    assert explicit_schedule.rules[0].rates.cache_write_input == 0.0


def test_rich_pricing_rejects_empty_missing_duplicate_or_extra_structure():
    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[])

    missing_evidence = _rich_pricing_rule()
    missing_evidence["evidence"] = []
    with pytest.raises(ValidationError):
        config_module.PricingSchedule(
            schema="pricing-rules-v1", rules=[missing_evidence]
        )

    missing_facts = _rich_pricing_rule()
    missing_facts["evidence"][0]["facts"] = []
    with pytest.raises(ValidationError):
        config_module.PricingSchedule(
            schema="pricing-rules-v1", rules=[missing_facts]
        )

    duplicate = _rich_pricing_rule()
    with pytest.raises(ValidationError):
        config_module.PricingSchedule(
            schema="pricing-rules-v1", rules=[duplicate, duplicate]
        )

    extra = _rich_pricing_rule()
    extra["unexpected"] = True
    with pytest.raises(ValidationError):
        config_module.PricingSchedule(schema="pricing-rules-v1", rules=[extra])


def test_rich_pricing_context_requires_exact_declared_dimensions():
    context = config_module.PricingContext(**_pricing_context())

    assert context.model_dump() == _pricing_context()

    with pytest.raises(ValidationError):
        config_module.PricingContext(
            **_pricing_context(), unexpected="not-accepted"
        )
    with pytest.raises(ValidationError):
        config_module.PricingContext(**_pricing_context(output_mode=" "))


def test_legacy_resolution_ignores_rich_inputs_and_preserves_flat_cost():
    resolved = config_module.resolve_model_pricing(
        "test-model",
        CUSTOM_CONFIG,
        pricing_context={"not": "a rich context"},
        input_tokens=-1,
        effective_at=datetime(2026, 7, 23),
    )
    cost = config_module.calculate_pricing_cost(
        resolved,
        input_tokens=10,
        output_tokens=20,
    )

    assert resolved.source == "legacy"
    assert resolved.rule_id is None
    assert resolved.cache_mode == "none"
    assert resolved.currency == "$"
    assert resolved.unit_tokens == 1_000_000
    assert resolved.rates.model_dump() == {
        "input": 1.0,
        "output": 2.0,
        "cache_read_input": None,
        "cache_write_input": None,
    }
    assert cost.input_cost == pytest.approx(0.00001)
    assert cost.output_cost == pytest.approx(0.00004)
    assert cost.total_cost == pytest.approx(0.00005)
    assert cost.currency == "$"


@pytest.mark.parametrize(
    ("input_tokens", "expected_rule"),
    [
        (1, "lower"),
        (100, "lower"),
        (101, "upper"),
        (1_000_000, "upper"),
    ],
)
def test_rich_resolution_uses_exact_open_lower_closed_upper_tiers(
    input_tokens, expected_rule
):
    lower = _rich_pricing_rule(
        rule_id="lower", effective_from=None, input_tokens_lte=100
    )
    upper = _rich_pricing_rule(
        rule_id="upper",
        effective_from=None,
        input_tokens_gt=100,
        input_tokens_lte=None,
    )
    config = _rich_pricing_config([lower, upper])

    resolved = config_module.resolve_model_pricing(
        "test-model",
        config,
        pricing_context=_pricing_context(),
        input_tokens=input_tokens,
        effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    assert resolved.source == "pricing-rules-v1"
    assert resolved.rule_id == expected_rule


def test_rich_resolution_fails_closed_below_every_tier():
    config = _rich_pricing_config([_rich_pricing_rule(effective_from=None)])

    with pytest.raises(config_module.PricingNoMatchError):
        config_module.resolve_model_pricing(
            "test-model",
            config,
            pricing_context=_pricing_context(),
            input_tokens=0,
            effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    ("effective_at", "matches"),
    [
        (datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc), False),
        (datetime(2026, 7, 1, tzinfo=timezone.utc), True),
        (
            datetime(2026, 7, 2, tzinfo=timezone.utc)
            - timedelta(microseconds=1),
            True,
        ),
        (datetime(2026, 7, 2, tzinfo=timezone.utc), False),
    ],
)
def test_rich_resolution_uses_closed_open_effective_interval(
    effective_at, matches
):
    rule = _rich_pricing_rule(
        rate_type="promotional",
        effective_from="2026-07-01T00:00:00Z",
        effective_until="2026-07-02T00:00:00Z",
    )
    config = _rich_pricing_config([rule])

    if matches:
        resolved = config_module.resolve_model_pricing(
            "test-model",
            config,
            pricing_context=_pricing_context(),
            input_tokens=10,
            effective_at=effective_at,
        )
        assert resolved.rule_id == "standard-us-realtime-none"
    else:
        with pytest.raises(config_module.PricingNoMatchError):
            config_module.resolve_model_pricing(
                "test-model",
                config,
                pricing_context=_pricing_context(),
                input_tokens=10,
                effective_at=effective_at,
            )


@pytest.mark.parametrize(
    ("dimension", "value"),
    [
        ("region", "eu-west"),
        ("service_scope", "regional"),
        ("deployment_type", "provisioned"),
        ("output_mode", "non-thinking"),
        ("request_mode", "batch"),
        ("cache_mode", "implicit"),
    ],
)
def test_rich_resolution_requires_exact_match_for_all_dimensions(
    dimension, value
):
    config = _rich_pricing_config([_rich_pricing_rule(effective_from=None)])

    with pytest.raises(config_module.PricingNoMatchError):
        config_module.resolve_model_pricing(
            "test-model",
            config,
            pricing_context=_pricing_context(**{dimension: value}),
            input_tokens=10,
            effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    ("pricing_context", "input_tokens", "effective_at"),
    [
        (None, 10, datetime(2026, 7, 23, tzinfo=timezone.utc)),
        (_pricing_context(), None, datetime(2026, 7, 23, tzinfo=timezone.utc)),
        (_pricing_context(), True, datetime(2026, 7, 23, tzinfo=timezone.utc)),
        (_pricing_context(), -1, datetime(2026, 7, 23, tzinfo=timezone.utc)),
        (_pricing_context(), 10, None),
        (_pricing_context(), 10, datetime(2026, 7, 23)),
    ],
)
def test_rich_resolution_requires_complete_valid_selection_inputs(
    pricing_context, input_tokens, effective_at
):
    config = _rich_pricing_config([_rich_pricing_rule(effective_from=None)])

    with pytest.raises(config_module.PricingContextRequiredError):
        config_module.resolve_model_pricing(
            "test-model",
            config,
            pricing_context=pricing_context,
            input_tokens=input_tokens,
            effective_at=effective_at,
        )


def test_rich_resolution_reports_ambiguity_without_order_tiebreak():
    first = _rich_pricing_rule(rule_id="first", effective_from=None)
    second = _rich_pricing_rule(rule_id="second", effective_from=None)
    config = _rich_pricing_config([first, second])

    with pytest.raises(config_module.PricingAmbiguityError) as error:
        config_module.resolve_model_pricing(
            "test-model",
            config,
            pricing_context=_pricing_context(),
            input_tokens=10,
            effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )

    assert "first" in str(error.value)
    assert "second" in str(error.value)


def test_rich_resolution_requires_identity_without_alias_fallback():
    rule = _rich_pricing_rule(effective_from=None)
    rule["billing_model_id"] = "provider-model-snapshot-2026-07-01"
    rule["capability_snapshot_id"] = "provider-model-id"
    config = _rich_pricing_config([rule])

    with pytest.raises(config_module.PricingNoMatchError):
        config_module.resolve_model_pricing(
            "test-model",
            config,
            pricing_context=_pricing_context(),
            input_tokens=10,
            effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )


def test_rich_raw_lookup_resolution_and_source_are_deeply_isolated():
    source = _rich_pricing_config([_rich_pricing_rule(effective_from=None)])
    _, _, first_model = get_model_details("test-model", source)
    first_model.pricing.rules[0].rates.input = 999.0
    first_model.pricing.rules[0].evidence[0].facts.append("mutated")

    resolved = config_module.resolve_model_pricing(
        "test-model",
        source,
        pricing_context=_pricing_context(),
        input_tokens=10,
        effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )
    resolved.rates.input = 888.0
    _, _, second_model = get_model_details("test-model", source)

    assert second_model.pricing.rules[0].rates.input == 2.0
    assert second_model.pricing.rules[0].evidence[0].facts == [
        "Prices vary by total request input tier."
    ]
    assert (
        source["test-provider"]["models"]["test-model"]["pricing"]["rules"][0][
            "rates"
        ]["input"]
        == 2.0
    )
    assert resolved.rates.input == 888.0

    with pytest.raises(ValidationError):
        resolved.currency = "EUR"


@pytest.mark.parametrize(
    ("cache_mode", "read_tokens", "write_tokens", "expected_input_cost"),
    [
        ("none", None, None, 0.2),
        ("implicit", 20, None, 0.17),
        ("explicit", 20, 30, 0.2),
    ],
)
def test_rich_pricing_calculator_uses_explicit_cache_bucket_rates(
    cache_mode, read_tokens, write_tokens, expected_input_cost
):
    rule = _rich_pricing_rule(cache_mode=cache_mode, effective_from=None)
    config = _rich_pricing_config([rule])
    resolved = config_module.resolve_model_pricing(
        "test-model",
        config,
        pricing_context=_pricing_context(cache_mode=cache_mode),
        input_tokens=100,
        effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    cost = config_module.calculate_pricing_cost(
        resolved,
        input_tokens=100,
        output_tokens=10,
        cache_read_input_tokens=read_tokens,
        cache_write_input_tokens=write_tokens,
    )

    assert cost.input_cost == pytest.approx(expected_input_cost)
    assert cost.output_cost == pytest.approx(0.04)
    assert cost.total_cost == pytest.approx(expected_input_cost + 0.04)
    assert cost.currency == "USD"


@pytest.mark.parametrize(
    ("cache_mode", "read_tokens", "write_tokens"),
    [
        ("none", 1, None),
        ("none", None, 1),
        ("implicit", None, None),
        ("implicit", 20, 1),
        ("explicit", None, 0),
        ("explicit", 0, None),
        ("explicit", 60, 41),
    ],
)
def test_rich_pricing_calculator_fails_closed_for_cache_mode_conflicts(
    cache_mode, read_tokens, write_tokens
):
    rule = _rich_pricing_rule(cache_mode=cache_mode, effective_from=None)
    config = _rich_pricing_config([rule])
    resolved = config_module.resolve_model_pricing(
        "test-model",
        config,
        pricing_context=_pricing_context(cache_mode=cache_mode),
        input_tokens=100,
        effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    with pytest.raises(config_module.PricingCalculationError):
        config_module.calculate_pricing_cost(
            resolved,
            input_tokens=100,
            output_tokens=10,
            cache_read_input_tokens=read_tokens,
            cache_write_input_tokens=write_tokens,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_tokens", True),
        ("input_tokens", -1),
        ("output_tokens", 1.5),
        ("output_tokens", "1"),
        ("cache_read_input_tokens", False),
        ("cache_read_input_tokens", -1),
        ("cache_write_input_tokens", 1.0),
        ("cache_write_input_tokens", "0"),
    ],
)
def test_rich_pricing_calculator_requires_plain_nonnegative_integers(
    field, value
):
    rule = _rich_pricing_rule(cache_mode="explicit", effective_from=None)
    config = _rich_pricing_config([rule])
    resolved = config_module.resolve_model_pricing(
        "test-model",
        config,
        pricing_context=_pricing_context(cache_mode="explicit"),
        input_tokens=100,
        effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )
    arguments = {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_read_input_tokens": 20,
        "cache_write_input_tokens": 30,
    }
    arguments[field] = value

    with pytest.raises(config_module.PricingCalculationError):
        config_module.calculate_pricing_cost(resolved, **arguments)


@pytest.mark.parametrize(
    "token_count",
    [
        pytest.param(10**308, id="non-finite-float"),
        pytest.param(10**400, id="integer-conversion-overflow"),
    ],
)
@pytest.mark.parametrize(
    ("component", "cache_mode"),
    [
        ("ordinary-input", "none"),
        ("output", "none"),
        ("cache-read", "implicit"),
        ("cache-write", "explicit"),
    ],
)
def test_rich_pricing_calculator_fails_closed_for_nonrepresentable_costs(
    component, cache_mode, token_count
):
    rule_arguments = {
        "cache_mode": cache_mode,
        "effective_from": None,
    }
    if component == "cache-read":
        rule_arguments["cache_read_input"] = 4.0
    rule = _rich_pricing_rule(**rule_arguments)
    config = _rich_pricing_config([rule])
    resolved = config_module.resolve_model_pricing(
        "test-model",
        config,
        pricing_context=_pricing_context(cache_mode=cache_mode),
        input_tokens=100,
        effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )
    arguments = {
        "input_tokens": 1,
        "output_tokens": 0,
        "cache_read_input_tokens": None,
        "cache_write_input_tokens": None,
    }
    if component == "ordinary-input":
        arguments["input_tokens"] = token_count
    elif component == "output":
        arguments["output_tokens"] = token_count
    elif component == "cache-read":
        arguments["input_tokens"] = token_count
        arguments["cache_read_input_tokens"] = token_count
    else:
        arguments["input_tokens"] = token_count
        arguments["cache_read_input_tokens"] = 0
        arguments["cache_write_input_tokens"] = token_count

    with pytest.raises(config_module.PricingCalculationError):
        config_module.calculate_pricing_cost(resolved, **arguments)


def test_rich_pricing_calculator_rejects_non_finite_total_of_finite_costs():
    token_count = 10**308
    component_cost = token_count * 0.9
    assert math.isfinite(component_cost)
    assert not math.isfinite(component_cost + component_cost)
    resolved = config_module.ResolvedPricing(
        source="pricing-rules-v1",
        rule_id="total-overflow",
        rates={"input": 0.9, "output": 0.9},
        cache_mode="none",
        currency="USD",
        unit_tokens=1,
    )

    with pytest.raises(config_module.PricingCalculationError):
        config_module.calculate_pricing_cost(
            resolved,
            input_tokens=token_count,
            output_tokens=token_count,
        )


def test_batch_rules_remain_available_to_pure_resolver_and_calculator():
    batch = _rich_pricing_rule(
        rule_id="batch",
        request_mode="batch",
        effective_from=None,
    )
    config = _rich_pricing_config([batch])

    resolved = config_module.resolve_model_pricing(
        "test-model",
        config,
        pricing_context=_pricing_context(request_mode="batch"),
        input_tokens=100,
        effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )
    cost = config_module.calculate_pricing_cost(
        resolved, input_tokens=100, output_tokens=10
    )

    assert resolved.rule_id == "batch"
    assert cost.total_cost == pytest.approx(0.24)


def test_built_wheel_exposes_canonical_rich_pricing_config_api(tmp_path):
    output_directory = tmp_path / "dist"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output_directory)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheel = next(output_directory.glob("*.whl"))
    symbols = [
        "PricingEvidence",
        "PricingRates",
        "PricingRule",
        "PricingSchedule",
        "CacheModePolicy",
        "CachePolicy",
        "PricingContext",
        "ResolvedPricing",
        "PricingCost",
        "PricingSelectionError",
        "PricingContextRequiredError",
        "PricingNoMatchError",
        "PricingAmbiguityError",
        "PricingCalculationError",
        "resolve_model_pricing",
        "calculate_pricing_cost",
    ]
    import_code = (
        "import inspect, sys; "
        f"sys.path.insert(0, {str(wheel)!r}); "
        "import llm_exec_core.config as config; "
        "from llm_exec_core.client import LLMClient; "
        f"assert all(hasattr(config, name) for name in {symbols!r}); "
        "schedule = config.PricingSchedule.model_construct("
        "schema_='pricing-rules-v1', rules=[]); "
        "assert schedule.model_dump(mode='json') == "
        "{'schema': 'pricing-rules-v1', 'rules': []}; "
        "assert 'pricing_context' in inspect.signature(LLMClient).parameters; "
        "import llm_exec_core; "
        "assert hasattr(llm_exec_core, 'AccountingStatus')"
    )
    imported = subprocess.run(
        [sys.executable, "-I", "-c", import_code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert imported.returncode == 0, imported.stdout + imported.stderr


def test_cache_policy_public_models_validate_and_serialize_canonically():
    policy = config_module.CachePolicy(
        support="supported",
        modes=[
            config_module.CacheModePolicy(
                cache_mode="implicit", activation="automatic"
            ),
            {
                "cache_mode": "explicit",
                "activation": "control-plane-and-request",
            },
            {"cache_mode": "none", "activation": "not-applicable"},
        ],
        evidence=_cache_policy_for_rules([])["evidence"],
    )

    assert policy.model_dump(mode="json") == {
        "support": "supported",
        "modes": [
            {"cache_mode": "implicit", "activation": "automatic"},
            {
                "cache_mode": "explicit",
                "activation": "control-plane-and-request",
            },
            {"cache_mode": "none", "activation": "not-applicable"},
        ],
        "evidence": [
            {
                "url": "https://example.invalid/cache-policy",
                "retrieved_on": "2026-07-23",
                "facts": ["Synthetic offline cache-policy evidence."],
            }
        ],
    }


@pytest.mark.parametrize(
    ("cache_mode", "activation"),
    [
        ("none", "automatic"),
        ("none", "request"),
        ("implicit", "not-applicable"),
        ("implicit", "request"),
        ("implicit", "control-plane-and-request"),
        ("explicit", "not-applicable"),
        ("explicit", "automatic"),
        ("explicit", "control-plane"),
    ],
)
def test_cache_mode_policy_rejects_incompatible_activation(
    cache_mode, activation
):
    with pytest.raises(ValidationError):
        config_module.CacheModePolicy(
            cache_mode=cache_mode, activation=activation
        )


def test_cache_policy_rejects_duplicate_modes_and_invalid_support_shapes():
    evidence = _cache_policy_for_rules([])["evidence"]
    invalid_policies = [
        {
            "support": "supported",
            "modes": [
                {"cache_mode": "implicit", "activation": "automatic"},
                {"cache_mode": "implicit", "activation": "control-plane"},
            ],
            "evidence": evidence,
        },
        {
            "support": "unsupported",
            "modes": [
                {"cache_mode": "implicit", "activation": "automatic"},
            ],
            "evidence": evidence,
        },
        {
            "support": "supported",
            "modes": [
                {"cache_mode": "none", "activation": "not-applicable"},
            ],
            "evidence": evidence,
        },
        {
            "support": "unsupported",
            "modes": [],
            "evidence": evidence,
        },
        {
            "support": "unsupported",
            "modes": [
                {"cache_mode": "none", "activation": "not-applicable"},
            ],
            "evidence": [],
        },
    ]

    for policy in invalid_policies:
        with pytest.raises(ValidationError):
            config_module.CachePolicy.model_validate(policy)


def test_supported_cache_policy_may_omit_or_include_none_exactly():
    evidence = _cache_policy_for_rules([])["evidence"]

    without_none = config_module.CachePolicy(
        support="supported",
        modes=[{"cache_mode": "implicit", "activation": "control-plane"}],
        evidence=evidence,
    )
    with_none = config_module.CachePolicy(
        support="supported",
        modes=[
            {"cache_mode": "implicit", "activation": "automatic"},
            {"cache_mode": "none", "activation": "not-applicable"},
        ],
        evidence=evidence,
    )

    assert [mode.cache_mode for mode in without_none.modes] == ["implicit"]
    assert [mode.cache_mode for mode in with_none.modes] == [
        "implicit",
        "none",
    ]


def test_rich_schedule_requires_route_profile_and_model_cache_policy():
    rule = _rich_pricing_rule()
    missing_profile = _rich_pricing_config([rule], usage_accounting=None)
    missing_policy = _rich_pricing_config([rule], cache_policy=None)

    with pytest.raises(ValidationError):
        load_all_settings(missing_profile)
    with pytest.raises(ValidationError):
        load_all_settings(missing_policy)


def test_legacy_flat_route_may_omit_profile_and_cache_policy_unchanged():
    settings = load_all_settings(CUSTOM_CONFIG)
    model = settings["test-provider"].models["test-model"]

    assert settings["test-provider"].usage_accounting is None
    assert model.cache_policy is None
    assert isinstance(model.pricing, Pricing)


def test_schedule_and_cache_policy_modes_are_exactly_cross_validated():
    implicit = _rich_pricing_rule(cache_mode="implicit")
    explicit = _rich_pricing_rule(
        rule_id="explicit",
        cache_mode="explicit",
        cache_write_input=None,
    )
    policy_missing_explicit = _cache_policy_for_rules([implicit])
    policy_has_unpriced_none = deepcopy(
        _cache_policy_for_rules([implicit, _rich_pricing_rule()])
    )

    with pytest.raises(ValidationError):
        load_all_settings(
            _rich_pricing_config(
                [implicit, explicit],
                usage_accounting="openai-chat-cached-tokens-v1",
                cache_policy=policy_missing_explicit,
            )
        )
    with pytest.raises(ValidationError):
        load_all_settings(
            _rich_pricing_config(
                [implicit],
                usage_accounting="openai-chat-cached-tokens-v1",
                cache_policy=policy_has_unpriced_none,
            )
        )


@pytest.mark.parametrize(
    ("profile", "rule"),
    [
        (
            "openai-chat-standard-v1",
            _rich_pricing_rule(cache_mode="implicit"),
        ),
        (
            "openai-chat-cached-tokens-v1",
            _rich_pricing_rule(cache_mode="explicit", cache_write_input=1.0),
        ),
        (
            "openai-chat-cache-hit-miss-v1",
            _rich_pricing_rule(cache_mode="implicit", cache_write_input=0.0),
        ),
        (
            "openai-chat-cache-creation-v1",
            _rich_pricing_rule(cache_mode="implicit", cache_write_input=None),
        ),
        (
            "openai-chat-cache-write-v1",
            _rich_pricing_rule(cache_mode="explicit", cache_write_input=None),
        ),
    ],
)
def test_route_profile_cross_validates_modes_and_direct_rates(profile, rule):
    with pytest.raises(ValidationError):
        load_all_settings(
            _rich_pricing_config([rule], usage_accounting=profile)
        )


@pytest.mark.parametrize(
    ("profile", "cache_mode", "cache_write"),
    [
        ("openai-chat-standard-v1", "none", None),
        ("openai-chat-cached-tokens-v1", "implicit", None),
        ("openai-chat-cache-hit-miss-v1", "explicit", None),
        ("openai-chat-cache-creation-v1", "implicit", 0.0),
        ("openai-chat-cache-write-v1", "explicit", 3.0),
    ],
)
def test_all_five_route_profiles_accept_exact_policy_and_rate_shapes(
    profile, cache_mode, cache_write
):
    rule = _rich_pricing_rule(
        cache_mode=cache_mode,
        cache_write_input=cache_write,
    )

    provider = load_all_settings(
        _rich_pricing_config([rule], usage_accounting=profile)
    )["test-provider"]

    assert provider.usage_accounting == profile
    assert provider.models["test-model"].cache_policy is not None


def test_cache_policy_and_source_are_deeply_isolated():
    rule = _rich_pricing_rule(cache_mode="implicit")
    source = _rich_pricing_config([rule])
    first = load_all_settings(source)["test-provider"].models["test-model"]
    assert first.cache_policy is not None
    first.cache_policy.modes[0].activation = "control-plane"
    first.cache_policy.evidence[0].facts.append("mutated")

    second = load_all_settings(source)["test-provider"].models["test-model"]
    assert second.cache_policy is not None
    assert second.cache_policy.modes[0].activation == "automatic"
    assert second.cache_policy.evidence[0].facts == [
        "Synthetic offline cache-policy evidence."
    ]
    assert (
        source["test-provider"]["models"]["test-model"]["cache_policy"][
            "modes"
        ][0]["activation"]
        == "automatic"
    )


def test_read_only_bucket_pricing_does_not_require_a_write_bucket_argument():
    rule = _rich_pricing_rule(
        cache_mode="explicit",
        effective_from=None,
        cache_write_input=None,
    )
    config = _rich_pricing_config(
        [rule], usage_accounting="openai-chat-cached-tokens-v1"
    )
    resolved = config_module.resolve_model_pricing(
        "test-model",
        config,
        pricing_context=_pricing_context(cache_mode="explicit"),
        input_tokens=100,
        effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    cost = config_module.calculate_pricing_cost(
        resolved,
        input_tokens=100,
        output_tokens=10,
        cache_read_input_tokens=20,
    )

    assert cost.input_cost == pytest.approx(0.17)
    assert cost.total_cost == pytest.approx(0.21)


def test_write_bucket_rate_requires_explicit_zero_or_nonzero_argument():
    rule = _rich_pricing_rule(
        cache_mode="implicit",
        effective_from=None,
        cache_write_input=0.0,
    )
    config = _rich_pricing_config(
        [rule], usage_accounting="openai-chat-cache-write-v1"
    )
    resolved = config_module.resolve_model_pricing(
        "test-model",
        config,
        pricing_context=_pricing_context(cache_mode="implicit"),
        input_tokens=100,
        effective_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    with pytest.raises(config_module.PricingCalculationError):
        config_module.calculate_pricing_cost(
            resolved,
            input_tokens=100,
            output_tokens=10,
            cache_read_input_tokens=0,
        )

    cost = config_module.calculate_pricing_cost(
        resolved,
        input_tokens=100,
        output_tokens=10,
        cache_read_input_tokens=0,
        cache_write_input_tokens=0,
    )
    assert cost.total_cost == pytest.approx(0.24)
