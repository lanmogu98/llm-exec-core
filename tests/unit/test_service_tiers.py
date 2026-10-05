"""Caller-owned service-tier declarations and request validation."""

from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest

from llm_exec_core.client import LLMClient
from llm_exec_core.config import ModelCapabilities, get_model_details

UNDECLARED = object()


def catalog(tiers=UNDECLARED, *, with_capabilities=True):
    capabilities = {
        "version": "unit-service-tiers",
        "source": "unit",
        "source_date": "2026-10-05",
    }
    if tiers is not UNDECLARED:
        capabilities["service_tiers"] = tiers
    model = {"id": "test-id", "pricing": {"input": 1, "output": 2}}
    if with_capabilities:
        model["capabilities"] = capabilities
    return {
        "test-provider": {
            "api_key_env_var": "TEST_API_KEY",
            "api_base_url": "https://example.invalid/v1/chat/completions",
            "max_tokens": 128,
            "temperature": 0.5,
            "pricing_currency": "$",
            "models": {"test-model": model},
        }
    }


def test_service_tier_declaration_has_three_distinct_states():
    for declared in (UNDECLARED, [], ["default", "fast"]):
        source = catalog(declared)
        original = deepcopy(source)
        _, _, model = get_model_details("test-model", source)
        expected = None if declared is UNDECLARED else declared
        assert model.capabilities.service_tiers == expected
        assert source == original
        if expected:
            model.capabilities.service_tiers.append("custom")
            _, _, reloaded = get_model_details("test-model", source)
            assert reloaded.capabilities.service_tiers == ["default", "fast"]


@pytest.mark.parametrize("tiers", ["fast", [1], [None]])
def test_service_tier_declaration_rejects_invalid_types(tiers):
    with pytest.raises(ValueError, match="service_tiers"):
        get_model_details("test-model", catalog(tiers))


@pytest.mark.parametrize("stream", [False, True])
def test_declared_service_tier_reaches_request_plan(monkeypatch, stream):
    monkeypatch.setenv("TEST_API_KEY", "offline-test-key")
    client = LLMClient(
        "test-model", config_source=catalog(["default", "fast"])
    )
    data, _ = client._build_request_plan(
        "Hello", stream, {"service_tier": "fast"}, None
    )
    assert data["service_tier"] == "fast"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["provider", "model", "call", "extra_body"])
@pytest.mark.parametrize("stream", [False, True])
async def test_unsupported_tier_is_rejected_before_http(
    monkeypatch, source, stream
):
    monkeypatch.setenv("TEST_API_KEY", "offline-test-key")
    config = catalog(["default"])
    options = None
    if source == "provider":
        config["test-provider"]["request_overrides"] = {"service_tier": "fast"}
    elif source == "model":
        config["test-provider"]["models"]["test-model"][
            "request_overrides"
        ] = {"service_tier": "fast"}
    elif source == "extra_body":
        options = {"extra_body": {"service_tier": "fast"}}
    else:
        options = {"service_tier": "fast"}
    client = LLMClient("test-model", config_source=config)

    def respond(request):
        usage = {"prompt_tokens": 1, "completion_tokens": 1}
        if stream:
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": usage},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as transport:
        with patch(
            "llm_exec_core.client.httpx.AsyncClient", return_value=transport
        ) as http_client:
            with pytest.raises(ValueError, match="service_tier"):
                await client.generate(
                    "Hello", stream=stream, request_options=options
                )
        http_client.assert_not_called()


@pytest.mark.parametrize(
    "value", ["fast", 1, None, ["fast"], {"tier": "fast"}]
)
def test_declared_tier_validates_value_and_type(monkeypatch, value):
    monkeypatch.setenv("TEST_API_KEY", "offline-test-key")
    client = LLMClient("test-model", config_source=catalog(["default"]))
    with pytest.raises(ValueError, match="service_tier"):
        client._build_request_plan(
            "Hello", False, {"service_tier": value}, None
        )


def test_empty_tier_declaration_rejects_explicit_tier_but_allows_omission(
    monkeypatch,
):
    monkeypatch.setenv("TEST_API_KEY", "offline-test-key")
    client = LLMClient("test-model", config_source=catalog([]))
    with pytest.raises(ValueError, match="service_tier"):
        client._build_request_plan(
            "Hello", False, {"service_tier": "fast"}, None
        )
    data, _ = client._build_request_plan("Hello", False, None, None)
    assert "service_tier" not in data


@pytest.mark.parametrize("with_capabilities", [False, True])
def test_undeclared_tiers_keep_legacy_passthrough(
    monkeypatch, with_capabilities
):
    monkeypatch.setenv("TEST_API_KEY", "offline-test-key")
    client = LLMClient(
        "test-model",
        config_source=catalog(with_capabilities=with_capabilities),
    )
    data, _ = client._build_request_plan(
        "Hello", False, {"service_tier": "custom", "custom_parameter": 1}, None
    )
    assert data["service_tier"] == "custom"
    assert data["custom_parameter"] == 1


def test_service_tier_validation_uses_effective_merged_value(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "offline-test-key")
    config = catalog(["default"])
    config["test-provider"]["request_overrides"] = {"service_tier": "fast"}
    client = LLMClient("test-model", config_source=config)
    data, _ = client._build_request_plan(
        "Hello", False, {"service_tier": "default"}, None
    )
    assert data["service_tier"] == "default"


def test_service_tiers_are_exposed_in_public_schema():
    assert (
        "service_tiers" in ModelCapabilities.model_json_schema()["properties"]
    )


@pytest.mark.parametrize("declared", [False, True])
def test_openrouter_tier_also_requires_supported_parameter(
    monkeypatch, declared
):
    monkeypatch.setenv("TEST_API_KEY", "offline-test-key")
    config = catalog(["fast"])
    provider = config.pop("test-provider")
    provider["models"]["test-model"]["capabilities"][
        "openrouter_supported_parameters"
    ] = (["service_tier"] if declared else [])
    config["openrouter-test"] = provider
    client = LLMClient("test-model", config_source=config)
    if declared:
        data, _ = client._build_request_plan(
            "Hello", False, {"service_tier": "fast"}, None
        )
        assert data["service_tier"] == "fast"
    else:
        with pytest.raises(
            ValueError, match="OpenRouter supported_parameters"
        ):
            client._build_request_plan(
                "Hello", False, {"service_tier": "fast"}, None
            )
