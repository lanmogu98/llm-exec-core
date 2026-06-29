"""LLM configuration loader."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel


class Pricing(BaseModel):
    input: float
    output: float


class ModelDetails(BaseModel):
    id: str
    pricing: Pricing


class RateLimitSettings(BaseModel):
    min_interval_seconds: float = 0.5
    max_requests_per_minute: int = 60


class ProviderSettings(BaseModel):
    api_key_env_var: str
    api_base_url: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    pricing_currency: str
    models: Dict[str, ModelDetails]
    request_overrides: Optional[Dict[str, Any]] = None
    rate_limit: Optional[RateLimitSettings] = None


_DEFAULT_PROVIDER_SETTINGS: Optional[Dict[str, ProviderSettings]] = None


def _get_default_config_path() -> Path:
    return Path(__file__).with_name("llm_config.yml")


def _load_raw_config(
    config_source: Path | Dict[str, Any] | None,
) -> Dict[str, Any]:
    if isinstance(config_source, dict):
        return config_source

    config_path = config_source or _get_default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_settings(
    config_source: Path | Dict[str, Any] | None,
) -> Dict[str, ProviderSettings]:
    config_data = _load_raw_config(config_source)
    return {
        provider_name: ProviderSettings(**provider_data)
        for provider_name, provider_data in config_data.items()
        if not provider_name.startswith("_")
    }


def load_all_settings(
    config_source: Path | Dict[str, Any] | None = None,
) -> Dict[str, ProviderSettings]:
    global _DEFAULT_PROVIDER_SETTINGS

    if config_source is None:
        if _DEFAULT_PROVIDER_SETTINGS is None:
            _DEFAULT_PROVIDER_SETTINGS = _build_settings(None)
        return _DEFAULT_PROVIDER_SETTINGS

    return _build_settings(config_source)


def get_supported_models(
    config_source: Path | Dict[str, Any] | None = None,
) -> List[str]:
    return [
        model_name
        for settings in load_all_settings(config_source).values()
        for model_name in settings.models
    ]


def get_model_details(
    model_name: str,
    config_source: Path | Dict[str, Any] | None = None,
) -> Tuple[str, ProviderSettings, ModelDetails]:
    settings_by_provider = load_all_settings(config_source)
    for provider_name, provider_settings in settings_by_provider.items():
        if model_name in provider_settings.models:
            return (
                provider_name,
                provider_settings,
                provider_settings.models[model_name],
            )

    available_models = ", ".join(get_supported_models(config_source))
    raise ValueError(
        f"Model '{model_name}' not found. Available models: {available_models}"
    )


def get_provider_settings(
    provider_name: str,
    config_source: Path | Dict[str, Any] | None = None,
) -> ProviderSettings:
    settings_by_provider = load_all_settings(config_source)
    if provider_name not in settings_by_provider:
        available_providers = ", ".join(settings_by_provider)
        raise ValueError(
            "Provider "
            f"'{provider_name}' not found. Available providers: "
            f"{available_providers}"
        )

    return settings_by_provider[provider_name]
