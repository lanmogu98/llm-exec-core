"""LLM configuration loader."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Pricing(BaseModel):
    input: float
    output: float


class ModelCapabilities(BaseModel):
    version: str
    source: str
    source_date: str
    strict_response_schema: bool = False
    json_object_response: bool = False
    tools: bool = False
    tool_streaming: bool = True
    tool_choice: bool = False
    parallel_tool_calls: bool = False
    reasoning_controls: List[str] = Field(default_factory=list)
    openrouter_supported_parameters: List[str] = Field(default_factory=list)


class ModelDetails(BaseModel):
    id: str
    pricing: Pricing
    capabilities: Optional[ModelCapabilities] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    request_overrides: Optional[Dict[str, Any]] = None
    output_token_field: Optional[
        Literal["max_tokens", "max_completion_tokens"]
    ] = None


class RateLimitSettings(BaseModel):
    min_interval_seconds: float = 0.5
    max_requests_per_minute: int = 60


class MaxTokensRetrySettings(BaseModel):
    status_code: int
    body_contains: str
    max_tokens_limit: int


class ProviderSettings(BaseModel):
    api_key_env_var: str
    api_key_env_aliases: List[str] = Field(default_factory=list)
    api_base_url: str
    api_base_url_env_var: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    output_token_field: Literal["max_tokens", "max_completion_tokens"] = (
        "max_tokens"
    )
    pricing_currency: str
    models: Dict[str, ModelDetails]
    request_overrides: Optional[Dict[str, Any]] = None
    max_tokens_retry: Optional[MaxTokensRetrySettings] = None
    rate_limit: Optional[RateLimitSettings] = None

    @field_validator("api_key_env_aliases")
    @classmethod
    def _validate_api_key_env_aliases(cls, aliases: List[str]) -> List[str]:
        for alias in aliases:
            cls._validate_new_environment_name(alias)
        return aliases

    @field_validator("api_base_url_env_var")
    @classmethod
    def _validate_api_base_url_env_var(
        cls, environment_name: Optional[str]
    ) -> Optional[str]:
        if environment_name is not None:
            cls._validate_new_environment_name(environment_name)
        return environment_name

    @model_validator(mode="after")
    def _validate_connection_environment_names(self) -> "ProviderSettings":
        if self.api_base_url_env_var is not None and (
            self.api_base_url_env_var == self.api_key_env_var
            or self.api_base_url_env_var in self.api_key_env_aliases
        ):
            raise ValueError(
                "api_base_url_env_var must differ from every API-key "
                "environment-variable name."
            )
        return self

    @staticmethod
    def _validate_new_environment_name(environment_name: str) -> None:
        if (
            not environment_name
            or "=" in environment_name
            or "\x00" in environment_name
            or any(character.isspace() for character in environment_name)
        ):
            raise ValueError(
                "New environment-variable declarations must be non-empty "
                "and contain no whitespace, '=', or NUL."
            )


def _load_raw_config(
    config_source: Path | Dict[str, Any],
) -> Dict[str, Any]:
    if isinstance(config_source, dict):
        return deepcopy(config_source)

    if not config_source.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_source}"
        )

    with config_source.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_settings(
    config_source: Path | Dict[str, Any],
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
    if config_source is None:
        raise ValueError(
            "config_source is required; pass a complete caller-owned "
            "catalog as a pathlib.Path or dict. See "
            "https://github.com/lanmogu98/llm-exec-core/issues/5."
        )

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

    available_models = ", ".join(
        model_name
        for provider_settings in settings_by_provider.values()
        for model_name in provider_settings.models
    )
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
