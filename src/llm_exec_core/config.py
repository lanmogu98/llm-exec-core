"""LLM configuration loader."""

import math
import re
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class Pricing(BaseModel):
    input: float
    output: float


class PricingSelectionError(ValueError):
    pass


class PricingContextRequiredError(PricingSelectionError):
    pass


class PricingNoMatchError(PricingSelectionError):
    pass


class PricingAmbiguityError(PricingSelectionError):
    pass


class PricingCalculationError(ValueError):
    pass


def _validate_identifier(value: str, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be nonempty and have no surrounding "
            "whitespace."
        )
    return value


class PricingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    retrieved_on: date
    facts: List[str] = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(
                "Pricing evidence requires an explicit HTTPS URL."
            )
        return value

    @field_validator("facts")
    @classmethod
    def _validate_facts(cls, facts: List[str]) -> List[str]:
        for fact in facts:
            _validate_identifier(fact, "evidence fact")
        return facts


class PricingRates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: float
    output: float
    cache_read_input: Optional[float] = None
    cache_write_input: Optional[float] = None

    @field_validator(
        "input",
        "output",
        "cache_read_input",
        "cache_write_input",
        mode="before",
    )
    @classmethod
    def _validate_rate(
        cls, value: Any, info: ValidationInfo
    ) -> Optional[float]:
        if value is None and info.field_name in {
            "cache_read_input",
            "cache_write_input",
        }:
            return None
        if type(value) not in {int, float}:
            raise ValueError(f"{info.field_name} must be a number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{info.field_name} must be finite and nonnegative."
            )
        return float(value)


class PricingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    billing_model_id: str
    capability_snapshot_id: Optional[str] = None
    region: str
    service_scope: str
    deployment_type: str
    output_mode: str
    request_mode: Literal["realtime", "batch"]
    cache_mode: Literal["none", "implicit", "explicit"]
    input_tokens_gt: int
    input_tokens_lte: Optional[int] = None
    currency: str
    unit_tokens: int
    effective_from: Optional[datetime] = None
    effective_until: Optional[datetime] = None
    rate_type: Literal["standard", "promotional"]
    rates: PricingRates
    evidence: List[PricingEvidence] = Field(min_length=1)

    @field_validator(
        "rule_id",
        "billing_model_id",
        "region",
        "service_scope",
        "deployment_type",
        "output_mode",
    )
    @classmethod
    def _validate_required_identifier(
        cls, value: str, info: ValidationInfo
    ) -> str:
        field_name = info.field_name or "identifier"
        value = _validate_identifier(value, field_name)
        if (
            info.field_name == "output_mode"
            and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is None
        ):
            raise ValueError("output_mode must be a nonempty slug.")
        return value

    @field_validator("capability_snapshot_id")
    @classmethod
    def _validate_optional_identifier(
        cls, value: Optional[str]
    ) -> Optional[str]:
        if value is not None:
            return _validate_identifier(value, "capability_snapshot_id")
        return value

    @field_validator(
        "input_tokens_gt", "input_tokens_lte", "unit_tokens", mode="before"
    )
    @classmethod
    def _validate_integer_field(
        cls, value: Any, info: ValidationInfo
    ) -> Optional[int]:
        if value is None and info.field_name == "input_tokens_lte":
            return None
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be a plain integer.")
        return value

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, value: str) -> str:
        if re.fullmatch(r"[A-Z]{3}", value) is None:
            raise ValueError(
                "currency must be an uppercase three-letter currency code."
            )
        return value

    @field_validator("effective_from", "effective_until")
    @classmethod
    def _validate_effective_datetime(
        cls, value: Optional[datetime], info: ValidationInfo
    ) -> Optional[datetime]:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def _validate_rule(self) -> "PricingRule":
        if self.input_tokens_gt < 0:
            raise ValueError("input_tokens_gt must be nonnegative.")
        if (
            self.input_tokens_lte is not None
            and self.input_tokens_lte <= self.input_tokens_gt
        ):
            raise ValueError(
                "input_tokens_lte must be greater than input_tokens_gt."
            )
        if self.unit_tokens <= 0:
            raise ValueError("unit_tokens must be positive.")
        if (
            self.effective_from is not None
            and self.effective_until is not None
            and self.effective_from >= self.effective_until
        ):
            raise ValueError(
                "effective_from must be earlier than effective_until."
            )
        if self.rate_type == "promotional" and (
            self.effective_from is None or self.effective_until is None
        ):
            raise ValueError(
                "Promotional pricing requires both exact effective bounds."
            )

        cache_read = self.rates.cache_read_input
        cache_write = self.rates.cache_write_input
        if self.cache_mode == "none" and (
            cache_read is not None or cache_write is not None
        ):
            raise ValueError("none cache mode cannot declare cache rates.")
        if self.cache_mode == "implicit" and (
            cache_read is None or cache_write is not None
        ):
            raise ValueError(
                "implicit cache mode requires only cache_read_input."
            )
        if self.cache_mode == "explicit" and (
            cache_read is None or cache_write is None
        ):
            raise ValueError(
                "explicit cache mode requires both cache input rates."
            )
        return self


class PricingSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    schema_: Literal["pricing-rules-v1"] = Field(alias="schema")
    rules: List[PricingRule] = Field(min_length=1)

    @property
    def schema(self) -> Literal["pricing-rules-v1"]:  # type: ignore[override]
        return self.schema_

    @model_validator(mode="after")
    def _validate_unique_rule_ids(self) -> "PricingSchedule":
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError(
                "Pricing rule IDs must be unique within a schedule."
            )
        return self


class PricingContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str
    service_scope: str
    deployment_type: str
    output_mode: str
    request_mode: Literal["realtime", "batch"]
    cache_mode: Literal["none", "implicit", "explicit"]

    @field_validator(
        "region", "service_scope", "deployment_type", "output_mode"
    )
    @classmethod
    def _validate_context_identifier(
        cls, value: str, info: ValidationInfo
    ) -> str:
        field_name = info.field_name or "identifier"
        value = _validate_identifier(value, field_name)
        if (
            info.field_name == "output_mode"
            and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is None
        ):
            raise ValueError("output_mode must be a nonempty slug.")
        return value


class ResolvedPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["legacy", "pricing-rules-v1"]
    rule_id: Optional[str] = None
    rates: PricingRates
    cache_mode: Literal["none", "implicit", "explicit"]
    currency: str
    unit_tokens: int = Field(gt=0)


class PricingCost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_cost: float
    output_cost: float
    total_cost: float
    currency: str


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
    pricing: Pricing | PricingSchedule
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


def _resolve_declared_pricing(
    model_name: str,
    billing_model_id: str,
    pricing: Pricing | PricingSchedule,
    legacy_currency: str,
    *,
    pricing_context: PricingContext | Mapping[str, Any] | None = None,
    input_tokens: Optional[int] = None,
    effective_at: Optional[datetime] = None,
) -> ResolvedPricing:
    if isinstance(pricing, Pricing):
        legacy_rates = PricingRates.model_construct(
            input=pricing.input,
            output=pricing.output,
            cache_read_input=None,
            cache_write_input=None,
        )
        return ResolvedPricing(
            source="legacy",
            rule_id=None,
            rates=legacy_rates,
            cache_mode="none",
            currency=legacy_currency,
            unit_tokens=1_000_000,
        )

    if pricing_context is None:
        raise PricingContextRequiredError(
            f"Pricing context is required for model '{model_name}'."
        )
    if type(input_tokens) is not int or input_tokens < 0:
        raise PricingContextRequiredError(
            "A valid total input-token count is required for model "
            f"'{model_name}'."
        )
    if not isinstance(effective_at, datetime) or (
        effective_at.tzinfo is None or effective_at.utcoffset() is None
    ):
        raise PricingContextRequiredError(
            "A timezone-aware effective timestamp is required for model "
            f"'{model_name}'."
        )

    if isinstance(pricing_context, PricingContext):
        context = pricing_context.model_copy(deep=True)
    elif isinstance(pricing_context, Mapping):
        context = PricingContext.model_validate(
            deepcopy(dict(pricing_context))
        )
    else:
        raise PricingContextRequiredError(
            f"Pricing context is required for model '{model_name}'."
        )

    matches = []
    for rule in pricing.rules:
        if rule.billing_model_id != billing_model_id:
            continue
        if any(
            getattr(rule, field_name) != getattr(context, field_name)
            for field_name in (
                "region",
                "service_scope",
                "deployment_type",
                "output_mode",
                "request_mode",
                "cache_mode",
            )
        ):
            continue
        if not rule.input_tokens_gt < input_tokens:
            continue
        if (
            rule.input_tokens_lte is not None
            and input_tokens > rule.input_tokens_lte
        ):
            continue
        if (
            rule.effective_from is not None
            and effective_at < rule.effective_from
        ):
            continue
        if (
            rule.effective_until is not None
            and effective_at >= rule.effective_until
        ):
            continue
        matches.append(rule)

    context_identifiers = ", ".join(
        f"{field_name}={getattr(context, field_name)!r}"
        for field_name in (
            "region",
            "service_scope",
            "deployment_type",
            "output_mode",
            "request_mode",
            "cache_mode",
        )
    )
    if not matches:
        raise PricingNoMatchError(
            f"No pricing rule matches model '{model_name}' and context "
            f"({context_identifiers})."
        )
    if len(matches) > 1:
        rule_ids = ", ".join(rule.rule_id for rule in matches)
        raise PricingAmbiguityError(
            f"Pricing rules [{rule_ids}] all match model '{model_name}' and "
            f"context ({context_identifiers})."
        )

    selected = matches[0]
    return ResolvedPricing(
        source="pricing-rules-v1",
        rule_id=selected.rule_id,
        rates=selected.rates.model_copy(deep=True),
        cache_mode=selected.cache_mode,
        currency=selected.currency,
        unit_tokens=selected.unit_tokens,
    )


def resolve_model_pricing(
    model_name: str,
    config_source: Path | Dict[str, Any] | None,
    *,
    pricing_context: PricingContext | Mapping[str, Any] | None = None,
    input_tokens: Optional[int] = None,
    effective_at: Optional[datetime] = None,
) -> ResolvedPricing:
    _, provider_settings, model_details = get_model_details(
        model_name, config_source
    )
    return _resolve_declared_pricing(
        model_name,
        model_details.id,
        model_details.pricing,
        provider_settings.pricing_currency,
        pricing_context=pricing_context,
        input_tokens=input_tokens,
        effective_at=effective_at,
    )


def _validate_token_count(
    field_name: str, value: Any, *, required: bool
) -> Optional[int]:
    if value is None and not required:
        return None
    if type(value) is not int or value < 0:
        raise PricingCalculationError(
            f"{field_name} must be a plain nonnegative integer."
        )
    return value


def calculate_pricing_cost(
    pricing: ResolvedPricing,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: Optional[int] = None,
    cache_write_input_tokens: Optional[int] = None,
) -> PricingCost:
    total_input = _validate_token_count(
        "input_tokens", input_tokens, required=True
    )
    total_output = _validate_token_count(
        "output_tokens", output_tokens, required=True
    )
    cache_read = _validate_token_count(
        "cache_read_input_tokens",
        cache_read_input_tokens,
        required=pricing.cache_mode in {"implicit", "explicit"},
    )
    cache_write = _validate_token_count(
        "cache_write_input_tokens",
        cache_write_input_tokens,
        required=pricing.cache_mode == "explicit",
    )
    assert total_input is not None
    assert total_output is not None

    read_tokens = cache_read or 0
    write_tokens = cache_write or 0
    read_rate = pricing.rates.cache_read_input
    write_rate = pricing.rates.cache_write_input

    if pricing.cache_mode == "none":
        if read_tokens != 0 or write_tokens != 0:
            raise PricingCalculationError(
                "none cache mode cannot consume cache token buckets."
            )
        if read_rate is not None or write_rate is not None:
            raise PricingCalculationError(
                "none cache mode cannot use cache rates."
            )
    elif pricing.cache_mode == "implicit":
        if write_tokens != 0:
            raise PricingCalculationError(
                "implicit cache mode cannot consume cache-write tokens."
            )
        if read_rate is None or write_rate is not None:
            raise PricingCalculationError(
                "implicit cache mode requires only a cache-read rate."
            )
    else:
        if read_rate is None or write_rate is None:
            raise PricingCalculationError(
                "explicit cache mode requires both cache rates."
            )

    if read_tokens + write_tokens > total_input:
        raise PricingCalculationError(
            "Cache token buckets cannot exceed total input tokens."
        )

    ordinary_tokens = total_input - read_tokens - write_tokens
    input_cost = ordinary_tokens * pricing.rates.input / pricing.unit_tokens
    if read_rate is not None:
        input_cost += read_tokens * read_rate / pricing.unit_tokens
    if write_rate is not None:
        input_cost += write_tokens * write_rate / pricing.unit_tokens
    output_cost = total_output * pricing.rates.output / pricing.unit_tokens
    return PricingCost(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        currency=pricing.currency,
    )
