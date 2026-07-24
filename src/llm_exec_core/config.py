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
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_validator,
    model_serializer,
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


UsageAccountingProfile = Literal[
    "openai-chat-standard-v1",
    "openai-chat-cached-tokens-v1",
    "openai-chat-cache-creation-v1",
    "openai-chat-cache-write-v1",
    "openai-chat-cache-hit-miss-v1",
]


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


class CacheModePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_mode: Literal["none", "implicit", "explicit"]
    activation: Literal[
        "not-applicable",
        "automatic",
        "request",
        "control-plane",
        "control-plane-and-request",
    ]

    @model_validator(mode="after")
    def _validate_activation(self) -> "CacheModePolicy":
        compatible_activations = {
            "none": {"not-applicable"},
            "implicit": {"automatic", "control-plane"},
            "explicit": {"request", "control-plane-and-request"},
        }
        if self.activation not in compatible_activations[self.cache_mode]:
            raise ValueError(
                f"{self.cache_mode} cache mode is incompatible with "
                f"{self.activation} activation."
            )
        return self


class CachePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    support: Literal["supported", "unsupported"]
    modes: List[CacheModePolicy] = Field(min_length=1)
    evidence: List[PricingEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_support(self) -> "CachePolicy":
        declared_modes = [mode.cache_mode for mode in self.modes]
        if len(declared_modes) != len(set(declared_modes)):
            raise ValueError("Cache-policy modes must be unique.")
        if self.support == "unsupported":
            if declared_modes != ["none"]:
                raise ValueError(
                    "Unsupported cache policy must declare only none mode."
                )
        elif not any(mode != "none" for mode in declared_modes):
            raise ValueError(
                "Supported cache policy requires a non-none mode."
            )
        return self


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
        if self.cache_mode != "none" and cache_read is None:
            raise ValueError("Non-none cache mode requires cache_read_input.")
        return self


class PricingSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: Literal["pricing-rules-v1"] = Field(alias="schema")
    rules: List[PricingRule] = Field(min_length=1)

    @property
    def schema(self) -> Literal["pricing-rules-v1"]:  # type: ignore[override]
        return self.schema_

    @model_serializer(mode="wrap")
    def _serialize_schedule(
        self, handler: SerializerFunctionWrapHandler
    ) -> Dict[str, Any]:
        serialized: Dict[str, Any] = handler(self)
        schema = serialized.pop("schema", self.schema_)
        serialized.pop("schema_", None)
        return {"schema": schema, **serialized}

    @model_validator(mode="after")
    def _validate_unique_rule_ids(self) -> "PricingSchedule":
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError(
                "Pricing rule IDs must be unique within a schedule."
            )
        return self


class PricingContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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


WirePath = Tuple[str, ...]
ReasoningAvailability = Literal[
    "unavailable", "optional", "adaptive", "always-on"
]
ReasoningMode = Literal["disabled", "enabled", "adaptive", "always-on"]
SelectableReasoningMode = Literal["disabled", "enabled", "adaptive"]
ActiveReasoningMode = Literal["enabled", "adaptive", "always-on"]
OmissionMode = Literal["provider-default", "provider-selected", "required"]


def _validate_wire_path(value: Any, field_name: str) -> WirePath:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} path must be a nonempty array.")
    if any(type(segment) is not str or not segment for segment in value):
        raise ValueError(
            f"{field_name} path segments must be nonempty strings."
        )
    return tuple(value)


def _path_is_prefix(left: WirePath, right: WirePath) -> bool:
    return len(left) < len(right) and right[: len(left)] == left


class NumericRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: float
    maximum: float
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True

    @field_validator("minimum", "maximum", mode="before")
    @classmethod
    def _validate_endpoint(cls, value: Any, info: ValidationInfo) -> float:
        if type(value) not in {int, float}:
            raise ValueError(f"{info.field_name} must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite.")
        return float(value)

    @model_validator(mode="after")
    def _validate_range(self) -> "NumericRange":
        if self.minimum > self.maximum:
            raise ValueError("Sampling range must be ordered.")
        if self.minimum == self.maximum and not (
            self.minimum_inclusive and self.maximum_inclusive
        ):
            raise ValueError("Sampling range cannot be empty.")
        return self


class SamplingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["supported", "fixed", "ignored", "deprecated", "forbidden"]
    range: Optional[NumericRange] = None
    fixed_value: Optional[float] = None

    @field_validator("fixed_value", mode="before")
    @classmethod
    def _validate_fixed_value(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        if type(value) not in {int, float}:
            raise ValueError("fixed_value must be a number.")
        if not math.isfinite(value):
            raise ValueError("fixed_value must be finite.")
        return float(value)

    @model_validator(mode="after")
    def _validate_state(self) -> "SamplingRule":
        if self.state == "supported":
            if self.range is None:
                raise ValueError("supported sampling requires a range.")
            if self.fixed_value is not None:
                raise ValueError(
                    "supported sampling cannot declare a fixed value."
                )
        elif self.state == "fixed":
            if self.fixed_value is None:
                raise ValueError("fixed sampling requires fixed_value.")
            if self.range is not None:
                raise ValueError("fixed sampling cannot declare a range.")
        elif self.range is not None or self.fixed_value is not None:
            raise ValueError(
                "ignored, deprecated, and forbidden sampling permit neither "
                "a range nor fixed_value."
            )
        return self


class SamplingControlPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: SamplingRule
    by_reasoning_mode: Dict[ReasoningMode, SamplingRule] = Field(
        default_factory=dict
    )


class SamplingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: SamplingControlPolicy
    top_p: SamplingControlPolicy


class ReasoningModeWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: WirePath
    values: Dict[SelectableReasoningMode, str | bool]

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> WirePath:
        return _validate_wire_path(value, "reasoning mode")

    @field_validator("values", mode="before")
    @classmethod
    def _validate_value_types(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("Reasoning mode values must be a mapping.")
        if any(type(item) not in {str, bool} for item in value.values()):
            raise ValueError(
                "Reasoning mode wire values must be a string or boolean."
            )
        return value

    @model_validator(mode="after")
    def _validate_unique_values(self) -> "ReasoningModeWire":
        typed_values = [(type(value), value) for value in self.values.values()]
        if len(typed_values) != len(set(typed_values)):
            raise ValueError("Reasoning mode wire values must be unique.")
        return self


class EffortModeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    omission: OmissionMode
    default: Optional[str] = None

    @field_validator("default")
    @classmethod
    def _validate_default_string(cls, value: Optional[str]) -> Optional[str]:
        if value == "":
            raise ValueError("Effort defaults must be nonempty.")
        return value

    @model_validator(mode="after")
    def _validate_omission(self) -> "EffortModeRule":
        if self.omission == "provider-default" and self.default is None:
            raise ValueError(
                "provider-default effort omission requires a default."
            )
        if self.omission != "provider-default" and self.default is not None:
            raise ValueError(
                "provider-selected and required effort omission forbid a "
                "default."
            )
        return self


class ReasoningEffortPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: WirePath
    allowed_values: Tuple[str, ...] = Field(min_length=1)
    aliases: Dict[str, str] = Field(default_factory=dict)
    modes: Dict[ActiveReasoningMode, EffortModeRule] = Field(min_length=1)

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> WirePath:
        return _validate_wire_path(value, "reasoning effort")

    @field_validator("allowed_values", mode="before")
    @classmethod
    def _validate_allowed_values(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("Effort allowed values must be nonempty.")
        if any(type(item) is not str or not item for item in value):
            raise ValueError("Effort allowed values must be nonempty strings.")
        return value

    @field_validator("aliases", mode="before")
    @classmethod
    def _validate_alias_strings(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("Effort aliases must be a mapping.")
        if any(
            type(alias) is not str
            or not alias
            or type(target) is not str
            or not target
            for alias, target in value.items()
        ):
            raise ValueError(
                "Effort aliases and targets must be nonempty strings."
            )
        return value

    @model_validator(mode="after")
    def _validate_values(self) -> "ReasoningEffortPolicy":
        canonical = set(self.allowed_values)
        if len(canonical) != len(self.allowed_values):
            raise ValueError("Effort allowed values must be unique.")
        if canonical.intersection(self.aliases):
            raise ValueError(
                "Effort alias keys cannot collide with canonical values."
            )
        if any(target not in canonical for target in self.aliases.values()):
            raise ValueError("Every effort alias target must be canonical.")
        for rule in self.modes.values():
            if rule.default is not None and rule.default not in canonical:
                raise ValueError("Every effort default must be canonical.")
        return self


class BudgetModeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    omission: OmissionMode
    default: Optional[int] = None
    output_limit_relation: Literal["none", "less-than", "less-than-or-equal"]

    @field_validator("default", mode="before")
    @classmethod
    def _validate_default_integer(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if type(value) is not int:
            raise ValueError("Budget default must be a plain integer.")
        return value

    @model_validator(mode="after")
    def _validate_omission(self) -> "BudgetModeRule":
        if self.omission == "provider-default" and self.default is None:
            raise ValueError(
                "provider-default budget omission requires a default."
            )
        if self.omission != "provider-default" and self.default is not None:
            raise ValueError(
                "provider-selected and required budget omission forbid a "
                "default."
            )
        return self


class ReasoningBudgetPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: WirePath
    minimum: int
    maximum: Optional[int] = None
    modes: Dict[ActiveReasoningMode, BudgetModeRule] = Field(min_length=1)

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> WirePath:
        return _validate_wire_path(value, "reasoning budget")

    @field_validator("minimum", "maximum", mode="before")
    @classmethod
    def _validate_integer(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None and info.field_name == "maximum":
            return None
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be a plain integer.")
        return value

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ReasoningBudgetPolicy":
        if self.minimum < 0:
            raise ValueError("Budget minimum must be nonnegative.")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError(
                "Budget maximum must be greater than or equal to minimum."
            )
        for rule in self.modes.values():
            if rule.default is None:
                continue
            if rule.default < self.minimum or (
                self.maximum is not None and rule.default > self.maximum
            ):
                raise ValueError("Budget default must be within bounds.")
        return self


class ReasoningPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    availability: ReasoningAvailability
    allowed_modes: Tuple[ReasoningMode, ...] = Field(min_length=1)
    default_mode: ReasoningMode
    can_disable: bool
    mode: Optional[ReasoningModeWire] = None
    effort: Optional[ReasoningEffortPolicy] = None
    budget_tokens: Optional[ReasoningBudgetPolicy] = None
    allow_effort_with_budget_in: Tuple[ActiveReasoningMode, ...] = ()

    @field_validator(
        "allowed_modes", "allow_effort_with_budget_in", mode="before"
    )
    @classmethod
    def _validate_mode_sequence(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{info.field_name} must be an array.")
        if info.field_name == "allowed_modes" and not value:
            raise ValueError("allowed_modes must be nonempty.")
        return value

    @model_validator(mode="after")
    def _validate_reasoning_policy(self) -> "ReasoningPolicy":
        allowed = tuple(self.allowed_modes)
        allowed_set = set(allowed)
        if len(allowed) != len(allowed_set):
            raise ValueError("Reasoning allowed modes must be unique.")
        coexist = tuple(self.allow_effort_with_budget_in)
        if len(coexist) != len(set(coexist)):
            raise ValueError("Effort/budget coexistence modes must be unique.")
        if self.default_mode not in allowed_set:
            raise ValueError("default_mode must be an allowed mode.")

        if self.availability == "unavailable":
            if (
                allowed != ("disabled",)
                or self.default_mode != "disabled"
                or self.can_disable
                or self.mode is not None
                or self.effort is not None
                or self.budget_tokens is not None
                or coexist
            ):
                raise ValueError(
                    "unavailable reasoning permits only disabled without "
                    "controls."
                )
        elif self.availability == "optional":
            if allowed_set != {"disabled", "enabled"} or len(allowed) != 2:
                raise ValueError(
                    "optional reasoning requires exactly disabled and "
                    "enabled modes."
                )
            if not self.can_disable or self.mode is None:
                raise ValueError(
                    "optional reasoning requires disable support and a mode "
                    "wire declaration."
                )
        elif self.availability == "adaptive":
            if (
                "adaptive" not in allowed_set
                or "always-on" in allowed_set
                or not allowed_set.issubset(
                    {"disabled", "enabled", "adaptive"}
                )
            ):
                raise ValueError(
                    "adaptive reasoning requires adaptive and permits only "
                    "disabled, enabled, and adaptive modes."
                )
            if self.default_mode not in {"disabled", "adaptive"}:
                raise ValueError(
                    "adaptive reasoning default must be adaptive or "
                    "disabled."
                )
            if self.can_disable != ("disabled" in allowed_set):
                raise ValueError(
                    "adaptive can_disable must match disabled availability."
                )
            if self.mode is None:
                raise ValueError(
                    "adaptive reasoning requires a mode wire declaration."
                )
        elif (
            allowed != ("always-on",)
            or self.default_mode != "always-on"
            or self.can_disable
            or self.mode is not None
        ):
            raise ValueError(
                "always-on reasoning requires only always-on without a mode "
                "wire field."
            )

        if self.mode is not None:
            selectable = {mode for mode in allowed_set if mode != "always-on"}
            if set(self.mode.values) != selectable:
                raise ValueError(
                    "Reasoning mode wire values must map every selectable "
                    "mode exactly."
                )

        active = {mode for mode in allowed_set if mode != "disabled"}
        effort_modes = set() if self.effort is None else set(self.effort.modes)
        budget_modes = (
            set()
            if self.budget_tokens is None
            else set(self.budget_tokens.modes)
        )
        if not effort_modes.issubset(active):
            raise ValueError(
                "Effort mode keys must be active reasoning modes."
            )
        if not budget_modes.issubset(active):
            raise ValueError(
                "Budget mode keys must be active reasoning modes."
            )
        coexist_set = set(coexist)
        if not coexist_set.issubset(active):
            raise ValueError("Effort/budget coexistence modes must be active.")
        if coexist_set and (self.effort is None or self.budget_tokens is None):
            raise ValueError(
                "Effort/budget coexistence requires both policies."
            )
        if not coexist_set.issubset(effort_modes.intersection(budget_modes)):
            raise ValueError(
                "Effort/budget coexistence modes must exist in both "
                "policies."
            )

        paths = []
        if self.mode is not None:
            paths.append(("mode", self.mode.path))
        if self.effort is not None:
            paths.append(("effort", self.effort.path))
        if self.budget_tokens is not None:
            paths.append(("budget", self.budget_tokens.path))
        protected_roots = {"model", "messages", "stream"}
        sampling_roots = {"temperature", "top_p"}
        for name, path in paths:
            if path[0] in sampling_roots:
                raise ValueError(
                    f"{name} path cannot root at a sampling-owned field."
                )
            if path[0] in protected_roots:
                raise ValueError(
                    f"{name} path cannot root at a protected field."
                )
        for index, (left_name, left) in enumerate(paths):
            for right_index, (right_name, right) in enumerate(paths):
                if right_index <= index:
                    continue
                if left == right:
                    raise ValueError(
                        f"{left_name} and {right_name} paths must be "
                        "distinct."
                    )
                if _path_is_prefix(left, right) or _path_is_prefix(
                    right, left
                ):
                    raise ValueError(
                        f"{left_name} and {right_name} paths cannot have a "
                        "prefix relationship."
                    )
        return self


class GenerationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: ReasoningPolicy
    sampling: SamplingPolicy

    @model_validator(mode="after")
    def _validate_sampling_modes(self) -> "GenerationPolicy":
        reachable = set(self.reasoning.allowed_modes)
        for name, control in (
            ("temperature", self.sampling.temperature),
            ("top_p", self.sampling.top_p),
        ):
            if not set(control.by_reasoning_mode).issubset(reachable):
                raise ValueError(
                    f"{name} sampling overrides must use reachable "
                    "reasoning modes."
                )
        return self


class ReasoningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Optional[SelectableReasoningMode] = None
    effort: Optional[str] = None
    budget_tokens: Optional[int] = None

    @field_validator("budget_tokens", mode="before")
    @classmethod
    def _validate_budget(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if type(value) is not int:
            raise ValueError("budget_tokens must be a plain integer.")
        return value


class SamplingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: Optional[float] = None
    top_p: Optional[float] = None

    @field_validator("temperature", "top_p", mode="before")
    @classmethod
    def _validate_sampling_value(
        cls, value: Any, info: ValidationInfo
    ) -> Optional[float]:
        if value is None:
            return None
        if type(value) not in {int, float}:
            raise ValueError(f"{info.field_name} must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite.")
        value = float(value)
        return 0.0 if value == 0.0 else value


class GenerationControls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: Optional[ReasoningRequest] = None
    sampling: Optional[SamplingRequest] = None


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
    generation_policy: Optional[GenerationPolicy] = None

    @model_serializer(mode="wrap")
    def _serialize_capabilities(
        self, handler: SerializerFunctionWrapHandler
    ) -> Dict[str, Any]:
        serialized: Dict[str, Any] = handler(self)
        if self.generation_policy is None:
            serialized.pop("generation_policy", None)
        return serialized


class ModelDetails(BaseModel):
    id: str
    pricing: Pricing | PricingSchedule
    cache_policy: Optional[CachePolicy] = None
    capabilities: Optional[ModelCapabilities] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    request_overrides: Optional[Dict[str, Any]] = None
    output_token_field: Optional[
        Literal["max_tokens", "max_completion_tokens"]
    ] = None

    @field_validator("pricing", mode="before")
    @classmethod
    def _select_declared_pricing(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "schema" in value:
            return PricingSchedule.model_validate(value)
        return value

    @model_validator(mode="after")
    def _validate_rich_cache_policy(self) -> "ModelDetails":
        if not isinstance(self.pricing, PricingSchedule):
            return self
        if self.cache_policy is None:
            raise ValueError(
                "Rich pricing schedules require a model cache policy."
            )
        pricing_modes = {rule.cache_mode for rule in self.pricing.rules}
        policy_modes = {mode.cache_mode for mode in self.cache_policy.modes}
        if pricing_modes != policy_modes:
            raise ValueError(
                "Pricing-rule cache modes and cache-policy modes must match "
                "exactly."
            )
        return self


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
    usage_accounting: Optional[UsageAccountingProfile] = None
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

    @model_validator(mode="after")
    def _validate_rich_usage_accounting(self) -> "ProviderSettings":
        schedules = [
            model.pricing
            for model in self.models.values()
            if isinstance(model.pricing, PricingSchedule)
        ]
        if not schedules:
            return self
        if self.usage_accounting is None:
            raise ValueError(
                "Routes containing rich pricing schedules require "
                "usage_accounting."
            )

        read_only_profiles = {
            "openai-chat-cached-tokens-v1",
            "openai-chat-cache-hit-miss-v1",
        }
        write_profiles = {
            "openai-chat-cache-creation-v1",
            "openai-chat-cache-write-v1",
        }
        for schedule in schedules:
            for rule in schedule.rules:
                if self.usage_accounting == "openai-chat-standard-v1":
                    if rule.cache_mode != "none":
                        raise ValueError(
                            "Standard usage accounting permits only none "
                            "cache mode."
                        )
                elif self.usage_accounting in read_only_profiles:
                    if rule.rates.cache_write_input is not None:
                        raise ValueError(
                            "Read-only usage accounting forbids a cache-write "
                            "rate."
                        )
                elif (
                    self.usage_accounting in write_profiles
                    and rule.cache_mode != "none"
                    and rule.rates.cache_write_input is None
                ):
                    raise ValueError(
                        "Write-bucket usage accounting requires an explicit "
                        "cache-write rate for every non-none rule."
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
        required=pricing.cache_mode != "none",
    )
    cache_write = _validate_token_count(
        "cache_write_input_tokens",
        cache_write_input_tokens,
        required=pricing.rates.cache_write_input is not None,
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
    else:
        if read_rate is None:
            raise PricingCalculationError(
                "Non-none cache mode requires a cache-read rate."
            )
        if write_tokens != 0 and write_rate is None:
            raise PricingCalculationError(
                "A nonzero cache-write bucket requires an explicit rate."
            )

    if read_tokens + write_tokens > total_input:
        raise PricingCalculationError(
            "Cache token buckets cannot exceed total input tokens."
        )

    ordinary_tokens = total_input - read_tokens - write_tokens
    try:
        input_cost = (
            ordinary_tokens * pricing.rates.input / pricing.unit_tokens
        )
        if read_rate is not None:
            input_cost += read_tokens * read_rate / pricing.unit_tokens
        if write_rate is not None:
            input_cost += write_tokens * write_rate / pricing.unit_tokens
        output_cost = total_output * pricing.rates.output / pricing.unit_tokens
        total_cost = input_cost + output_cost
    except OverflowError as error:
        raise PricingCalculationError(
            "Pricing cost calculation must remain finite."
        ) from error
    if not all(
        math.isfinite(value) for value in (input_cost, output_cost, total_cost)
    ):
        raise PricingCalculationError(
            "Pricing cost calculation must remain finite."
        )
    return PricingCost(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        currency=pricing.currency,
    )
