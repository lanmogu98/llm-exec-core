"""Public value objects for llm execution results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal

AccountingReason = Literal[
    "missing_usage",
    "malformed_usage",
    "profile_contradiction",
    "pricing_no_match",
    "pricing_ambiguity",
    "calculation_failure",
    "aggregate_currency_conflict",
]

_ACCOUNTING_REASONS = {
    "missing_usage",
    "malformed_usage",
    "profile_contradiction",
    "pricing_no_match",
    "pricing_ambiguity",
    "calculation_failure",
    "aggregate_currency_conflict",
}


@dataclass(frozen=True, slots=True)
class AccountingStatus:
    """Availability of authoritative token and cost accounting."""

    tokens_available: bool = True
    cost_available: bool = True
    reason: AccountingReason | None = None

    def __post_init__(self) -> None:
        if (
            type(self.tokens_available) is not bool
            or type(self.cost_available) is not bool
        ):
            raise ValueError("Accounting availability flags must be booleans.")
        if not self.tokens_available and self.cost_available:
            raise ValueError(
                "Cost cannot be available when token counts are unavailable."
            )
        is_available = self.tokens_available and self.cost_available
        if is_available and self.reason is not None:
            raise ValueError(
                "Available accounting cannot declare an unavailable reason."
            )
        if not is_available and self.reason not in _ACCOUNTING_REASONS:
            raise ValueError(
                "Unavailable accounting requires a finite reason code."
            )


@dataclass(slots=True)
class TokenUsage:
    """Usage metrics returned from an LLM call."""

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    input_cost: float | None
    output_cost: float | None
    total_cost: float | None
    currency: str | None
    requests: List[Dict[str, Any]] = field(default_factory=list)
    process_times: Dict[str, Any] = field(default_factory=dict)
    accounting: AccountingStatus = field(default_factory=AccountingStatus)

    def to_legacy_dict(self, duration_seconds: float = 0.0) -> Dict[str, Any]:
        process_times: Dict[str, Any] = dict(self.process_times)
        if "request_times" not in process_times:
            process_times["request_times"] = []
        process_times["total_time"] = duration_seconds
        legacy: Dict[str, Any] = {
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "requests": self.requests,
            "process_times": process_times,
            "cost": {
                "input_cost": self.input_cost,
                "output_cost": self.output_cost,
                "total_cost": self.total_cost,
            },
        }
        if not (
            self.accounting.tokens_available and self.accounting.cost_available
        ):
            legacy["accounting"] = asdict(self.accounting)
        return legacy


@dataclass(slots=True)
class ExecutionMetadata:
    """Execution metadata for a completed LLM request."""

    request_id: str | None
    run_id: str | None
    request_name: str
    model_name: str
    model_id: str
    provider_name: str
    started_at: str
    finished_at: str
    duration_seconds: float
    trace_context: Dict[str, Any]
    planning: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResult:
    """Container for an LLM text response plus usage and metadata."""

    text: str
    usage: TokenUsage
    metadata: ExecutionMetadata
    structured: Any | None = None

    def to_legacy_tuple(self) -> tuple[str, Dict[str, Any]]:
        legacy_usage: Dict[str, Any] = self.usage.to_legacy_dict(
            duration_seconds=self.metadata.duration_seconds
        )
        legacy_usage["metadata"] = asdict(self.metadata)
        return self.text, legacy_usage
