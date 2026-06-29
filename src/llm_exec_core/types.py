"""Public value objects for llm execution results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class TokenUsage:
    """Usage metrics returned from an LLM call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str
    requests: List[Dict[str, Any]] = field(default_factory=list)
    process_times: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionMetadata:
    """Execution metadata for a completed LLM request."""

    request_id: str
    run_id: str
    request_name: str
    model_name: str
    model_id: str
    provider_name: str
    started_at: str
    finished_at: str
    duration_seconds: float
    trace_context: Dict[str, Any]


@dataclass(slots=True)
class LLMResult:
    """Container for an LLM text response plus usage and metadata."""

    text: str
    usage: TokenUsage
    metadata: ExecutionMetadata

    def to_legacy_tuple(self) -> tuple[str, Dict[str, Any]]:
        legacy_usage: Dict[str, Any] = {
            "total_input_tokens": self.usage.input_tokens,
            "total_output_tokens": self.usage.output_tokens,
            "requests": self.usage.requests,
            "process_times": {
                **self.usage.process_times,
                "total_time": self.metadata.duration_seconds,
            },
            "cost": {
                "input_cost": self.usage.input_cost,
                "output_cost": self.usage.output_cost,
                "total_cost": self.usage.total_cost,
            },
            "metadata": asdict(self.metadata),
        }
        return self.text, legacy_usage
