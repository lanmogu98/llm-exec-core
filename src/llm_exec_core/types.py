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

    def to_legacy_dict(self, duration_seconds: float = 0.0) -> Dict[str, Any]:
        process_times: Dict[str, Any] = dict(self.process_times)
        if "request_times" not in process_times:
            process_times["request_times"] = []
        process_times["total_time"] = duration_seconds
        return {
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
