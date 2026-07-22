"""OpenAI-compatible LLM execution core."""

from .client import LLMClient, StructuredOutputValidationError
from .types import ExecutionMetadata, LLMResult, TokenUsage
from .usage import format_usage_report

__version__ = "0.4.1"

__all__ = [
    "__version__",
    "LLMClient",
    "StructuredOutputValidationError",
    "ExecutionMetadata",
    "LLMResult",
    "TokenUsage",
    "format_usage_report",
]
