"""OpenAI-compatible LLM execution core."""

from .client import LLMClient, StructuredOutputValidationError
from .types import AccountingStatus, ExecutionMetadata, LLMResult, TokenUsage
from .usage import format_usage_report

__version__ = "0.4.1"

__all__ = [
    "__version__",
    "LLMClient",
    "StructuredOutputValidationError",
    "AccountingStatus",
    "ExecutionMetadata",
    "LLMResult",
    "TokenUsage",
    "format_usage_report",
]
