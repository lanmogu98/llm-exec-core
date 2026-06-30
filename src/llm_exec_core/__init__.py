"""OpenAI-compatible LLM execution core."""

from .client import LLMClient
from .types import ExecutionMetadata, LLMResult, TokenUsage
from .usage import format_usage_report

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LLMClient",
    "ExecutionMetadata",
    "LLMResult",
    "TokenUsage",
    "format_usage_report",
]
