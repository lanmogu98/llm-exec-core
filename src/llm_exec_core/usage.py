"""Usage formatting helpers."""

from __future__ import annotations

import time
from typing import Any, Dict


def _as_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_usage_report(
    project_name: str,
    model: str,
    model_name: str,
    pricing_currency: str,
    token_usage: Dict[str, Any],
    timestamp: str | None = None,
) -> str:
    """Return the human-readable token usage report as text only."""

    generated_at = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    total_input_tokens = _as_int(token_usage.get("total_input_tokens"))
    total_output_tokens = _as_int(token_usage.get("total_output_tokens"))
    total_tokens = total_input_tokens + total_output_tokens

    process_times = token_usage.get("process_times", {})
    if not isinstance(process_times, dict):
        process_times = {}
    total_process_time = _as_number(process_times.get("total_time"))

    cost = token_usage.get("cost", {})
    if not isinstance(cost, dict):
        cost = {}
    input_cost = _as_number(cost.get("input_cost"))
    output_cost = _as_number(cost.get("output_cost"))
    total_cost = _as_number(cost.get("total_cost"))

    lines = [
        f"Token Usage Report for {project_name}",
        f"Generated on: {generated_at}",
        f"Model: {model} ({model_name})",
        "",
        "Summary:",
        f"  Total Input Tokens: {total_input_tokens}",
        f"  Total Output Tokens: {total_output_tokens}",
        f"  Total Tokens: {total_tokens}",
        f"  Total Process Time: {total_process_time:.2f} seconds",
        f"  Input Cost: {pricing_currency}{input_cost:.6f}",
        f"  Output Cost: {pricing_currency}{output_cost:.6f}",
        f"  Total Cost: {pricing_currency}{total_cost:.6f}",
        "",
        "Detailed Usage by Request:",
    ]

    requests = token_usage.get("requests", [])
    if isinstance(requests, list):
        for index, request in enumerate(requests, start=1):
            if not isinstance(request, dict):
                request = {}

            request_cost = request.get("input_cost", 0.0)
            output_request_cost = request.get("output_cost", 0.0)
            total_request_cost = request.get("total_cost", 0.0)
            process_time = request.get("process_time", 0.0)

            lines.append(f"  Request {index}: {request.get('name')}")
            lines.append(f"    Timestamp: {request.get('timestamp', 'N/A')}")
            lines.append(f"    Input Tokens: {request.get('input_tokens', 0)}")
            lines.append(f"    Output Tokens: {request.get('output_tokens', 0)}")
            lines.append(f"    Total Tokens: {request.get('total_tokens', 0)}")
            lines.append(f"    Process Time: {_as_number(process_time):.2f} seconds")
            lines.append(
                f"    Input Cost: {pricing_currency}{_as_number(request_cost):.6f}"
            )
            lines.append(
                f"    Output Cost: {pricing_currency}{_as_number(output_request_cost):.6f}"
            )
            lines.append(
                f"    Total Cost: {pricing_currency}{_as_number(total_request_cost):.6f}"
            )
            lines.append("")

    return "\n".join(lines)
