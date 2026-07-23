"""Usage formatting helpers."""

from __future__ import annotations

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


def _integer_text(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(_as_int(value))


def _cost_text(pricing_currency: str | None, value: Any) -> str:
    if value is None or pricing_currency is None:
        return "N/A"
    return f"{pricing_currency}{_as_number(value):.6f}"


def format_usage_report(
    project_name: str,
    model: str,
    model_name: str,
    pricing_currency: str | None,
    token_usage: Dict[str, Any],
    timestamp: str | None = None,
) -> str:
    """Return the human-readable token usage report as text only."""

    generated_at = timestamp or "N/A"
    raw_input_tokens = token_usage.get("total_input_tokens", 0)
    raw_output_tokens = token_usage.get("total_output_tokens", 0)
    total_input_tokens = (
        None if raw_input_tokens is None else _as_int(raw_input_tokens)
    )
    total_output_tokens = (
        None if raw_output_tokens is None else _as_int(raw_output_tokens)
    )
    total_tokens = (
        None
        if total_input_tokens is None or total_output_tokens is None
        else total_input_tokens + total_output_tokens
    )

    process_times = token_usage.get("process_times", {})
    if not isinstance(process_times, dict):
        process_times = {}
    total_process_time = _as_number(process_times.get("total_time"))

    cost = token_usage.get("cost", {})
    if not isinstance(cost, dict):
        cost = {}

    lines = [
        f"Token Usage Report for {project_name}",
        f"Generated on: {generated_at}",
        f"Model: {model} ({model_name})",
        "",
        "Summary:",
        f"  Total Input Tokens: {_integer_text(total_input_tokens)}",
        f"  Total Output Tokens: {_integer_text(total_output_tokens)}",
        f"  Total Tokens: {_integer_text(total_tokens)}",
        f"  Total Process Time: {total_process_time:.2f} seconds",
        "  Input Cost: "
        f"{_cost_text(pricing_currency, cost.get('input_cost', 0.0))}",
        "  Output Cost: "
        f"{_cost_text(pricing_currency, cost.get('output_cost', 0.0))}",
        "  Total Cost: "
        f"{_cost_text(pricing_currency, cost.get('total_cost', 0.0))}",
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
            request_currency = request.get("currency", pricing_currency)
            if not isinstance(request_currency, str):
                request_currency = None
            lines.append(f"  Request {index}: {request.get('name')}")
            lines.append(f"    Timestamp: {request.get('timestamp', 'N/A')}")
            lines.append(
                "    Input Tokens: "
                f"{_integer_text(request.get('input_tokens', 0))}"
            )
            lines.append(
                "    Output Tokens: "
                f"{_integer_text(request.get('output_tokens', 0))}"
            )
            lines.append(
                "    Total Tokens: "
                f"{_integer_text(request.get('total_tokens', 0))}"
            )
            lines.append(
                f"    Process Time: {_as_number(process_time):.2f} seconds"
            )
            lines.append(
                "    Input Cost: "
                f"{_cost_text(request_currency, request_cost)}"
            )
            lines.append(
                "    Output Cost: "
                f"{_cost_text(request_currency, output_request_cost)}"
            )
            lines.append(
                "    Total Cost: "
                f"{_cost_text(request_currency, total_request_cost)}"
            )
            lines.append("")

    return "\n".join(lines)
