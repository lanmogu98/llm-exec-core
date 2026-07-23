from llm_exec_core.usage import format_usage_report


def test_format_usage_report_returns_text_without_writing_file():
    report = format_usage_report(
        project_name="Test Project",
        model="provider-model-id",
        model_name="test-model",
        pricing_currency="$",
        token_usage={
            "total_input_tokens": 10,
            "total_output_tokens": 20,
            "requests": [
                {
                    "name": "brief",
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "process_time": 0.5,
                    "input_cost": 0.001,
                    "output_cost": 0.002,
                    "total_cost": 0.003,
                    "timestamp": "2026-06-18 12:00:00",
                }
            ],
            "process_times": {"total_time": 0.5, "request_times": []},
            "cost": {
                "input_cost": 0.001,
                "output_cost": 0.002,
                "total_cost": 0.003,
            },
        },
        timestamp="2026-06-18 12:00:01",
    )

    assert "Token Usage Report for Test Project" in report
    assert "Model: provider-model-id (test-model)" in report
    assert "Total Tokens: 30" in report
    assert "Request 1: brief" in report


def test_format_usage_report_uses_placeholder_without_timestamp():
    report = format_usage_report(
        project_name="Test Project",
        model="provider-model-id",
        model_name="test-model",
        pricing_currency="$",
        token_usage={
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "requests": [],
            "process_times": {"total_time": 0},
            "cost": {"input_cost": 0, "output_cost": 0, "total_cost": 0},
        },
    )

    assert "Generated on: N/A" in report


def test_format_usage_report_renders_unavailable_values():
    report = format_usage_report(
        project_name="Test Project",
        model="provider-model-id",
        model_name="test-model",
        pricing_currency="USD",
        token_usage={
            "total_input_tokens": None,
            "total_output_tokens": None,
            "requests": [
                {
                    "name": "rich",
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "process_time": 0.5,
                    "input_cost": None,
                    "output_cost": None,
                    "total_cost": None,
                    "timestamp": "2026-07-23 12:00:00",
                    "accounting": {
                        "tokens_available": False,
                        "cost_available": False,
                        "reason": "missing_usage",
                    },
                }
            ],
            "process_times": {"total_time": 0.5, "request_times": []},
            "cost": {
                "input_cost": None,
                "output_cost": None,
                "total_cost": None,
            },
            "accounting": {
                "tokens_available": False,
                "cost_available": False,
                "reason": "missing_usage",
            },
        },
        timestamp="2026-07-23 12:00:01",
    )

    assert "Total Input Tokens: N/A" in report
    assert "Total Output Tokens: N/A" in report
    assert "Total Tokens: N/A" in report
    assert "Input Cost: N/A" in report
    assert "Output Cost: N/A" in report
    assert "Total Cost: N/A" in report
    assert "    Input Tokens: N/A" in report
    assert "    Output Tokens: N/A" in report
    assert "    Total Tokens: N/A" in report
    assert "    Input Cost: N/A" in report
    assert "    Output Cost: N/A" in report
    assert "    Total Cost: N/A" in report


def test_format_usage_report_avoids_currency_guess_after_conflict():
    report = format_usage_report(
        project_name="Test Project",
        model="provider-model-id",
        model_name="test-model",
        pricing_currency=None,
        token_usage={
            "total_input_tokens": 21,
            "total_output_tokens": 20,
            "requests": [
                {
                    "name": "usd",
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "total_tokens": 20,
                    "process_time": 0.5,
                    "input_cost": 0.02,
                    "output_cost": 0.04,
                    "total_cost": 0.06,
                },
                {
                    "name": "eur",
                    "input_tokens": 11,
                    "output_tokens": 10,
                    "total_tokens": 21,
                    "process_time": 0.5,
                    "input_cost": 0.022,
                    "output_cost": 0.04,
                    "total_cost": 0.062,
                },
            ],
            "process_times": {"total_time": 1.0, "request_times": []},
            "cost": {
                "input_cost": None,
                "output_cost": None,
                "total_cost": None,
            },
            "accounting": {
                "tokens_available": True,
                "cost_available": False,
                "reason": "aggregate_currency_conflict",
            },
        },
    )

    assert "  Total Cost: N/A" in report
    assert report.count("    Input Cost: N/A") == 2
    assert report.count("    Output Cost: N/A") == 2
    assert report.count("    Total Cost: N/A") == 2
