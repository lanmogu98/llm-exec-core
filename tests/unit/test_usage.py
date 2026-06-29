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
