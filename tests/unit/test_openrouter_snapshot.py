from copy import deepcopy
import json
from pathlib import Path
import runpy

import yaml

ROOT = Path(__file__).parents[2]
CHECK_SCRIPT = ROOT / "scripts" / "check_openrouter_capabilities.py"
CONFIG_PATH = ROOT / "src" / "llm_exec_core" / "llm_config.yml"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "openrouter_models.json"


def _load_compare_function():
    assert (
        CHECK_SCRIPT.exists()
    ), "OpenRouter capability check script is missing"
    namespace = runpy.run_path(str(CHECK_SCRIPT))
    return namespace["compare_openrouter_capabilities"]


def _load_inputs():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        models_payload = json.load(handle)
    return config, models_payload


def test_openrouter_snapshot_fixture_matches_tracked_target_fields():
    compare = _load_compare_function()
    config, models_payload = _load_inputs()

    assert compare(config, models_payload) == []


def test_openrouter_snapshot_comparison_reports_parameter_drift():
    compare = _load_compare_function()
    config, models_payload = _load_inputs()
    drifted_payload = deepcopy(models_payload)
    gpt = next(
        model
        for model in drifted_payload["data"]
        if model["id"] == "openai/gpt-5.5"
    )
    gpt["supported_parameters"].remove("reasoning_effort")

    assert compare(config, drifted_payload) == [
        "openai/gpt-5.5: tracked-only=['reasoning_effort']; live-only=[]"
    ]


def test_openrouter_snapshot_comparison_reports_new_live_parameter():
    compare = _load_compare_function()
    config, models_payload = _load_inputs()
    drifted_payload = deepcopy(models_payload)
    gpt = next(
        model
        for model in drifted_payload["data"]
        if model["id"] == "openai/gpt-5.5"
    )
    gpt["supported_parameters"].append("temperature")

    assert compare(config, drifted_payload) == [
        "openai/gpt-5.5: tracked-only=[]; live-only=['temperature']"
    ]
