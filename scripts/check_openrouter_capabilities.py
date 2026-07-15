#!/usr/bin/env python3
"""Compare tracked OpenRouter target fields with the models API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping
from urllib.request import Request, urlopen

import yaml

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CONFIG_PATH = (
    Path(__file__).parents[1] / "src" / "llm_exec_core" / "llm_config.yml"
)
TARGET_MODELS = {
    "anthropic/claude-opus-4.8": (
        "anthropic-openrouter",
        "claude-opus-4.8-or",
    ),
    "anthropic/claude-sonnet-5": (
        "anthropic-openrouter",
        "claude-sonnet-5-or",
    ),
    "openai/gpt-5.5": ("openai-openrouter", "gpt-5.5-or"),
    "z-ai/glm-5.2": ("zhipu-openrouter", "glm-5.2-or"),
}
TRACKED_FIELDS = frozenset(
    {
        "include_reasoning",
        "max_completion_tokens",
        "max_tokens",
        "parallel_tool_calls",
        "reasoning",
        "reasoning_effort",
        "response_format",
        "structured_outputs",
        "temperature",
        "tool_choice",
        "tools",
        "verbosity",
    }
)


def compare_openrouter_capabilities(
    config: Mapping[str, Any],
    models_payload: Mapping[str, Any],
) -> list[str]:
    live_models = {
        model.get("id"): model
        for model in models_payload.get("data", [])
        if isinstance(model, Mapping)
    }
    differences = []

    for model_id, (provider_name, model_name) in TARGET_MODELS.items():
        live_model = live_models.get(model_id)
        if live_model is None:
            differences.append(f"{model_id}: missing from models API")
            continue

        capabilities = config[provider_name]["models"][model_name][
            "capabilities"
        ]
        tracked = set(capabilities["openrouter_supported_parameters"])
        live = set(live_model.get("supported_parameters", []))
        tracked_only = sorted((tracked & TRACKED_FIELDS) - live)
        live_only = sorted((live & TRACKED_FIELDS) - tracked)
        if tracked_only or live_only:
            differences.append(
                f"{model_id}: tracked-only={tracked_only}; "
                f"live-only={live_only}"
            )

    return differences


def _load_config(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_models_payload(path: Path | None) -> Mapping[str, Any]:
    if path is not None:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    request = Request(
        OPENROUTER_MODELS_URL,
        headers={"User-Agent": "llm-exec-core-capability-check"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--models-json",
        type=Path,
        help="Use a local models API fixture instead of the live endpoint.",
    )
    args = parser.parse_args()

    try:
        config = _load_config(args.config)
        models_payload = _load_models_payload(args.models_json)
        differences = compare_openrouter_capabilities(config, models_payload)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"OpenRouter capability check failed: {error}", file=sys.stderr)
        return 2

    if differences:
        for difference in differences:
            print(difference, file=sys.stderr)
        return 1

    print(
        "OpenRouter capability snapshot matches tracked fields for "
        f"{len(TARGET_MODELS)} target models."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
