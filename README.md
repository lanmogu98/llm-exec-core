# llm-exec-core

`llm-exec-core` is a small provider-agnostic Python package for async LLM
execution, model-catalog schema and loading infrastructure, streaming assembly,
and typed usage/result objects.

It provides:

- `LLMClient` for async request execution and streaming
- schema and loading helpers for caller-owned provider/model catalogs
- `llm_exec_core/llm_config.yml` as a temporary legacy fallback
- typed result and usage objects for legacy and new call sites

## Install

```bash
uv add llm-exec-core
```

## Basic usage

```python
from pathlib import Path

from llm_exec_core.client import LLMClient

catalog = Path("path/to/llm_config.yml")
client = LLMClient("your-model-name", config_source=catalog)
models = LLMClient.get_supported_models(catalog)
```

`config_source` accepts either a catalog path or a raw configuration dictionary
across the client and public configuration helpers. Caller-owned catalogs are
the recommended runtime policy boundary.

For compatibility, omitting `config_source` still loads the bundled catalog and
emits `DeprecationWarning`. This fallback has no approved removal version;
removal is governed by the separate
[final-removal child #10](https://github.com/lanmogu98/llm-exec-core/issues/10)
under the [model-catalog migration contract
#5](https://github.com/lanmogu98/llm-exec-core/issues/5).

## Request options

Use `request_options` for raw per-call OpenAI-compatible Chat Completions
payload fields. For target models with capability metadata, core validates known
high-risk fields such as `response_format`, `tools`, tool streaming, and
reasoning controls before sending the request.

Check the target route's docs or supported-parameter metadata before relying on
strict schemas, tools, reasoning controls, sampling controls, or routing
objects. For example, if an OpenRouter model reports `structured_outputs` /
`response_format`, pin or require compatible routing when correctness depends on
that feature:

```python
result = await client.generate(
    "Extract the title.",
    request_options={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "title_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
        },
        "provider": {"require_parameters": True},
    },
)
```

That raw request example is route/model-specific; it should not be read as
universal support across every configured model. Some routes expose JSON mode
without strict schema support, and Qwen/Bailian compatible docs currently say
`tools` cannot be used with `stream=True`.
On OpenRouter routes with capability metadata, unsupported core defaults such as
`temperature` are omitted before sending; explicit unsupported per-call values
fail fast.

`generate_response()` accepts the same `request_options` argument for legacy
tuple-returning call sites. Core-owned fields (`model`, `messages`, and
`stream`) are rejected in request options; provider-specific fields pass
through unchanged. See
[`docs/design_docs/openai_compat_request_options.md`](docs/design_docs/openai_compat_request_options.md)
for the current target-model matrix and fallback boundary.

For semantic structured-output fallback, use the explicit `structured_output`
argument instead of hiding fallback policy inside `request_options`.

```python
result = await client.generate(
    "Extract the title.",
    structured_output={
        "schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
        "mode": "prefer",
    },
)

title = result.structured["title"]
```

`mode="require"` sends strict schema when supported and otherwise fails before
the request. `mode="prefer"` falls back to JSON mode or prompt-only JSON
instructions when strict schema is unavailable. Both modes parse the response as
JSON and validate it with `jsonschema` before returning success. Without a hook,
`result.structured` is the parsed value. In this semantic path,
`structured_output_hook` keeps its text-input signature but runs only after
validation as an optional transform.
Validation failures raise `StructuredOutputValidationError`, with the original
JSON or schema error preserved as the cause.

Structured parsing, validation, and hook processing complete before a network
response is cached, and cached structured responses are validated again on each
hit. Planning metadata reports `validation_status="client_validated"` and a
separate `hook_applied` flag.

Gemini paid/free aliases for the reviewed models share capabilities. On Gemini
routes, `reasoning_effort` cannot be combined with `thinking_level` or
`thinking_budget` under either normalized `google.thinking_config` payload
shape; `include_thoughts` alone is allowed.

OpenRouter target capabilities remain static at runtime. Refresh/review them
explicitly with:

```bash
uv run python scripts/check_openrouter_capabilities.py
```

This repo is also used for coordinated local development with sibling
checkouts during the current extraction/migration work.

## Development

The package supports Python 3.10 and newer. Required CI runs on Python 3.10 and
3.13 and is deterministic, locked, and mock-only.

```bash
uv sync --locked --group dev
uv run --frozen pytest -q
uv run --frozen black --check src tests
uv run --frozen flake8 src tests
uv run --frozen mypy src
uv build
```

See [AGENTS.md](AGENTS.md) for the project workflow and quality contract.

## Issues and contributions

This repository accepts public Issues only. Code preparation and contribution
require an accepted Issue plus an explicit maintainer dispatch naming the
semantic scope, actor/session, and branch/PR delivery. Unsolicited external code
is not inspected or executed. See [CONTRIBUTING.md](CONTRIBUTING.md).

Report vulnerabilities through GitHub's private vulnerability reporting path,
not a public Issue. See [SECURITY.md](SECURITY.md).

## License

Proprietary. See [LICENSE](LICENSE).
