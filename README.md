# llm-exec-core

`llm-exec-core` is a small Python package for async LLM execution, shared
provider/model catalog loading, streaming assembly, and typed usage/result
objects.

It provides:

- `LLMClient` for async request execution and streaming
- `llm_exec_core/llm_config.yml` as the shared model catalog
- typed result and usage objects for legacy and new call sites

## Install

```bash
uv add llm-exec-core
```

## Basic usage

```python
from llm_exec_core.client import LLMClient

client = LLMClient("your-model-name")
```

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

That example is route/model-specific; it should not be read as universal support
across every configured model. Some routes expose JSON mode without strict
schema support, some require caller-side validation, and Qwen/Bailian compatible
docs currently say `tools` cannot be used with `stream=True`.
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
```

`mode="require"` sends strict schema when supported and otherwise fails before
the request. `mode="prefer"` falls back to JSON mode or prompt-only JSON
instructions when strict schema is unavailable. The selected strategy and
fallback status are available on `result.metadata.planning`.

This repo is also used for coordinated local development with sibling
checkouts during the current extraction/migration work.

## License

Proprietary. See [LICENSE](LICENSE).
