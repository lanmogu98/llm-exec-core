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

Use `request_options` for per-call OpenAI-compatible Chat Completions fields
such as structured output schemas, sampling controls, tools, and routing
objects:

```python
result = await client.generate(
    "Extract the title as JSON.",
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
        "top_p": 0.2,
    },
)
```

`generate_response()` accepts the same `request_options` argument for legacy
tuple-returning call sites. Core-owned fields (`model`, `messages`, and
`stream`) are rejected in request options; provider-specific fields pass
through unchanged.

This repo is also used for coordinated local development with sibling
checkouts during the current extraction/migration work.

## License

Proprietary. See [LICENSE](LICENSE).
