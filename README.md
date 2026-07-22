# llm-exec-core

`llm-exec-core` is a small provider-agnostic Python package for async LLM
execution, model-catalog schema and loading infrastructure, streaming assembly,
and typed usage/result objects.

It provides:

- `LLMClient` for async request execution and streaming
- schema and loading helpers for caller-owned provider/model catalogs
- `llm_exec_core/llm_config.yml` as a maintained complete reference template,
  never runtime policy
- typed result and usage objects for legacy and new call sites

## Install

```bash
uv add llm-exec-core
```

## Basic usage

```python
from importlib.resources import files
from pathlib import Path

from llm_exec_core.client import LLMClient

reference = files("llm_exec_core").joinpath("llm_config.yml")
catalog = Path("config/llm_config.yml")
catalog.write_bytes(reference.read_bytes())

client = LLMClient("your-model-name", config_source=catalog)
models = LLMClient.get_supported_models(catalog)
```

As of 0.4.0, `config_source` is required across the client and public
configuration helpers and accepts either a complete catalog `Path` or raw
dictionary. Omitting it or passing `None` raises `ValueError`. The packaged
catalog is only a maintained, complete reference template that callers may
copy explicitly; it is never selected as runtime policy.

Adopt or refresh the reference by copying it manually or automating a full
replacement, then optionally edit the complete application-owned copy. Every
supplied catalog must be complete: core provides no partial overlay or
inheritance semantics with the packaged template.

## Provider connection overrides

Caller-owned catalogs can declare ordered API-key aliases and one endpoint
override variable without adding provider-specific logic to `LLMClient`:

```yaml
synthetic-provider:
  api_key_env_var: SYNTHETIC_PRIMARY_API_KEY
  api_key_env_aliases:
    - SYNTHETIC_FALLBACK_API_KEY
  api_base_url: https://static.example.invalid/v1/chat/completions
  api_base_url_env_var: SYNTHETIC_API_BASE_URL
  pricing_currency: "$"
  models:
    synthetic-model:
      id: synthetic-model-id
      pricing: {input: 0.0, output: 0.0}
```

The primary key is checked first, followed by aliases in declaration order.
Unset, empty, and Unicode-whitespace-only values are skipped. The first value
with non-whitespace content is used unchanged, including ordinary surrounding
spaces, after rejecting C0/DEL control characters. Resolution stops at that
winner; errors name declarations but never credential values.

When the endpoint variable is absent, unset, or Unicode-whitespace-only, the
static `api_base_url` is used byte-for-byte. A populated override may be either
an HTTPS SDK base ending in `/v1` or a full HTTPS URL ending in
`/chat/completions`; surrounding ASCII whitespace and trailing slashes are
removed, and `/chat/completions` is appended to an SDK base. Overrides with
userinfo, a query or fragment delimiter, invalid ports, unsupported paths,
backslashes, or remaining whitespace/control characters fail before HTTP
client or request construction.

There is deliberately no hostname allowlist. Explicit catalogs and environment
variables are controlled by the caller or deployer, so regional, workspace,
private-gateway, IP-literal, and self-hosted compatible endpoints remain
possible. The deterministic HTTPS/path/parser checks reduce accidental secret
leakage and ambiguous parsing; they are not an SSRF boundary against a
malicious deployer.

The declarations are optional. `ProviderSettings.model_fields`, JSON schema,
and default `model_dump()` output add `api_key_env_aliases=[]` and
`api_base_url_env_var=None`; existing field meanings and ordinary static-route
behavior remain unchanged apart from fail-closed whitespace/control-bearing
credentials.

## Request options

Use `request_options` for raw per-call OpenAI-compatible Chat Completions
payload fields. For target models with capability metadata, core validates known
high-risk fields such as `response_format`, `tools`, tool streaming, and
reasoning controls before sending the request.

Catalog lookup remains raw: `get_model_details()` returns the declared
`ProviderSettings` and `ModelDetails` without synthesizing or mutating an
effective settings object. A model may optionally override `temperature`,
`max_tokens`, `context_window`, `request_overrides`, and `output_token_field`.
`LLMClient` uses each non-`None` model scalar before its provider default and
deep-clones caller-owned catalog data across loads and clients.

Request defaults are normalized and merged provider first, then model, with
per-call `request_options` highest. One-level `extra_body` promotion, protected
core fields, and deep provider → model → per-call `stream_options` merging are
preserved. Legacy catalogs generate `max_tokens`; a provider or model can select
`max_completion_tokens` instead. Any explicit token-limit field in request
defaults or per-call options suppresses the generated default, while a final
payload containing both token-limit fields fails before HTTP. Configured retry
lowering updates whichever single integer token-limit field is present.

The `thinking_level` convenience argument maps to `reasoning_effort` when model
capabilities are absent or explicitly list that control. Capability metadata
that is present but does not list `reasoning_effort` rejects `thinking_level`
before request construction; core does not translate it to provider-specific
thinking fields.

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

On Gemini
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
