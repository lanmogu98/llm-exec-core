# OpenAI-Compatible Chat Completions Request Options

This document is the design source of truth for `LLMClient.generate(...,
request_options=...)` and `generate_response(..., request_options=...)`.

## Scope

This feature transports OpenAI-compatible `/chat/completions` request payload
fields. `request_options` is the raw transport layer: callers can pass
provider/model-specific fields through with correct merge, cache, stream, and
protected-field semantics.

This feature includes a capability-aware request planner for known high-risk
extensions. It does not execute tools, parse tool-call responses, perform live
provider probes, or implement the OpenAI Responses API.

## Target Model Scope

Capability review for Issue #2 targets the intended current model set, not
every historical model left in `llm_config.yml`.

| Provider / route | Target model(s) |
| --- | --- |
| Zhipu official | `glm-5.2` only |
| Zhipu via OpenRouter | `z-ai/glm-5.2` |
| DeepSeek official | `deepseek-v4-flash`, `deepseek-v4-pro` |
| DeepSeek via Volcengine | `deepseek-v3.2`, `deepseek-r1` |
| Qwen / Bailian | `qwen3.6-flash`; `qwen-max` as a moving latest-max alias |
| Gemini OpenAI compatibility | `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview` |
| OpenRouter OpenAI | `openai/gpt-5.5` |
| OpenRouter Anthropic | `anthropic/claude-sonnet-5`, `anthropic/claude-opus-4.8` |
| Volcengine Doubao | `doubao-seed-2-1-pro-260628` |

Qwen endpoint migration note: the current config may still use a legacy
DashScope-compatible endpoint. Bailian now documents workspace-scoped
compatible endpoints such as
`https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`. That
migration is tracked separately and is not required for this capability-aware
work.

## Capability Matrix

This matrix describes known route/model capability boundaries. Core uses the
matching `llm_config.yml` capability metadata to validate known high-risk
request fields and to plan `structured_output`.

For this issue, the reviewed target registry is the set of model entries with
`capabilities` metadata. Legacy catalog entries without that metadata remain raw
passthrough routes and are outside the capability-aware guarantee.

| Route / model | Strict response schema | JSON mode | Tools | Reasoning / thinking | Important constraints |
| --- | --- | --- | --- | --- | --- |
| Zhipu official `glm-5.2` | No official strict API `json_schema` response format found. | Yes: `response_format={"type":"json_object"}`. | Yes, with provider/model-specific tool-choice semantics. | Yes: `thinking` and `reasoning_effort`. | JSON Schema in Zhipu docs is not OpenAI-style strict API schema; the semantic package path validates locally. |
| OpenRouter `z-ai/glm-5.2` | Yes: OpenRouter models API reports `structured_outputs` and `response_format`. | Yes via `response_format`. | Yes: `tools`, `tool_choice`, `parallel_tool_calls`. | Yes: `reasoning`, `include_reasoning`, `reasoning_effort`. | Check `supported_parameters`; use `provider.require_parameters=true` or pin routing when correctness depends on schema/tools. `temperature` and `max_tokens` are listed. |
| Qwen / Bailian `qwen3.6-flash` | No positive evidence in the Bailian OpenAI-compatible table. | No `response_format` listed in the compatible parameter table. | No confirmed tool support in the compatible table. | Provider/model-specific; do not assume OpenAI `reasoning_effort`. | Keep `tools` disabled until the Bailian migration issue confirms support. |
| Qwen / Bailian `qwen-max` | No positive evidence for strict schema in the compatible table. | No `response_format` listed in the compatible table. | Yes for the `qwen-max` family in the documented compatible table. | Provider/model-specific. | Treat `qwen-max` as a moving latest-max alias, not a frozen historical model id; `tools + stream=True` is unsupported and fails fast. |
| Gemini OpenAI compatibility `gemini-3-flash-preview` | Use compatibility-layer `response_format` / SDK parse support where documented. | Compatibility-layer behavior only. | Yes through OpenAI-compatible tool-calling examples. | Yes: `reasoning_effort` maps to Gemini thinking controls for the Gemini 3 family. | Paid/free aliases share capabilities. Do not mix `reasoning_effort` with `thinking_level` or `thinking_budget` under `google.thinking_config`. |
| Gemini OpenAI compatibility `gemini-3.1-flash-lite-preview` | Same compatibility-layer boundary as Gemini 3 Flash. | Same as Gemini 3 Flash. | Same as Gemini 3 Flash. | Yes: documented compatibility mapping includes Gemini 3.1 Flash-Lite. | Paid/free aliases share capabilities; use exactly what the compatibility framework exposes. |
| OpenRouter `openai/gpt-5.5` | Yes: OpenRouter models API reports `structured_outputs` and `response_format`. | Yes. | Yes: `tools`, `tool_choice`. | Yes: `reasoning`, `include_reasoning`, `reasoning_effort`. | Current `supported_parameters` include token-limit controls but not generic `temperature/top_p`; docs/examples must not claim sampling controls are universal. |
| OpenRouter `anthropic/claude-sonnet-5` | Yes: OpenRouter models API reports `structured_outputs` and `response_format`. | Yes. | Yes: `tools`, `tool_choice`. | Yes: `reasoning`, `include_reasoning`, `reasoning_effort`; also `verbosity`. | Token-limit controls are listed, but generic sampling knobs are not assumed unless listed. |
| OpenRouter `anthropic/claude-opus-4.8` | Yes: OpenRouter models API reports `structured_outputs` and `response_format`. | Yes. | Yes: `tools`, `tool_choice`. | Yes: `reasoning`, `include_reasoning`, `reasoning_effort`; also `verbosity`. | Current `supported_parameters` include `temperature`, `max_tokens`, and `max_completion_tokens`; use the model-specific snapshot. |
| Volcengine `doubao-seed-2-1-pro-260628` | Yes: Volcengine Chat API documents `response_format.json_schema`; model list marks this model with structured-output support. | Yes: `json_object` mode documented. | Yes: model list marks tool support; Chat API documents `tools` / `tool_choice`. | Yes: model list marks deep thinking; Chat API documents thinking/reasoning controls. | Support is model-specific; API-level support does not imply all Ark models support the same extension set. |
| DeepSeek official `deepseek-v4-flash`, `deepseek-v4-pro` | No strict API `json_schema` response format. | Yes: `response_format.type=json_object`. | Yes: tools/tool_choice and JSON Schema tool parameters are documented. | Yes: `thinking` and `reasoning_effort`. | JSON mode requires explicit JSON instruction; the semantic package path validates locally. |
| DeepSeek via Volcengine `deepseek-v3.2`, `deepseek-r1` | Treat as acceptable for the current issue, with model-specific support. | Model-specific JSON mode. | Model-specific tool support. | Model-specific thinking controls. | Do not infer structured output support from Ark Chat API alone; check the model row. |

## Public API

Both request methods accept keyword-only
`request_options: Mapping[str, Any] | None = None` and
`structured_output: Mapping[str, Any] | None = None`.

`request_options` is for HTTP request payload fields only. Metadata such as
`request_name`, callbacks, `trace_context`, `run_id`, `request_id`, and
`structured_output_hook` never enters the HTTP payload or cache key.

`structured_output` is an explicit semantic-planner entry point, not raw
transport. `{"mode": "off"}` is a no-op. `mode="require"` sends strict schema
when supported and otherwise fails before the request. `mode="prefer"` sends
strict schema when supported, falls back to JSON mode when supported, and then
falls back to prompt-only JSON instructions. The mode controls transport
selection only: every active `structured_output` response is parsed as JSON and
validated against the supplied schema before success.

Without `structured_output_hook`, `LLMResult.structured` contains the parsed
value. The existing `Callable[[str], Any]` hook signature is preserved; in the
semantic structured-output path it receives the raw response text only after
schema validation and acts as a post-validation transform. A hook used without
`structured_output` retains its legacy transform behavior but does not imply
validation.

## Payload Construction

The effective request payload is built in this order:

1. Core defaults: `model`, `messages`, `temperature`, `stream`, and usually
   `max_tokens`.
2. Normalized provider `request_overrides`.
3. Normalized per-call `request_options`.

Each provider and per-call layer normalizes SDK-style `extra_body` one level
before merging. Values under `extra_body` are promoted into that layer's
top-level request body, and direct top-level keys in the same layer win. The
normalization does not recurse, so `extra_body.extra_body` becomes the final
top-level raw `extra_body` field.

`model`, `messages`, and `stream` are core-owned protected fields. They are
rejected after normalization in provider `request_overrides` and per-call
`request_options`. `stream_options` is the supported way to customize streaming
request behavior.

If normalized provider or per-call options contain `max_completion_tokens`, the
core-generated default `max_tokens` is omitted. An explicit `max_tokens` in
provider or per-call options remains in the payload and follows normal
precedence.

On OpenRouter routes with capability metadata, core-generated `temperature`,
`max_tokens`, and `max_completion_tokens` are planned against
`supported_parameters`. Unsupported generated defaults are omitted; a generated
`max_tokens` is converted to `max_completion_tokens` when only that token-limit
parameter is listed. Explicit provider/per-call values for unsupported fields
fail fast instead of being sent silently.

`stream_options` is deep-merged as provider first, then per-call. When
`stream=True`, `include_usage: true` is added when final `stream_options` is
absent. Mapping-valued `stream_options` keeps caller values and receives
`include_usage: true` only when that key is absent. Other explicit values are
considered present and are not modified. An explicit per-call
`stream_options: None` clears any provider `stream_options` and suppresses the
core `include_usage` default. Omit `stream_options` to receive the default
streaming usage request.

For non-streaming calls, `stream_options` is included only when explicitly
provided by provider or per-call options.

## Capability-Aware Planning

Known high-risk request fields are validated when the selected model has
capability metadata. Raw unknown/provider-specific fields still pass through.

- `response_format.type=json_schema` requires strict schema support.
- `response_format.type=json_object` requires JSON-object mode support.
- `tools` requires tool support; `tools + stream=True` requires tool streaming
  support.
- `tool_choice`, `parallel_tool_calls`, and known reasoning/thinking/verbosity
  controls are validated when present.
- On OpenRouter routes, correctness-dependent `response_format` and `tools`
  requests set `provider.require_parameters=true` in the planned payload.
  Missing OpenRouter `supported_parameters` metadata fails closed for those
  correctness-dependent fields.
- On OpenRouter routes, core defaults for `temperature`, `max_tokens`, and
  `max_completion_tokens` are also checked against `supported_parameters` before
  sending the request.
- On Gemini routes, the final normalized payload rejects `reasoning_effort`
  combined with `thinking_level` or `thinking_budget` under either
  `google.thinking_config` or `extra_body.google.thinking_config`.
  `include_thoughts` by itself is not conflicting. This applies to paid and free
  aliases.

`structured_output` planning chooses one of these strategies:

1. `strict_schema`: send strict `response_format.type=json_schema`, then parse
   and validate the returned JSON locally.
2. `json_object`: send `response_format.type=json_object`, append explicit JSON
   Schema instructions to the user message, then parse and validate locally.
3. `prompt_json`: append explicit JSON Schema instructions without
   `response_format`, then parse and validate locally.
4. `raw`: no semantic planner was requested.

`ExecutionMetadata.planning` exposes `strategy`, `fallback_reason`,
`validation_status`, `capability_version`, and `hook_applied`. A successful
semantic response reports `validation_status="client_validated"`; hook presence
is reported independently. Parse and schema failures raise
`StructuredOutputValidationError` and retain the original error as `__cause__`.

## Cache Behavior

Only non-streaming calls use the response cache. The cache key is a
deterministic JSON serialization of the final effective HTTP payload after
normalization, merges, and capability-aware planning, plus internal planning
metadata (`strategy`, `fallback_reason`, and `capability_version`). Streaming
calls bypass cache lookup and cache writes so callbacks and streaming assembly
always run.

For semantic structured output, JSON parsing, schema validation, and the
optional post-validation hook run before cache write. Parse, schema, or hook
failure therefore leaves no new cache entry. Cache hits hold raw response text
and repeat parsing, schema validation, and hook processing before returning.

Because the key is derived from the HTTP payload, request-affecting fields such
as `response_format`, sampling controls, reasoning controls, routing objects,
and `stream` are included. Non-request metadata is excluded.

## Semantic Fallback Boundary

Graceful structured-output fallback is explicit and separate from raw
`request_options`:

```python
structured_output = {
    "schema": schema,
    "mode": "require",  # or "prefer" / "off"
}
```

Tool-call extraction fallback is not implemented in this PR because the package
does not yet execute tools or parse tool-call responses.

## OpenRouter Snapshot Check

The tracked OpenRouter capability rows are deterministic runtime snapshots dated
2026-07-15. Normal requests and normal unit tests never fetch capability
metadata. Reviewers can explicitly compare the package-relevant target-field
intersection with the live models API:

```bash
uv run python scripts/check_openrouter_capabilities.py
```

The comparison logic is unit-tested with a local models API fixture so drift
detection remains deterministic without network access.

## Sources

- Zhipu model overview: https://docs.bigmodel.cn/cn/guide/start/model-overview
- Zhipu GLM new-model migration: https://docs.bigmodel.cn/cn/guide/start/migrate-to-glm-new
- Zhipu structured output: https://docs.bigmodel.cn/cn/guide/capabilities/struct-output
- OpenRouter models API: https://openrouter.ai/api/v1/models
- OpenRouter models endpoint docs: https://openrouter.ai/docs/api/api-reference/models/get-models
- OpenRouter structured outputs: https://openrouter.ai/docs/guides/features/structured-outputs
- Bailian / DashScope OpenAI compatibility: https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
- Gemini OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- DeepSeek Chat Completions API: https://api-docs.deepseek.com/api/create-chat-completion
- Volcengine Ark Chat API: https://www.volcengine.com/docs/82379/1494384
- Volcengine model list: https://www.volcengine.com/docs/82379/1330310
