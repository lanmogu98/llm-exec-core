# OpenAI-Compatible Chat Completions Request Options

This document is the design source of truth for `LLMClient.generate(...,
request_options=...)` and `generate_response(..., request_options=...)`.

## Scope

This feature transports OpenAI-compatible `/chat/completions` request payload
fields. `request_options` is the raw transport layer: callers can pass
provider/model-specific fields through with correct merge, cache, stream, and
protected-field semantics.

This feature is not a semantic structured-output planner. It does not choose
strict schema vs. JSON mode vs. tool-call extraction, rewrite unsupported fields,
execute tools, parse tool-call responses, add runtime capability filtering, or
implement the OpenAI Responses API. The explicit `structured_output` argument is
a guardrail for that future planner: `mode="off"` is a no-op, while
`mode="require"` and `mode="prefer"` fail fast until the semantic planner exists.

## Target Model Scope

Capability review for Issue #2 targets the intended current model set, not
every historical model left in `llm_config.yml`.

| Provider / route | Target model(s) |
| --- | --- |
| Zhipu official | `glm-5.2` only |
| Zhipu via OpenRouter | `z-ai/glm-5.2` |
| DeepSeek official | Current official DeepSeek models, preserving JSON-mode caveats |
| DeepSeek via Volcengine | Current Volcengine DeepSeek route, preserving model-specific caveats |
| Qwen / Bailian | `qwen3.6-flash`; `qwen-max` as a moving latest-max alias |
| Gemini OpenAI compatibility | `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview` |
| OpenRouter OpenAI | `openai/gpt-5.5` |
| OpenRouter Anthropic | `anthropic/claude-sonnet-5`, `anthropic/claude-opus-4.8` |
| Volcengine Doubao | `doubao-seed-2-1-pro-260628` |

Qwen endpoint migration note: the current config may still use a legacy
DashScope-compatible endpoint. Bailian now documents workspace-scoped
compatible endpoints such as
`https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`. That
migration is tracked separately and is not required for this raw transport PR.

## Capability Matrix

This matrix describes known route/model capability boundaries for callers using
raw `request_options`. Core does not enforce these capabilities.

| Route / model | Strict response schema | JSON mode | Tools | Reasoning / thinking | Important constraints |
| --- | --- | --- | --- | --- | --- |
| Zhipu official `glm-5.2` | No official strict API `json_schema` response format found. | Yes: `response_format={"type":"json_object"}`. | Yes, with provider/model-specific tool-choice semantics. | Yes: `thinking` and `reasoning_effort`. | JSON Schema in Zhipu structured-output docs is caller-side prompt/validation guidance, not OpenAI-style strict API schema. |
| OpenRouter `z-ai/glm-5.2` | Yes: OpenRouter models API reports `structured_outputs` and `response_format`. | Yes via `response_format`. | Yes: `tools`, `tool_choice`, `parallel_tool_calls`. | Yes: `reasoning`, `include_reasoning`, `reasoning_effort`. | Check `supported_parameters`; use `provider.require_parameters=true` or pin routing when correctness depends on schema/tools. |
| Qwen / Bailian `qwen3.6-flash` | No positive evidence in the Bailian OpenAI-compatible table. | No `response_format` listed in the compatible parameter table. | Tool support should be confirmed in the Bailian migration issue. | Provider/model-specific; do not assume OpenAI `reasoning_effort`. | Documented compatible table says `tools` currently cannot be used with `stream=True`. |
| Qwen / Bailian `qwen-max` | No positive evidence for strict schema in the compatible table. | No `response_format` listed in the compatible table. | Yes for the `qwen-max` family in the documented compatible table. | Provider/model-specific. | Treat `qwen-max` as a moving latest-max alias, not a frozen historical model id. |
| Gemini OpenAI compatibility `gemini-3-flash-preview` | Use compatibility-layer `response_format` / SDK parse support where documented. | Compatibility-layer behavior only. | Yes through OpenAI-compatible tool-calling examples. | Yes: `reasoning_effort` maps to Gemini thinking controls for the Gemini 3 family. | Do not mix OpenAI `reasoning_effort` with Gemini-specific `extra_body.google.thinking_config` in the same request. |
| Gemini OpenAI compatibility `gemini-3.1-flash-lite-preview` | Same compatibility-layer boundary as Gemini 3 Flash. | Same as Gemini 3 Flash. | Same as Gemini 3 Flash. | Yes: documented compatibility mapping includes Gemini 3.1 Flash-Lite. | Use exactly what the compatibility framework exposes. |
| OpenRouter `openai/gpt-5.5` | Yes: OpenRouter models API reports `structured_outputs` and `response_format`. | Yes. | Yes: `tools`, `tool_choice`. | Yes: `reasoning`, `include_reasoning`. | Current `supported_parameters` do not include generic `temperature/top_p`; docs/examples must not claim sampling controls are universal. |
| OpenRouter `anthropic/claude-sonnet-5` | Yes: OpenRouter models API reports `structured_outputs` and `response_format`. | Yes. | Yes: `tools`, `tool_choice`. | Yes: `reasoning`, `include_reasoning`; also `verbosity`. | Use `supported_parameters`; do not assume OpenAI sampling knobs unless listed. |
| OpenRouter `anthropic/claude-opus-4.8` | Yes: OpenRouter models API reports `structured_outputs` and `response_format`. | Yes. | Yes: `tools`, `tool_choice`. | Yes: `reasoning`, `include_reasoning`; also `verbosity`. | `max_completion_tokens` support differs between Anthropic OpenRouter variants; use model-specific `supported_parameters`. |
| Volcengine `doubao-seed-2-1-pro-260628` | Yes: Volcengine Chat API documents `response_format.json_schema`; model list marks this model with structured-output support. | Yes: `json_object` mode documented. | Yes: model list marks tool support; Chat API documents `tools` / `tool_choice`. | Yes: model list marks deep thinking; Chat API documents thinking/reasoning controls. | Support is model-specific; API-level support does not imply all Ark models support the same extension set. |
| DeepSeek official | No strict API `json_schema` response format. | Yes: `response_format.type=json_object`. | Yes: tools/tool_choice and JSON Schema tool parameters are documented. | Yes: `thinking` and `reasoning_effort`. | JSON mode requires explicit JSON instruction; schema adherence must be caller-validated. |
| DeepSeek via Volcengine | Treat as acceptable for the current issue, with model-specific support. | Model-specific. | Model-specific. | Model-specific. | Do not infer structured output support from Ark Chat API alone; check the model row. |

## Public API

Both request methods accept keyword-only
`request_options: Mapping[str, Any] | None = None` and
`structured_output: Mapping[str, Any] | None = None`.

`request_options` is for HTTP request payload fields only. Metadata such as
`request_name`, callbacks, `trace_context`, `run_id`, `request_id`, and
`structured_output_hook` never enters the HTTP payload or cache key.

`structured_output` is an explicit semantic-planner entry point, not raw
transport. In this PR only `{"mode": "off"}` is accepted as a no-op. The
`require` and `prefer` modes raise `NotImplementedError` before any HTTP request
so callers cannot mistake raw passthrough for graceful fallback.

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

## Cache Behavior

Only non-streaming calls use the response cache. The cache key is a
deterministic JSON serialization of the final effective HTTP payload after
normalization and merges. Streaming calls bypass cache lookup and cache writes
so callbacks and streaming assembly always run.

Because the key is derived from the HTTP payload, request-affecting fields such
as `response_format`, sampling controls, reasoning controls, routing objects,
and `stream` are included. Non-request metadata is excluded.

## Semantic Fallback Boundary

Graceful structured-output fallback should be a separate semantic layer, not
hidden inside raw `request_options`. PR #4 reserves that boundary with a
fail-fast `structured_output` argument, but does not implement the planner.

A future API can make that policy explicit, for example:

```python
structured_output = {
    "schema": schema,
    "mode": "require",  # or "prefer" / "off"
}
```

The planner should make visible choices:

1. If the target route supports strict schema, send strict `response_format`.
2. Else if it supports JSON mode and mode is `prefer`, send JSON mode plus
   explicit JSON instructions, then validate with `structured_output_hook` or
   caller-side schema validation.
3. Else if tools with suitable parameters are supported and tool-call
   extraction is acceptable, encode the extraction target as a tool call.
4. Else use prompt-only JSON plus post-validation when mode is `prefer`, or
   fail before request when mode is `require`.
5. For OpenRouter, set `provider.require_parameters=true` whenever correctness
   depends on `structured_outputs`, `tools`, or another model-specific
   parameter.
6. For Qwen/Bailian, `tools + stream=True` must fail fast or visibly disable
   streaming under the fallback policy. It must not be sent silently.

Fallback metadata should be visible to callers, including chosen strategy,
fallback reason, validation status, and whether unsupported fields were removed
or preserved.

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
