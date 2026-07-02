# OpenAI-Compatible Chat Completions Request Options

This document is the design source of truth for `LLMClient.generate(...,
request_options=...)` and `generate_response(..., request_options=...)`.

## Scope

This feature transports OpenAI-compatible `/chat/completions` request fields.
It does not implement the OpenAI Responses API, execute tools, parse tool-call
responses, or add runtime provider capability filtering.

## Provider Status

Provider research as of 2026-06-30:

| Provider / route | Structured output support | Tools support | Common request controls worth transporting | Evidence / notes |
| --- | --- | --- | --- | --- |
| OpenAI Chat Completions | `response_format.type=json_schema` with `json_schema.strict=true` for Structured Outputs; `response_format.type=json_object` is older JSON mode and guarantees valid JSON, not schema adherence. | `tools`, `tool_choice`, `parallel_tool_calls`. | `temperature`, `top_p`, preferred `max_completion_tokens`, legacy/deprecated `max_tokens`, `stream`, `stream_options`, `stop`, `seed`, `logprobs`, `top_logprobs`, penalties. | https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create and https://developers.openai.com/api/docs/guides/structured-outputs |
| Gemini OpenAI compatibility | OpenAI SDK `parse()` helpers document structured output via `response_format`; direct raw REST `chat.completions.create(response_format={...})` should be probe-tested before treating it as generic payload support. | `tools`, `tool_choice="auto"`. | `stream`, `reasoning_effort`, provider-specific raw top-level `extra_body` fields such as `extra_body.google.thinking_config`. | https://ai.google.dev/gemini-api/docs/openai |
| DeepSeek official | `response_format.type=text|json_object`; `json_object` is valid-JSON mode only, requires explicit JSON instruction in messages, and has no documented `json_schema` response format. | `tools`, `tool_choice` (`none`/`auto`/`required` or named function), `tools[].function.parameters` as JSON Schema; `tools[].function.strict` is beta. | `thinking`, `reasoning_effort`, `max_tokens`, `stop`, `stream_options`, `temperature`, `top_p`, `logprobs`, `top_logprobs`. | https://api-docs.deepseek.com/api/create-chat-completion |
| DashScope / Qwen OpenAI-compatible mode | No `response_format` listed on the OpenAI-compatible parameter table. | `tools`; docs say tools cannot currently be used with `stream=True`. | `top_p`, `temperature`, `presence_penalty`, `n`, `max_tokens`, `seed`, `stream`, `stop`, `stream_options`, SDK `extra_body.enable_search`. | https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope |
| Zhipu / GLM | `response_format={"type":"json_object"}` JSON mode; JSON Schema is shown as caller-side validation/prompt guidance, not strict API schema. | `tools`; `tool_choice` defaults to and only supports `auto`. | `do_sample`, `temperature`, `top_p`, `max_tokens`, `stream`, `thinking`, `reasoning_effort`. | https://docs.bigmodel.cn/cn/guide/capabilities/struct-output, https://docs.bigmodel.cn/cn/guide/capabilities/function-calling, https://docs.bigmodel.cn/cn/guide/start/concept-param |
| OpenRouter | `response_format.type=json_schema` for compatible models; support varies by selected model/provider route. | Standardized `tools` interface; support varies by model/provider. | Preferred `max_completion_tokens`, deprecated `max_tokens`, `logprobs`, `seed`, `reasoning`, `provider`, `top_k`, `min_p`, `top_a`; check model `supported_parameters` and use `provider.require_parameters=true` when required. | https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request, https://openrouter.ai/docs/guides/features/structured-outputs, https://openrouter.ai/docs/guides/features/tool-calling, https://openrouter.ai/docs/api/api-reference/models/get-models |
| Volcengine Ark | OpenAI SDK/API compatibility is documented, but chat parameter support for `response_format`, tools, streaming options, and common controls remains unknown from fetchable official text. | Unknown/probe-required. | Unknown/probe-required. | https://www.volcengine.com/docs/82379/1494384?lang=zh&redirect=1 |

Important corrections:

- OpenRouter is a routing layer; supported parameters vary by selected
  upstream provider/model.
- DeepSeek, Zhipu, and DashScope JSON mode are not equivalent to OpenAI
  strict `response_format.type=json_schema`.
- JSON Schema references in tool/function parameters or caller-side validation
  do not imply strict response schema support.
- Volcengine Ark remains probe-required until an inspectable official
  parameter table or live probe confirms behavior.

## Public API

Both request methods accept keyword-only `request_options:
Mapping[str, Any] | None = None`.

`request_options` is for HTTP request payload fields only. Metadata such as
`request_name`, callbacks, `trace_context`, `run_id`, `request_id`, and
`structured_output_hook` never enters the HTTP payload or cache key.

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
