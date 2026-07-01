# OpenAI-Compatible Chat Completions Request Options

This document is the design source of truth for `LLMClient.generate(...,
request_options=...)` and `generate_response(..., request_options=...)`.

## Scope

This feature transports OpenAI-compatible `/chat/completions` request fields.
It does not implement the OpenAI Responses API, execute tools, parse tool-call
responses, or add runtime provider capability filtering.

## Provider Status

| Provider / route | Structured output status | Notes |
| --- | --- | --- |
| OpenAI Chat Completions | Supports strict `response_format.type=json_schema`; JSON mode remains separate. | Also supports common tool and sampling fields. |
| Gemini OpenAI compatibility | SDK helpers document structured output through `response_format`; direct raw REST support should be probe-tested per model. | Provider raw fields can be sent through SDK-style `extra_body`. |
| DeepSeek official | Documents JSON mode, not strict response JSON Schema. | Tool parameters can still use JSON Schema. |
| DashScope / Qwen OpenAI-compatible mode | No documented `response_format` field in the checked compatibility table. | Common controls and `extra_body.enable_search` can pass through. |
| Zhipu / GLM | Documents JSON mode, not strict response JSON Schema. | Some controls are provider-specific. |
| OpenRouter | Supports structured outputs for compatible model/provider routes. | Support varies by routed model; routing objects pass through. |
| Volcengine Ark | OpenAI-compatible API exists, but detailed option support needs a live probe or clearer official table. | Treat as probe-required. |

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
`stream=True`, `include_usage: true` is added only when absent after the merge.
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
