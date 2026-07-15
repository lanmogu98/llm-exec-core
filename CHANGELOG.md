# Changelog

## 0.2.0

### Added

- Add per-call `request_options` for OpenAI-compatible Chat Completions
  payload fields, including provider/model-specific structured-output request
  fields, tool request pass-through, sampling controls, routing options, and
  stream options.
- Add capability-aware planning for known high-risk request options and
  client-validated `structured_output` fallback modes: strict schema when
  supported, JSON mode or prompt-only fallback for `prefer`, and fail-fast
  behavior for unsupported `require`.
- Add OpenRouter supported-parameter planning for core-generated
  `temperature`, `max_tokens`, and `max_completion_tokens` defaults.
- Add a deterministic, explicit OpenRouter target-capability comparison script
  backed by fixture tests; normal requests and unit tests remain offline.
- Cache non-streaming responses by deterministic request payload instead of
  only model and prompt, and bypass response cache for streaming calls.

### Fixed

- Parse and validate every active structured-output response with `jsonschema`,
  populate `LLMResult.structured`, and report validation separately from hook
  application.
- Run structured validation and post-processing before cache writes, and repeat
  validation on cache hits so invalid or failed responses do not populate cache.
- Reject conflicting Gemini reasoning/thinking controls before HTTP and share
  capability metadata across the paid/free aliases for reviewed Gemini models.
- Refresh the 2026-07-15 OpenRouter target snapshots for reasoning effort,
  temperature, and completion-token support.

## 0.1.0

### Added

- Initial extracted `llm-exec-core` package with async LLM execution, shared model catalog, and compatibility result types.
