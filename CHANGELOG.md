# Changelog

## 0.2.0

### Added

- Add per-call `request_options` for OpenAI-compatible Chat Completions
  payload fields, including provider/model-specific structured-output request
  fields, tool request pass-through, sampling controls, routing options, and
  stream options.
- Add capability-aware planning for known high-risk request options and
  `structured_output` fallback modes: strict schema when supported, JSON mode or
  prompt-only fallback for `prefer`, and fail-fast behavior for unsupported
  `require`.
- Add OpenRouter supported-parameter planning for core-generated
  `temperature`, `max_tokens`, and `max_completion_tokens` defaults.
- Cache non-streaming responses by deterministic request payload instead of
  only model and prompt, and bypass response cache for streaming calls.

## 0.1.0

### Added

- Initial extracted `llm-exec-core` package with async LLM execution, shared model catalog, and compatibility result types.
