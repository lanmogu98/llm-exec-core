# Changelog

## 0.2.0

### Added

- Add per-call `request_options` for OpenAI-compatible Chat Completions
  payload fields, including provider/model-specific structured-output request
  fields, tool request pass-through, sampling controls, routing options, and
  stream options.
- Add an explicit `structured_output` semantic-planner guardrail: `mode="off"`
  is a no-op, while `require` and `prefer` fail fast until fallback planning is
  implemented.
- Cache non-streaming responses by deterministic request payload instead of
  only model and prompt, and bypass response cache for streaming calls.

## 0.1.0

### Added

- Initial extracted `llm-exec-core` package with async LLM execution, shared model catalog, and compatibility result types.
