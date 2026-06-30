# llm-exec-core

Shared LLM execution package extracted from Editor Assistant.

It provides:

- `LLMClient` for async request execution and streaming
- `llm_exec_core/llm_config.yml` as the shared model catalog
- typed result and usage objects for legacy and new call sites

This repo is currently consumed from sibling checkouts during the migration branch. After publish, downstream apps can depend on the released `llm-exec-core` package directly.
