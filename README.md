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

This repo is also used for coordinated local development with sibling
checkouts during the current extraction/migration work.

## License

Proprietary. See [LICENSE](LICENSE).
