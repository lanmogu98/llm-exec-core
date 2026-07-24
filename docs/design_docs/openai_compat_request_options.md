# OpenAI-Compatible Chat Completions Request Options

This document is the design source of truth for `LLMClient.generate(...,
request_options=..., generation_controls=...)` and
`generate_response(..., request_options=..., generation_controls=...)`.

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
`structured_output: Mapping[str, Any] | None = None`. They also accept
`generation_controls: GenerationControls | Mapping[str, Any] | None = None`.

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

## Generation Policy and Typed Controls

`ModelCapabilities.generation_policy` is an optional, complete,
provider-neutral declaration of the selected route's reasoning and sampling
contract. The policy is authoritative when present. It contains no provider,
model-name, or endpoint dispatch logic; two routes with the same upstream model
ID may declare different policies, and unrelated routes may declare identical
ones.

The public types live in `llm_exec_core.config` and are not re-exported from
the package root:

```text
GenerationPolicy
  reasoning: ReasoningPolicy
  sampling: SamplingPolicy

ReasoningPolicy
  availability: unavailable | optional | adaptive | always-on
  allowed_modes: tuple[disabled | enabled | adaptive | always-on, ...]
  default_mode: disabled | enabled | adaptive | always-on
  can_disable: bool
  mode: ReasoningModeWire | None
  effort: ReasoningEffortPolicy | None
  budget_tokens: ReasoningBudgetPolicy | None
  allow_effort_with_budget_in:
    tuple[enabled | adaptive | always-on, ...] = ()

ReasoningModeWire
  path: tuple[str, ...]
  values: dict[disabled | enabled | adaptive, str | bool]

ReasoningEffortPolicy
  path: tuple[str, ...]
  allowed_values: tuple[str, ...]
  aliases: dict[str, str] = {}
  modes: dict[enabled | adaptive | always-on, EffortModeRule]

EffortModeRule
  omission: provider-default | provider-selected | required
  default: str | None

ReasoningBudgetPolicy
  path: tuple[str, ...]
  minimum: int
  maximum: int | None
  modes: dict[enabled | adaptive | always-on, BudgetModeRule]

BudgetModeRule
  omission: provider-default | provider-selected | required
  default: int | None
  output_limit_relation: none | less-than | less-than-or-equal

SamplingPolicy
  temperature: SamplingControlPolicy
  top_p: SamplingControlPolicy

SamplingControlPolicy
  base: SamplingRule
  by_reasoning_mode:
    dict[disabled | enabled | adaptive | always-on, SamplingRule] = {}

SamplingRule
  state: supported | fixed | ignored | deprecated | forbidden
  range: NumericRange | None
  fixed_value: float | None

NumericRange
  minimum: float
  maximum: float
  minimum_inclusive: bool = True
  maximum_inclusive: bool = True
```

The reasoning declaration has four availability shapes:

| Availability | Allowed shape |
| --- | --- |
| `unavailable` | Exactly `disabled`, with no mode wire field, effort, budget, or caller disable operation. |
| `optional` | Exactly `disabled` and `enabled`, with a complete mode wire mapping and disable support. |
| `adaptive` | Includes `adaptive`; may also include `disabled` and evidence-backed manual `enabled`, with a complete mode wire mapping. |
| `always-on` | Exactly `always-on`, with no caller-selectable mode wire field or disable operation. |

Every policy declares `allowed_modes`, `default_mode`, and `can_disable`.
Selectable modes use a route-declared nested JSON path and an exact wire value
for every selectable mode. Wire values are distinct strings or booleans and
are decoded with exact type equality, so `true` is not interchangeable with
`1`.

Effort policy declares a nested wire path, canonical string values, optional
alias-to-canonical mappings, and per-active-mode omission behavior. Budget
policy declares a nested wire path, inclusive integer bounds, and
per-active-mode omission behavior plus one final output-limit relation:
`none`, `less-than`, or `less-than-or-equal`. A policy separately lists the
active modes in which effort and budget may coexist.

The three omission states are:

- `provider-default`: omission has a declared canonical value for validation
  and documentation, but the field remains absent on the wire.
- `provider-selected`: omission delegates selection to the provider and has no
  declared default.
- `required`: callers must resolve a value for that active mode.

Sampling policy is complete for both `temperature` and `top_p`. Each control
has a base rule and optional overrides keyed by reachable reasoning mode.
Supported rules declare an inclusive/exclusive finite numeric range; fixed
rules declare one finite value.

| Sampling state | Provider/model configured value | Per-call raw or typed value |
| --- | --- | --- |
| `supported` | Validate, canonicalize, and send. | Validate, canonicalize, and send. |
| `fixed` | Omit. | Accept only the fixed value, then omit. |
| `ignored` | Omit. | Require a finite number, then omit. |
| `deprecated` | Omit. | Fail before execution. |
| `forbidden` | Omit. | Fail before execution. |

All new schema objects reject unknown fields. Wire paths are nonempty arrays of
nonempty strings. Reasoning paths cannot root at core-owned `model`,
`messages`, or `stream`, cannot root at sampling-owned `temperature` or
`top_p`, and cannot be equal to or a prefix of another reasoning path. Effort,
budget, coexistence, and sampling-mode keys must be reachable in the declared
reasoning shape. Aliases, defaults, ranges, and bounds are validated when the
catalog is loaded.

`GenerationControls` has optional `reasoning` and `sampling` groups:

```python
generation_controls = {
    "reasoning": {
        "mode": "enabled",
        "effort": "high",
        "budget_tokens": None,
    },
    "sampling": {
        "temperature": 0.2,
        "top_p": None,
    },
}
```

Both mapping and Pydantic inputs preserve field presence. An absent field means
"inherit"; an explicitly present `None` is a tombstone that clears all
lower-precedence values for that semantic control. A present group set to
`None` tombstones every control in that group. An empty top-level controls
object is a no-op; any nonempty typed controls require a generation policy.

Policy controls resolve independently, in this immutable order:

1. provider scalar, then provider request override;
2. model scalar, then model request override;
3. constructor `thinking_level` for effort only;
4. per-call raw `request_options` or typed `generation_controls`.

A request cannot provide the same semantic control through both raw and typed
per-call surfaces. Different controls may use the two surfaces together.
Direct keys continue to win over promoted `extra_body` keys before semantic
resolution. Canonical writing removes lower-precedence declarations at the
route-declared path, preserves unknown siblings, and fails if a declared path
would traverse a scalar.

Effort aliases are converted to their canonical value before payload and cache
construction. Explicitly disabling reasoning suppresses lower-precedence
effort and budget defaults; a conflicting effort or budget in the same
per-call layer fails. Effort/budget mode support, required omissions,
coexistence, integer bounds, and relations to the single final
`max_tokens`/`max_completion_tokens` value all fail closed.

After structured-output, token-field, OpenRouter, and legacy capability
planning, the final payload is decoded and validated again against the policy.
This happens before cache lookup, HTTP client construction, rate limiting, or
HTTP. Token-limit retry mutation repeats policy validation before a second HTTP
attempt. Cache identity therefore uses the final canonical payload plus the
existing planning capability identity: aliases and no-effect sampling choices
share a key, while distinct effective payloads or capability versions do not.

Unknown raw fields remain passthrough. A missing generation policy preserves
legacy request, result, usage, streaming, structured-output, token-retry, and
tuple behavior. The only policy-absent correction is that a core-generated
unset temperature is omitted instead of being sent as JSON `null`; an explicit
unknown raw `temperature: null` remains passthrough.

### Generation-policy evidence refreshed 2026-07-24

The schema covers the following current, official primary-source facts without
adding any catalog route in this change:

- Google documents model-specific thinking levels/defaults and, beginning with
  Gemini 3.6 Flash and Gemini 3.5 Flash-Lite, deprecated and ignored
  `temperature`/`top_p` parameters that should be removed; future model
  generations will reject them:
  https://ai.google.dev/gemini-api/docs/latest-model and
  https://ai.google.dev/gemini-api/docs/thinking
- Anthropic documents model-dependent manual, adaptive, and always-on thinking
  contracts; effort availability/defaults; a 1,024-token manual-budget
  minimum; a general `budget_tokens < max_tokens` rule with a documented
  interleaved-thinking exception; incompatibility with modified
  `temperature`/`top_k`; and a thinking-enabled `top_p` range of 0.95–1:
  https://platform.claude.com/docs/en/build-with-claude/effort and
  https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- DeepSeek documents `thinking.type` with enabled/disabled wire values,
  enabled-by-default behavior, canonical `high`/`max` effort, and compatibility
  aliases `low`/`medium -> high` and `xhigh -> max`; in thinking mode its
  sampling and penalty parameters are accepted for compatibility but ignored:
  https://api-docs.deepseek.com/guides/thinking_mode
- Kimi K3 documents always-on reasoning, top-level `reasoning_effort` values
  `low`/`high`/`max` with default `max`, and fixed
  `temperature=1.0`/`top_p=0.95` values that callers should omit:
  https://platform.kimi.ai/docs/guide/kimi-k3-quickstart
- Alibaba Bailian's OpenAI-compatible parameter table documents
  `temperature` in `[0, 2)` and `top_p` in `(0, 1.0)`, while also identifying
  model-specific parameter support:
  https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
- Volcengine's `ContextChatCompletions` endpoint documentation lists
  `reasoning_effort` values `low`/`medium`/`high` and
  `temperature`/`top_p` ranges `[0, 1]`:
  https://api.volcengine.com/api-docs/view?action=ContextChatCompletions&serviceCode=ark&version=2024-01-01

Generic endpoint documentation does not prove support for every model route.
Each future catalog declaration still requires current official evidence for
the exact provider, endpoint, and model ID. No authenticated console was used.

## Provider Connection Resolution

`ProviderSettings` keeps the required `api_key_env_var` and `api_base_url`
fields and adds two optional, provider-neutral declarations:

```yaml
synthetic-provider:
  api_key_env_var: SYNTHETIC_PRIMARY_API_KEY
  api_key_env_aliases:
    - SYNTHETIC_FALLBACK_API_KEY
  api_base_url: https://static.example.invalid/v1/chat/completions
  api_base_url_env_var: SYNTHETIC_API_BASE_URL
```

Each new environment-variable name must be a non-empty string with no Unicode
whitespace, `=`, or NUL. The endpoint variable must differ from the primary key
name and every alias. Alias order and duplicates are preserved, and an alias
may repeat the primary name. These checks intentionally do not tighten the
legacy primary declaration.

Credential resolution checks `[api_key_env_var, *api_key_env_aliases]` in
order. An unset, empty, or Unicode-whitespace-only value does not win. The first
value containing non-whitespace content wins and terminates lookup. If its
original value contains a C0 control or DEL, resolution fails without checking
a later alias; otherwise the original value is used unchanged in the
Authorization header. Missing-key and invalid-key errors contain declared
variable names only. With no aliases, a missing or blank primary preserves the
legacy `ValueError` message exactly.

Endpoint resolution uses only `api_base_url_env_var`:

1. An absent declaration or an unset/Unicode-whitespace-only value returns the
   static URL byte-for-byte without validation or normalization.
2. A populated value is stripped of exactly surrounding ASCII space, tab, LF,
   CR, VT, and FF, then any remaining ASCII control/space, DEL, Unicode
   whitespace, or backslash is rejected.
3. Literal query and fragment delimiters are rejected. Standard-library URL
   parsing must yield an absolute case-insensitive HTTPS URL with authority and
   hostname, no userinfo, and either no explicit port or a valid port in
   `1..65535`; an empty port delimiter is invalid.
4. Trailing slashes are removed. A path ending exactly in
   `/chat/completions` is used as-is; a path ending exactly in `/v1` receives
   `/chat/completions`; all other paths are rejected.
5. Apart from the defined edge trim, trailing-slash removal, and suffix append,
   the scheme, authority, and path bytes are preserved.

| Declared override value | Resolved request URL |
| --- | --- |
| `https://api.example.invalid/v1` | `https://api.example.invalid/v1/chat/completions` |
| `https://api.example.invalid/v1/chat/completions` | unchanged |
| `https://workspace.example.invalid/compatible-mode/v1///` | `https://workspace.example.invalid/compatible-mode/v1/chat/completions` |
| unset or blank | the exact static `api_base_url` |

Invalid overrides fail before HTTP client/request construction. Exceptions may
identify the endpoint variable and a reason category but never reproduce its
value, hostname, port, path, query, or fragment. Resolution does not mutate
`ProviderSettings`, caller dictionaries, or lists, so repeated Path/dictionary
loads and independent model instances remain isolated.

There is intentionally no hostname allowlist or provider-name branch. Catalogs
and environment variables are deployer-controlled, and regional, workspace,
private-gateway, IP-literal, and self-hosted OpenAI-compatible endpoints must
remain possible. HTTPS/path/control/userinfo/query/fragment checks prevent
common accidental leakage and cross-parser ambiguity, but do not form an SSRF
boundary against a malicious deployer.

The optional fields are additive public Pydantic schema/serialization surface:
legacy default `model_dump()` output gains `api_key_env_aliases=[]` and
`api_base_url_env_var=None`. Existing valid credentials and static endpoints
retain their meanings; whitespace-only and C0/DEL-bearing selected credentials
now fail closed. A focused revert must remove both declarations, both resolver
paths, their tests, and this documentation together to restore the prior
single-key/static-URL behavior.

## Raw Catalog and Effective Client Policy

`ProviderSettings` retains its existing `temperature`, `max_tokens`,
`context_window`, and `request_overrides` declarations and adds
`output_token_field`, which defaults to `max_tokens`. `ModelDetails` adds
optional `temperature`, `max_tokens`, `context_window`, `request_overrides`, and
`output_token_field` declarations. The two valid output-token field names are
`max_tokens` and `max_completion_tokens`.

`get_model_details()` continues to return the raw
`(provider_name, ProviderSettings, ModelDetails)` tuple. It does not fill model
fields from provider defaults or write merged request defaults into either
object. Dictionary catalogs are deep-copied at the load boundary, and Path
catalogs are parsed afresh, so mutations to one returned settings tree or one
client's effective defaults do not affect the caller's source or a later load.

For client construction, `temperature`, `max_tokens`, `context_window`, and
`output_token_field` use the model value when it is non-`None`; otherwise they
use the provider value. This keeps legacy catalogs provider-wide while allowing
heterogeneous models under one provider without provider-name branches.

## Payload Construction

The effective request payload is built in this order:

1. Core defaults: `model`, `messages`, effective `temperature`, `stream`, and
   the effective generated output-token field/value when no explicit token
   field controls the payload.
2. Normalized provider `request_overrides`.
3. Normalized model `request_overrides`.
4. Normalized per-call `request_options`.

Each provider, model, and per-call layer normalizes SDK-style `extra_body` one
level before merging. Values under `extra_body` are promoted into that layer's
top-level request body, and direct top-level keys in the same layer win. The
normalization does not recurse, so `extra_body.extra_body` becomes the final
top-level raw `extra_body` field.

`model`, `messages`, and `stream` are core-owned protected fields. They are
rejected after normalization in provider/model `request_overrides` and per-call
`request_options`. `stream_options` is the supported way to customize streaming
request behavior.

Legacy catalogs select a generated `max_tokens`. A provider or model may select
generated `max_completion_tokens` through `output_token_field` without placing
the numeric value in raw request overrides. If any normalized provider, model,
or per-call layer contains either token-limit field, that explicit field
suppresses the generated default and follows normal same-key precedence. A
final payload containing both `max_tokens` and `max_completion_tokens` is
contradictory and fails before HTTP with an actionable `ValueError`.

Configured max-token retry matching still uses the declared status, body
substring, configured limit, existing lower bound, retry count, and delay. It
inspects the single token field in the final payload; when that value is an
integer above the configured limit, retry lowering logs the field name and
numeric limits and updates that same field. An absent or non-integer limit does
not enter the lowering path.

On OpenRouter routes with capability metadata, core-generated `temperature`,
`max_tokens`, and `max_completion_tokens` are planned against
`supported_parameters`. Unsupported generated defaults are omitted; a generated
`max_tokens` is converted to `max_completion_tokens` when only that token-limit
parameter is listed. Explicit provider/model/per-call values for unsupported
fields fail fast instead of being sent silently.

`stream_options` is deep-merged as provider first, then model, then per call.
When `stream=True`, `include_usage: true` is added when final `stream_options` is
absent. Mapping-valued `stream_options` keeps caller values and receives
`include_usage: true` only when that key is absent. Other explicit values are
considered present and are not modified. An explicit model or per-call
`stream_options: None` clears lower-precedence `stream_options`; a per-call
`None` also suppresses the core `include_usage` default. Omit `stream_options`
to receive the default streaming usage request.

For non-streaming calls, `stream_options` is included only when explicitly
provided by provider, model, or per-call options.

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
- With no capability metadata, the `thinking_level` constructor convenience
  keeps legacy behavior and emits `reasoning_effort`. With metadata present, it
  emits that field only when `reasoning_controls` lists `reasoning_effort`;
  otherwise client construction fails before request planning. Direct raw
  request controls continue through the existing payload capability validation.
  Core never translates `thinking_level` to provider-specific thinking fields.

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
