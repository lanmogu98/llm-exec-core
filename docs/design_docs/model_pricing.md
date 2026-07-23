# Tiered Regional Mode-Aware Model Pricing

This document is the design source of truth for opt-in
`pricing-rules-v1` schedules, deterministic price selection, cost calculation,
and `LLMClient.pricing_context`.

## Scope and compatibility

Existing catalogs keep using `Pricing(input, output)`. Its validation,
two-key serialization, provider-level `pricing_currency`, one-million-token
basis, client estimation, result objects, and cost behavior are unchanged.
Rich pricing is enabled only when one model replaces that flat object with a
schedule whose explicit schema is `pricing-rules-v1`.

`get_model_details()` remains a raw lookup. It returns the declared
`Pricing | PricingSchedule` without selecting a rule, filling dimensions, or
mutating catalog data. Catalog provider/model order and rule/evidence order are
preserved. Rich types are public from `llm_exec_core.config`; they are not
re-exported from the package root.

## Public schema

The rich public models are:

- `PricingEvidence(url, retrieved_on, facts)`
- `PricingRates(input, output, cache_read_input?, cache_write_input?)`
- `PricingRule(rule_id, billing_model_id, capability_snapshot_id?, region,
  service_scope, deployment_type, output_mode, request_mode, cache_mode,
  input_tokens_gt, input_tokens_lte?, currency, unit_tokens, effective_from?,
  effective_until?, rate_type, rates, evidence)`
- `PricingSchedule(schema, rules)`
- `PricingContext(region, service_scope, deployment_type, output_mode,
  request_mode, cache_mode)`
- frozen `ResolvedPricing(source, rule_id?, rates, cache_mode, currency,
  unit_tokens)`
- `PricingCost(input_cost, output_cost, total_cost, currency)`

Every nested rich model forbids extra fields. Schedules contain at least one
rule and rule IDs are globally unique within one schedule, exact, and
case-sensitive. Rule IDs, billing identities, optional capability snapshot
identities, categorical dimensions, and evidence facts must be nonempty and
have no surrounding whitespace. `output_mode` is a lowercase slug.

Every rule has at least one evidence record; every record has an explicit HTTPS
URL, retrieval date, and at least one fact. Rates are finite and nonnegative,
currency is an uppercase three-letter code, `unit_tokens` is positive, and
token bounds are plain integers with
`0 <= input_tokens_gt < input_tokens_lte` when an upper bound exists.

Effective timestamps must be timezone-aware. Intervals are
`[effective_from, effective_until)`. Standard rates may have open bounds;
promotional rates require both exact bounds. Promotions without public exact
bounds are evidence only and cannot be actionable rules.

Cache-rate structure is validated while loading the catalog:

| `cache_mode` | `cache_read_input` | `cache_write_input` |
| --- | --- | --- |
| `none` | `None` | `None` |
| `implicit` | required | `None` |
| `explicit` | required | required |

Zero is a valid documented rate. `None` means the bucket is inapplicable.

Canonical rich serialization is `model_dump(mode="json")`. The serialized key
is `schema`; dates use ISO date strings; datetimes use timezone-bearing RFC
3339 strings; numbers remain numbers; and absent optional values serialize as
`null`. Legacy `Pricing.model_dump()` remains exactly `{"input": ...,
"output": ...}`.

## Identity and deterministic selection

The catalog mapping key remains the caller-facing model alias.
`billing_model_id` must match `ModelDetails.id` for selection.
`capability_snapshot_id` records reviewed alias/snapshot evidence only; it never
substitutes an identity or creates fallback.

`PricingContext` requires exact `region`, `service_scope`, `deployment_type`,
`output_mode`, `request_mode`, and `cache_mode`. The request modes are
`realtime | batch`; cache modes are `none | implicit | explicit`. A genuinely
inapplicable dimension must be declared as the same explicit value, such as
`not-applicable`, in both rule and context. Core never infers a dimension from
provider name, endpoint, `thinking_level`, or request options.

`resolve_model_pricing()` preserves raw lookup behavior and applies these rules:

1. Flat pricing resolves with provider `pricing_currency`, no rule ID,
   `cache_mode="none"`, and `unit_tokens=1_000_000`; rich selection inputs are
   ignored.
2. A schedule requires a `PricingContext`, a plain nonnegative total input-token
   count, and a timezone-aware effective timestamp.
3. Candidate rules must match billing identity and all six categorical
   dimensions exactly.
4. Token selection uses total request input and
   `input_tokens_gt < total_input_tokens <= input_tokens_lte`; a missing upper
   bound is infinity.
5. Effective selection uses the closed-open interval above.
6. Exactly one candidate succeeds. Zero raises `PricingNoMatchError`; multiple
   candidates raise `PricingAmbiguityError` and identify every matching rule.

Declaration order, specificity, evidence date, aliases, snapshots, and prices
never break a tie. Unknown-model lookup keeps its existing plain `ValueError`.
All selection failures inherit `PricingSelectionError`; missing selection input
uses `PricingContextRequiredError`.

## Calculation

`calculate_pricing_cost()` accepts a `ResolvedPricing` and authoritative plain
nonnegative integer counts. The input count is total request input. Ordinary
input is:

```text
total input - cache-read input - cache-write input
```

Each bucket uses its explicit selected rate and the same `unit_tokens` basis;
output is calculated separately. The helper does not convert currency, apply a
percentage, derive a rate, or round. Required cache arguments must be supplied
even when zero. A missing rate, contradictory mode/bucket, invalid type or
count, or cache sum exceeding total input raises `PricingCalculationError`.

Pure resolution and calculation support both realtime and batch rules. The
current `LLMClient` transport remains realtime-only and rejects a rich batch
context.

## Client usage contract

Schedule clients require the keyword-only `pricing_context` argument. The
client deep-copies it and uses request-start UTC as `effective_at`. Flat clients
need no context and retain their current behavior.

Rich client pricing recognizes only OpenAI-compatible Chat Completions usage:

```text
usage.prompt_tokens
usage.completion_tokens
usage.prompt_tokens_details.cached_tokens
usage.prompt_tokens_details.cache_creation_input_tokens
```

No `input_tokens`, `output_tokens`, Anthropic Messages, DashScope-native, or
Responses API aliases are accepted. Consumed fields must be present when their
mode requires them, have `type(value) is int`, and be nonnegative. Booleans,
floats, strings, `null`, and numeric-like objects fail.

- `none`: details may be absent or `null`; present cache values must be zero.
- `implicit`: details and `cached_tokens` are required, including explicit
  zero; cache creation may be absent or zero but cannot be nonzero.
- `explicit`: details and both cache keys are required, including explicit
  zeros.

For implicit and explicit modes, cache buckets cannot exceed
`prompt_tokens`. Unknown keys inside `prompt_tokens_details` are ignored by
pricing.

Non-streaming reads `response_json["usage"]`. Streaming retains the last
non-null `chunk["usage"]` before `[DONE]` and passes it to the same parser.
Missing or malformed authoritative rich usage never falls back to estimation.
Parsing, rule selection, cost calculation, and mixed-currency checks all
complete before aggregate token/cost state changes.

Legacy non-streaming still uses missing-field zero defaults. Legacy streaming
still estimates a counter when its final value is zero, and cache detail fields
do not change flat input pricing. The local `ResponseCache` is unrelated to
provider cache pricing: a hit keeps zero tokens and zero cost and does not
mutate aggregate usage.

`TokenUsage`, `LLMResult`, `ExecutionMetadata`, legacy tuples, request detail
keys, `get_token_usage()`, and `format_usage_report()` are unchanged. A rich
request's `TokenUsage.currency` is the selected three-letter rule currency. A
client refuses to accumulate a later request in a different currency.

## Opt-in example

```python
from llm_exec_core.client import LLMClient
from llm_exec_core.config import PricingContext

context = PricingContext(
    region="us-east",
    service_scope="global",
    deployment_type="serverless",
    output_mode="thinking",
    request_mode="realtime",
    cache_mode="explicit",
)

client = LLMClient(
    "application-model-alias",
    config_source=catalog,
    pricing_context=context,
)
```

The model's catalog entry must contain a complete schedule with direct reviewed
rates for that exact context. There is no automatic conversion from a flat
price.

## Official evidence refreshed 2026-07-23

- [Model pricing](https://help.aliyun.com/zh/model-studio/model-pricing): total
  request input selects closed-upper token tiers; tables separate input/output,
  service scope and region, thinking mode, aliases/snapshots, batch/cache
  treatment, and promotional presentation.
- [Context Cache](https://help.aliyun.com/zh/model-studio/context-cache):
  explicit and implicit caches are mutually exclusive, have distinct creation
  and read treatment, and OpenAI Chat Completions examples expose
  `prompt_tokens_details.cached_tokens` and
  `cache_creation_input_tokens`.
- [OpenAI-compatible Batch](https://help.aliyun.com/zh/model-studio/batch-interfaces-compatible-with-openai/):
  batch is asynchronous, priced separately from realtime, and exposed through
  region-specific service endpoints.
- [Text-generation model matrix](https://help.aliyun.com/en/model-studio/text-generation-model/):
  capability tables distinguish model IDs and snapshots, context limits,
  thinking support, and batch availability.
- [Deep thinking](https://help.aliyun.com/en/model-studio/deep-thinking): hybrid
  models switch thinking per request, thinking output is billed, and some
  models price thinking and non-thinking differently.
- [OpenAI-compatible Chat](https://help.aliyun.com/en/model-studio/compatibility-of-openai-with-dashscope):
  endpoints differ by region and `stream_options={"include_usage": true}`
  requests streaming token counts.

These records contain URLs, retrieval date, and verified semantic facts only.
No authenticated console, credential, live request, provider response, or
representation hash is evidence.

## Prohibitions, migration, and rollback

Do not infer missing dimensions, guess a fallback, flatten tiers or modes,
convert currencies, apply undocumented discounts, derive rates, round costs,
substitute aliases, or use declaration order to resolve overlap. Actual rich
catalog adoption requires direct reviewed rates and dated official evidence.

Migration is per model and opt-in; existing flat catalogs remain valid. A
focused rollback removes schedule/client support, its tests, this document, the
README guidance, and the matching changelog entry together. Because this
change does not migrate the maintained catalog, existing consumers remain on
flat behavior throughout rollback.
