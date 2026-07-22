# Cross-Repository LLM Execution Boundaries Design

- Status: design draft amended after read-only counterexample review;
  implementation not dispatched
- Date: 2026-07-22
- Tracking Issue: [llm-exec-core#26](https://github.com/lanmogu98/llm-exec-core/issues/26)
- Planning dispatch: [#26 maintainer dispatch record](https://github.com/lanmogu98/llm-exec-core/issues/26#issuecomment-5043424563)
- Target completion window: 2026-07-26

## Objective

Give each personal LLM application one clear abstraction boundary so provider
transport, structured-output validation, document-task behavior, and domain
workflow orchestration are not copied across repositories.

The program must make the highest-value applications implementation-ready in
this maintainer priority order:

1. `editor-assistant`
2. `arxiv-batch-processor`
3. `hormuz-watch`
4. `open-translate`
5. Remaining consumers, ordered by migration value and available capacity

Priority is not a technical dependency. ArXiv, Hormuz, and Open Translate must
not be marked `blocked-by` Editor unless a later repository-local design finds
a real contract dependency.

## Scope

This design covers:

- the target abstraction layer for each identified personal application;
- end-to-end structured output in Editor's Python and CLI surfaces;
- atomic structured paper jobs in ArXiv processing;
- migration of eligible Python transports to `llm-exec-core`;
- the intentional JavaScript-native boundary for Open Translate;
- issue decomposition, ordering, compatibility, validation, and rollback;
- a program-level readiness audit before implementation begins.

## Non-Goals

- Implementing any child repository migration in the #26 design change.
- Making Editor a mandatory middle layer for Python applications.
- Integrating the organization-owned Java administration application with
  Python code. Its learning-materials schema is evidence only.
- Adding a core multimodal/PDF input API for `print-pub-ai` in this program.
- Changing domain prompts, taxonomies, investment thresholds, or editorial
  behavior during transport migration.
- Adding live-provider CI, secrets, releases, package publication, merge
  automation, repository settings changes, or direct `main` updates.

## Evidence Summary

| Consumer | Current execution shape | Boundary problem |
| --- | --- | --- |
| `editor-assistant` | Calls legacy `generate_response()` and stores text | Core validates structured data but Editor discards `LLMResult.structured` |
| `arxiv-batch-processor` | Four independent semantic tasks, later grouped by `arxiv_id` | Repeated full-text input, partial results, inconsistent semantic generation, and copied Editor internals |
| `hormuz-watch` | Local role-separated HTTP client plus tolerant JSON parsing and fallback | Transport can move to core only after core preserves the system/user role boundary; strict schema validation is a separate domain decision |
| `open-translate` | Browser-native JavaScript streaming client | Python runtime reuse is inappropriate; capability and protocol assumptions need an explicit native boundary |
| `global-science-db` | Role-separated client adapted from Editor plus Pydantic tasks | Provider transport is copied, but its system prompt and layered domain validation must survive migration |
| `cell-press-paper-classifier` | Native Gemini and OpenRouter classifiers | OpenRouter duplicates core behavior; replacing the native Gemini route requires separate parity evidence |
| `print-pub-ai` | Editor client for text plus direct multimodal HTTP for PDF | Text calls can use core after catalog ownership is explicit; PDF message content is outside core's current prompt-only contract |

The organization Java sample demonstrates useful schema characteristics:
nested objects and arrays, required keys, exact item counts, enums, and
`additionalProperties: false`. No prompt, persistence state machine, runtime
dependency, or cross-repository implementation relationship is imported from
that application.

## Layer Ownership

### `llm-exec-core`

Core owns Python LLM execution mechanics:

- catalog schema/loading and reviewed reference capability metadata;
- provider request construction and rate limiting;
- request-option and structured-output planning;
- JSON parsing and JSON Schema validation;
- typed `LLMResult`, usage, and execution metadata;
- deterministic mocked/faked provider tests.

Core does not own domain prompts, domain schemas, application checkpoints,
document conversion, business fallbacks, or result persistence.

Core currently accepts one user prompt and protects the raw `messages` request
field. This program adds only the smallest role-preserving prerequisite needed
by real consumers: an optional `system_prompt` alongside the existing user
prompt on `generate()` and its compatibility `generate_response()` surface. A
general raw-messages or multimodal API is outside this design.

### `editor-assistant`

Editor owns reusable document-oriented behavior:

- conversion to `MDArticle`;
- built-in document tasks and prompt composition;
- local CLI and Python convenience surfaces;
- Editor run history and output persistence;
- a public, additive typed task-execution result.

Editor is a reference consumer of core, not a universal application runtime.

### Domain Applications

An application should depend directly on core when it already owns its input
model, prompt, schema, batching, persistence, and retry semantics. This applies
to ArXiv, Hormuz, gsdb, and the Cell classifier.

### Non-Python Applications

Open Translate remains JavaScript-native because it executes in a browser,
stores user-supplied credentials in extension-managed storage, and streams
directly to a content script. It may align design evidence and catalog
provenance without importing or serving Python.

## Cross-Cutting Prerequisites

### Role-Preserving Core Input

Hormuz and gsdb currently depend on distinct system and user messages. Joining
them into one string would be a model-behavior change, so both migrations are
blocked by a new core Issue that adds an optional `system_prompt` to the public
execution surface.

The additive core change must:

- leave existing prompt-only calls and payloads unchanged;
- keep caller-supplied raw `messages` protected;
- include the system prompt in request construction, cache identity, redacted
  diagnostics, and deterministic tests;
- define behavior for streaming and structured-output modes;
- document compatibility and security effects; and
- receive the high-risk independent exact-head review required for a public API
  change.

This prerequisite does not make Hormuz or gsdb depend on Editor.

### Caller-Owned Model Catalogs

Core requires an explicit complete `config_source`; removing an Editor wrapper
does not remove that responsibility. Every repository-local migration Issue
must name its catalog owner, source evidence, and update responsibility.

- Editor retains its existing caller-owned catalog contract.
- ArXiv, Hormuz, gsdb, and print-pub each own a minimal catalog containing only
  models they actually use.
- The Cell classifier keeps the native Gemini configuration until parity is
  proven and owns a minimal catalog for any core-backed route.
- Open Translate owns its browser-usable JavaScript catalog and does not copy a
  Python artifact mechanically.

Catalog changes use dated official evidence, offline fixtures, alias/fallback
analysis, and a fresh independent review on the exact implementation head.

### Repository-Local Ownership Checklist

Every child Issue must identify the catalog, prompt, schema, persistence, and
failure-semantics owner; compatibility entry points; offline fixtures and fake
HTTP coverage; and the rollback switch or PR path. A transport migration is not
allowed to silently change timeouts, retry exhaustion, partial-result behavior,
or domain fallback semantics.

## Editor Structured-Output Design

### Additive Result Contract

Editor must stop using the legacy tuple internally when a typed result is
needed. Add explicit typed output and success-only application results similar
to:

```python
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

@dataclass(slots=True)
class OutputArtifact:
    value: str | JSONValue
    content_type: Literal["text/plain", "application/json"]
    serialized_text: str

@dataclass(slots=True)
class TaskExecutionResult:
    task_name: str
    run_id: int
    outputs: Mapping[str, OutputArtifact]
    llm_result: LLMResult
```

A new low-level Python execution entry point returns `TaskExecutionResult` and
propagates typed exceptions. Existing `process_mds() -> tuple[bool, int]` and
`process_multiple()` behavior remain compatibility wrappers that translate
success or failure into their legacy surfaces.

This avoids a breaking return-type change while giving new callers access to
`LLMResult.text`, `LLMResult.structured`, typed usage, and planning metadata.

### Task Contract

The base `Task` gains an optional structured-output declaration with a default
of `None`. Existing `brief`, `outline`, and `translate` tasks therefore remain
text tasks without code changes to their semantic behavior.

A runtime-configured `StructuredExtractTask` owns:

- the caller-supplied prompt;
- the caller-supplied JSON Schema;
- a stable schema name;
- `require` or `prefer` planning mode.

The caller explicitly chooses this task. No automatic task classification
selects or enables JSON Schema.

Existing registry tasks remain no-argument classes. The new Python entry point
accepts a configured task instance, so CLI and library callers construct
`StructuredExtractTask(options)` directly. Runtime prompt/schema/name/mode
values must not be stored in a global mutable registry or flattened back into a
string-only `post_process()` result.

The first Editor child explicitly updates `pyproject.toml` and the lockfile from
the current core `<0.4` range to a released version containing typed structured
results. Dependency and lockfile changes are reviewed as part of that child,
not treated as an implicit setup step.

### Python and CLI Surfaces

The Python surface accepts a configured task instance and returns a
`TaskExecutionResult`.

The CLI exposes a dedicated structured extraction command with input, prompt
file, schema file, mode, and output path. It writes valid JSON to the requested
file or machine-readable stdout. Progress and diagnostic messages must not
corrupt the JSON stream.

For this command stdout contains only the final JSON value. Progress and errors
go to stderr, failures return a nonzero exit code, and `--output` uses an atomic
replace so a failed process cannot leave a successful-looking partial file.
Subprocess tests verify these stream and exit-code contracts.

Structured execution does not print partial streaming JSON. It validates the
schema file before provider execution and uses core's semantic planner for
response parsing and validation.

Preflight validation checks the selected JSON Schema metaschema, schema name,
and JSON serializability before creating a successful run or sending provider
HTTP. An invalid schema therefore cannot consume a paid request and cannot be
misreported as a model-response mismatch. The task also reserves output tokens
when checking the context budget.

### Persistence

Editor persists content type explicitly. It must not infer JSON from the first
character of a string. The validated structured value is serialized with a
stable encoding; the raw response and core metadata remain available as
execution evidence.

### Conformance Sample

The functional sample is a sanitized, repository-owned learning-materials
example that preserves the useful constraints of the Java evidence without
copying its business prompt or production schema verbatim.

Required offline coverage includes:

- nested arrays and objects;
- exact item-count validation with a compact fixture;
- enum validation;
- required and additional-property rejection;
- propagation of the parsed value through core, Editor, Python API, CLI, and
  persistence;
- unchanged behavior for all existing text tasks.

### Editor Issue Tree

`editor-assistant#28` becomes a tracking Epic with three ordered children:

1. Add typed task-execution results and compatibility entry points.
2. Add Task, Python, CLI, and persistence structured-output contracts.
3. Add `structured-extract` and the learning-materials conformance sample.

Child 2 depends on child 1. Child 3 depends on child 2. Existing
`editor-assistant#11` moves to Later/Research and is not a blocker.

## ArXiv Atomic Paper-Job Design

### Problem Statement

The current application treats translated title, translated abstract, brief,
and outline as four independent tasks. Brief and outline each send the full
paper, while later aggregation groups unrelated text completions into one JSON
record. The aggregate has no atomic schema or shared prompt/model revision.

Programmatic assembly is not itself wrong. Deterministic source identity and
execution provenance should still be assembled by code. The generated semantic
components should be produced and validated together when context permits.

### Processing Profiles

ArXiv defines two stable profiles:

- `metadata`: title and abstract input; one structured call returns
  `title_zh` and `abstract_zh`; no PDF conversion is required.
- `full`: title, abstract, and cached full-text Markdown input; normally one
  structured call returns `title_zh`, `abstract_zh`, `brief`, and `outline`.

The initial schema keeps `brief` and `outline` as strings so existing exports
remain compatible. Nested outline semantics are deferred until a real
field-level consumer exists.

### Result Envelope

The model produces only the schema-validated `generated` object. The
application writes deterministic source and execution fields:

```json
{
  "schema_version": "1.0",
  "source": {
    "arxiv_id": "2507.00002",
    "title": "Original title",
    "abstract": "Original abstract",
    "authors": [],
    "subjects": [],
    "pdf_url": "https://arxiv.org/pdf/2507.00002"
  },
  "generated": {
    "title_zh": "...",
    "abstract_zh": "...",
    "brief": "...",
    "outline": "..."
  },
  "execution": {
    "model": "...",
    "prompt_version": "...",
    "schema_version": "1.0"
  }
}
```

The exporter continues exposing the existing `title_zh`, `abstract_zh`,
`brief`, and `outline` keys to downstream readers.

### Direct Core Dependency

ArXiv removes its Editor runtime dependency after the new path is validated.
It owns the composite prompt and schema because they describe an ArXiv product
record, not a generic Editor task.

The LLM phase uses an asynchronous queue, an application-owned global rate
limiter, and bounded concurrency with one reusable core client per model. It
must not create one client per paper or rely on client-local core rate state as
a process-wide limit. The PDF download/conversion phase remains a separate,
high-concurrency preprocessing phase. A process pool is not used merely to wait
on LLM HTTP calls.

An offline concurrent-admission test starts multiple coroutines together and
proves that the application limiter still enforces the configured minimum
interval and requests-per-minute boundary.

ArXiv owns a minimal reviewed core catalog for the models it actually runs. Its
prompt and generated schema remain application-owned and versioned.

### Checkpoint V2

The checkpoint unit changes from one component row to one paper/profile job in
a side-by-side v2 table or database; the existing component checkpoint is not
mutated in place. Each reusable result is keyed by at least:

- `arxiv_id`;
- processing profile;
- source digest;
- prompt version;
- schema version;
- model identity.

Changing any correctness-relevant value must not silently reuse a stale
completion. A successful job stores a complete validated result; a failed job
does not store partial semantic output.

On restart, an abandoned `in_progress` v2 job returns to pending under an
explicit recovery rule. Status, attempts, and the complete structured result
commit in one transaction. The v1 checkpoint is read only for comparison and
cannot satisfy a v2 cache lookup.

### Errors and Oversized Input

HTTP, parse, or schema failures retry the atomic paper job under bounded
application policy after core exhausts its request policy. The final failure
records the typed cause and attempt count.

Context budget is checked with an application-owned conservative estimate of
the complete prompt, schema, and output reserve. Only deterministic budget
overflow triggers a split path. Ordinary `full` papers use one call. An
oversized `full` paper uses exactly two structured calls: metadata returns
`title_zh` and `abstract_zh`; document returns `brief` and `outline`. Both must
validate before one transaction marks the paper job complete, and neither half
is exported alone. The selected strategy is recorded in execution metadata.

Comparison evidence records the one-call and split proportions, schema failure
and retry counts, input/output tokens and estimated cost per paper, latency, and
human consistency review across all four generated fields.

### Rollout

ArXiv uses two ordered implementation children:

1. Add the minimal catalog, composite schemas, prompts, profiles, a v2 executor
   behind an explicit opt-in/shadow path, and representative comparison
   evidence.
2. Add async paper jobs, checkpoint v2, compatible export, default cutover,
   and removal of the Editor/legacy aggregator path after validation.

The first change is additive and supplies the rollback path for the second.

## Hormuz Migration Design

Hormuz eventually depends directly on core, but transport replacement and
strict structured classification are intentionally separate changes. The
application continues owning:

- the geopolitical classification prompt and calibration anchors;
- category and duration semantics;
- keyword fallback;
- database writes and signal lifecycle;
- benchmark data and investment-domain thresholds.

The first Hormuz Issue is blocked by the core `system_prompt` prerequisite. It
replaces only provider transport and retry plumbing, retains the existing
tolerant parser and per-item normalization, preserves the current 60-second
failure latency unless explicitly accepted otherwise, and introduces the
minimal caller-owned catalog. This makes transport parity measurable without
changing batch fallback semantics.

A second Issue decides the domain contract for strict structured output. It
must explicitly choose between all-or-nothing schema failure and a looser shape
gate followed by existing normalization. It covers category enums, bounded
relevance, duration enum, actor strings, reasoning, and dynamic batch item
counts. It cannot switch the default until the maintainer confirms the labelled
benchmark, migration comparison meets the accepted thresholds, and a human
domain review finds no prompt/taxonomy drift.

Because output contributes to investment monitoring, the final migration head
requires a fresh independent review even if the code change appears mechanical.

## Open Translate Native-Boundary Design

Open Translate remains a direct browser client. Its workstream audits and, only
where evidence supports a change, aligns:

- provider/model catalog provenance appropriate to browser availability;
- OpenAI-compatible streaming parsing assumptions;
- capability-specific request controls;
- cancellation, timeout, and sanitized error behavior;
- documentation of why Python core is not a runtime dependency.

It must not adopt core's catalog mechanically when a model or endpoint is not
usable from a browser. This workstream may validly conclude that a documented
intentional difference is preferable to code sharing.

The current generated catalog exposes fields such as token limits, rate limits,
and request overrides that the background runtime does not consistently use.
The Issue must either connect each claimed control with tests or remove the
misleading field. Documentation must also describe the actual
`chrome.storage.sync` credential behavior rather than calling it merely local.
ADR and evidence work can begin immediately; behavior-changing transport work
is blocked by the repository's existing E2E gate Issue #18.

## Remaining Consumer Migrations

### `global-science-db`

After existing gsdb Issue #3 stabilizes the schema/prompt source of truth and
the core `system_prompt` prerequisite lands, replace its copied transport with
core structured output. Keep PyMuPDF preprocessing, Pydantic domain models,
task registry, prompts, layered domain validation, and human-review gates in
gsdb. Core schema validation is only a transport/shape gate.

### `cell-press-paper-classifier`

Replace the eligible OpenRouter execution path with one core adapter. Keep
article models, classification schema, batch concurrency, checkpointing, and
CSV output in the application. The native Gemini SDK path remains available
until a separate parity benchmark proves that routing the same model through an
OpenAI-compatible endpoint preserves required behavior; this program does not
assume that equivalence.

### `print-pub-ai`

First add a minimal caller-owned catalog, then migrate text-only review,
translation, and evaluation calls to core. Keep the current provider-native PDF
multimodal extraction as an explicit exception. The PDF path must continue to
resolve its model identity without Editor before the Editor runtime dependency
can be removed. Core's prompt-only API protects `messages`, so adding
multimodal message input would be a separate public-API design requiring
another real consumer.

## Program Issue Graph

The #26 Epic is tracking-only. Each repository receives a local accepted Issue
or local Epic/children before implementation. Repository-local issues own their
objective, scope, non-goals, compatibility, validation, rollback, and ownership
checklist. A separate core child tracks the Program Readiness Audit.

The graph distinguishes:

- **Priority:** maintainer ordering in the Project roadmap.
- **Depends on:** a real artifact or contract that must land first.
- **Related to:** shared program context without a blocking edge.

The accepted work-item tree is:

```text
llm-exec-core #26 (tracking only)
├── Audit Program #26 for implementation readiness
├── Add additive system_prompt support to execution APIs
├── editor-assistant #28 (rewrite existing Issue as Epic)
│   ├── Expose typed task execution results without breaking legacy callers
│   ├── Add Task, Python, CLI, and persistence structured-output contracts
│   └── Add generic structured-extract and an offline conformance sample
├── arxiv-batch-processor Epic: atomic schema-validated paper jobs
│   ├── Introduce composite metadata/full profiles with core structured output
│   └── Migrate durable execution to async paper jobs and checkpoint v2
├── hormuz-watch Epic: migrate classification without semantic drift
│   ├── Move role-preserving transport to core while retaining tolerant parsing
│   └── Decide and validate the structured classification failure contract
├── open-translate: define and enforce the browser-native transport boundary
├── global-science-db: adopt core structured output for TOC tasks
├── cell-press-paper-classifier: migrate eligible routes and benchmark parity
└── print-pub-ai: migrate text-only editorial calls to core
```

The organization Java application gets no Issue or relationship. Editor #11 is
not a child; it stays in Later/Research until a real taxonomy consumer exists.
Existing core #23 is unrelated and remains unchanged.

Expected dependencies are:

- Every first implementation Issue depends on a completed `PASS` Program
  Readiness Audit; this is a program gate, not the maintainer priority order.
- Editor child 2 depends on Editor child 1.
- Editor child 3 depends on Editor child 2.
- ArXiv cutover depends on the additive ArXiv v2 executor and comparison.
- Hormuz transport depends on the new core `system_prompt` API; its structured
  classification decision depends on transport parity and the confirmed
  benchmark.
- gsdb transport depends on core `system_prompt` and gsdb #3.
- Open Translate behavior changes depend on Open Translate #18; its ADR and
  evidence audit do not.
- Repository migrations depend on a named compatible core release, never an
  unreviewed sibling-workspace commit.

ArXiv, Hormuz, and Open Translate do not depend on Editor #28 under this design.

## Roadmap

All program items target completion by Sunday, 2026-07-26. The four named main
workstreams and the core prerequisite are `p1`; priority controls attention and
review order, while technically independent implementation can run in parallel
after readiness passes. Editor #11 remains `p2 / Later` without a target date,
and unrelated core #23 remains unchanged.

| Stage | Target | Outcome |
| --- | --- | --- |
| Program setup and readiness audit | 2026-07-22 | Spec, repository-local issues, relations, roadmap, audit PASS |
| Editor | 2026-07-22 to 2026-07-23 | End-to-end Python and CLI structured output |
| Core role-preserving input | 2026-07-22 to 2026-07-23 | Additive `system_prompt` prerequisite for Hormuz and gsdb |
| ArXiv | 2026-07-23 to 2026-07-24 | Atomic profiles and paper-job cutover |
| Hormuz | 2026-07-24 to 2026-07-25 | Transport parity, then accepted structured semantics |
| Open Translate | 2026-07-25 to 2026-07-26 | Native-boundary audit and scoped optimization |
| Remaining consumers | 2026-07-25 to 2026-07-26 where accepted and independent | Core migrations or documented exceptions |

Project writes happen only after reading the existing Project #5 field and
option IDs. The coordinator must not guess fields or create new fields under
this dispatch. If Start/Target fields already exist, the roadmap uses the dates
above; otherwise it records the available target dates and reports the missing
view capability. Priority order is never encoded as false `blocked-by` edges.

## Validation and Review

### Program Readiness Audit

After the spec, repository-local Issue tree, relationships, and roadmap are
stable, launch one fresh independent read-only auditor. The auditor reviews the
actual planning artifacts and reports `PASS` or `FAIL`, findings, and blockers.
It must not write Issues, Project fields, files, or implementation revisions.

The program audit covers layer ownership, issue completeness, real dependency
edges, role-preserving prompts, catalog ownership, failure semantics,
compatibility, rollback, and non-goals. It does not pre-audit future code heads
or replace repository-local exact-head review.

### Implementation Validation

Every migration establishes an offline baseline before changing transport.
Required tests mock or fake provider HTTP and cover relevant parse, schema,
retry, and compatibility failures. No required test performs a live request.

Public API, compatibility, workflow, security, external-code, catalog, or other
high-risk changes receive a fresh independent read-only review on the actual PR
head. A new commit makes that review stale.

## Rollback Principles

- Editor keeps legacy entry points while the typed path is introduced.
- ArXiv keeps the current executor until the v2 comparison and checkpoint path
  pass; cutover is a separate change.
- Hormuz keeps the keyword fallback and does not remove the old transport until
  benchmark parity is recorded.
- Text-only consumer migrations do not absorb unsupported multimodal behavior.
- Each repository rollback is performed through its own PR; the tracking Epic
  never grants cross-repository merge or direct-push authority.

## Program Acceptance

The program is implementation-ready when:

- the design is reviewed on its exact PR head;
- every planned workstream has a repository-local accepted Issue;
- Project ordering and dates reflect the maintainer priority;
- only real dependencies are marked as blockers;
- Java integration and core multimodal expansion remain excluded;
- the independent Program Readiness Audit reports `PASS` with no unresolved
  blockers;
- Editor's first child has a matching maintainer implementation dispatch.
