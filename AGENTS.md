# AGENTS.md — llm-exec-core project contract

## Project

`llm-exec-core` is a provider-agnostic Python package for asynchronous LLM
execution, model catalog loading, streaming assembly, and typed result/usage
objects. It is proprietary source that is publicly visible and consumed by
other repositories.

- Supported package runtime: Python 3.10 and newer.
- Required CI representatives: Python 3.10 and 3.13.
- Package manager: `uv`; `uv.lock` is authoritative and must remain locked.
- Public API lives under `src/llm_exec_core/`.
- GitHub Issues are the source of truth for accepted work.

## Current shape

| Area | Responsibility |
|---|---|
| `src/llm_exec_core/` | Package implementation, types, model configuration, and `py.typed` marker |
| `src/llm_exec_core/llm_config.yml` | Reviewed provider/model catalog and capability evidence |
| `tests/unit/` | Offline product tests with provider HTTP mocked or faked |
| `tests/fixtures/` | Tracked deterministic test snapshots |
| `tests/governance/` | Offline governance-contract and workflow tests |
| `scripts/` | Explicit maintenance and offline governance-check helpers |
| `docs/design_docs/` | Public API and compatibility design records |
| `docs/governance/` | Tabletop governance and recovery evidence |
| `.github/` | Structured intake, PR guidance, and mock-only CI |

## Canonical commands

```bash
uv sync --locked --group dev
uv run --frozen pytest -q
uv run --frozen black --check src tests
uv run --frozen flake8 src tests
uv run --frozen mypy src
uv build
```

Formatting changes use `uv run --frozen black src tests`. Do not run the live
mode of `scripts/check_openrouter_capabilities.py` as part of ordinary tests;
use its tracked fixture mode or the unit test.

## Change discipline

1. Clarify the accepted Issue scope and map exact file insertion points before
   editing.
2. Touch only files required by the accepted Issue. Do not add abstractions,
   cleanup, logging, comments, or error handling unrelated to it.
3. For behavior changes, write tests first and observe the expected failure.
   Every bug fix requires a reproducing regression test.
4. Review correctness, downstream compatibility, security, and scope after the
   change. Public API changes require explicit compatibility analysis.
5. Run all canonical commands and deliver a clean commit plus exact handoff
   evidence.

Never discard or rewrite unrelated user changes. Use conventional commits:
`feat`, `fix`, `refactor`, `docs`, `test`, or `chore`.

## Task entry and contribution authorization

Public GitHub Issues are the only public intake channel. A visible repository
is not an invitation to submit code. Any outside fork, patch, or pull request
requires a prior owner-authored durable authorization in its accepted public
Issue (or private advisory for confidential work):

```yaml
CONTRIBUTION-AUTHORIZATION:
  contributor: github-login
  work_item: permalink
  allowed_scope: exact scope
  allowed_delivery: fork/branch/PR description
  issued_at: timestamp
  expires_at: timestamp-or-completion
```

The record must predate code preparation and delivery and match the contributor,
work item, scope, and delivery. It does not relicense the project. Do not
checkout, download, inspect substantively, build, test, execute, or approve a
workflow for unauthorized external code. Close the PR with a link to
`CONTRIBUTING.md`. Previously known external contributors follow the same rule.
The owner comment must remain unedited (`created_at == updated_at`), its declared
`issued_at` must equal the API `created_at`, and that durable timestamp must be
strictly earlier than the earliest preparation/delivery event being authorized.
Timestamp expiry is inclusive. For `expires_at: completion`, record completion
state explicitly and reject any new preparation/delivery after completion.

Maintainer/API-created or incomplete Issues may exist for triage, but they do
not authorize implementation. Accepted work must be audit-ready and carry one
of `gate0:pending`, `gate0:failed`, or `gate0:passed`; the label mirrors the
evidence and never replaces it.

## Gate 0 public workflow

### States and permitted work

Before PASS, only read-only discovery, contract drafting/remediation, audit
preparation, and independent audit are allowed. Do not assign an implementation
agent, create an implementation branch, change tracked files or settings, or
execute external code. A material change to objective, scope, non-goals,
acceptance, permissions, security, compatibility, order, CI/governance,
rollback, or recovery resets the Issue to pending and requires a fresh audit.

A valid dispatch requires all of the following on the exact Issue revision:

1. a fresh independent adversarial Agent report with `verdict: PASS`;
2. a separate unedited owner-authored maintainer attestation;
3. recomputed Issue and report hashes that match;
4. report and attestation comments that still exist and have
   `created_at == updated_at`; and
5. `gate0:passed` as the single Gate 0 label.

PASS establishes readiness only; it grants no operation by itself. Every
implementation action must also be in the dispatched task and the contribution
plane for that actor. Merge, auto-merge, settings, secrets, bypass, and
publishing remain denied to implementation agents even if a callable tool or a
malformed task-action set names them.

Any missing, edited, deleted, stale, mismatched, or unverified blocking evidence
fails closed. A FAIL is remediated in the Issue body, then a new independent run
audits the new hash. Never edit old reports or attestations.

### Roles and independence

Each audit record names `maintainer_trust_root`, `issue_publisher`, all
`contract_authors`, all `contract_remediators`, `intended_implementer`,
`auditor_run_id`, and `orchestrator_task_id`. The auditor must be a fresh run
that is not a publisher, author, remediator, intended implementer, or a
continuation of those roles. It is read-only, receives the frozen public
contract and sanitized repository evidence, performs no writes, does not inspect
host-local credentials, and can never implement that revision. Inaccessible
evidence is reported as unverified, not inferred safe.

### Canonical hashes

Use `issue_revision_sha256` everywhere. Fetch API `title` and `body` (`null`
body becomes empty), replace CRLF and lone CR with LF independently, and hash
UTF-8 bytes of `normalized_title + "\n\n" + normalized_body` with SHA-256.
Do not trim, append a newline, add a BOM, or normalize Unicode. Comments use the
same rules over the API-returned body alone. Labels, comments, reactions, and
`updated_at` are not part of the Issue revision.

Required vectors:

- title `T`, body `B` ->
  `c467a8e4991deeb4a45eda82181eeb4bf23107e7798dd197cb5432cf9df3f268`
- title `T`, body `A\r\nB` ->
  `72a94906a55d109eee887e883a28c7b9f87bb0ab740b88241827da110f8063f9`
- comment body `B` ->
  `df7e70e5021544f4834bbee64a9e3789febc4be81470df629cad6ddb03320a5c`

### Public evidence

The maintainer publishes the auditor's capability-safe report verbatim, then
posts a separate owner attestation containing the Issue hash, run/task IDs,
roles, report permalink, report body hash, report timestamp, and verdict.
Reports must not contain secret values or host-local credential paths, scopes,
identity counts, or unrelated usernames. Immediately before merge, re-fetch
and verify the Issue, PASS report, attestation, authors, timestamps, hashes,
roles, and exact PR head.

The report uses this schema:

```yaml
gate0_version: 1
issue: owner/repository#N
issue_revision:
  observed_updated_at: timestamp
  issue_revision_sha256: sha256
repository_revision:
  default_branch: name
  base_commit: sha
roles:
  maintainer_trust_root: lanmogu98
  issue_publisher: identity
  contract_authors: [identity]
  contract_remediators: [identity]
  intended_implementer: identity-or-unassigned
  auditor_run_id: id
  orchestrator_task_id: id
independence_attestation: text
limitations: []
verdict: PASS|FAIL
findings: []
blocking_findings: []
```

The owner attestation is a separate comment:

```yaml
GATE0-MAINTAINER-ATTESTATION:
  issue_revision_sha256: sha256
  auditor_run_id: id
  orchestrator_task_id: id
  report_comment_url: permalink
  report_comment_sha256: sha256
  report_created_at: timestamp
  roles:
    issue_publisher: identity
    contract_authors: [identity]
    contract_remediators: [identity]
    intended_implementer: identity-or-unassigned
  verdict: PASS|FAIL
```

The adversarial report covers scope and non-goals; dependencies and ordering;
compatibility and security; role separation; edit/deletion/forgery paths;
public permissions, apps and collaborators; fork and external-code boundaries;
CI provenance, names, permissions, secrets and self-modification; confidential
remediation; rollback/recovery; and material-change invalidation. Any unresolved
Critical or High is FAIL. Medium blocks unless the report accepts an explicit
disposition with rationale, risk owner, and follow-up trigger/deadline.

## Private security Gate 0

Security reports use GitHub Private Vulnerability Reporting, never a public
Issue. The advisory is the private work-item source of truth. Record the
reporter and sorted collaborators. Advisory access does not authorize code;
each reporter/collaborator patch needs a prior owner-authored
`CONTRIBUTION-AUTHORIZATION` inside the advisory before inspection or execution.

The owner first posts an immutable `PRIVATE-GATE0-CONTRACT-V1` discussion
comment containing advisory ID, objective, scope, non-goals, compatibility and
security impact, acceptance, validation, rollback/recovery, reporter, sorted
collaborators, authorized contributors, and intended implementer. Compute
`private_contract_sha256` over its exact API body using the comment hash rule.
Dispatch a fresh independent sanitized read-only audit. The report and separate
owner attestation are new advisory comments with the public schema adapted to
the private hash.

```yaml
PRIVATE-GATE0-CONTRACT-V1:
  advisory_id: GHSA-id
  objective: text
  scope: []
  non_goals: []
  compatibility_security_impact: text
  acceptance_criteria: []
  validation: []
  rollback_recovery: text
  reporter: github-login
  collaborators_sorted: [github-login]
  authorized_code_contributors: []
  intended_implementer: identity-or-unassigned
```

The adapted report must parse as a complete schema and bind the exact contract
digest, advisory, access roles, intended implementer, auditor run ID,
orchestrator task ID, and PASS verdict. The immutable owner attestation binds
those same values plus the report's stable API comment ID, permalink, immutable
body digest, author and association, created/updated timestamps, and exact head.

Immediately before merge, the owner re-fetches and re-hashes the contract,
report, and attestation, checks sorted access and exact head, then posts a fourth
immutable owner-authored `PRIVATE-GATE0-OWNER-VERIFICATION-V1` comment. Its body
binds the exact frozen digest, API comment ID, URL, author, association, and
created/updated timestamps for the contract, report, and attestation; reporter,
sorted collaborators, authorized contributors; the complete role map including
auditor/orchestrator IDs; and the exact head. Its `created_at` must be strictly
later than the attestation's. The supplied independent snapshot also freezes the
verification comment's own digest, ID, URL, author, association, and timestamp.
All four evidence comments must exist and satisfy `created_at == updated_at`.
Matching substrings or a coordinated replacement that recomputes every local
body and downstream digest never satisfies the independent frozen snapshot.

Any contract/report/attestation/verification edit, deletion, missing comment,
provenance mismatch, reporter/collaborator change, authorization change,
contract change, or private-fork head change invalidates PASS. Temporary private
security forks do not supply normal required-check evidence; run the canonical
commands in an isolated local environment against the recorded head, record
base/head/commands/results and rollback privately, and let only the maintainer
use the advisory merge action. Publish only sanitized notes after coordinated
disclosure.

An active-exploitation exception must be private, owner-authored, time-bounded,
name the waived step and reason, owner, expiry, rollback, and retrospective
independent audit due within 24 hours. Agents cannot authorize it, and direct
push remains prohibited while GitHub is operational.

## GitHub authority boundary

Codex and Claude GitHub Apps/connectors may be used for an explicitly dispatched
task's contribution plane: read evidence; create/update operational Issues and
comments; create and push `codex/issue-<number>-<slug>` topic branches; change
source, tests, docs, config, and workflow files within scope; create/update PRs;
and inspect, trigger, or retry mock-only CI. Connector action exposure is not
authorization.

Only `lanmogu98` may merge, queue, enable auto-merge, bypass or update `main`,
change repository/ruleset/branch/Actions/App/collaborator/webhook/environment/
variable/secret settings, approve protected deployments, attest Gate 0, issue
owner-only authorization records, or publish tags/releases/packages. Those
actions require a distinct explicit maintainer instruction even if a connector
technically exposes them. Agents need no PAT, deploy key, server SSH credential,
or host-local credential metadata.

All work reaches `main` through a PR. The branch ruleset has zero required human
approvals for this single-maintainer repository, requires resolved conversations,
blocks force-push/deletion, and has no bypass actors. Auto-merge stays off. The
maintainer reviews and merges through GitHub UI only after exact-head evidence.

Any PR touching `.github/workflows/**` must be explicitly authorized by the
audited Issue. After reviewing the exact diff, the maintainer posts the unedited
comment `WORKFLOW-CHANGE-AUTHORIZED: <head SHA>`. A new commit invalidates it.
Never merge a workflow change without a matching current-head comment, even if
checks are green.

## CI, secrets, and compatibility

Required checks are uniquely named `CI / python-3.10` and
`CI / python-3.13`. They run on PRs targeting `main` and pushes to `main` with a
read-only token, full-SHA actions, locked dependencies, no path filters, no
`pull_request_target`, no provider/release secrets, and no live request. CI may
never merge, approve external workflows, attest Gate 0, or write repository
contents. Both first-time and previously-known external fork runs await owner
approval, which occurs only after current authorization and workflow diffs are
checked.

Provider HTTP is mocked/faked in required tests. Never inject live provider keys
into agent or CI processes, print secret values, commit `.env`, or create a live
provider workflow/environment. A future live-provider test needs a separate
public Issue, Gate 0 audit, controlled runner/secret design, maintainer approval,
and rotation/revocation plan.

Preserve backward compatibility unless an accepted Issue explicitly authorizes
a breaking change. Changes to `llm_config.yml` require dated source evidence,
fixture updates, explicit alias/fallback analysis, and offline tests. Do not
silently refresh the lockfile; dependency changes and `uv.lock` diffs must be in
scope and reviewed.

## Bootstrap recovery

The preliminary no-bypass `main` ruleset remains active as the recovery anchor.
Missing/not-started, skipped, cancelled, timed-out/stuck beyond 30 minutes,
failed/error, duplicate/wrong check names, wrong SHA/source, or a final-setting
readback failure freezes all merges. Owner `lanmogu98` may retry once, then uses
a focused repair PR. If not green and fully verified within 24 hours of bootstrap
merge, revert through a PR and leave protected `main` as default.
If preliminary-ruleset readback itself fails, treat the anchor as unverified and
restore-required: freeze both merges and PR recovery until the exact exported
preliminary configuration is restored and verified.

The final required-check ruleset is separate. If it is wrong or cannot be read
back, disable/delete only that final ruleset through the authorized settings
plane and verify the exported preliminary configuration remains active before
any PR repair. If rollback cannot be verified, freeze merges and restore the
preliminary settings; never direct-push or bypass.

## Definition of Done

- The change matches one accepted, current Gate 0-passed Issue and one concern.
- Tests were added first where behavior changed, and all canonical commands pass.
- Provider calls are mocked/faked; no secret or live-provider configuration is
  present.
- Public API, model-catalog evidence, docs, security, and downstream effects are
  reviewed and recorded.
- The branch is a topic branch, the commit is conventional, and the PR links the
  Issue, PASS report, owner attestation, revision hash, commands/results, risk,
  rollback, and exact head.
- Workflow changes have current-head owner authorization. Required checks are
  green on that head. Only the maintainer merges through GitHub UI.
- Handoff lists every modified file, test result, commit SHA, branch, PR URL,
  assumptions, and remaining risks; the worktree is clean.
