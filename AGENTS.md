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

## Issue contract and dispatch

Public GitHub Issues are the only public intake channel. An accepted Issue is
the semantic source of truth and defines the objective, scope, non-goals,
acceptance criteria, compatibility and security impact, validation, rollback,
and authority boundaries. Labels, Project state, and ordinary comments record
progress; do not put workflow-state bookkeeping in the Issue body.

Implementation requires an explicit plain-language maintainer dispatch that
names:

- the Issue or private advisory work item;
- the semantic scope;
- the implementation actor or session;
- the topic branch and pull-request delivery; and
- the withheld owner-only actions.

The dispatch must match the accepted Issue. It never grants merge, auto-merge,
release or package publication, settings, secrets, permissions, protected
deployment, bypass, or direct `main` authority.

A new dispatch is required when the objective, semantic scope, behavior,
compatibility, security, or authority changes materially. Dates, links,
formatting, typo corrections, evidence refreshes, labels, and status updates do
not invalidate an otherwise matching dispatch. Machines may gather and verify
mechanical evidence; semantic dispatch remains a maintainer decision.

## Implementation and review

Ordinary source, test, documentation, and configuration work requires an
accepted Issue, a matching dispatch, tests, a topic branch and PR, required CI
on the exact head, review of the actual diff and artifacts, and maintainer UI
merge. A universal pre-implementation audit is not required.

Workflow, permission, secret, security, external-code, breaking-public-API, and
model-catalog changes are high risk. They additionally require a new independent
read-only reviewer on the actual PR head. The reviewer must not have implemented
that head and reports a concise `PASS` or `FAIL`, findings, and blockers. A new
commit makes that review stale. Unresolved blockers prevent merge.

Keep one Issue, implementation session, worktree, `codex/` topic branch, and PR
per concern. Preserve backward compatibility unless the accepted Issue
explicitly authorizes a breaking change. Changes to `llm_config.yml` require
dated source evidence, fixture updates, explicit alias/fallback analysis, and
offline tests. Do not silently refresh the lockfile; dependency and `uv.lock`
diffs must be in scope and reviewed.

## Evidence and program ordering

For official web evidence, record the source URL, retrieval date, and verified
semantic facts. Do not use representation hashes for web pages, Issues,
comments, reports, or attestations. SHA-256 remains appropriate for built
release artifacts; the commit SHA identifies the exact PR head; package-manager
lock integrity remains native to the package manager.

Only completed upstream stages block a child work item. Refresh downstream
evidence and assumptions when that child's turn arrives instead of fully
auditing future work in advance. A parent and child must not duplicate approval
of the same semantic decision.

## External contributions

A visible repository is not an invitation to submit code. An outside fork,
patch, or pull request requires an accepted public Issue and an explicit
maintainer dispatch naming that contributor or session, exact semantic scope,
and permitted branch/PR delivery before code preparation or submission. The
dispatch does not relicense the project or transfer to another person, Issue,
scope, or delivery.

Do not checkout, download, inspect substantively, build, test, execute, or
approve a workflow for unauthorized external code. Close an unauthorized PR
with a link to `CONTRIBUTING.md`. First-time and previously known contributors
follow the same rule. After valid dispatch, external code remains high risk and
requires independent exact-head review. The maintainer approves an external
fork workflow only after checking the current dispatch and workflow diff.

## Private security work

Security reports use GitHub Private Vulnerability Reporting, never a public
Issue. The advisory is the private semantic work item and should state the
objective, scope, non-goals, compatibility/security impact, acceptance,
validation, rollback, reporter, collaborators, authorized contributors, and
intended implementer.

Advisory access does not authorize code. Each reporter or collaborator who will
prepare a patch needs an explicit maintainer dispatch inside the advisory that
names the work item, actor/session, semantic scope, and branch/PR delivery.
Unauthorized submitted code is not inspected or executed.

Security implementation is high risk. A new independent read-only reviewer who
did not implement the reviewed head must review the actual private PR head and
report `PASS` or `FAIL`, findings, and blockers. Validate the recorded head in
an isolated local environment with the canonical commands; provider access is
mocked or faked. The maintainer alone merges through the advisory workflow and
coordinates sanitized disclosure and publication.

An active-exploitation exception must be private, owner-authored, time-bounded,
name the waived step and reason, owner, expiry, rollback, and retrospective
independent review due within 24 hours. Agents cannot authorize it, and direct
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
variable/secret settings, approve protected deployments, or publish tags,
releases, or packages. Those actions require a distinct explicit maintainer
instruction even if a connector exposes them. Agents need no PAT, deploy key,
server SSH credential, or host-local credential metadata.

All work reaches `main` through a PR. The branch ruleset has zero required human
approvals for this single-maintainer repository, requires resolved conversations,
blocks force-push/deletion, and has no bypass actors. Auto-merge stays off. The
maintainer reviews and merges through GitHub UI only after exact-head evidence.

## CI, secrets, and compatibility

Required checks are uniquely named `CI / python-3.10` and
`CI / python-3.13`. They run on PRs targeting `main` and pushes to `main` with a
read-only token, full-SHA actions, locked dependencies, no path filters, no
`pull_request_target`, no provider/release secrets, and no live request. CI may
never merge, approve external workflows, change settings, or write repository
contents.

Provider HTTP is mocked/faked in required tests. Never inject live provider keys
into agent or CI processes, print secret values, commit `.env`, or create a live
provider workflow/environment. A future live-provider test needs a separate
accepted Issue, matching dispatch, controlled runner/secret design, maintainer
approval, and rotation/revocation plan.

## Bootstrap recovery

The preliminary no-bypass `main` ruleset remains active as the recovery anchor.
Missing/not-started, skipped, cancelled, timed-out/stuck beyond 30 minutes,
failed/error, duplicate/wrong check names, wrong SHA/source, or a final-setting
readback failure freezes all merges. Owner `lanmogu98` may retry once, then uses
a focused repair PR. If not green and fully verified within 24 hours of the
bootstrap merge, revert through a PR and leave protected `main` as default.

If preliminary-ruleset readback itself fails, treat the anchor as unverified and
restore-required: freeze both merges and PR recovery until the exact exported
preliminary configuration is restored and verified.

The final required-check ruleset is separate. If it is wrong or cannot be read
back, disable/delete only that final ruleset through the owner settings plane
and verify the exported preliminary configuration remains active before any PR
repair. If rollback cannot be verified, freeze merges and restore the
preliminary settings; never direct-push or bypass.

## Definition of Done

- The change matches one accepted Issue, one matching dispatch, and one concern.
- Tests were added first where behavior changed, and all canonical commands pass.
- Provider calls are mocked/faked; no secret or live-provider configuration is
  present.
- Public API, model-catalog evidence, docs, security, and downstream effects are
  reviewed and recorded.
- The branch is a topic branch, the commit is conventional, and the PR links the
  Issue and dispatch, and records commands/results, risk, rollback, and exact
  head.
- Any high-risk change has a new independent `PASS` review on that exact head.
- Required checks are green on that head. Only the maintainer merges in GitHub UI.
- Handoff lists every modified file, test result, commit SHA, branch, PR URL,
  assumptions, and remaining risks; the worktree is clean.
