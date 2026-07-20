# Governance tabletop evidence

These cases are synthetic and offline. They do not create external accounts,
submit code, approve a workflow, inspect credentials, call a provider, or
exercise a live security advisory. Executable cases live in
`tests/governance/test_governance_contract.py`.

## Dispatch cases

| Case | Issue and dispatch state | Required result |
|---|---|---|
| Ordinary source/test/docs/config change | Accepted Issue; dispatch matches scope, actor/session, and branch/PR delivery | Implementation may proceed; tests, exact-head CI/review, and owner UI merge remain required |
| Objective, scope, behavior, compatibility, security, or authority changes materially | Existing dispatch predates the semantic change | Stop and obtain a new maintainer dispatch |
| Date, link, formatting, typo, evidence, or status changes | Existing dispatch still matches semantic work | Preserve dispatch; continue without ceremony |
| High-risk PR | Matching dispatch; workflow, permissions, secrets, security, external code, breaking API, or catalog is involved | New independent read-only `PASS` review on the actual PR head |
| High-risk PR head changes | Previous review names an older commit | Review is stale; obtain a new independent review |
| Owner-only action | Implementation agent is otherwise fully dispatched | Deny merge, auto-merge, release/publication, settings, secrets, permissions, protected deployment, bypass, and direct `main` update |

No Issue/comment hash, attestation chain, timestamp chain, or pre-implementation
universal audit is part of these cases. The Issue is the semantic source of
truth; the commit SHA identifies the actual PR head.

## External and security boundaries

| Contributor | Accepted work item and matching dispatch | Required handling |
|---|---:|---|
| First-time external | No | Do not inspect or execute submitted code; close the PR |
| First-time external | Yes | Check current scope and workflow diff; treat code as high risk |
| Previously known external | No | Do not inspect or execute submitted code; close the PR |
| Previously known external | Yes | Apply the same current-scope and high-risk review rules |
| Security reporter/collaborator | Advisory access only | Access is not code authority; require an advisory dispatch |
| Security implementer | Matching advisory dispatch | Isolated validation plus independent review on the actual private PR head |

The independent reviewer must not have implemented the reviewed head and
reports concise `PASS` or `FAIL`, findings, and blockers. The maintainer alone
merges through the GitHub UI or advisory workflow and controls coordinated
publication. Unauthorized external code is never checkout, downloaded,
substantively inspected, built, tested, executed, or approved for workflow use.

## Evidence and ordering cases

- Official web facts record the URL, retrieval date, and verified semantic facts;
  representation hashes are not governance evidence.
- SHA-256 is retained for built release artifacts, commit SHA for exact PR-head
  identity, and package-manager-native lock integrity for dependencies.
- Only completed upstream stages block a child. Refresh downstream evidence
  when that child's turn arrives; do not pre-audit future work.
- Parent and child work items do not duplicate approval of the same semantic
  decision.
- Rollback is a focused revert PR. Direct push and bypass are never recovery
  paths.

## Bootstrap recovery matrix

| Observed condition | Detection | Required terminal state |
|---|---|---|
| Missing or not-started check | 30 minutes | Freeze merges; preliminary ruleset active |
| Skipped, cancelled, timed out, or stuck check | 30 minutes | Freeze merges; preliminary ruleset active |
| Failure or error | Immediate / 30 minutes max | Retry once, then focused repair PR |
| Wrong or duplicate check name | 30 minutes | Freeze merges; repair workflow through PR |
| Check bound to wrong SHA | 30 minutes | Freeze merges; require exact-head evidence |
| Preliminary ruleset readback fails | Final readback | Mark anchor unverified; freeze PR recovery until restored |
| Nonexistent/wrong-source final required check | Final readback | Disable/delete only final ruleset; verify preliminary anchor |
| Any other final-setting readback failure | Final readback | Freeze merges; retry/repair/revert path |

Owner is always `lanmogu98`; detection is bounded at 30 minutes and repair or
revert at 24 hours after bootstrap merge. If final ruleset rollback cannot be
verified, no PR repair begins: restore and re-read the preliminary settings
through the maintainer settings plane first. Every terminal state keeps
protected `main` under the preliminary no-bypass ruleset.
