# Governance tabletop evidence

These cases are synthetic and offline. They do not create external accounts,
submit code, approve a workflow, inspect credentials, call a provider, or
exercise a live security advisory. Executable cases live in
`tests/governance/test_governance_contract.py`.

## External fork event states

| Contributor | Prior exact authorization | Fork workflow | Maintainer action |
|---|---:|---|---|
| First-time external | No | Awaiting approval | Do not inspect/execute; close the PR |
| First-time external | Yes, current | Awaiting approval | Re-check author/scope/time and workflow diff, then owner may approve |
| Previously-known external | No | Awaiting approval | Do not inspect/execute; close the PR |
| Previously-known external | Yes, current | Awaiting approval | Re-check author/scope/time and workflow diff, then owner may approve |

The offline authorization fixture proves that an owner-authored record must
predate hypothetical delivery and match contributor and work item. Repository
setting readback, not this table, proves GitHub is configured with
`approval_policy=all_external_contributors` before CI is reachable.

## Gate 0 and workflow-change cases

- Pending Gate 0 permits contract read/edit, audit, and remediation only.
- Pending Gate 0 rejects implementation assignment/branch, tracked writes,
  settings changes, and external-code execution.
- PASS permits scoped implementation only after all immutable evidence matches.
- A synthetic CI replacement that returns success is still not mergeable unless
  the owner posts exactly `WORKFLOW-CHANGE-AUTHORIZED: <current head SHA>`.
- A new head SHA, edited comment, non-owner comment, or extra comment text
  invalidates workflow authorization; only the maintainer may merge in the UI.

## Private advisory cases

The synthetic advisory fixture computes the exact private contract digest with
Python `hashlib` and OpenSSL. Its unchanged baseline passes. Each of these
mutations fails closed independently:

- reporter changes;
- sorted collaborator snapshot changes;
- contract/report/attestation is edited or deleted;
- contract body changes without matching immutable evidence;
- authorized contributor set changes; or
- private-fork head SHA changes.

The tabletop does not claim normal GitHub checks run in a temporary private
security fork. The required path is isolated execution of the canonical command
set against the recorded head, private evidence, maintainer-only advisory merge,
and sanitized coordinated disclosure.

## Bootstrap recovery matrix

| Observed condition | Detection | Required terminal state |
|---|---|---|
| Missing or not-started check | 30 minutes | Freeze merges; preliminary ruleset active |
| Skipped, cancelled, timed out, or stuck check | 30 minutes | Freeze merges; preliminary ruleset active |
| Failure or error | Immediate / 30 minutes max | Retry once, then focused repair PR |
| Wrong or duplicate check name | 30 minutes | Freeze merges; repair workflow through PR |
| Check bound to wrong SHA | 30 minutes | Freeze merges; require exact-head evidence |
| Nonexistent/wrong-source final required check | Final readback | Disable/delete only final ruleset; verify preliminary anchor |
| Any other final-setting readback failure | Final readback | Freeze merges; retry/repair/revert path |

Owner is always `lanmogu98`; detection is bounded at 30 minutes and repair or
revert at 24 hours after bootstrap merge. If final ruleset rollback cannot be
verified, no PR repair begins: restore and re-read the preliminary settings
through the maintainer settings plane first. Every terminal state keeps
protected `main` under the preliminary no-bypass ruleset.
