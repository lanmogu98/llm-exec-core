# Contributing

This public repository accepts public Issue reports and change requests only.
It does not accept unsolicited code contributions.

Before preparing a fork, patch, or pull request, open a structured public Issue
and wait until the maintainer accepts it and gives an explicit maintainer
dispatch. The plain-language dispatch must name the work item, contributor or
implementation session, exact semantic scope, and permitted branch/PR delivery.
It does not change the proprietary license or grant permission beyond that
dispatch.

An Issue defines the objective, scope, non-goals, acceptance criteria,
compatibility and security impact, validation, rollback, and authority
boundaries. A material change to objective, semantic scope, behavior,
compatibility, security, or authority requires a new dispatch. Dates, links,
formatting, typo corrections, evidence refreshes, and status bookkeeping do not.

Unauthorized external pull requests are closed without checkout, download,
build, test, execution, workflow approval, or substantive review. First-time
and previously known external contributors follow the same rule. The maintainer
approves an external fork workflow only after checking the current dispatch and
workflow-file diff.

For a dispatched pull request:

- use one Issue, session, worktree, `codex/` branch, and PR for one concern;
- link the accepted Issue and dispatch;
- keep changes inside the dispatched semantic scope and preserve compatibility;
- use mocked/fake provider calls and never include keys, tokens, `.env`, or live
  provider requests;
- run the commands listed in `AGENTS.md` and record the exact PR head;
- obtain a new independent exact-head review for workflow, permission, secret,
  security, external-code, breaking-public-API, or catalog work; and
- never update `main` directly, enable auto-merge, merge, publish, or change
  settings, secrets, or permissions.

Required CI must pass on the exact head. Only `lanmogu98` merges through the
GitHub UI. Rollback uses a focused revert PR, never direct push or bypass.

Security vulnerabilities must not be filed publicly. Use
[private vulnerability reporting](https://github.com/lanmogu98/llm-exec-core/security/advisories/new)
and follow [SECURITY.md](SECURITY.md). Advisory access by itself does not
authorize code; an explicit maintainer dispatch inside the advisory is required.

Public visibility grants no rights beyond [LICENSE](LICENSE).
