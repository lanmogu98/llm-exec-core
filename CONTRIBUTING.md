# Contributing

This public repository accepts public Issue reports and change requests only.
It does not accept unsolicited code contributions.

Before preparing a fork, patch, or pull request, open a structured public Issue
and wait for a prior owner-authored written authorization in that Issue. The
authorization must name you, the exact scope, allowed delivery, issue permalink,
issue time, and expiry/completion boundary. It does not change the proprietary
license or grant permission beyond the authorized contribution.

```yaml
CONTRIBUTION-AUTHORIZATION:
  contributor: github-login
  work_item: permalink
  allowed_scope: exact scope
  allowed_delivery: fork/branch/PR description
  issued_at: timestamp
  expires_at: timestamp-or-completion
```

After authorization, the Issue must be completed as an audit-ready work
contract, pass an independent adversarial Gate 0 audit, and receive a separate
maintainer attestation before implementation starts. Authorization for one
Issue, person, scope, or delivery does not transfer to another.

Unauthorized external pull requests are closed without checkout, download,
build, test, execution, workflow approval, or substantive review. First-time
and previously-known external contributors both require owner approval before
fork workflows run. The maintainer approves a run only after revalidating the
authorization and workflow-file diff for the current delivery.

For an authorized pull request:

- use one PR for one concern;
- link the accepted Issue, PASS report, maintainer attestation, revision hash,
  and authorization permalink;
- keep changes inside the authorized scope and preserve compatibility;
- use mocked/fake provider calls and never include keys, tokens, `.env`, or live
  provider requests;
- run the commands listed in `AGENTS.md`; and
- never update `main` directly or enable auto-merge.

Security vulnerabilities must not be filed publicly. Use
[private vulnerability reporting](https://github.com/lanmogu98/llm-exec-core/security/advisories/new)
and follow the private process in [SECURITY.md](SECURITY.md). Advisory access by
itself does not authorize code.

Public visibility grants no rights beyond [LICENSE](LICENSE).
