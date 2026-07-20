# Security policy

## Supported versions

Security fixes target the current `main` branch and the latest published package
version when a release exists. Older versions are not guaranteed to receive
backports.

## Report privately

Do not disclose a suspected vulnerability in a public Issue, discussion, pull
request, or log. Use GitHub's
[private vulnerability reporting form](https://github.com/lanmogu98/llm-exec-core/security/advisories/new).
Include affected versions, impact, reproduction details, and a safe contact
method. Do not include real provider credentials or unrelated user data.

The maintainer will acknowledge the report when practicable, assess severity and
scope, coordinate validation/remediation privately, and provide status updates
when there is material progress. No fixed response or disclosure deadline is
promised.

The advisory is the private semantic work item. It records the objective, scope,
non-goals, compatibility/security impact, acceptance, validation, rollback,
reporter, collaborators, authorized contributors, and intended implementer.
Reporter or collaborator access does not authorize code.

Do not prepare or submit a patch unless `lanmogu98` gives an explicit maintainer
dispatch inside the advisory naming the work item, contributor or implementation
session, exact semantic scope, and branch/PR delivery. Unauthorized submitted
code is not inspected or executed.

Security work requires a new independent read-only reviewer on the actual private
PR head. The reviewer must not have implemented that head and reports a concise
`PASS` or `FAIL`, findings, and blockers. Validate the recorded head in an
isolated local environment with mocked/fake provider access and the canonical
commands in `AGENTS.md`. A new commit requires a new review.

Only the maintainer merges through the advisory workflow and coordinates
sanitized disclosure, release, and publication. Rollback is a focused revert PR;
direct push, bypass, and disclosure of confidential evidence are prohibited.
