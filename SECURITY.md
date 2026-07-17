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

The security advisory is the private work item. Reporter/collaborator access
does not authorize code. Do not prepare or submit a patch unless `lanmogu98`
first posts an owner-authored `CONTRIBUTION-AUTHORIZATION` in the advisory that
names you, the exact scope and delivery, and its expiry. Unauthorized submitted
code is not inspected or executed.

Confidential work follows the private Gate 0 contract in `AGENTS.md`: immutable
owner contract, fresh independent read-only audit, separate owner attestation,
exact-body hashes, access/head revalidation, isolated offline checks, and
maintainer-only advisory merge. Any evidence, access, authorization, contract,
or head mutation invalidates PASS. Public disclosure is sanitized and
coordinated after remediation.
