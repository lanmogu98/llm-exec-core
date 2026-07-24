# PyPI release governance

This runbook defines the implementation and owner boundaries for
`.github/workflows/release.yml`. Public PyPI is the package distribution
authority. A GitHub Release is optional release-note presentation and is not a
dependency, integrity, or provenance source.

## Fixed identity and inputs

The only accepted Trusted Publisher identity is:

- GitHub owner/repository: `lanmogu98/llm-exec-core`;
- workflow file: `release.yml`, whose repository path is
  `.github/workflows/release.yml`; and
- GitHub Actions environment `pypi`.

PyPI's publisher and Integrity APIs expose the workflow filename as
`release.yml`. The Sigstore certificate is additionally required to identify
the full workflow path at `refs/heads/main`, the protected environment, and the
exact source commit SHA.

Every manual dispatch supplies an exact normalized stable epoch-0 `version`, an
exact 40-character current `main` `expected_sha`, and `dry_run`. The workflow
rejects another ref, a stale `main`, a nonzero PEP 440 epoch, disagreement among
source/lock/distribution versions, or a version already present on PyPI.

The release toolchain is deliberately fixed:

- `actions/checkout` 7.0.1;
- `actions/upload-artifact` 7.0.1;
- `actions/download-artifact` 8.0.1;
- `astral-sh/setup-uv` 9.0.0;
- `pypa/gh-action-pypi-publish` 1.14.1;
- uv 0.11.31;
- release Python 3.13.14, with smoke coverage on Python 3.10.20 and 3.13.14;
- setuptools 83.0.0 with accepted wheel and sdist SHA-256 hashes in
  `.github/release-build-constraints.txt`; and
- `pypi-attestations` 0.0.29 for post-publication cryptographic verification.

Every external Action is referenced by its full commit SHA in the workflow.

## Dry run and publication

A dry run is the default. It performs source preflight, locked quality gates,
one hash-constrained isolated build, distribution metadata and hash checks, and
separate clean-install/import checks for the wheel and sdist on both supported
Python representatives. It retains the validated distributions as an ordinary
workflow artifact and performs no PyPI upload, tag, or GitHub Release action.

Publication uses those same validated files. Before the protected job can
start, repository variable `LLM_EXEC_CORE_PYPI_PUBLISH_ENABLED` must read
exactly `true`. The `publish` job is the only job with `id-token: write`; it has
only a validated-artifact download step and the official PyPA publishing
Action. It has no checkout, build, arbitrary shell step, username, password,
secret, PAT, or PyPI API token. Attestations remain enabled and
`skip-existing` is forbidden.

After every attempted non-dry-run upload, whether the `publish` job succeeds,
fails, or is cancelled, a read-only job audits public PyPI state. A skipped
`publish` job means no upload was attempted and does not trigger this audit.
The auditor waits for bounded index consistency, classifies the public file set
as absent, partial, or complete, and records that state in the workflow summary.
A successful `publish` result requires the complete expected set; a failed or
cancelled result may expose zero, one, or both files.

After the bounded wait, a failed or cancelled publish does not discard evidence
merely because the version JSON and Simple APIs still disagree. The audit
records each API's file set, marks `index-fallback`, and conservatively treats
their union as public. This fallback is forbidden after a successful publish,
which must end with a complete and consistent two-file set.

For every file observed as public, the auditor validates every available claim
from the version JSON and Simple JSON APIs, downloads only bounded HTTPS
responses from `pypi.org` and `files.pythonhosted.org`, and verifies its
filename, size, non-yanked state, SHA-256 digest, Integrity subject, and
publisher identity. When both APIs expose a file, their URLs must agree. A
120-second-bounded verifier then parses each saved provenance document with
`pypi-attestations` 0.0.29, cryptographically verifies its attestations against
that exact downloaded distribution, and inspects the certificate on that same
verified publish-attestation object. It requires the exact repository,
workflow, `main` ref, protected environment, and source SHA.

The Integrity publisher object is open-ended. Its index-retained `claims`
member may be omitted, null, or an object and is never treated as authenticated
release identity. Required publisher fields are matched individually; the
cryptographically verified Sigstore certificate extensions are authoritative
for workflow, ref, environment, and source-SHA claims.

## Ordered owner gates

This workflow PR is Core #39 and changes no setting or package. Subsequent owner
actions remain separate:

1. Core #41 creates and protects environment `pypi`, restricts deployment to
   `main`, and records the reviewer/readback evidence.
2. Core #42 prepares the first legitimate public version through a focused
   version-only PR.
3. Core #43 registers the exact pending Trusted Publisher immediately before
   first publication. The pending publisher does not reserve the PyPI name.
4. Core #40 explicitly enables
   `LLM_EXEC_CORE_PYPI_PUBLISH_ENABLED`, approves the protected deployment,
   publishes the first version, verifies it, then disables the switch unless
   another authorized publication window is active.

Only `lanmogu98` may merge, change the environment or variable, register the
pending Trusted Publisher, approve a protected deployment, dispatch publish
mode, or publish a package. A workflow implementation dispatch must not be
treated as authority for any of those actions.

No long-lived API token is created or stored. Trusted Publishing failure is
repaired at the exact workflow/environment/publisher identity; it must not be
worked around with a password, repository secret, PAT, or manual upload.

## Failure and recovery

Any preflight, quality, build, install, index, file, or provenance mismatch
fails closed. A failed dry run publishes nothing. A failed OIDC exchange leaves
no long-lived credential to rotate.

PyPI package filenames and versions are not reusable. If the audit reports a
partial publication, or a published version is bad, do not delete and retry,
overwrite, use `skip-existing`, or reuse the version. Set
`LLM_EXEC_CORE_PYPI_PUBLISH_ENABLED=false`, preserve the workflow summary and
public evidence, yank every affected file/version as appropriate, and supersede
it with a new legitimate version through the same reviewed process. Never reuse
a public version.

If the PyPI project name is claimed before the first owner-controlled
publication, stop the program and redesign the package identity. A pending
Trusted Publisher is not a name reservation.

If publisher identity compromise is suspected, disable the publication switch,
disable the `pypi` environment, and revoke the Trusted Publisher before repair.
Direct `main` pushes, protection bypass, tag rewriting, manual upload, tokens,
and artifact replacement are never recovery mechanisms.

## Official references

Retrieved and verified on 2026-07-23:

- https://docs.pypi.org/trusted-publishers/using-a-publisher/
- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- https://docs.pypi.org/trusted-publishers/security-model/
- https://docs.pypi.org/attestations/producing-attestations/
- https://docs.pypi.org/attestations/consuming-attestations/
- https://docs.pypi.org/api/integrity/
- https://docs.astral.sh/uv/concepts/projects/build/
- https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-pypi
