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

Every manual dispatch supplies an exact normalized stable `version`, an exact
40-character current `main` `expected_sha`, and `dry_run`. The workflow rejects
another ref, a stale `main`, disagreement among source/lock/distribution
versions, or a version already present on PyPI.

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

After upload, a read-only job compares the local files against PyPI's version
JSON API and Simple JSON API, downloads only bounded HTTPS responses from
`pypi.org` and `files.pythonhosted.org`, and verifies filenames, sizes,
non-yanked state, SHA-256 digests, Integrity subjects, publisher identity, and
certificate claims. It then runs `pypi-attestations verify pypi` for each file
and requires cryptographic provenance from the exact repository, workflow,
`main` ref, protected environment, and source SHA.

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

PyPI package filenames and versions are not reusable. If publication is partial
or a published version is bad, do not delete and retry, overwrite, use
`skip-existing`, or reuse the version. Set
`LLM_EXEC_CORE_PYPI_PUBLISH_ENABLED=false`, preserve the workflow and public
evidence, yank every affected file/version as appropriate, and supersede it
with a new legitimate version through the same reviewed process. Never reuse a
public version.

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
