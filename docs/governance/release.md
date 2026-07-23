# GitHub Release runbook

The `Release` workflow is a manual, default-safe validator for one exact
`main` commit. GitHub only exposes a `workflow_dispatch` workflow after its
file exists on the default branch, so the implementation PR must first pass
Core CI and independent exact-head review, then receive a separate owner merge
decision.

## Inputs and validation

Dispatch from the default branch with an exact normalized PEP 440 `version`,
the lower-case 40-character `expected_sha`, and normally `dry_run=true`. The
workflow fails closed unless the event is a default-branch dispatch and
`expected_sha` equals both the dispatched SHA and current remote `main`. The
version must match `pyproject.toml`, the AST-read package `__version__`, the
editable root in `uv.lock`, the no-`v` tag, and both built distributions and
their metadata.

Tag, Release, and target-Release asset collisions are read without mutation.
In `dry_run=true`, collisions are reported as `publish blocked` while all safe
validation continues. The existing `0.4.1` tag/Release state is the expected
safe collision fixture. In publish mode, any collision stops before mutation;
API failures also fail closed.

## Candidate flow and assets

Locked quality checks run on Python 3.10 and 3.13. After both cells pass, one
authoritative build creates exactly one wheel and one sdist. The same candidate
is reused by the four clean-install cells: Python 3.10 and 3.13, each installing
the wheel and sdist independently outside the checkout.

The exact workflow artifact and future Release asset set is:

1. one wheel;
2. one sdist;
3. `SHA256SUMS`;
4. `RELEASE_NOTES.md`; and
5. `PROVENANCE.json`.

Every consumer verifies `SHA256SUMS`. Release notes come deterministically from
the matching changelog section and previous reachable tag range. Provenance
records the exact source, workflow/run, actor, and distribution checksums.

## Separate owner gates

Immutable Releases settings, merge, dispatch, and publication are four
separate owner gates. The workflow never enables immutable Releases. A
maintainer must verify that setting separately, merge separately, authorize
post-merge dry-run evidence separately, and later authorize publication
separately. Issue #39 creates no stable Release.

Only the publication job has `contents: write`. Immediately before mutation it
rechecks both actors, remote `main`, the immutable Releases setting,
collisions, checksums, and the exact five files. It then creates the no-`v` tag
and an incomplete draft, uploads without clobbering, verifies the draft body,
asset names, and digests, publishes only when complete, and reads back the
public tag target, assets, digests, and immutability.

## Failure and recovery

A validation or dry-run failure creates no tag or Release and requires a
focused repair PR. If a future publish attempt leaves an incomplete draft or
tag, stop and preserve it for maintainer inspection; never delete or repair it
automatically.

Rollback the workflow itself by revert through a normal PR; never rewrite an
immutable public Release, tag, or asset. Supersede a faulty public artifact
with a new version.

The accepted residual risk is that the canonical build uses the repository's
existing `setuptools>=64` build requirement. Issue #39 intentionally makes no
dependency or lockfile change.
