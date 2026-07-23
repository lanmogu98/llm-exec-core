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

Tag, published Release, and target-Release asset collisions are read without
mutation. The read-only validation job uses GitHub's published-only by-tag
endpoint, which is sufficient for the existing `0.4.1` dry-run collision
fixture but does not claim to discover drafts. In `dry_run=true`, collisions
are reported as `publish blocked` while all safe validation continues. Before
any future publish mutation, the write-authorized job separately paginates all
Releases and rejects an exact matching `tag_name`, including a draft. In
publish mode, any collision stops before mutation; API failures also fail
closed.

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

Every consumer first requires exact basename-set equality, then parses
`SHA256SUMS` as exactly four unique, lower-case SHA-256 records for the four
non-manifest basenames, rejecting traversal, duplicates, and extras before
running the checksum tool. Release notes come deterministically from the
matching changelog section and previous reachable tag range. Provenance records
the exact source, workflow/run, actor, and distribution checksums.

## Credentialed immutable-setting verification

The immutable-Releases repository endpoint requires `Administration: read`.
Core #43 is the separate owner-only provisioning gate for a GitHub App with
that read permission and an installation limited to the current repository.
The owner places its client ID in the
`CORE_RELEASE_PREFLIGHT_APP_CLIENT_ID` variable and its private key in the
`CORE_RELEASE_PREFLIGHT_APP_PRIVATE_KEY` secret of the protected `release`
environment. The workflow does not create or configure the App, installation,
environment, variable, secret, or immutable Releases setting.

Only `publish` binds the protected `release` environment and has
`contents: write`; every other job remains read-only. After its ordinary actor,
candidate checksum, remote-main, Release-listing, and tag-collision preflight,
the job mints a short-lived App installation token. Because the token Action
omits `owner` and `repositories`, it is scoped to the current repository; its
default revocation remains enabled. The token is passed only to the immediately
following immutable-setting step, which performs only immutable-Releases setting read.
It is not used by checkout, outputs, artifacts, summaries, tag
or Release operations, or public readback; all those operations continue with
`GITHUB_TOKEN`/`GH_TOKEN`.

Dry-run does not require the App credential because it skips the entire
`publish` job. In publish mode, a missing or misconfigured environment
credential, denied immutable-setting request, malformed response, or a false
`enabled` readback fails before tag or Release mutation.

## Separate owner gates

Core #43 provisioning, immutable Releases settings, merge, dispatch, and
publication are separate owner gates. The workflow never enables immutable
Releases. A maintainer must verify the setting and credential provisioning
separately, merge separately, authorize post-merge dry-run evidence separately,
and later authorize publication separately. Issue #39 creates no stable
Release.

Immediately before mutation, the publication job has rechecked both actors,
remote `main`, collisions, checksums, and the exact five files, then completed
the isolated immutable-setting read. It creates the no-`v` tag and an
incomplete draft through REST, captures its numeric release ID, uploads without
clobbering, verifies the draft by ID, publishes that same ID only when complete,
and reserves the by-tag endpoint for public readback of the tag target, assets,
digests, and immutability.

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
