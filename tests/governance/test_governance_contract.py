from pathlib import Path
import re
import runpy

import pytest
import yaml

ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
EXPECTED_CHECKS = {"CI / python-3.10", "CI / python-3.13"}
CONTRACT = runpy.run_path(str(ROOT / "scripts" / "governance_contract.py"))
assess_ci_evidence = CONTRACT["assess_ci_evidence"]
assess_final_ruleset_failure = CONTRACT["assess_final_ruleset_failure"]
dispatch_requires_refresh = CONTRACT["dispatch_requires_refresh"]
implementation_action_allowed = CONTRACT["implementation_action_allowed"]
independent_review_is_valid = CONTRACT["independent_review_is_valid"]
independent_review_required = CONTRACT["independent_review_required"]


def test_ordinary_change_uses_dispatch_without_hash_or_attestation() -> None:
    assert implementation_action_allowed(
        issue_accepted=True,
        dispatch_matches=True,
        action="tracked_file_write",
    )
    assert not implementation_action_allowed(
        issue_accepted=False,
        dispatch_matches=True,
        action="tracked_file_write",
    )
    assert not implementation_action_allowed(
        issue_accepted=True,
        dispatch_matches=False,
        action="tracked_file_write",
    )


@pytest.mark.parametrize("action", ["delete_repository", "unknown_action"])
def test_implementation_actions_fail_closed_to_explicit_allowlist(
    action: str,
) -> None:
    assert implementation_action_allowed(
        issue_accepted=True,
        dispatch_matches=True,
        action="tracked_file_write",
    )
    assert not implementation_action_allowed(
        issue_accepted=True,
        dispatch_matches=True,
        action=action,
    )


@pytest.mark.parametrize(
    "change_kind",
    [
        "objective",
        "scope",
        "behavior",
        "compatibility",
        "security",
        "authority",
    ],
)
def test_semantic_scope_change_requires_new_dispatch(change_kind: str) -> None:
    assert dispatch_requires_refresh(change_kind)


@pytest.mark.parametrize(
    "change_kind",
    ["date", "link", "formatting", "typo", "evidence", "status"],
)
def test_bookkeeping_and_evidence_updates_preserve_dispatch(
    change_kind: str,
) -> None:
    assert not dispatch_requires_refresh(change_kind)


@pytest.mark.parametrize(
    "change_kind",
    [
        "workflow",
        "permissions",
        "secrets",
        "security",
        "external_code",
        "breaking_api",
        "catalog",
    ],
)
def test_high_risk_review_is_fresh_independent_and_exact_head(
    change_kind: str,
) -> None:
    head_sha = "a" * 40

    assert independent_review_required(change_kind)
    assert independent_review_is_valid(
        change_kind=change_kind,
        head_sha=head_sha,
        review_head_sha=head_sha,
        reviewer_is_implementer=False,
        verdict="PASS",
    )
    assert not independent_review_is_valid(
        change_kind=change_kind,
        head_sha=head_sha,
        review_head_sha="b" * 40,
        reviewer_is_implementer=False,
        verdict="PASS",
    )
    assert not independent_review_is_valid(
        change_kind=change_kind,
        head_sha=head_sha,
        review_head_sha=head_sha,
        reviewer_is_implementer=True,
        verdict="PASS",
    )
    assert not independent_review_is_valid(
        change_kind=change_kind,
        head_sha=head_sha,
        review_head_sha=head_sha,
        reviewer_is_implementer=False,
        verdict="FAIL",
    )


@pytest.mark.parametrize(
    "action",
    [
        "merge",
        "auto_merge",
        "release",
        "package_publication",
        "settings_change",
        "secrets_change",
        "permissions_change",
        "protected_deployment",
        "main_bypass",
        "direct_main_update",
    ],
)
def test_owner_only_actions_remain_denied_to_implementation_agents(
    action: str,
) -> None:
    assert not implementation_action_allowed(
        issue_accepted=True,
        dispatch_matches=True,
        action=action,
    )


@pytest.mark.parametrize(
    "condition",
    [
        "missing",
        "not_started",
        "skipped",
        "cancelled",
        "timed_out",
        "stuck",
        "failure",
        "error",
        "wrong_name",
        "duplicate_name",
        "wrong_sha",
        "wrong_source",
        "final_setting_readback_failed",
    ],
)
def test_ci_non_success_keeps_preliminary_ruleset(condition: str) -> None:
    result = assess_ci_evidence(
        condition=condition,
        owner="lanmogu98",
        elapsed_minutes=30,
        repair_deadline_hours=24,
    )

    assert result == {
        "merge_frozen": True,
        "preliminary_ruleset": "active",
        "owner": "lanmogu98",
        "detection_minutes": 30,
        "repair_deadline_hours": 24,
    }


@pytest.mark.parametrize(
    "condition",
    [
        "default_branch_readback_failed",
        "app_inventory_readback_failed",
        "collaborator_key_webhook_readback_failed",
        "authorization_model_readback_failed",
        "actions_permissions_readback_failed",
        "external_approval_readback_failed",
        "bypass_readback_failed",
        "required_check_source_readback_failed",
        "merge_settings_readback_failed",
        "branch_deletion_readback_failed",
        "labels_readback_failed",
        "secret_scanning_readback_failed",
        "push_protection_readback_failed",
        "private_reporting_readback_failed",
    ],
)
def test_each_final_setting_failure_freezes_merge(condition: str) -> None:
    result = assess_ci_evidence(
        condition=condition,
        owner="lanmogu98",
        elapsed_minutes=30,
        repair_deadline_hours=24,
    )

    assert result["merge_frozen"] is True
    assert result["preliminary_ruleset"] == "active"


def test_preliminary_ruleset_readback_failure_blocks_pr_recovery() -> None:
    result = assess_ci_evidence(
        condition="preliminary_ruleset_readback_failed",
        owner="lanmogu98",
        elapsed_minutes=30,
        repair_deadline_hours=24,
    )

    assert result == {
        "merge_frozen": True,
        "pr_repair_allowed": False,
        "preliminary_ruleset": "unverified_restore_required",
        "owner": "lanmogu98",
        "detection_minutes": 30,
        "repair_deadline_hours": 24,
    }


@pytest.mark.parametrize(
    "condition", ["missing_check", "wrong_name", "wrong_source", "mismatch"]
)
def test_final_ruleset_failure_removes_only_final_before_pr_repair(
    condition: str,
) -> None:
    assert assess_final_ruleset_failure(condition) == {
        "final_ruleset": "disable_or_delete",
        "preliminary_ruleset": "verify_exported_anchor_active",
        "merge_frozen": True,
        "pr_repair_allowed": False,
    }


def test_ci_workflow_has_stable_locked_mock_only_contract() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    job = workflow["jobs"]["test"]
    versions = job["strategy"]["matrix"]["python-version"]
    job_name = job["name"]
    checks = {
        job_name.replace("${{ matrix.python-version }}", version)
        for version in versions
    }

    assert workflow["name"] == "CI"
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["on"]["pull_request"]["branches"] == ["main"]
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert checks == EXPECTED_CHECKS
    assert job_name == "CI / python-${{ matrix.python-version }}"
    assert "pull_request_target" not in text
    assert "secrets." not in text
    assert "API_KEY" not in text
    assert "uv sync --locked --group dev" in text
    assert "uv run --frozen pytest -q" in text
    assert "uv run --frozen black --check src tests" in text
    assert "uv run --frozen flake8 src tests" in text
    assert "uv run --frozen mypy src" in text
    assert "uv build" in text
    action_refs = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", text)
    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)


def test_release_workflow_has_safe_manual_interface_and_pins() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    dispatch = workflow["on"]["workflow_dispatch"]
    inputs = dispatch["inputs"]

    assert workflow["name"] == "Release"
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert set(inputs) == {"version", "expected_sha", "dry_run"}
    assert inputs["version"]["required"] == "true"
    assert inputs["version"]["type"] == "string"
    assert inputs["expected_sha"]["required"] == "true"
    assert inputs["expected_sha"]["type"] == "string"
    assert inputs["dry_run"] == {
        "description": inputs["dry_run"]["description"],
        "required": "true",
        "type": "boolean",
        "default": "true",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "llm-exec-core-release",
        "queue": "max",
        "cancel-in-progress": "false",
    }

    expected_actions = {
        "actions/checkout": (
            "3d3c42e5aac5ba805825da76410c181273ba90b1",
            "v7.0.1",
        ),
        "astral-sh/setup-uv": (
            "c771a70e6277c0a99b617c7a806ffedaca235ff9",
            "v9.0.0",
        ),
        "actions/upload-artifact": (
            "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
            "v7.0.1",
        ),
        "actions/download-artifact": (
            "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
            "v8.0.1",
        ),
    }
    action_lines = re.findall(
        r"uses:\s*([^@\s]+)@([0-9a-f]{40})\s+#\s+(\S+)", text
    )
    assert action_lines
    assert set(action for action, _, _ in action_lines) == set(
        expected_actions
    )
    assert all(
        (sha, version) == expected_actions[action]
        for action, sha, version in action_lines
    )

    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            assert not re.search(r"\$\{\{\s*inputs\.", run)
            if step.get("uses", "").startswith("astral-sh/setup-uv@"):
                assert step["with"]["version"] == "0.11.30"

    assert "pull_request_target" not in text
    assert "secrets." not in text
    assert "API_KEY" not in text
    assert "OPENAI" not in text
    assert "ANTHROPIC" not in text
    assert "live LLM" not in text


def test_release_workflow_fails_closed_on_source_version_and_collisions() -> (
    None
):
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    validate = workflow["jobs"]["validate"]
    checkout = next(
        step
        for step in validate["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    validation = next(
        step for step in validate["steps"] if step.get("id") == "validate"
    )
    validation_run = validation["run"]

    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "fetch-depth": "0",
        "persist-credentials": "false",
    }
    assert validation["env"]["RAW_VERSION"] == "${{ inputs.version }}"
    assert validation["env"]["EXPECTED_SHA"] == "${{ inputs.expected_sha }}"
    assert validation["env"]["DRY_RUN"] == "${{ inputs.dry_run }}"
    assert validation["env"]["DISPATCHED_SHA"] == "${{ github.sha }}"
    for required in (
        "workflow_dispatch",
        "refs/heads/main",
        '"ls-remote"',
        "packaging.version",
        "Version(",
        "pyproject.toml",
        "ast.parse",
        "llm_exec_core/__init__.py",
        "uv.lock",
        'editable = "."',
        "40-character lower-case SHA",
        "git/ref/tags",
        "releases/tags",
        "assets",
        "publish blocked",
    ):
        assert required in validation_run
    assert "str(parsed_version) != raw_version" in validation_run
    assert set(validate["outputs"]) == {
        "version",
        "tag",
        "expected_sha",
        "artifact_name",
        "publish_blocked",
        "collision_report",
    }


def test_release_workflow_builds_once_and_reuses_exact_candidate() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    assert set(jobs) == {
        "validate",
        "quality",
        "build",
        "install",
        "publish",
        "summary",
    }

    quality = jobs["quality"]
    assert quality["needs"] == "validate"
    assert quality["strategy"]["fail-fast"] == "false"
    assert quality["strategy"]["matrix"]["python-version"] == [
        "3.10",
        "3.13",
    ]
    quality_run = "\n".join(step.get("run", "") for step in quality["steps"])
    for command in (
        "uv sync --locked --group dev",
        "uv run --frozen pytest -q",
        "uv run --frozen black --check src tests",
        "uv run --frozen flake8 src tests",
        "uv run --frozen mypy src",
    ):
        assert command in quality_run
    assert "uv build" not in quality_run

    build = jobs["build"]
    assert set(build["needs"]) == {"validate", "quality"}
    assert set(build["outputs"]) == {
        "wheel_filename",
        "wheel_sha256",
        "sdist_filename",
        "sdist_sha256",
        "notes_sha256",
        "provenance_sha256",
    }
    install = jobs["install"]
    assert set(install["needs"]) == {"validate", "build"}
    assert install["strategy"]["fail-fast"] == "false"
    assert install["strategy"]["matrix"] == {
        "python-version": ["3.10", "3.13"],
        "distribution": ["wheel", "sdist"],
    }
    all_run = "\n".join(
        step.get("run", "")
        for job in jobs.values()
        for step in job.get("steps", [])
    )
    assert len(re.findall(r"(?m)^\s*uv build\s*$", all_run)) == 1

    build_run = "\n".join(step.get("run", "") for step in build["steps"])
    for required in (
        "exactly one wheel",
        "exactly one sdist",
        "METADATA",
        "PKG-INFO",
        "CHANGELOG.md",
        "previous_tag",
        "RELEASE_NOTES.md",
        "PROVENANCE.json",
        "sort_keys=True",
        "SHA256SUMS",
        "exact five-file candidate set",
    ):
        assert required in build_run
    assert build_run.index("PROVENANCE.json") < build_run.index("SHA256SUMS")

    upload_steps = [
        step
        for step in build["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(upload_steps) == 1
    assert set(upload_steps[0]["with"]["path"].splitlines()) == {
        "candidate/*.whl",
        "candidate/*.tar.gz",
        "candidate/SHA256SUMS",
        "candidate/RELEASE_NOTES.md",
        "candidate/PROVENANCE.json",
    }

    integrity_steps = {
        "install": next(
            step
            for step in install["steps"]
            if step["name"] == "Verify and install selected distribution"
        ),
        "publish": next(
            step
            for step in jobs["publish"]["steps"]
            if step.get("id") == "publish_preflight"
        ),
    }
    for job_name, integrity_step in integrity_steps.items():
        steps = jobs[job_name]["steps"]
        assert any(
            step.get("uses", "").startswith("actions/download-artifact@")
            for step in steps
        )
        assert (
            integrity_step["env"]["VERSION"]
            == "${{ needs.validate.outputs.version }}"
        )
        run = integrity_step["run"]
        assert 'f"llm_exec_core-{filename_version}-py3-none-any.whl"' in run
        assert 'f"llm_exec_core-{filename_version}.tar.gz"' in run
        for filename in (
            "SHA256SUMS",
            "RELEASE_NOTES.md",
            "PROVENANCE.json",
        ):
            assert f'"{filename}"' in run
        assert "actual_files != expected_files" in run
        assert "len(manifest_lines) != 4" in run
        assert 're.fullmatch(r"[0-9a-f]{64}  [^\\r\\n]+", line)' in run
        assert "Path(filename).name != filename" in run
        assert "filename in manifest" in run
        assert "set(manifest) != expected_manifest_files" in run
        assert '["sha256sum", "--check", "SHA256SUMS"]' in run
        assert "check=True" in run

    install_run = "\n".join(step.get("run", "") for step in install["steps"])
    install_step = next(
        step
        for step in install["steps"]
        if step["name"] == "Verify and install selected distribution"
    )
    assert "${{ runner.temp }}" in install_step["env"]["CANDIDATE_DIR"]
    assert "${{ runner.temp }}" in install_step["env"]["INSTALL_ENV"]
    assert "uv venv" in install_run
    assert "uv pip install" in install_run
    assert 'metadata.version("llm-exec-core")' in install_run
    assert "llm_exec_core.__version__" in install_run


def test_release_workflow_gates_immutable_draft_first_publication() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    publish = jobs["publish"]

    assert set(publish["needs"]) == {"validate", "build", "install"}
    assert publish["if"] == "${{ inputs.dry_run == false }}"
    assert publish["permissions"] == {"contents": "write"}
    assert all(
        job.get("permissions", {}).get("contents") != "write"
        for name, job in jobs.items()
        if name != "publish"
    )

    publish_steps = publish["steps"]
    publish_ids = [
        step["id"] for step in publish_steps if step.get("id") is not None
    ]
    assert publish_ids[-5:] == [
        "publish_preflight",
        "create_draft",
        "verify_draft",
        "publish_release",
        "verify_public",
    ]
    preflight = next(
        step for step in publish_steps if step.get("id") == "publish_preflight"
    )
    assert preflight["env"]["ORIGINAL_ACTOR"] == "${{ github.actor }}"
    assert (
        preflight["env"]["TRIGGERING_ACTOR"]
        == "${{ github.triggering_actor }}"
    )
    preflight_run = preflight["run"]
    for required in (
        "lanmogu98",
        "immutable-releases",
        "remote main",
        "git/ref/tags",
        "releases?per_page=100&page=",
        'release.get("tag_name") == target_tag',
        "target Release asset collision",
    ):
        assert required in preflight_run
    assert "releases/tags" not in preflight_run
    assert preflight_run.index(
        "releases?per_page=100&page="
    ) < preflight_run.index("git/ref/tags")

    create_draft_step = next(
        step for step in publish_steps if step.get("id") == "create_draft"
    )
    create_draft = create_draft_step["run"]
    assert "git/refs" in create_draft
    assert 'gh api --method POST "repos/$REPOSITORY/releases"' in create_draft
    assert "-F draft=true" in create_draft
    assert "release_id=" in create_draft
    assert "release_id=$release_id" in create_draft
    assert "GITHUB_OUTPUT" in create_draft
    assert "gh release create" not in create_draft
    assert "--clobber" not in create_draft

    verify_draft_step = next(
        step for step in publish_steps if step.get("id") == "verify_draft"
    )
    assert (
        verify_draft_step["env"]["RELEASE_ID"]
        == "${{ steps.create_draft.outputs.release_id }}"
    )
    verify_draft = verify_draft_step["run"]
    for required in (
        "draft",
        "RELEASE_NOTES.md",
        "asset names",
        "digest",
        "releases/{release_id}",
    ):
        assert required in verify_draft
    assert "releases/tags" not in verify_draft

    publish_release_step = next(
        step for step in publish_steps if step.get("id") == "publish_release"
    )
    assert (
        publish_release_step["env"]["RELEASE_ID"]
        == "${{ steps.create_draft.outputs.release_id }}"
    )
    publish_release = publish_release_step["run"]
    expected_publish_api = (
        'gh api --method PATCH "repos/$REPOSITORY/releases/$RELEASE_ID"'
    )
    assert expected_publish_api in publish_release
    assert "-F draft=false" in publish_release
    assert "gh release edit" not in publish_release
    verify_public = next(
        step for step in publish_steps if step.get("id") == "verify_public"
    )["run"]
    for required in (
        "immutable",
        "git/ref/tags",
        "asset names",
        "digest",
        "time.sleep",
        "releases/tags/{encoded_tag}",
    ):
        assert required in verify_public

    summary = jobs["summary"]
    assert summary["if"] == "${{ always() }}"
    assert set(summary["needs"]) == {
        "validate",
        "quality",
        "build",
        "install",
        "publish",
    }
    summary_step = summary["steps"][0]
    assert summary_step["env"]["RAW_VERSION"] == "${{ inputs.version }}"
    assert summary_step["env"]["DISPATCHED_SHA"] == "${{ github.sha }}"
    assert summary_step["env"]["DISPATCHED_REF"] == "${{ github.ref }}"
    assert (
        summary_step["env"]["VALIDATED_VERSION"]
        == "${{ needs.validate.outputs.version }}"
    )
    assert (
        summary_step["env"]["WHEEL_FILENAME"]
        == "${{ needs.build.outputs.wheel_filename }}"
    )
    assert (
        summary_step["env"]["WHEEL_SHA256"]
        == "${{ needs.build.outputs.wheel_sha256 }}"
    )
    summary_run = summary_step["run"]
    assert "publish blocked" in summary_run
    assert "validation" in summary_run
    assert "Requested version/tag" in summary_run
    assert "Dispatched commit/ref" in summary_run
    assert "Validated identity: unavailable" in summary_run
    assert 'if [[ "$BUILD_STATE" == "success" ]]' in summary_run
    assert "Candidate: not produced" in summary_run
    assert 'if [[ "$INSTALL_STATE" == "success" ]]' in summary_run
    assert "Consumer checksum verification: complete" in summary_run
    assert "Consumer checksum verification: incomplete" in summary_run

    build_branch = summary_run.split(
        'if [[ "$BUILD_STATE" == "success" ]]', 1
    )[1]
    assert build_branch.index("$WHEEL_FILENAME") < build_branch.index(
        "Candidate: not produced"
    )
    install_branch = summary_run.split(
        'if [[ "$INSTALL_STATE" == "success" ]]', 1
    )[1]
    assert install_branch.index(
        "Consumer checksum verification: complete"
    ) < install_branch.index("Consumer checksum verification: incomplete")


def test_release_runbook_documents_owner_gates_and_recovery() -> None:
    text = (ROOT / "docs" / "governance" / "release.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "default branch",
        "workflow_dispatch",
        "expected_sha",
        "remote `main`",
        "`dry_run=true`",
        "`0.4.1`",
        "publish blocked",
        "Python 3.10 and 3.13",
        "four clean-install",
        "`SHA256SUMS`",
        "`RELEASE_NOTES.md`",
        "`PROVENANCE.json`",
        "immutable Releases",
        "merge",
        "dispatch",
        "publication",
        "incomplete draft",
        "revert",
        "never rewrite",
        "new version",
        "`setuptools>=64`",
        "Administration: read",
        "cannot currently authorize",
        "published-only",
    ):
        assert required in text


def test_public_intake_and_policy_files_are_consistent() -> None:
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    chooser_text = (ROOT / ".github/ISSUE_TEMPLATE/config.yml").read_text(
        encoding="utf-8"
    )
    chooser = yaml.load(chooser_text, Loader=yaml.BaseLoader)
    normalized_readme = " ".join(readme.split())
    normalized_contributing = " ".join(contributing.split())
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert claude == "@AGENTS.md\n"
    assert "accepts public Issues only" in normalized_readme
    assert "explicit maintainer dispatch" in normalized_readme
    assert (
        "accepts public Issue reports and change requests only"
        in normalized_contributing
    )
    assert "explicit maintainer dispatch" in normalized_contributing
    assert chooser["blank_issues_enabled"] == "false"
    assert "Public Issues only" in chooser_text
    assert "explicit maintainer dispatch" in chooser_text
    assert "security/advisories/new" in chooser_text
    assert "security/advisories/new" in security
    assert ".env" in gitignore

    for name in ("bug.yml", "change.yml"):
        form = yaml.load(
            (ROOT / ".github/ISSUE_TEMPLATE" / name).read_text(
                encoding="utf-8"
            ),
            Loader=yaml.BaseLoader,
        )
        assert form["body"]
