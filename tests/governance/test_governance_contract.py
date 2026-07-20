from pathlib import Path
import re
import runpy

import pytest
import yaml

ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
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
