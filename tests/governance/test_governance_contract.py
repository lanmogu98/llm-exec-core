from datetime import datetime, timezone
from pathlib import Path
import re
import runpy
import subprocess

import pytest
import yaml

ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
EXPECTED_CHECKS = {"CI / python-3.10", "CI / python-3.13"}
CONTRACT = runpy.run_path(str(ROOT / "scripts" / "governance_contract.py"))
assess_ci_evidence = CONTRACT["assess_ci_evidence"]
assess_final_ruleset_failure = CONTRACT["assess_final_ruleset_failure"]
canonical_issue_sha256 = CONTRACT["canonical_issue_sha256"]
comment_sha256 = CONTRACT["comment_sha256"]
contribution_is_authorized = CONTRACT["contribution_is_authorized"]
gate0_action_allowed = CONTRACT["gate0_action_allowed"]
private_gate_is_valid = CONTRACT["private_gate_is_valid"]
workflow_change_is_authorized = CONTRACT["workflow_change_is_authorized"]


def test_canonical_issue_hash_vectors() -> None:
    assert canonical_issue_sha256("T", "B") == (
        "c467a8e4991deeb4a45eda82181eeb4bf23107e7798dd197cb5432cf9df3f268"
    )
    assert canonical_issue_sha256("T", "A\r\nB") == (
        "72a94906a55d109eee887e883a28c7b9f87bb0ab740b88241827da110f8063f9"
    )
    assert comment_sha256("B") == (
        "df7e70e5021544f4834bbee64a9e3789febc4be81470df629cad6ddb03320a5c"
    )


def test_private_contract_digest_matches_openssl() -> None:
    body = "PRIVATE-GATE0-CONTRACT-V1:\r\n  advisory_id: GHSA-test"
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    openssl = subprocess.run(
        ["openssl", "dgst", "-sha256"],
        input=normalized.encode(),
        capture_output=True,
        check=True,
    )
    openssl_digest = openssl.stdout.decode().strip().rsplit(" ", 1)[-1]

    assert comment_sha256(body) == openssl_digest


@pytest.mark.parametrize(
    "action",
    [
        "implementation_assignment",
        "implementation_branch",
        "tracked_file_write",
        "settings_change",
        "external_code_execution",
    ],
)
def test_gate0_blocks_execution_before_pass(action: str) -> None:
    assert not gate0_action_allowed("pending", action)
    assert gate0_action_allowed("passed", action)


@pytest.mark.parametrize(
    "action", ["contract_read", "contract_edit", "audit", "audit_remediation"]
)
def test_gate0_allows_contract_preparation_before_pass(action: str) -> None:
    assert gate0_action_allowed("pending", action)


def _authorization() -> dict[str, str]:
    return {
        "author": "lanmogu98",
        "author_association": "OWNER",
        "contributor": "external-user",
        "work_item": "https://github.com/lanmogu98/llm-exec-core/issues/9",
        "allowed_scope": "src/example.py",
        "allowed_delivery": "fork PR",
        "issued_at": "2026-07-01T09:00:00Z",
        "expires_at": "2026-07-03T09:00:00Z",
        "created_at": "2026-07-01T09:00:00Z",
        "updated_at": "2026-07-01T09:00:00Z",
    }


def test_external_delivery_requires_prior_exact_owner_authorization() -> None:
    authorization = _authorization()
    delivery_at = datetime(2026, 7, 2, tzinfo=timezone.utc)

    assert contribution_is_authorized(
        authorization,
        contributor="external-user",
        work_item=authorization["work_item"],
        delivery_at=delivery_at,
    )
    assert not contribution_is_authorized(
        authorization,
        contributor="different-user",
        work_item=authorization["work_item"],
        delivery_at=delivery_at,
    )
    assert not contribution_is_authorized(
        authorization,
        contributor="external-user",
        work_item=authorization["work_item"],
        delivery_at=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )


def test_workflow_change_authorization_is_owner_authored_and_head_exact() -> (
    None
):
    head = "a" * 40
    comment = {
        "author": "lanmogu98",
        "author_association": "OWNER",
        "body": f"WORKFLOW-CHANGE-AUTHORIZED: {head}",
        "created_at": "2026-07-02T09:00:00Z",
        "updated_at": "2026-07-02T09:00:00Z",
    }

    assert workflow_change_is_authorized(comment, head)
    assert not workflow_change_is_authorized(comment, "b" * 40)
    edited = {**comment, "updated_at": "2026-07-02T09:01:00Z"}
    assert not workflow_change_is_authorized(edited, head)


def _private_gate_snapshot() -> dict[str, object]:
    contract_body = "PRIVATE-GATE0-CONTRACT-V1:\n  advisory_id: GHSA-test"
    head = "b" * 40
    return {
        "expected_reporter": "reporter",
        "current_reporter": "reporter",
        "expected_collaborators": ["helper", "reporter"],
        "current_collaborators": ["helper", "reporter"],
        "expected_authorizations": ["helper"],
        "current_authorizations": ["helper"],
        "expected_head_sha": head,
        "current_head_sha": head,
        "contract": {
            "exists": True,
            "author": "lanmogu98",
            "author_association": "OWNER",
            "body": contract_body,
            "sha256": comment_sha256(contract_body),
            "created_at": "2026-07-02T09:00:00Z",
            "updated_at": "2026-07-02T09:00:00Z",
        },
        "report": {
            "exists": True,
            "body": "verdict: PASS",
            "sha256": comment_sha256("verdict: PASS"),
            "created_at": "2026-07-02T09:01:00Z",
            "updated_at": "2026-07-02T09:01:00Z",
        },
        "attestation": {
            "exists": True,
            "author": "lanmogu98",
            "author_association": "OWNER",
            "body": "verdict: PASS\nhead: " + head,
            "sha256": comment_sha256("verdict: PASS\nhead: " + head),
            "created_at": "2026-07-02T09:02:00Z",
            "updated_at": "2026-07-02T09:02:00Z",
        },
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("current_reporter",), "other"),
        (("current_collaborators",), ["reporter"]),
        (("contract", "updated_at"), "2026-07-02T09:03:00Z"),
        (("report", "exists"), False),
        (("contract", "body"), "changed"),
        (("current_authorizations",), []),
        (("current_head_sha",), "c" * 40),
    ],
)
def test_private_gate_mutations_fail_closed(
    path: tuple[str, ...], value: object
) -> None:
    snapshot = _private_gate_snapshot()
    target = snapshot
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]

    assert not private_gate_is_valid(snapshot)


def test_private_gate_accepts_unchanged_synthetic_snapshot() -> None:
    assert private_gate_is_valid(_private_gate_snapshot())


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
        "preliminary_ruleset_readback_failed",
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
    checks = {f"CI / python-{version}" for version in versions}

    assert workflow["name"] == "CI"
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["on"]["pull_request"]["branches"] == ["main"]
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert checks == EXPECTED_CHECKS
    assert job["name"] == "python-${{ matrix.python-version }}"
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
    assert "prior written maintainer authorization" in normalized_readme
    assert (
        "accepts public Issue reports and change requests only"
        in normalized_contributing
    )
    assert (
        "prior owner-authored written authorization" in normalized_contributing
    )
    assert chooser["blank_issues_enabled"] == "false"
    assert "Public Issues only" in chooser_text
    assert "prior written authorization" in chooser_text
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
