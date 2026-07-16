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
gate0_is_ready = CONTRACT["gate0_is_ready"]
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
    assert not gate0_action_allowed(
        "pending",
        action,
        actor_role="implementation_agent",
        task_authorized_actions={action},
    )


@pytest.mark.parametrize(
    "action", ["contract_read", "contract_edit", "audit", "audit_remediation"]
)
def test_gate0_allows_contract_preparation_before_pass(action: str) -> None:
    actor_role = "auditor" if action == "audit" else "contract_remediator"
    assert gate0_action_allowed(
        "pending",
        action,
        actor_role=actor_role,
        task_authorized_actions=set(),
    )


def test_gate0_readiness_is_separate_from_action_authority() -> None:
    assert not gate0_is_ready("pending")
    assert gate0_is_ready("passed")

    authorized = {"tracked_file_write", "pr_update"}
    assert gate0_action_allowed(
        "passed",
        "tracked_file_write",
        actor_role="implementation_agent",
        task_authorized_actions=authorized,
    )
    assert not gate0_action_allowed(
        "passed",
        "topic_branch",
        actor_role="implementation_agent",
        task_authorized_actions=authorized,
    )


@pytest.mark.parametrize(
    "action",
    [
        "merge",
        "auto_merge",
        "settings_change",
        "secrets_change",
        "main_bypass",
        "publishing",
    ],
)
def test_gate0_pass_never_grants_implementation_authority_plane(
    action: str,
) -> None:
    assert not gate0_action_allowed(
        "passed",
        action,
        actor_role="implementation_agent",
        task_authorized_actions={action},
    )


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
        expected_scope="src/example.py",
        expected_delivery="fork PR",
        delivery_at=delivery_at,
    )
    assert not contribution_is_authorized(
        authorization,
        contributor="different-user",
        work_item=authorization["work_item"],
        expected_scope="src/example.py",
        expected_delivery="fork PR",
        delivery_at=delivery_at,
    )
    assert not contribution_is_authorized(
        authorization,
        contributor="external-user",
        work_item=authorization["work_item"],
        expected_scope="src/example.py",
        expected_delivery="fork PR",
        delivery_at=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    ("mutation", "expected_scope", "expected_delivery", "delivery_at"),
    [
        ({}, "different scope", "fork PR", "2026-07-02T00:00:00Z"),
        ({}, "src/example.py", "direct patch", "2026-07-02T00:00:00Z"),
        (
            {
                "created_at": "2026-07-02T01:00:00Z",
                "updated_at": "2026-07-02T01:00:00Z",
                "issued_at": "2026-07-02T01:00:00Z",
            },
            "src/example.py",
            "fork PR",
            "2026-07-02T00:00:00Z",
        ),
        (
            {"issued_at": "2026-06-30T09:00:00Z"},
            "src/example.py",
            "fork PR",
            "2026-07-02T00:00:00Z",
        ),
        (
            {"updated_at": "2026-07-01T09:01:00Z"},
            "src/example.py",
            "fork PR",
            "2026-07-02T00:00:00Z",
        ),
        (
            {"expires_at": "2026-07-01T12:00:00Z"},
            "src/example.py",
            "fork PR",
            "2026-07-02T00:00:00Z",
        ),
    ],
)
def test_external_authorization_rejects_mismatch_or_stale_record(
    mutation: dict[str, str],
    expected_scope: str,
    expected_delivery: str,
    delivery_at: str,
) -> None:
    authorization = {**_authorization(), **mutation}

    assert not contribution_is_authorized(
        authorization,
        contributor="external-user",
        work_item=authorization["work_item"],
        expected_scope=expected_scope,
        expected_delivery=expected_delivery,
        delivery_at=datetime.fromisoformat(delivery_at.replace("Z", "+00:00")),
    )


def test_completion_bound_authorization_tracks_completion_state() -> None:
    authorization = {**_authorization(), "expires_at": "completion"}
    delivery_at = datetime(2026, 7, 2, tzinfo=timezone.utc)

    assert contribution_is_authorized(
        authorization,
        contributor="external-user",
        work_item=authorization["work_item"],
        expected_scope="src/example.py",
        expected_delivery="fork PR",
        delivery_at=delivery_at,
        completed_at=None,
    )
    assert contribution_is_authorized(
        authorization,
        contributor="external-user",
        work_item=authorization["work_item"],
        expected_scope="src/example.py",
        expected_delivery="fork PR",
        delivery_at=delivery_at,
        completed_at=datetime(2026, 7, 2, 1, tzinfo=timezone.utc),
    )
    assert not contribution_is_authorized(
        authorization,
        contributor="external-user",
        work_item=authorization["work_item"],
        expected_scope="src/example.py",
        expected_delivery="fork PR",
        delivery_at=delivery_at,
        completed_at=datetime(2026, 7, 1, 23, tzinfo=timezone.utc),
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
    head = "b" * 40
    report_url = "https://github.com/advisories/GHSA-test/comments/2"
    contract_data = {
        "PRIVATE-GATE0-CONTRACT-V1": {
            "advisory_id": "GHSA-test",
            "objective": "Validate a synthetic vulnerability fix.",
            "scope": ["src/example.py"],
            "non_goals": ["live provider testing"],
            "compatibility_security_impact": "Synthetic tabletop only.",
            "acceptance_criteria": ["offline checks pass"],
            "validation": ["uv run --frozen pytest -q"],
            "rollback_recovery": "Revert through a protected PR.",
            "reporter": "reporter",
            "collaborators_sorted": ["helper", "reporter"],
            "authorized_code_contributors": ["helper"],
            "intended_implementer": "security-implementer",
        }
    }
    contract_body = yaml.safe_dump(contract_data, sort_keys=False)
    contract_digest = comment_sha256(contract_body)
    roles = {
        "reporter": "reporter",
        "collaborators_sorted": ["helper", "reporter"],
        "authorized_code_contributors": ["helper"],
        "intended_implementer": "security-implementer",
        "auditor_run_id": "auditor-run-1",
        "orchestrator_task_id": "orchestrator-task-1",
    }
    report_data = {
        "private_gate0_version": 1,
        "advisory_id": "GHSA-test",
        "private_contract_sha256": contract_digest,
        "roles": roles,
        "independence_attestation": "Fresh read-only synthetic audit.",
        "verdict": "PASS",
        "findings": [],
        "blocking_findings": [],
    }
    report_body = yaml.safe_dump(report_data, sort_keys=False)
    report_digest = comment_sha256(report_body)
    attestation_data = {
        "PRIVATE-GATE0-MAINTAINER-ATTESTATION": {
            "private_contract_sha256": contract_digest,
            "auditor_run_id": "auditor-run-1",
            "orchestrator_task_id": "orchestrator-task-1",
            "report_comment_url": report_url,
            "report_comment_sha256": report_digest,
            "report_created_at": "2026-07-02T09:01:00Z",
            "roles": roles,
            "verdict": "PASS",
            "head_sha": head,
            "reverification": {
                "verified_at": "2026-07-02T09:03:00Z",
                "contract_sha256": contract_digest,
                "report_sha256": report_digest,
                "reporter": "reporter",
                "collaborators_sorted": ["helper", "reporter"],
                "authorized_code_contributors": ["helper"],
                "head_sha": head,
            },
        }
    }
    attestation_body = yaml.safe_dump(attestation_data, sort_keys=False)
    return {
        "expected_reporter": "reporter",
        "current_reporter": "reporter",
        "expected_collaborators": ["helper", "reporter"],
        "current_collaborators": ["helper", "reporter"],
        "expected_authorizations": ["helper"],
        "current_authorizations": ["helper"],
        "expected_head_sha": head,
        "current_head_sha": head,
        "expected_auditor_run_id": "auditor-run-1",
        "expected_orchestrator_task_id": "orchestrator-task-1",
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
            "url": report_url,
            "body": report_body,
            "sha256": report_digest,
            "created_at": "2026-07-02T09:01:00Z",
            "updated_at": "2026-07-02T09:01:00Z",
        },
        "attestation": {
            "exists": True,
            "author": "lanmogu98",
            "author_association": "OWNER",
            "body": attestation_body,
            "sha256": comment_sha256(attestation_body),
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
        (("current_authorizations",), []),
        (("current_head_sha",), "c" * 40),
        (("contract", "sha256"), "0" * 64),
        (("report", "sha256"), "1" * 64),
        (("report", "created_at"), "2026-07-02T09:04:00Z"),
        (("report", "url"), "https://example.invalid/wrong-report"),
    ],
)
def test_private_gate_metadata_mutations_fail_closed(
    path: tuple[str, ...], value: object
) -> None:
    snapshot = _private_gate_snapshot()
    target = snapshot
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]

    assert not private_gate_is_valid(snapshot)


def _mutate_comment_body(
    snapshot: dict[str, object],
    comment_name: str,
    path: tuple[str, ...],
    value: object,
) -> None:
    comment = snapshot[comment_name]
    assert isinstance(comment, dict)
    parsed = yaml.safe_load(comment["body"])
    target = parsed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    body = yaml.safe_dump(parsed, sort_keys=False)
    comment["body"] = body
    comment["sha256"] = comment_sha256(body)


@pytest.mark.parametrize(
    ("comment_name", "path", "value"),
    [
        (
            "contract",
            ("PRIVATE-GATE0-CONTRACT-V1", "reporter"),
            "other-reporter",
        ),
        ("report", ("private_contract_sha256",), "2" * 64),
        ("report", ("roles", "reporter"), "other-reporter"),
        ("report", ("roles", "collaborators_sorted"), ["reporter"]),
        ("report", ("roles", "authorized_code_contributors"), []),
        ("report", ("roles", "auditor_run_id"), "different-auditor"),
        (
            "report",
            ("roles", "orchestrator_task_id"),
            "different-orchestrator",
        ),
        ("report", ("verdict",), "FAIL"),
        (
            "attestation",
            (
                "PRIVATE-GATE0-MAINTAINER-ATTESTATION",
                "private_contract_sha256",
            ),
            "3" * 64,
        ),
        (
            "attestation",
            ("PRIVATE-GATE0-MAINTAINER-ATTESTATION", "auditor_run_id"),
            "different-auditor",
        ),
        (
            "attestation",
            (
                "PRIVATE-GATE0-MAINTAINER-ATTESTATION",
                "report_comment_sha256",
            ),
            "4" * 64,
        ),
        (
            "attestation",
            ("PRIVATE-GATE0-MAINTAINER-ATTESTATION", "report_created_at"),
            "2026-07-02T09:04:00Z",
        ),
        (
            "attestation",
            ("PRIVATE-GATE0-MAINTAINER-ATTESTATION", "roles", "reporter"),
            "other-reporter",
        ),
        (
            "attestation",
            ("PRIVATE-GATE0-MAINTAINER-ATTESTATION", "head_sha"),
            "c" * 40,
        ),
        (
            "attestation",
            (
                "PRIVATE-GATE0-MAINTAINER-ATTESTATION",
                "reverification",
                "verified_at",
            ),
            "2026-07-02T09:00:00Z",
        ),
        (
            "attestation",
            (
                "PRIVATE-GATE0-MAINTAINER-ATTESTATION",
                "reverification",
                "reporter",
            ),
            "other-reporter",
        ),
        (
            "attestation",
            (
                "PRIVATE-GATE0-MAINTAINER-ATTESTATION",
                "reverification",
                "collaborators_sorted",
            ),
            ["reporter"],
        ),
        (
            "attestation",
            (
                "PRIVATE-GATE0-MAINTAINER-ATTESTATION",
                "reverification",
                "authorized_code_contributors",
            ),
            [],
        ),
        (
            "attestation",
            (
                "PRIVATE-GATE0-MAINTAINER-ATTESTATION",
                "reverification",
                "head_sha",
            ),
            "c" * 40,
        ),
    ],
)
def test_private_gate_schema_mutations_fail_after_local_rehash(
    comment_name: str, path: tuple[str, ...], value: object
) -> None:
    snapshot = _private_gate_snapshot()
    _mutate_comment_body(snapshot, comment_name, path, value)

    assert not private_gate_is_valid(snapshot)


def test_private_report_cross_binding_survives_local_rehash_attempt() -> None:
    snapshot = _private_gate_snapshot()
    _mutate_comment_body(
        snapshot, "report", ("private_contract_sha256",), "5" * 64
    )
    report = snapshot["report"]
    assert isinstance(report, dict)
    report_digest = report["sha256"]
    _mutate_comment_body(
        snapshot,
        "attestation",
        (
            "PRIVATE-GATE0-MAINTAINER-ATTESTATION",
            "report_comment_sha256",
        ),
        report_digest,
    )
    _mutate_comment_body(
        snapshot,
        "attestation",
        (
            "PRIVATE-GATE0-MAINTAINER-ATTESTATION",
            "reverification",
            "report_sha256",
        ),
        report_digest,
    )

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
