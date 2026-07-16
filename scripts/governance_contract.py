"""Pure, offline checks for the repository governance contract."""

from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Any, Mapping

_PRE_PASS_ACTIONS = {
    "contract_read",
    "contract_edit",
    "audit",
    "audit_remediation",
}
_NON_SUCCESS_CONDITIONS = {
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
}
_FINAL_RULESET_FAILURES = {
    "missing_check",
    "wrong_name",
    "wrong_source",
    "mismatch",
}


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_issue_sha256(title: str, body: str | None) -> str:
    """Hash an Issue title/body using the Gate 0 canonicalization rule."""
    normalized_title = _normalize_newlines(title)
    normalized_body = _normalize_newlines(body or "")
    return _sha256(f"{normalized_title}\n\n{normalized_body}")


def comment_sha256(body: str) -> str:
    """Hash an API-returned comment body without trimming it."""
    return _sha256(_normalize_newlines(body))


def gate0_action_allowed(state: str, action: str) -> bool:
    """Return whether an action is allowed by the public Gate 0 state."""
    return state == "passed" or action in _PRE_PASS_ACTIONS


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def contribution_is_authorized(
    authorization: Mapping[str, str],
    *,
    contributor: str,
    work_item: str,
    delivery_at: datetime,
) -> bool:
    """Check a synthetic outside-contribution authorization record."""
    try:
        issued_at = _parse_timestamp(authorization.get("issued_at"))
        expires_at = _parse_timestamp(authorization.get("expires_at"))
    except (TypeError, ValueError):
        return False

    return all(
        (
            authorization.get("author") == "lanmogu98",
            authorization.get("author_association") == "OWNER",
            authorization.get("created_at") == authorization.get("updated_at"),
            authorization.get("contributor") == contributor,
            authorization.get("work_item") == work_item,
            bool(authorization.get("allowed_scope")),
            bool(authorization.get("allowed_delivery")),
            issued_at <= delivery_at <= expires_at,
        )
    )


def workflow_change_is_authorized(
    comment: Mapping[str, str], head_sha: str
) -> bool:
    """Check the exact-head owner comment required for workflow changes."""
    return all(
        (
            comment.get("author") == "lanmogu98",
            comment.get("author_association") == "OWNER",
            comment.get("created_at") == comment.get("updated_at"),
            comment.get("body") == f"WORKFLOW-CHANGE-AUTHORIZED: {head_sha}",
        )
    )


def _comment_is_unchanged(
    comment: object, *, owner_required: bool = False
) -> bool:
    if not isinstance(comment, Mapping) or not comment.get("exists"):
        return False
    body = comment.get("body")
    digest = comment.get("sha256")
    if not isinstance(body, str) or digest != comment_sha256(body):
        return False
    if comment.get("created_at") != comment.get("updated_at"):
        return False
    if owner_required:
        return (
            comment.get("author") == "lanmogu98"
            and comment.get("author_association") == "OWNER"
        )
    return True


def private_gate_is_valid(snapshot: Mapping[str, Any]) -> bool:
    """Validate an unchanged synthetic private-advisory evidence snapshot."""
    contract = snapshot.get("contract")
    report = snapshot.get("report")
    attestation = snapshot.get("attestation")
    current_head = snapshot.get("current_head_sha")

    if snapshot.get("expected_reporter") != snapshot.get("current_reporter"):
        return False
    expected_collaborators = snapshot.get("expected_collaborators")
    current_collaborators = snapshot.get("current_collaborators")
    if expected_collaborators != current_collaborators:
        return False
    if not isinstance(
        current_collaborators, list
    ) or current_collaborators != sorted(current_collaborators):
        return False
    if snapshot.get("expected_authorizations") != snapshot.get(
        "current_authorizations"
    ):
        return False
    if snapshot.get("expected_head_sha") != current_head:
        return False
    if not _comment_is_unchanged(contract, owner_required=True):
        return False
    if not _comment_is_unchanged(report):
        return False
    if not _comment_is_unchanged(attestation, owner_required=True):
        return False
    if not isinstance(report, Mapping) or "verdict: PASS" not in str(
        report.get("body")
    ):
        return False
    if not isinstance(attestation, Mapping):
        return False
    return "verdict: PASS" in str(attestation.get("body")) and str(
        current_head
    ) in str(attestation.get("body"))


def assess_ci_evidence(
    *,
    condition: str,
    owner: str,
    elapsed_minutes: int,
    repair_deadline_hours: int,
) -> dict[str, object]:
    """Return the fail-closed recovery state for CI/settings non-success."""
    if condition not in _NON_SUCCESS_CONDITIONS:
        raise ValueError(f"unknown non-success condition: {condition}")
    return {
        "merge_frozen": True,
        "preliminary_ruleset": "active",
        "owner": owner,
        "detection_minutes": elapsed_minutes,
        "repair_deadline_hours": repair_deadline_hours,
    }


def assess_final_ruleset_failure(condition: str) -> dict[str, object]:
    """Return the settings-plane rollback required before PR recovery."""
    if condition not in _FINAL_RULESET_FAILURES:
        raise ValueError(f"unknown final-ruleset failure: {condition}")
    return {
        "final_ruleset": "disable_or_delete",
        "preliminary_ruleset": "verify_exported_anchor_active",
        "merge_frozen": True,
        "pr_repair_allowed": False,
    }
