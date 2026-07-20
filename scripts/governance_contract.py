"""Pure, offline checks for the repository governance contract."""

from __future__ import annotations


_SEMANTIC_CHANGE_KINDS = {
    "objective",
    "scope",
    "behavior",
    "compatibility",
    "security",
    "authority",
}
_HIGH_RISK_CHANGE_KINDS = {
    "workflow",
    "permissions",
    "secrets",
    "security",
    "external_code",
    "breaking_api",
    "catalog",
}
_OWNER_ONLY_ACTIONS = {
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
}
_IMPLEMENTATION_CONTRIBUTION_ACTIONS = {
    "read_evidence",
    "issue_comment",
    "topic_branch",
    "commit",
    "push_topic_branch",
    "tracked_file_write",
    "workflow_file_write",
    "pr_update",
    "mock_ci_control",
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


def implementation_action_allowed(
    *, issue_accepted: bool, dispatch_matches: bool, action: str
) -> bool:
    """Return whether an implementation agent may perform an action."""
    return (
        issue_accepted
        and dispatch_matches
        and action in _IMPLEMENTATION_CONTRIBUTION_ACTIONS
        and action not in _OWNER_ONLY_ACTIONS
    )


def dispatch_requires_refresh(change_kind: str) -> bool:
    """Return whether a material semantic change requires new dispatch."""
    return change_kind in _SEMANTIC_CHANGE_KINDS


def independent_review_required(change_kind: str) -> bool:
    """Return whether a change requires independent exact-head review."""
    return change_kind in _HIGH_RISK_CHANGE_KINDS


def independent_review_is_valid(
    *,
    change_kind: str,
    head_sha: str,
    review_head_sha: str | None,
    reviewer_is_implementer: bool,
    verdict: str | None,
) -> bool:
    """Return whether the required high-risk review is valid."""
    if not independent_review_required(change_kind):
        return True
    return (
        verdict == "PASS"
        and review_head_sha == head_sha
        and not reviewer_is_implementer
    )


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
    result: dict[str, object] = {
        "merge_frozen": True,
        "preliminary_ruleset": "active",
        "owner": owner,
        "detection_minutes": elapsed_minutes,
        "repair_deadline_hours": repair_deadline_hours,
    }
    if condition == "preliminary_ruleset_readback_failed":
        result["preliminary_ruleset"] = "unverified_restore_required"
        result["pr_repair_allowed"] = False
    return result


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
