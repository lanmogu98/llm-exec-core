"""Pure, offline checks for the repository governance contract."""

from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml

_PRE_PASS_ACTIONS = {
    "contract_read",
    "contract_edit",
    "audit",
    "audit_remediation",
}
_PRE_PASS_ROLES = {
    "contract_read": {"auditor", "contract_remediator"},
    "contract_edit": {"contract_remediator"},
    "audit": {"auditor"},
    "audit_remediation": {"contract_remediator"},
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
_PRIVATE_CONTRACT_FIELDS = {
    "advisory_id",
    "objective",
    "scope",
    "non_goals",
    "compatibility_security_impact",
    "acceptance_criteria",
    "validation",
    "rollback_recovery",
    "reporter",
    "collaborators_sorted",
    "authorized_code_contributors",
    "intended_implementer",
}
_PRIVATE_REPORT_FIELDS = {
    "private_gate0_version",
    "advisory_id",
    "private_contract_sha256",
    "roles",
    "independence_attestation",
    "verdict",
    "findings",
    "blocking_findings",
}
_PRIVATE_ROLE_FIELDS = {
    "reporter",
    "collaborators_sorted",
    "authorized_code_contributors",
    "intended_implementer",
    "auditor_run_id",
    "orchestrator_task_id",
}
_PRIVATE_ATTESTATION_FIELDS = {
    "private_contract_sha256",
    "auditor_run_id",
    "orchestrator_task_id",
    "report_comment_id",
    "report_comment_url",
    "report_comment_sha256",
    "report_author",
    "report_author_association",
    "report_created_at",
    "report_updated_at",
    "roles",
    "verdict",
    "head_sha",
}
_PRIVATE_PROVENANCE_FIELDS = {
    "id",
    "url",
    "author",
    "author_association",
    "created_at",
    "updated_at",
    "sha256",
}
_PRIVATE_VERIFICATION_FIELDS = {
    "evidence",
    "reporter",
    "collaborators_sorted",
    "authorized_code_contributors",
    "roles",
    "head_sha",
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


def gate0_is_ready(state: str) -> bool:
    """Return whether immutable Gate 0 evidence is ready for dispatch."""
    return state == "passed"


def gate0_action_allowed(
    state: str,
    action: str,
    *,
    actor_role: str,
    task_authorized_actions: set[str],
) -> bool:
    """Apply Gate 0, actor-role, task-scope, and contribution-plane gates."""
    if action in _PRE_PASS_ACTIONS:
        return actor_role in _PRE_PASS_ROLES[action]
    if not gate0_is_ready(state) or actor_role != "implementation_agent":
        return False
    return (
        action in _IMPLEMENTATION_CONTRIBUTION_ACTIONS
        and action in task_authorized_actions
    )


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return parsed


def contribution_is_authorized(
    authorization: Mapping[str, str],
    *,
    contributor: str,
    work_item: str,
    expected_scope: str,
    expected_delivery: str,
    earliest_event_at: datetime,
    completed_at: datetime | None = None,
) -> bool:
    """Check a synthetic outside-contribution authorization record."""
    try:
        issued_at = _parse_timestamp(authorization.get("issued_at"))
        created_at = _parse_timestamp(authorization.get("created_at"))
    except (TypeError, ValueError):
        return False

    if (
        earliest_event_at.tzinfo is None
        or earliest_event_at.utcoffset() is None
    ):
        return False
    if issued_at != created_at or created_at >= earliest_event_at:
        return False

    expires_at_value = authorization.get("expires_at")
    if expires_at_value == "completion":
        if completed_at is not None:
            if completed_at.tzinfo is None or completed_at.utcoffset() is None:
                return False
            if earliest_event_at > completed_at:
                return False
    else:
        try:
            expires_at = _parse_timestamp(expires_at_value)
        except (TypeError, ValueError):
            return False
        if earliest_event_at > expires_at:
            return False

    return all(
        (
            authorization.get("author") == "lanmogu98",
            authorization.get("author_association") == "OWNER",
            authorization.get("created_at") == authorization.get("updated_at"),
            authorization.get("contributor") == contributor,
            authorization.get("work_item") == work_item,
            authorization.get("allowed_scope") == expected_scope,
            authorization.get("allowed_delivery") == expected_delivery,
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


def _exact_mapping(value: object, fields: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == fields


def _sorted_string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) for item in value)
        and value == sorted(value)
        and len(value) == len(set(value))
    )


def _parse_yaml_mapping(comment: object) -> Mapping[str, Any] | None:
    if not isinstance(comment, Mapping):
        return None
    body = comment.get("body")
    if not isinstance(body, str):
        return None
    try:
        parsed = yaml.safe_load(body)
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _comment_matches_frozen_provenance(
    comment: object,
    frozen: object,
    *,
    owner_required: bool,
) -> bool:
    if not _exact_mapping(frozen, _PRIVATE_PROVENANCE_FIELDS):
        return False
    assert isinstance(frozen, Mapping)
    stable_id = frozen.get("id")
    url = frozen.get("url")
    author = frozen.get("author")
    author_association = frozen.get("author_association")
    digest = frozen.get("sha256")
    if (
        not isinstance(stable_id, int)
        or isinstance(stable_id, bool)
        or stable_id <= 0
    ):
        return False
    if not isinstance(url, str) or not url.strip():
        return False
    parsed_url = urlparse(url)
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or not parsed_url.path
    ):
        return False
    if not isinstance(author, str) or not author.strip():
        return False
    if (
        not isinstance(author_association, str)
        or not author_association.strip()
    ):
        return False
    if owner_required and author_association != "OWNER":
        return False
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return False
    try:
        created_at = _parse_timestamp(frozen.get("created_at"))
        updated_at = _parse_timestamp(frozen.get("updated_at"))
    except (TypeError, ValueError):
        return False
    if created_at != updated_at:
        return False
    if not _comment_is_unchanged(comment, owner_required=owner_required):
        return False
    assert isinstance(comment, Mapping)
    return all(
        comment.get(field) == frozen.get(field)
        for field in _PRIVATE_PROVENANCE_FIELDS
    )


def private_gate_is_valid(snapshot: Mapping[str, Any]) -> bool:
    """Validate independently frozen private-advisory evidence."""
    contract = snapshot.get("contract")
    report = snapshot.get("report")
    attestation = snapshot.get("attestation")
    verification = snapshot.get("verification")
    frozen_evidence = snapshot.get("frozen_evidence")
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
    if not _sorted_string_list(snapshot.get("current_authorizations")):
        return False
    if snapshot.get("expected_head_sha") != current_head:
        return False
    if not _exact_mapping(
        frozen_evidence,
        {"contract", "report", "attestation", "verification"},
    ):
        return False
    assert isinstance(frozen_evidence, Mapping)
    for name, comment, owner_required in (
        ("contract", contract, True),
        ("report", report, False),
        ("attestation", attestation, True),
        ("verification", verification, True),
    ):
        if not _comment_matches_frozen_provenance(
            comment,
            frozen_evidence.get(name),
            owner_required=owner_required,
        ):
            return False
    assert isinstance(contract, Mapping)
    assert isinstance(report, Mapping)
    assert isinstance(attestation, Mapping)
    assert isinstance(verification, Mapping)

    contract_document = _parse_yaml_mapping(contract)
    report_document = _parse_yaml_mapping(report)
    attestation_document = _parse_yaml_mapping(attestation)
    verification_document = _parse_yaml_mapping(verification)
    if not _exact_mapping(contract_document, {"PRIVATE-GATE0-CONTRACT-V1"}):
        return False
    if not _exact_mapping(report_document, _PRIVATE_REPORT_FIELDS):
        return False
    if not _exact_mapping(
        attestation_document, {"PRIVATE-GATE0-MAINTAINER-ATTESTATION"}
    ):
        return False
    if not _exact_mapping(
        verification_document, {"PRIVATE-GATE0-OWNER-VERIFICATION-V1"}
    ):
        return False

    assert contract_document is not None
    assert report_document is not None
    assert attestation_document is not None
    assert verification_document is not None
    contract_data = contract_document["PRIVATE-GATE0-CONTRACT-V1"]
    attestation_data = attestation_document[
        "PRIVATE-GATE0-MAINTAINER-ATTESTATION"
    ]
    verification_data = verification_document[
        "PRIVATE-GATE0-OWNER-VERIFICATION-V1"
    ]
    if not _exact_mapping(contract_data, _PRIVATE_CONTRACT_FIELDS):
        return False
    if not _exact_mapping(attestation_data, _PRIVATE_ATTESTATION_FIELDS):
        return False
    if not _exact_mapping(verification_data, _PRIVATE_VERIFICATION_FIELDS):
        return False
    assert isinstance(contract_data, Mapping)
    assert isinstance(attestation_data, Mapping)
    assert isinstance(verification_data, Mapping)

    string_contract_fields = {
        "advisory_id",
        "objective",
        "compatibility_security_impact",
        "rollback_recovery",
        "reporter",
        "intended_implementer",
    }
    if any(
        not isinstance(contract_data.get(field), str)
        or not contract_data.get(field)
        for field in string_contract_fields
    ):
        return False
    if any(
        not isinstance(contract_data.get(field), list)
        for field in {
            "scope",
            "non_goals",
            "acceptance_criteria",
            "validation",
        }
    ):
        return False
    contract_collaborators = contract_data.get("collaborators_sorted")
    contract_authorizations = contract_data.get("authorized_code_contributors")
    if not _sorted_string_list(contract_collaborators):
        return False
    if not _sorted_string_list(contract_authorizations):
        return False
    if contract_data.get("reporter") != snapshot.get("current_reporter"):
        return False
    if contract_collaborators != snapshot.get("current_collaborators"):
        return False
    if contract_authorizations != snapshot.get("current_authorizations"):
        return False

    contract_body = contract.get("body")
    report_body = report.get("body")
    if not isinstance(contract_body, str) or not isinstance(report_body, str):
        return False
    contract_digest = comment_sha256(contract_body)
    report_digest = comment_sha256(report_body)
    expected_roles = {
        "reporter": contract_data.get("reporter"),
        "collaborators_sorted": contract_collaborators,
        "authorized_code_contributors": contract_authorizations,
        "intended_implementer": contract_data.get("intended_implementer"),
        "auditor_run_id": snapshot.get("expected_auditor_run_id"),
        "orchestrator_task_id": snapshot.get("expected_orchestrator_task_id"),
    }
    report_roles = report_document.get("roles")
    if not _exact_mapping(report_roles, _PRIVATE_ROLE_FIELDS):
        return False
    if report_roles != expected_roles:
        return False
    if report_document.get("private_gate0_version") != 1:
        return False
    if report_document.get("advisory_id") != contract_data.get("advisory_id"):
        return False
    if report_document.get("private_contract_sha256") != contract_digest:
        return False
    if not isinstance(report_document.get("independence_attestation"), str):
        return False
    if not report_document.get("independence_attestation"):
        return False
    if report_document.get("verdict") != "PASS":
        return False
    if not isinstance(report_document.get("findings"), list):
        return False
    if report_document.get("blocking_findings") != []:
        return False

    attestation_roles = attestation_data.get("roles")
    if not _exact_mapping(attestation_roles, _PRIVATE_ROLE_FIELDS):
        return False
    if attestation_roles != expected_roles:
        return False
    if (
        attestation_data.get("auditor_run_id")
        != expected_roles["auditor_run_id"]
    ):
        return False
    if (
        attestation_data.get("orchestrator_task_id")
        != expected_roles["orchestrator_task_id"]
    ):
        return False
    if attestation_data.get("private_contract_sha256") != contract_digest:
        return False
    if attestation_data.get("report_comment_id") != report.get("id"):
        return False
    if attestation_data.get("report_comment_url") != report.get("url"):
        return False
    if attestation_data.get("report_comment_sha256") != report_digest:
        return False
    if attestation_data.get("report_author") != report.get("author"):
        return False
    if attestation_data.get("report_author_association") != report.get(
        "author_association"
    ):
        return False
    report_created_at = report.get("created_at")
    if attestation_data.get("report_created_at") != report_created_at:
        return False
    if attestation_data.get("report_updated_at") != report.get("updated_at"):
        return False
    if attestation_data.get("verdict") != "PASS":
        return False
    if attestation_data.get("head_sha") != current_head:
        return False

    verification_evidence = verification_data.get("evidence")
    if not _exact_mapping(
        verification_evidence, {"contract", "report", "attestation"}
    ):
        return False
    assert isinstance(verification_evidence, Mapping)
    for name in ("contract", "report", "attestation"):
        provenance = verification_evidence.get(name)
        if not _exact_mapping(provenance, _PRIVATE_PROVENANCE_FIELDS):
            return False
        if provenance != frozen_evidence.get(name):
            return False
    if verification_data.get("reporter") != snapshot.get("current_reporter"):
        return False
    if verification_data.get("collaborators_sorted") != snapshot.get(
        "current_collaborators"
    ):
        return False
    if verification_data.get("authorized_code_contributors") != snapshot.get(
        "current_authorizations"
    ):
        return False
    if verification_data.get("roles") != expected_roles:
        return False
    if verification_data.get("head_sha") != current_head:
        return False

    try:
        contract_created = _parse_timestamp(contract.get("created_at"))
        report_created = _parse_timestamp(report_created_at)
        attestation_created = _parse_timestamp(attestation.get("created_at"))
        verification_created = _parse_timestamp(verification.get("created_at"))
    except (AttributeError, TypeError, ValueError):
        return False
    if any(
        timestamp.tzinfo is None or timestamp.utcoffset() is None
        for timestamp in (
            contract_created,
            report_created,
            attestation_created,
            verification_created,
        )
    ):
        return False
    if not (
        contract_created
        < report_created
        < attestation_created
        < verification_created
    ):
        return False
    return True


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
