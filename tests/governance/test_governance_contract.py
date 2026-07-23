import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
RELEASE_BUILD_CONSTRAINTS = ROOT / ".github" / "release-build-constraints.txt"
RELEASE_RUNBOOK = ROOT / "docs" / "governance" / "pypi-release.md"
EXPECTED_CHECKS = {"CI / python-3.10", "CI / python-3.13"}
EXPECTED_RELEASE_ACTIONS = {
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    "actions/download-artifact": "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
    "astral-sh/setup-uv": "c771a70e6277c0a99b617c7a806ffedaca235ff9",
    "pypa/gh-action-pypi-publish": (
        "ba38be9e461d3875417946c167d0b5f3d385a247"
    ),
}
CONTRACT = runpy.run_path(str(ROOT / "scripts" / "governance_contract.py"))
assess_ci_evidence = CONTRACT["assess_ci_evidence"]
assess_final_ruleset_failure = CONTRACT["assess_final_ruleset_failure"]
dispatch_requires_refresh = CONTRACT["dispatch_requires_refresh"]
implementation_action_allowed = CONTRACT["implementation_action_allowed"]
independent_review_is_valid = CONTRACT["independent_review_is_valid"]
independent_review_required = CONTRACT["independent_review_required"]


def _release_step_run(step_name: str) -> str:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    matches = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == step_name
    ]
    assert len(matches) == 1
    return matches[0]


def _embedded_python(step_name: str) -> str:
    run = _release_step_run(step_name)
    marker = "python - <<'PY'\n"
    assert run.count(marker) == 1
    script = run.split(marker, 1)[1]
    assert script.endswith("\nPY\n")
    return script[: -len("\nPY\n")]


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


def test_release_dispatch_and_source_identity_fail_closed() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    preflight = workflow["jobs"]["preflight"]
    preflight_runs = "\n".join(
        step["run"] for step in preflight["steps"] if "run" in step
    )
    all_runs = "\n".join(
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "run" in step
    )

    assert workflow["name"] == "Release to PyPI"
    assert workflow["permissions"] == {"contents": "read"}
    assert set(inputs) == {"version", "expected_sha", "dry_run"}
    assert inputs["version"]["required"] == "true"
    assert inputs["version"]["type"] == "string"
    assert inputs["expected_sha"]["required"] == "true"
    assert inputs["expected_sha"]["type"] == "string"
    assert inputs["dry_run"]["type"] == "boolean"
    assert inputs["dry_run"]["default"] == "true"
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert workflow["concurrency"]["group"] == "llm-exec-core-pypi-release"
    assert preflight["permissions"] == {"contents": "read"}
    assert "refs/heads/main" in preflight_runs
    assert "EXPECTED_SHA" in preflight_runs
    assert "GITHUB_SHA" in preflight_runs
    assert "^[0-9a-f]{40}$" in preflight_runs
    assert "api.github.com" in text
    assert "/commits/main" in text
    assert "pyproject.toml" in preflight_runs
    assert "src/llm_exec_core/__init__.py" in preflight_runs
    assert "uv.lock" in preflight_runs
    assert "Version(" in preflight_runs
    assert "is_prerelease" in preflight_runs
    assert "https://pypi.org/pypi/llm-exec-core/" in preflight_runs
    assert "pull_request_target" not in text
    assert "secrets." not in text
    assert "${{ inputs." not in all_runs


def test_release_build_is_exact_and_hash_constrained() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    build = workflow["jobs"]["build"]
    build_runs = "\n".join(
        step["run"] for step in build["steps"] if "run" in step
    )
    constraints = RELEASE_BUILD_CONSTRAINTS.read_text(encoding="utf-8")
    all_action_refs = re.findall(r"uses:\s*([^@\s]+)@([^\s#]+)", text)
    action_refs = dict(all_action_refs)
    setup_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    ]

    assert all(
        re.fullmatch(r"[0-9a-f]{40}", ref) for _, ref in all_action_refs
    )
    assert action_refs == EXPECTED_RELEASE_ACTIONS
    assert all(step["with"]["version"] == "0.11.31" for step in setup_steps)
    assert all(step["with"]["enable-cache"] == "false" for step in setup_steps)
    assert "3.10.20" in text
    assert "3.13.14" in text
    assert text.count("uv build ") == 1
    assert "uv build --no-sources" in build_runs
    assert (
        "--build-constraints .github/release-build-constraints.txt"
        in build_runs
    )
    assert "--require-hashes" in build_runs
    assert "--python 3.13.14" in build_runs
    assert "--clear" in build_runs
    assert "--out-dir release-artifact/packages" in build_runs
    assert constraints.startswith("setuptools==83.0.0")
    hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})", constraints)
    assert set(hashes) == {
        "29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3",
        "025bccbbf0fa05b6192bc64ae1e7b16e001fd6d6d4d5de03c97b1c1ade523bef",
    }


def test_release_validates_two_distributions_and_clean_installs_each() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    build = workflow["jobs"]["build"]
    build_runs = "\n".join(
        step["run"] for step in build["steps"] if "run" in step
    )
    smoke = workflow["jobs"]["smoke"]
    smoke_runs = "\n".join(
        step["run"] for step in smoke["steps"] if "run" in step
    )
    matrix = smoke["strategy"]["matrix"]["include"]
    upload = next(
        step
        for step in build["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )

    assert "llm_exec_core-" in build_runs
    assert "py3-none-any.whl" in build_runs
    assert ".tar.gz" in build_runs
    assert "METADATA" in build_runs
    assert "PKG-INFO" in build_runs
    assert "SHA256SUMS" in build_runs
    assert "hashlib.file_digest" in build_runs
    assert upload["with"]["name"] == "validated-distributions"
    assert upload["with"]["path"] == "release-artifact"
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["compression-level"] == "0"
    assert {
        (item["label"], item["python-version"], item["distribution"])
        for item in matrix
    } == {
        ("3.10-wheel", "3.10.20", "wheel"),
        ("3.10-sdist", "3.10.20", "sdist"),
        ("3.13-wheel", "3.13.14", "wheel"),
        ("3.13-sdist", "3.13.14", "sdist"),
    }
    assert "sha256sum --check" in smoke_runs
    assert "uv export --frozen --no-dev --no-emit-project" in smoke_runs
    assert "uv venv --python" in smoke_runs
    assert "uv pip install" in smoke_runs
    assert "--no-deps" in smoke_runs
    assert "--build-constraints .github/release-build-constraints.txt" in (
        smoke_runs
    )
    assert "import llm_exec_core" in smoke_runs
    assert "importlib.metadata.version" in smoke_runs


def test_publish_job_is_minimal_oidc_and_protected_environment_only() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    gate = jobs["publication_gate"]
    gate_runs = "\n".join(
        step["run"] for step in gate["steps"] if "run" in step
    )
    publish = jobs["publish"]
    publish_steps = publish["steps"]
    publisher = publish_steps[1]

    assert gate["if"] == "${{ !inputs.dry_run }}"
    assert "LLM_EXEC_CORE_PYPI_PUBLISH_ENABLED" in gate_runs
    assert '!= "true"' in gate_runs
    assert publish["if"] == "${{ !inputs.dry_run }}"
    assert publish["needs"] == "publication_gate"
    assert publish["environment"]["name"] == "pypi"
    assert publish["permissions"] == {"id-token": "write"}
    assert len(publish_steps) == 2
    assert publish_steps[0]["uses"].startswith("actions/download-artifact@")
    assert publisher["uses"].startswith("pypa/gh-action-pypi-publish@")
    assert publisher["with"]["packages-dir"] == "release-artifact/packages/"
    assert publisher["with"]["attestations"] == "true"
    assert publisher["with"]["print-hash"] == "true"
    assert "user" not in publisher["with"]
    assert "password" not in publisher["with"]
    assert "skip-existing" not in publisher["with"]
    assert all("run" not in step for step in publish_steps)
    assert all(
        job.get("permissions", {}).get("id-token") != "write"
        for name, job in jobs.items()
        if name != "publish"
    )


def test_post_publish_verification_is_bounded_and_cryptographic() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    verify = workflow["jobs"]["verify_publication"]
    verify_runs = "\n".join(
        step["run"] for step in verify["steps"] if "run" in step
    )

    assert verify["needs"] == "publish"
    assert verify["permissions"] == {"contents": "read"}
    assert "https://pypi.org/pypi/llm-exec-core/" in verify_runs
    assert "https://pypi.org/simple/llm-exec-core/" in verify_runs
    assert "application/vnd.pypi.simple.v1+json" in verify_runs
    assert "/integrity/llm-exec-core/" in verify_runs
    assert "application/vnd.pypi.integrity.v1+json" in verify_runs
    assert "files.pythonhosted.org" in verify_runs
    assert "MAX_METADATA_BYTES" in verify_runs
    assert "MAX_FILE_BYTES" in verify_runs
    assert "TIMEOUT_SECONDS" in verify_runs
    assert "MAX_ATTEMPTS" in verify_runs
    assert "MAX_REDIRECTS = 0" in verify_runs
    assert "yanked" in verify_runs
    assert "size" in verify_runs
    assert "sha256" in verify_runs
    assert "hashlib.sha256" in verify_runs
    assert 'publisher.get("repository")' in verify_runs
    assert 'publisher.get("workflow")' in verify_runs
    assert 'publisher.get("environment")' in verify_runs
    assert "1.3.6.1.4.1.57264.1.13" in verify_runs
    assert "EXPECTED_SHA" in verify_runs
    assert "pypi-attestations==0.0.29" in verify_runs
    assert "Provenance.model_validate" in verify_runs
    assert "attestation.verify(" in verify_runs
    assert "verification_material.certificate" in verify_runs
    assert "timeout 120s" in verify_runs


@pytest.mark.parametrize(
    "publisher_extra",
    [
        pytest.param({}, id="claims-omitted"),
        pytest.param({"claims": None}, id="claims-null"),
        pytest.param(
            {"claims": {"index-retained": "not-authoritative"}},
            id="claims-object",
        ),
    ],
)
def test_saved_provenance_is_verified_before_its_claims_are_inspected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publisher_extra: dict,
) -> None:
    script = _embedded_python(
        "Cryptographically verify saved provenance and signed claims"
    )
    expected_sha = "a" * 40
    expected_signer = (
        "https://github.com/lanmogu98/llm-exec-core/"
        ".github/workflows/release.yml@refs/heads/main"
    )
    expected_extensions = {
        "1.3.6.1.4.1.57264.1.8": (
            "https://token.actions.githubusercontent.com"
        ),
        "1.3.6.1.4.1.57264.1.9": expected_signer,
        "1.3.6.1.4.1.57264.1.10": expected_sha,
        "1.3.6.1.4.1.57264.1.11": "github-hosted",
        "1.3.6.1.4.1.57264.1.12": (
            "https://github.com/lanmogu98/llm-exec-core"
        ),
        "1.3.6.1.4.1.57264.1.13": expected_sha,
        "1.3.6.1.4.1.57264.1.14": "refs/heads/main",
        "1.3.6.1.4.1.57264.1.18": expected_signer,
        "1.3.6.1.4.1.57264.1.19": expected_sha,
        "1.3.6.1.4.1.57264.1.20": "workflow_dispatch",
        "1.3.6.1.4.1.57264.1.22": "public",
        "1.3.6.1.4.1.57264.1.23": "pypi",
    }
    verified_attestations = []
    inspected_attestations = []

    class FakeDistribution:
        def __init__(self, path: Path) -> None:
            self.name = path.name
            self.digest = hashlib.sha256(path.read_bytes()).hexdigest()

        @classmethod
        def from_file(cls, path: Path) -> "FakeDistribution":
            return cls(path)

    class FakeGitHubPublisher:
        def __init__(
            self,
            *,
            kind: str,
            repository: str,
            workflow: str,
            environment: str,
        ) -> None:
            self.kind = kind
            self.repository = repository
            self.workflow = workflow
            self.environment = environment

    class FakeVerificationMaterial:
        def __init__(self, attestation: "FakeAttestation") -> None:
            self.attestation = attestation

        @property
        def certificate(self) -> bytes:
            assert self.attestation in verified_attestations
            inspected_attestations.append(self.attestation)
            return self.attestation.certificate

    class FakeAttestation:
        def __init__(self, certificate: bytes) -> None:
            self.certificate = certificate
            self.verification_material = FakeVerificationMaterial(self)

        def verify(
            self,
            publisher: FakeGitHubPublisher,
            distribution: FakeDistribution,
        ) -> tuple[str, None]:
            assert publisher.repository == "lanmogu98/llm-exec-core"
            assert distribution.digest
            verified_attestations.append(self)
            return (
                "https://docs.pypi.org/attestations/publish/v1",
                None,
            )

    class FakeProvenance:
        def __init__(self, bundles: list) -> None:
            self.attestation_bundles = bundles

        @classmethod
        def model_validate(cls, raw: dict) -> "FakeProvenance":
            bundles = []
            for raw_bundle in raw["attestation_bundles"]:
                raw_publisher = raw_bundle["publisher"]
                publisher = FakeGitHubPublisher(
                    kind=raw_publisher["kind"],
                    repository=raw_publisher["repository"],
                    workflow=raw_publisher["workflow"],
                    environment=raw_publisher["environment"],
                )
                attestations = [
                    FakeAttestation(item["certificate"].encode("utf-8"))
                    for item in raw_bundle["attestations"]
                ]
                bundles.append(
                    SimpleNamespace(
                        publisher=publisher,
                        attestations=attestations,
                    )
                )
            return cls(bundles)

    class FakeObjectIdentifier:
        def __init__(self, dotted_string: str) -> None:
            self.dotted_string = dotted_string

    class FakeExtensionOID:
        SUBJECT_ALTERNATIVE_NAME = FakeObjectIdentifier("san")

    class FakeUniformResourceIdentifier:
        pass

    class FakeSubjectAlternativeName:
        def get_values_for_type(self, value_type: type) -> list[str]:
            assert value_type is FakeUniformResourceIdentifier
            return [expected_signer]

    class FakeExtensions:
        def get_extension_for_oid(
            self, oid: FakeObjectIdentifier
        ) -> SimpleNamespace:
            if oid.dotted_string == "san":
                return SimpleNamespace(value=FakeSubjectAlternativeName())
            value = expected_extensions[oid.dotted_string].encode("utf-8")
            assert len(value) < 128
            der_value = b"\x0c" + bytes([len(value)]) + value
            return SimpleNamespace(value=SimpleNamespace(value=der_value))

    class FakeCertificate:
        extensions = FakeExtensions()

    def fake_load_der_x509_certificate(value: bytes) -> FakeCertificate:
        assert value.startswith(b"certificate-")
        return FakeCertificate()

    pypi_module = ModuleType("pypi_attestations")
    pypi_module.Distribution = FakeDistribution
    pypi_module.GitHubPublisher = FakeGitHubPublisher
    pypi_module.Provenance = FakeProvenance
    cryptography_module = ModuleType("cryptography")
    x509_module = ModuleType("cryptography.x509")
    oid_module = ModuleType("cryptography.x509.oid")
    x509_module.UniformResourceIdentifier = FakeUniformResourceIdentifier
    x509_module.load_der_x509_certificate = fake_load_der_x509_certificate
    oid_module.ExtensionOID = FakeExtensionOID
    oid_module.ObjectIdentifier = FakeObjectIdentifier
    cryptography_module.x509 = x509_module

    monkeypatch.setitem(sys.modules, "pypi_attestations", pypi_module)
    monkeypatch.setitem(sys.modules, "cryptography", cryptography_module)
    monkeypatch.setitem(sys.modules, "cryptography.x509", x509_module)
    monkeypatch.setitem(sys.modules, "cryptography.x509.oid", oid_module)

    runner_temp = tmp_path / "runner"
    verification_dir = runner_temp / "public-verification"
    verification_dir.mkdir(parents=True)
    evidence_files = {}
    for index, name in enumerate(
        (
            "llm_exec_core-0.4.2-py3-none-any.whl",
            "llm_exec_core-0.4.2.tar.gz",
        )
    ):
        distribution = verification_dir / name
        distribution.write_bytes(f"distribution-{index}".encode("utf-8"))
        evidence_files[name] = hashlib.sha256(
            distribution.read_bytes()
        ).hexdigest()
        publisher = {
            "environment": "pypi",
            "kind": "GitHub",
            "repository": "lanmogu98/llm-exec-core",
            "workflow": "release.yml",
            **publisher_extra,
        }
        provenance = {
            "version": 1,
            "attestation_bundles": [
                {
                    "publisher": publisher,
                    "attestations": [{"certificate": f"certificate-{index}"}],
                }
            ],
        }
        (verification_dir / f"{name}.provenance.json").write_text(
            json.dumps(provenance),
            encoding="utf-8",
        )

    evidence_dir = tmp_path / "release-artifact"
    evidence_dir.mkdir()
    (evidence_dir / "release-evidence.json").write_text(
        json.dumps({"files": evidence_files}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EXPECTED_SHA", expected_sha)
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    exec(compile(script, "<release-verifier>", "exec"), {})

    assert len(verified_attestations) == 2
    assert inspected_attestations == verified_attestations


def test_release_runbook_records_owner_gates_and_safe_recovery() -> None:
    runbook = RELEASE_RUNBOOK.read_text(encoding="utf-8")

    assert "lanmogu98/llm-exec-core" in runbook
    assert ".github/workflows/release.yml" in runbook
    assert "environment `pypi`" in runbook
    assert "pending Trusted Publisher" in runbook
    assert "LLM_EXEC_CORE_PYPI_PUBLISH_ENABLED" in runbook
    assert "dry run" in runbook.lower()
    assert "API token" in runbook
    assert "must not" in runbook
    assert "yank" in runbook.lower()
    assert "never reuse" in runbook.lower()
    assert "Core #41" in runbook
    assert "Core #42" in runbook
    assert "Core #43" in runbook
    assert "Core #40" in runbook


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
