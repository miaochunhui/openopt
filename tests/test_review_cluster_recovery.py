"""Portable audit controls plus explicitly opted-in historical evidence replay.

OPENOPT_RUN_RECOVERY_EVIDENCE_TESTS=1 enables the real 3FS/Git campaign tests.
Default skips are not campaign qualification. Missing opted-in dependencies
fail loudly before fixture copying or historical code loading.
"""

import copy
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from xml.etree import ElementTree as ET

import pytest

from scripts import review_cluster_recovery as audit

REPORT = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/failure-recovery-real-v3/recovery.json")
REPORT_PIN = "cd911cf2cbe4f1488511157b02cb8c81c33b6ad740ea5f6437915bf4dbbb6610"
LIVE = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/live-recovery-v1/live-095407908933.json")
LIVE_PIN = "09bb2842415c903046258ecff9f26f59a4bfdc21fb3fd50fcd8dd9ed16deb6a1"
HISTORICAL_COMMIT = "e618b68c5209794c8cb1ec4cef727e7e13ef1ada"
HISTORICAL_PIN = "784eba817bea9d2dfe48288827bfb4392068a564a01f6fba6cd65e4fb86b4bbc"
ANCHORS = {"historical_commit": HISTORICAL_COMMIT, "historical_sha256": HISTORICAL_PIN}
CLI_ANCHORS = ["--historical-commit", HISTORICAL_COMMIT,
               "--historical-verifier-sha256", HISTORICAL_PIN]
EVIDENCE_ENV = "OPENOPT_RUN_RECOVERY_EVIDENCE_TESTS"
EVIDENCE_SKIP = (
    "historical recovery evidence NOT RUN; set OPENOPT_RUN_RECOVERY_EVIDENCE_TESTS=1 "
    "with approved 3FS artifacts and Git objects; no campaign qualification"
)


def require_evidence_opt_in():
    value = os.environ.get(EVIDENCE_ENV)
    if value in (None, "", "0"):
        pytest.skip(EVIDENCE_SKIP)
    if value != "1":
        pytest.fail(f"{EVIDENCE_ENV} must be exactly 1, 0 or unset", pytrace=False)


@pytest.fixture(scope="module")
def real_evidence():
    require_evidence_opt_in()
    # Validate caller report/live pins and every committed snapshot before any
    # copied fixture or authenticated historical-module execution is attempted.
    original = audit.review(REPORT, REPORT_PIN, LIVE, LIVE_PIN, **ANCHORS)
    assert original["verdict"] == "RECOVERY_REVIEW_PASS"
    assert original["reviewed_source"]["commit"] == HISTORICAL_COMMIT
    assert original["production_verifier_used"] is original["historical_verifier_executed"] is False
    return original


@pytest.fixture
def local_storage(tmp_path, monkeypatch):
    # Only portable unit inputs use the pytest sandbox; the real gate's 3FS
    # policy is unchanged and no historical fixture uses this monkeypatch.
    monkeypatch.setattr(audit, "SHARED", tmp_path.resolve())
    return tmp_path


@pytest.fixture
def summary_fixture(local_storage):
    report_path = local_storage / "synthetic-report.json"
    report = {
        "owner": "infra-openopt", "cause": "refinement snapshot missing research document",
        "failed": {"job": "synthetic-failed", "exit_code": 1,
                   "finished_at": "2026-10-10T09:36:53Z"},
        "recovered": {"finished_at": "2026-10-10T09:40:00Z"},
    }
    audit.write_json(report_path, report)
    pin = audit.descriptor(report_path)["sha256"]
    live = {
        "verdict": "PASS", "apiserver_ready": True, "ready": {"h3c2": "True"},
        "active": [], "cordoned": [], "read_at": "2026-10-10T09:54:07+00:00",
        "recovery_reports": [{"path": str(report_path), "sha256": pin}],
        "retained_failures": [{
            "job": "synthetic-failed", "owner": report["owner"], "cause": report["cause"],
            "containers": [{"restart_count": 0, "termination": {
                "exitCode": 1, "finishedAt": report["failed"]["finished_at"],
            }}],
        }],
    }
    return report, live, report_path, pin


@pytest.mark.parametrize("value", [None, "", "0"])
def test_default_evidence_fixture_skips_before_any_historical_io(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(EVIDENCE_ENV, raising=False)
    else:
        monkeypatch.setenv(EVIDENCE_ENV, value)

    def forbidden(*args, **kwargs):
        pytest.fail("disabled evidence fixture must not read historical inputs")

    monkeypatch.setattr(audit, "review", forbidden)
    with pytest.raises(pytest.skip.Exception, match="no campaign qualification"):
        real_evidence.__wrapped__()


@pytest.mark.parametrize("value", ["true", "yes", "2"])
def test_invalid_evidence_opt_in_fails_not_silent_skip(monkeypatch, value):
    monkeypatch.setenv(EVIDENCE_ENV, value)
    with pytest.raises(pytest.fail.Exception, match="must be exactly"):
        require_evidence_opt_in()


@pytest.mark.parametrize("missing", ["report", "live", "git"])
def test_opted_in_missing_dependencies_fail_before_copy_or_code_execution(
        local_storage, monkeypatch, missing):
    report, live = local_storage / "report.json", local_storage / "live.json"
    for path in (report, live):
        path.write_bytes(b'{"synthetic":true}\n')
    monkeypatch.setattr(sys.modules[__name__], "REPORT", report)
    monkeypatch.setattr(sys.modules[__name__], "LIVE", live)
    monkeypatch.setattr(sys.modules[__name__], "REPORT_PIN", audit.descriptor(report)["sha256"])
    monkeypatch.setattr(sys.modules[__name__], "LIVE_PIN", audit.descriptor(live)["sha256"])
    monkeypatch.setenv(EVIDENCE_ENV, "1")

    def forbidden(*args, **kwargs):
        pytest.fail("unvalidated evidence cannot be copied or executed")

    def missing_git(*args):
        raise ValueError("read-only Git source lookup failed")

    monkeypatch.setattr(audit, "clone_fixture", forbidden)
    monkeypatch.setattr(audit, "load_historical_recovery", forbidden)
    monkeypatch.setattr(audit, "git", missing_git)
    if missing == "report":
        monkeypatch.setattr(sys.modules[__name__], "REPORT", local_storage / "absent-report.json")
    elif missing == "live":
        monkeypatch.setattr(sys.modules[__name__], "LIVE", local_storage / "absent-live.json")
    with pytest.raises(ValueError, match=(
            "Git source lookup failed" if missing == "git" else "external evidence missing")):
        real_evidence.__wrapped__()


@pytest.fixture(autouse=True)
def no_default_historical_io(monkeypatch):
    if os.environ.get(EVIDENCE_ENV) == "1":
        return
    original_read = audit.read
    historical_paths = (REPORT, LIVE, REPORT.resolve(), LIVE.resolve())

    def portable_read(path):
        if Path(path) in historical_paths:
            pytest.fail("default research tests must not read historical campaign evidence")
        return original_read(path)

    def forbidden_git(*args):
        pytest.fail("default research tests must not require historical Git objects")

    monkeypatch.setattr(audit, "read", portable_read)
    monkeypatch.setattr(audit, "git", forbidden_git)


def xml(failed):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", tests="7", failures="7" if failed else "0",
                          errors="0", skipped="0")
    for classname, name in audit.EXPECTED:
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        if failed:
            ET.SubElement(case, "failure").text = f"FileNotFoundError: {audit.DOC}"
    return root


@pytest.mark.parametrize("failed", [True, False])
def test_exact_junit_complete_seven_and_three_negative_cases(failed):
    identities = audit.strict_junit(ET.tostring(xml(failed)), failed)
    assert identities == [list(item) for item in audit.EXPECTED]
    assert len([name for _, name in identities if "negative_controls" in name]) == 3


@pytest.mark.parametrize("mutation", [
    "replace_name", "replace_class", "missing", "duplicate", "declared_count",
    "declared_failure", "skip", "error", "nested_case", "extra_suite",
    "wrong_cause", "duplicate_failure", "doctype",
])
def test_coordinated_or_malformed_junit_cannot_fake_complete_selection(mutation):
    root = xml(True)
    suite = root[0]
    if mutation == "replace_name":
        for index, case in enumerate(suite):
            case.set("name", f"unrelated{index}")
    elif mutation == "replace_class":
        for case in suite:
            case.set("classname", "unrelated")
    elif mutation == "missing":
        suite.remove(suite[-1])
    elif mutation == "duplicate":
        suite[-1].set("name", suite[0].get("name"))
    elif mutation == "declared_count":
        suite.set("tests", "0")
    elif mutation == "declared_failure":
        suite.set("failures", "0")
    elif mutation in ("skip", "error"):
        ET.SubElement(suite[0], "skipped" if mutation == "skip" else "error")
    elif mutation == "nested_case":
        case = suite[-1]
        suite.remove(case)
        ET.SubElement(suite, "unknown").append(case)
    elif mutation == "extra_suite":
        ET.SubElement(root, "testsuite")
    elif mutation == "wrong_cause":
        suite[0][0].text = "numerical failure"
    elif mutation == "duplicate_failure":
        ET.SubElement(suite[0], "failure").text = f"FileNotFoundError {audit.DOC}"
    raw = ET.tostring(root)
    if mutation == "doctype":
        raw = b"<!DOCTYPE testsuites>" + raw
    with pytest.raises(ValueError):
        audit.strict_junit(raw, True)


def test_committed_old_junit_accepts_fake_names_independently_of_current_production(
        real_evidence, monkeypatch):
    poisoned = ModuleType("scripts.verify_cluster_recovery")

    def forbidden(*args, **kwargs):
        pytest.fail("independent review must never call current production recovery")

    poisoned.junit = poisoned.verify = poisoned.build = forbidden
    monkeypatch.setitem(sys.modules, "scripts.verify_cluster_recovery", poisoned)
    historical = audit.load_historical_recovery(HISTORICAL_COMMIT, HISTORICAL_PIN)
    assert historical is not poisoned
    assert historical.historical_source["commit"] == HISTORICAL_COMMIT
    assert historical.historical_source["sha256"][audit.HISTORICAL_VERIFIER] == HISTORICAL_PIN
    assert historical.__file__ == f"git:{HISTORICAL_COMMIT}:{audit.HISTORICAL_VERIFIER}"
    assert historical.junit.__code__.co_filename == historical.__file__
    failed, succeeded = xml(True), xml(False)
    for root in (failed, succeeded):
        for index, case in enumerate(root.iter("testcase")):
            case.set("name", f"fake{index}")
    assert historical.junit(ET.tostring(failed), True) == historical.junit(ET.tostring(succeeded), False)
    with pytest.raises(ValueError, match="exact expected selection"):
        audit.strict_junit(ET.tostring(succeeded), False)


@pytest.mark.parametrize("commit,pin", [
    ("HEAD", HISTORICAL_PIN), ("refs/heads/main", HISTORICAL_PIN),
    ("e618b68", HISTORICAL_PIN), ("0" * 40, HISTORICAL_PIN),
    (HISTORICAL_COMMIT, "0" * 64), (HISTORICAL_COMMIT, HISTORICAL_PIN.upper()),
])
def test_historical_toy_commit_or_mutable_ref_not_approved(commit, pin):
    with pytest.raises(ValueError, match="caller historical commit/hash"):
        audit.load_historical_recovery(commit, pin)


def test_changed_git_blob_not_executed_even_with_caller_anchor(monkeypatch):
    called = []

    def fake_git(*args):
        called.append(args)
        if args == ("rev-parse", "--verify", f"{HISTORICAL_COMMIT}^{{commit}}"):
            return HISTORICAL_COMMIT.encode() + b"\n"
        if args == ("show", f"{HISTORICAL_COMMIT}:{audit.HISTORICAL_VERIFIER}"):
            return b"raise RuntimeError('untrusted code must never execute')\n"
        pytest.fail(f"unexpected Git query in synthetic negative: {args}")

    monkeypatch.setattr(audit, "git", fake_git)
    with pytest.raises(ValueError, match="approved historical Git source mismatch"):
        audit.load_historical_recovery(HISTORICAL_COMMIT, HISTORICAL_PIN)
    assert called


def test_reviewed_historical_source_does_not_require_old_current_checkout(
        real_evidence, monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    for relative in audit.REVIEWED:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"current production P2 fixed independently\n")
    # Bind Git lookup to the real trusted repository, not the dummy checkout.
    import subprocess

    repository = Path(audit.__file__).resolve().parents[1]

    def committed_git(*args):
        result = subprocess.run(["git", *args], cwd=repository, capture_output=True, check=True)
        return result.stdout

    monkeypatch.setattr(audit, "git", committed_git)
    source = audit.reviewed_source(HISTORICAL_COMMIT, HISTORICAL_PIN)
    assert source["current_checkout_equality_required"] is False
    assert source["sha256"] == audit.REVIEWED


@pytest.mark.parametrize("mutation", [
    "erase_failed", "duplicate_failed", "changed_owner", "changed_exit",
    "restart", "changed_finished", "wrong_report_pin", "predates_recovery",
    "active", "not_ready", "no_ready", "apiserver",
])
def test_live_summary_rejects_coordinated_false_preserve_or_health(summary_fixture, mutation):
    report, live, report_path, pin = summary_fixture
    row = next(item for item in live["retained_failures"] if item["owner"] == "infra-openopt")
    if mutation == "erase_failed":
        live["retained_failures"].remove(row)
    elif mutation == "duplicate_failed":
        live["retained_failures"].append(copy.deepcopy(row))
    elif mutation == "changed_owner":
        row["owner"] = "other"
    elif mutation == "changed_exit":
        row["containers"][0]["termination"]["exitCode"] = 0
    elif mutation == "restart":
        row["containers"][0]["restart_count"] = 1
    elif mutation == "changed_finished":
        row["containers"][0]["termination"]["finishedAt"] = "2026-10-10T09:40:00Z"
    elif mutation == "wrong_report_pin":
        live["recovery_reports"][0]["sha256"] = "0" * 64
    elif mutation == "predates_recovery":
        live["read_at"] = "2026-10-10T09:30:00+00:00"
    elif mutation == "active":
        live["active"] = ["still-running"]
    elif mutation == "not_ready":
        live["ready"]["h3c2"] = "False"
    elif mutation == "no_ready":
        live["ready"] = {}
    elif mutation == "apiserver":
        live["apiserver_ready"] = False
    with pytest.raises(ValueError):
        audit.check_live(live, report, report_path, pin)


def test_live_missing_uid_is_unverified_not_filled_from_old_status(summary_fixture):
    report, live, report_path, pin = summary_fixture
    result = audit.check_live(live, report, report_path, pin)
    assert result["summary"] == "CONSISTENT"
    assert result["pod_job_uid_binding"] == "UNVERIFIED"
    assert result["current_cluster_health"] == "NOT_READ"


@pytest.mark.parametrize("pin", ["", "0" * 64, "F" * 64, "0" * 63, True])
def test_caller_pin_required_and_exact(local_storage, pin):
    path = local_storage / "pinned.json"
    path.write_bytes(b'{"synthetic":true}\n')
    with pytest.raises(ValueError):
        audit.pinned(path, pin)


def test_mount_alias_normalization_and_traversal_refused(monkeypatch):
    alias = Path("/hf3fs-jd")
    path = alias / "data/tmp/nonexistent-synthetic-recovery.json"
    monkeypatch.setattr(audit, "SHARED", alias.resolve() / "data/tmp")
    assert audit.external(path, exists=False) == alias.resolve() / "data/tmp" / path.name
    with pytest.raises(ValueError):
        audit.external(Path("/hf3fs-jd/data/tmp/a/../b"), exists=False)
    with pytest.raises(ValueError):
        audit.external(Path("/data/dev/openopt"))


def test_symlink_input_is_rejected(local_storage):
    original = local_storage / "synthetic.json"
    original.write_bytes(b'{"synthetic":true}\n')
    path = local_storage / "linked.json"
    path.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        audit.read(path)


def test_json_duplicates_and_nonfinite_are_not_defaulted(local_storage):
    for index, raw in enumerate((b'{"a":1,"a":2}', b'{"a":NaN}', b"[]")):
        path = local_storage / f"bad{index}.json"
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            audit.pinned(path, audit.digest(raw))


@pytest.fixture(scope="module")
def real_counterexamples(real_evidence, tmp_path_factory):
    assert audit.descriptor(REPORT)["sha256"] == REPORT_PIN
    assert audit.descriptor(LIVE)["sha256"] == LIVE_PIN
    directory = tmp_path_factory.mktemp("real-recovery-counterexamples")
    result = audit.counterexamples(REPORT, REPORT_PIN, LIVE, LIVE_PIN, directory, **ANCHORS)
    audit.write_json(directory / "test-counterexamples.json", result)
    assert audit.descriptor(REPORT)["sha256"] == REPORT_PIN
    assert audit.descriptor(LIVE)["sha256"] == LIVE_PIN
    return result


def test_real_e618b68_recovery_and_unchanged_copy_pass_without_numerical_claim(real_counterexamples):
    result = real_counterexamples
    assert result["originals_unchanged"] is True
    original = result["original_review"]
    assert original["packaging_recovery"] == "PASS"
    assert original["old_failure"] == "PRESERVED_FAIL"
    assert original["numerical_cuda_qualification"] == "NOT_GRANTED"
    assert original["commercial_qualification"] == original["publication"] == "UNVERIFIED"
    assert original["production_verifier_used"] is original["historical_verifier_executed"] is False
    assert result["historical_source"]["commit"] == HISTORICAL_COMMIT
    positive = result["cases"][0]
    assert positive["mutation"] == "unchanged_copy"
    assert positive["upstream_e618b68"] == positive["independent_review"] == "ACCEPTED"


@pytest.mark.parametrize("mutation", audit.MUTANTS)
def test_real_resealed_coordinated_negative_controls_rejected(real_counterexamples, mutation):
    row = next(item for item in real_counterexamples["cases"] if item["mutation"] == mutation)
    assert row["independent_review"] == "REJECTED"
    assert row["reason"]
    assert row["upstream_e618b68"] == (
        "REJECTED" if mutation == "promote_qualification" else "ACCEPTED")
    assert audit.descriptor(row["report"]["path"]) == row["report"]
    assert audit.descriptor(row["live"]["path"]) == row["live"]


def test_real_canonical_replay_does_not_execute_historical_or_production_verifier(
        real_evidence, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("canonical review cannot execute the historical or production verifier")

    monkeypatch.setattr(audit, "load_historical_recovery", forbidden)
    poisoned = ModuleType("scripts.verify_cluster_recovery")
    poisoned.build = poisoned.verify = poisoned.junit = forbidden
    monkeypatch.setitem(sys.modules, "scripts.verify_cluster_recovery", poisoned)
    result = audit.review(REPORT, REPORT_PIN, LIVE, LIVE_PIN, **ANCHORS)
    assert result["verdict"] == "RECOVERY_REVIEW_PASS"
    assert result["production_verifier_used"] is result["historical_verifier_executed"] is False


@pytest.mark.parametrize("mutation", [
    "unchanged_copy", "replace_selection", "contradict_immutable_result",
    "contradict_job_template", "wrong_runner", "inflate_telemetry",
])
def test_repaired_production_replays_actual_copies_without_old_gate_dependency(
        real_counterexamples, mutation):
    from scripts import verify_cluster_recovery as current

    row = next(item for item in real_counterexamples["cases"] if item["mutation"] == mutation)
    report = audit.pinned(row["report"]["path"], row["report"]["sha256"])
    if mutation == "unchanged_copy":
        rebuilt = current.build(report["failed_run"], report["recovered_run"])
        assert rebuilt["schema_version"] == 2
        assert rebuilt["verdict"] == "RECOVERY_VERIFIED"
        assert rebuilt["failed"]["executed_sha256"] == rebuilt["recovered"]["executed_sha256"]
    else:
        with pytest.raises(ValueError):
            current.build(report["failed_run"], report["recovered_run"])
    assert audit.descriptor(row["report"]["path"]) == row["report"]
    assert row["historical_source"]["commit"] == HISTORICAL_COMMIT
    assert row["historical_source"]["sha256"][audit.HISTORICAL_VERIFIER] == HISTORICAL_PIN


@pytest.mark.parametrize("mutation", ["summary_uid", "summary_count", "summary_type",
                                     "empty_inputs", "extra_field", "promote_scope"])
def test_resealed_report_cannot_change_canonical_summary_or_input_inventory(
        real_evidence, tmp_path, mutation):
    report = audit.pinned(REPORT, REPORT_PIN)
    if mutation == "summary_uid":
        report["failed"]["job_uid"] = "fabricated"
    elif mutation == "summary_count":
        report["recovered"]["cases"] = []
    elif mutation == "summary_type":
        report["failed"]["exit_code"] = True
    elif mutation == "empty_inputs":
        report["inputs"] = []
    elif mutation == "extra_field":
        report["certified"] = True
    else:
        report["scope"] = "certified commercial CUDA qualification"
    path = tmp_path / "resealed-report.json"
    audit.write_json(path, report)
    with pytest.raises(ValueError, match="canonical|scope"):
        audit.review(path, audit.descriptor(path)["sha256"], LIVE, LIVE_PIN, **ANCHORS)


def test_cli_failure_preserves_first_cause_and_external_log(local_storage, monkeypatch):
    report = local_storage / "report.json"
    live = local_storage / "live.json"
    report.write_bytes(b'{"synthetic":true}\n')
    live.write_bytes(b'{"synthetic":true}\n')
    output = local_storage / "failed-cli"

    def forbidden(*args, **kwargs):
        pytest.fail("bad caller pin must fail before historical Git lookup")

    monkeypatch.setattr(audit, "git", forbidden)
    rc = audit.main([
        "review", "--report", str(report), "--report-sha256", "0" * 64,
        "--live", str(live), "--live-sha256", audit.descriptor(live)["sha256"],
        "--artifact-dir", str(output),
        *CLI_ANCHORS,
    ])
    assert rc == 1
    failure = json.loads((output / "failure.json").read_bytes())
    assert failure["exit_code"] == 1
    assert failure["first_cause"] == "ValueError: caller evidence pin mismatch"
    assert "verdict=FAIL" in (output / "execution.log").read_text()
    assert not (output / "review.json").exists()
    with pytest.raises(FileExistsError):
        audit.main([
            "review", "--report", str(report),
            "--report-sha256", audit.descriptor(report)["sha256"],
            "--live", str(live), "--live-sha256", audit.descriptor(live)["sha256"],
            "--artifact-dir", str(output),
            *CLI_ANCHORS,
        ])
