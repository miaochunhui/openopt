"""Independent packaging audit controls, including real disk-evidence replay.

The real fixtures are copied to the verifier's external TMPDIR. CPU inspection
does not execute CUDA tests and never changes original artifacts or cluster.
"""

import copy
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from scripts import review_cluster_recovery as audit

REPORT = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/failure-recovery-real-v3/recovery.json")
REPORT_PIN = "cd911cf2cbe4f1488511157b02cb8c81c33b6ad740ea5f6437915bf4dbbb6610"
LIVE = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/live-recovery-v1/live-095407908933.json")
LIVE_PIN = "09bb2842415c903046258ecff9f26f59a4bfdc21fb3fd50fcd8dd9ed16deb6a1"


def xml(failed):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", tests="7", failures="7" if failed else "0",
                          errors="0", skipped="0")
    for classname, name in audit.EXPECTED:
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        if failed:
            ET.SubElement(case, "failure").text = f"FileNotFoundError: {audit.recovery.DOC}"
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
        ET.SubElement(suite[0], "failure").text = f"FileNotFoundError {audit.recovery.DOC}"
    raw = ET.tostring(root)
    if mutation == "doctype":
        raw = b"<!DOCTYPE testsuites>" + raw
    with pytest.raises(ValueError):
        audit.strict_junit(raw, True)


def test_upstream_junit_accepts_coordinated_nonselected_names_but_review_rejects():
    failed, succeeded = xml(True), xml(False)
    for root in (failed, succeeded):
        for index, case in enumerate(root.iter("testcase")):
            case.set("name", f"fake{index}")
    assert audit.recovery.junit(ET.tostring(failed), True) == audit.recovery.junit(ET.tostring(succeeded), False)
    with pytest.raises(ValueError, match="exact expected selection"):
        audit.strict_junit(ET.tostring(succeeded), False)


@pytest.mark.parametrize("mutation", [
    "erase_failed", "duplicate_failed", "changed_owner", "changed_exit",
    "restart", "changed_finished", "wrong_report_pin", "predates_recovery",
    "active", "not_ready", "no_ready", "apiserver",
])
def test_live_summary_rejects_coordinated_false_preserve_or_health(mutation):
    report = audit.pinned(REPORT, REPORT_PIN)
    live = audit.pinned(LIVE, LIVE_PIN)
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
        audit.check_live(live, report, REPORT, REPORT_PIN)


def test_live_missing_uid_is_unverified_not_filled_from_old_status():
    result = audit.check_live(audit.pinned(LIVE, LIVE_PIN), audit.pinned(REPORT, REPORT_PIN),
                              REPORT, REPORT_PIN)
    assert result["summary"] == "CONSISTENT"
    assert result["pod_job_uid_binding"] == "UNVERIFIED"
    assert result["current_cluster_health"] == "NOT_READ"


@pytest.mark.parametrize("pin", ["", "0" * 64, "F" * 64, "0" * 63, True])
def test_caller_pin_required_and_exact(pin):
    with pytest.raises(ValueError):
        audit.pinned(REPORT, pin)


def test_mount_alias_normalization_and_traversal_refused():
    assert audit.external(REPORT) == audit.external(REPORT.resolve())
    with pytest.raises(ValueError):
        audit.external(Path("/hf3fs-jd/data/tmp/a/../b"), exists=False)
    with pytest.raises(ValueError):
        audit.external(Path("/data/dev/openopt"))


def test_symlink_input_is_rejected(tmp_path):
    path = tmp_path / "linked.json"
    path.symlink_to(REPORT)
    with pytest.raises(ValueError, match="symlink"):
        audit.read(path)


def test_json_duplicates_and_nonfinite_are_not_defaulted(tmp_path):
    for index, raw in enumerate((b'{"a":1,"a":2}', b'{"a":NaN}', b"[]")):
        path = tmp_path / f"bad{index}.json"
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            audit.pinned(path, audit.digest(raw))


@pytest.fixture(scope="module")
def real_counterexamples(tmp_path_factory):
    assert audit.descriptor(REPORT)["sha256"] == REPORT_PIN
    assert audit.descriptor(LIVE)["sha256"] == LIVE_PIN
    directory = tmp_path_factory.mktemp("real-recovery-counterexamples")
    result = audit.counterexamples(REPORT, REPORT_PIN, LIVE, LIVE_PIN, directory)
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


def test_cli_failure_preserves_first_cause_and_external_log(tmp_path):
    output = tmp_path / "failed-cli"
    rc = audit.main([
        "review", "--report", str(REPORT), "--report-sha256", "0" * 64,
        "--live", str(LIVE), "--live-sha256", LIVE_PIN, "--artifact-dir", str(output),
    ])
    assert rc == 1
    failure = json.loads((output / "failure.json").read_bytes())
    assert failure["exit_code"] == 1
    assert failure["first_cause"] == "ValueError: caller evidence pin mismatch"
    assert "verdict=FAIL" in (output / "execution.log").read_text()
    assert not (output / "review.json").exists()
    with pytest.raises(FileExistsError):
        audit.main([
            "review", "--report", str(REPORT), "--report-sha256", REPORT_PIN,
            "--live", str(LIVE), "--live-sha256", LIVE_PIN, "--artifact-dir", str(output),
        ])
