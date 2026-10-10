"""First-cause recovery checks do not grant numerical/performance qualification."""

import json
import copy
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import verify_cluster_recovery as recovery
from scripts.qp_campaign import failure_attribution, pod_identity


def xml(failed, cause="FileNotFoundError: docs/research/refinement.md"):
    payload = f'<testsuite tests="7" failures="{7 if failed else 0}" errors="0" skipped="0">'
    for classname, name in recovery.EXPECTED_CASES:
        payload += f'<testcase classname="{classname}" name="{name}">'
        if failed:
            payload += f"<failure>{cause}</failure>"
        payload += "</testcase>"
    return (payload + "</testsuite>").encode()


def test_complete_failure_and_recovery_case_membership():
    assert recovery.junit(xml(True), True) == recovery.junit(xml(False), False)


def test_kubernetes_gpu_quantity_serialization_is_lossless():
    assert recovery.resources_match(
        {"limits": {"cpu": "4", "nvidia.com/gpu": "1"}},
        {"limits": {"cpu": "4", "nvidia.com/gpu": 1}},
    )
    assert not recovery.resources_match(
        {"limits": {"cpu": "4", "nvidia.com/gpu": "2"}},
        {"limits": {"cpu": "4", "nvidia.com/gpu": 1}},
    )
    assert not recovery.resources_match(
        {"limits": {"cpu": "4", "nvidia.com/gpu": True}},
        {"limits": {"cpu": "4", "nvidia.com/gpu": 1}},
    )


@pytest.mark.parametrize("raw,failed", [
    (xml(True, "numerical failure"), True),
    (xml(False), True),
    (xml(True), False),
    (xml(False).replace(b"wrong_sign", b"wrong_active_set"), False),
    (xml(False).replace(b"test_cuda_refinement", b"unrelated"), False),
    (xml(False).replace(b'failures="0"', b'failures="7"'), False),
    (xml(False).replace(b"</testcase>", b"<skipped/></testcase>", 1), False),
    (b"<!DOCTYPE x>" + xml(False), False),
    (b"<testsuite/>", False),
])
def test_incomplete_or_wrong_first_cause_rejected(raw, failed):
    with pytest.raises(ValueError):
        recovery.junit(raw, failed)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'[]'])
def test_recovery_json_fails_loud(raw):
    with pytest.raises(ValueError):
        recovery.object_json(raw)


def test_unchanged_solver_and_tests_required(monkeypatch):
    failed = {"core_sha256": {"qp": "a"}, "cases": ["x"], "source_sha": "old",
              "executed_sha256": {"qp": "a"},
              "job_uid": "old", "image_id": "same"}
    fixed = {**failed, "source_sha": "new", "job_uid": "new", "core_sha256": {"qp": "b"}}
    monkeypatch.setattr(recovery, "leg", lambda path, old, inputs: failed if old else fixed)
    with pytest.raises(ValueError, match="solver/test changed"):
        recovery.build("old", "new")


def test_report_is_specific_recovery_not_gpu_qualification(monkeypatch):
    failed = {"core_sha256": {"qp": "a"}, "cases": ["x"], "source_sha": "old",
              "executed_sha256": {"qp": "a"},
              "job_uid": "old", "image_id": "same"}
    fixed = {**failed, "source_sha": "new", "job_uid": "new"}
    monkeypatch.setattr(recovery, "leg", lambda path, old, inputs: failed if old else fixed)
    report = recovery.build("old", "new")
    assert report["verdict"] == "RECOVERY_VERIFIED"
    assert "qualification remains separate" in report["scope"]
    assert json.loads(json.dumps(report)) == report


def test_contradictory_immutable_result_is_not_coordinator_success():
    with pytest.raises(ValueError, match="immutable"):
        recovery.matching_results(b'{"verdict":"PASS"}', b'{"verdict":"FAIL"}')
    assert recovery.matching_results(b'{"verdict":"PASS"}', b'{"verdict":"PASS"}')["verdict"] == "PASS"


def execution_fixture(tmp_path):
    directory, output = tmp_path, tmp_path / "case-0"
    container = {
        "name": "benchmark", "image": "existing:runtime",
        "command": ["python3", str(directory / "source" / recovery.RUNNER),
                    "--artifact-dir", str(output), "--target", recovery.TARGET,
                    "--expression", recovery.SELECTION],
        "resources": {kind: {"cpu": "4", "memory": "12Gi", "nvidia.com/gpu": 1}
                      for kind in ("requests", "limits")},
        "env": [{"name": "PYTHONPATH", "value": str(directory / "source/src")},
                {"name": "TMPDIR", "value": str(output)},
                {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                {"name": "OPENOPT_NODE", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}],
    }
    spec = {"containers": [container], "restartPolicy": "Never"}
    manifest = {"spec": {"backoffLimit": 0, "template": {"spec": spec}}}
    return manifest, copy.deepcopy(manifest), {"spec": copy.deepcopy(spec)}, directory, output


def test_job_api_fieldref_defaults_do_not_change_execution(tmp_path):
    values = execution_fixture(tmp_path)
    values[2]["spec"]["containers"][0]["env"][-1]["valueFrom"]["fieldRef"]["apiVersion"] = "v1"
    assert recovery.execution_contract(*values)


@pytest.mark.parametrize("mutation", ["job_command", "joint_runner", "environment",
                                     "extra_container", "retry", "args", "gpu"])
def test_resealed_execution_substitutions_fail(tmp_path, mutation):
    manifest, job, pod, directory, output = execution_fixture(tmp_path)
    if mutation == "job_command":
        job["spec"]["template"]["spec"]["containers"][0]["command"] = ["false"]
    elif mutation == "joint_runner":
        for spec in (manifest["spec"]["template"]["spec"], job["spec"]["template"]["spec"], pod["spec"]):
            spec["containers"][0]["command"][1] = "/not-frozen/runner.py"
    elif mutation == "environment":
        pod["spec"]["containers"][0]["env"][0]["value"] = "/not-frozen/src"
    elif mutation == "extra_container":
        pod["spec"]["containers"].append(copy.deepcopy(pod["spec"]["containers"][0]))
    elif mutation == "retry":
        job["spec"]["backoffLimit"] = 1
    elif mutation == "args":
        pod["spec"]["containers"][0]["args"] = ["substitute"]
    else:
        pod["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = 2
    with pytest.raises(ValueError):
        recovery.execution_contract(manifest, job, pod, directory, output)


@pytest.mark.parametrize("count,raw", [
    (2, b'[{"rc":0,"gpu":"sample","cpu":{"usage_usec":1}}]'),
    (1, b'[{"rc":1,"gpu":"sample","cpu":{"usage_usec":1}}]'),
    (1, b'[{"rc":0,"gpu":"sample"}]'),
    (0, b'[]'),
    (True, b'[{"rc":0,"gpu":"sample","cpu":{"usage_usec":1}}]'),
])
def test_telemetry_is_actual_nonempty_count_not_claim(count, raw):
    with pytest.raises(ValueError):
        recovery.check_telemetry(raw, count)


def live_fixture():
    report = {"owner": "infra-openopt", "cause": "snapshot doc missing",
              "failed": {"pod_uid": "pod1", "pod": "failed", "job": "job1",
                         "job_uid": "job-uid", "exit_code": 1}}
    pod = {"metadata": {"uid": "pod1", "name": "failed",
                        "labels": {"owner": "infra-openopt", "job-name": "job1"},
                        "ownerReferences": [{"controller": True, "uid": "job-uid"}]},
           "status": {"containerStatuses": [
               {"restartCount": 0, "state": {"terminated": {"exitCode": 1}}}]}}
    return pod, report


def test_live_requires_verified_report_and_actual_failed_pod():
    pod, report = live_fixture()
    assert failure_attribution(pod, []) == ("UNKNOWN", "unattributed failure")
    assert failure_attribution(pod, [report]) == ("infra-openopt", "snapshot doc missing")


def test_live_identity_is_observed_not_inferred_from_names():
    pod, _ = live_fixture()
    pod["metadata"]["ownerReferences"][0]["name"] = "job1"
    assert pod_identity(pod, "job1") == {"pod": "failed", "pod_uid": "pod1", "job_uid": "job-uid"}
    pod["metadata"]["ownerReferences"][0]["name"] = "other"
    with pytest.raises(ValueError, match="identity"):
        pod_identity(pod, "job1")


@pytest.mark.parametrize("mutation", ["missing_pod_uid", "missing_job_uid", "extra_controller"])
def test_live_missing_or_ambiguous_identity_fails(mutation):
    pod, _ = live_fixture()
    pod["metadata"]["ownerReferences"][0]["name"] = "job1"
    if mutation == "missing_pod_uid":
        pod["metadata"].pop("uid")
    elif mutation == "missing_job_uid":
        pod["metadata"]["ownerReferences"][0].pop("uid")
    else:
        pod["metadata"]["ownerReferences"].append({"controller": True, "name": "job1", "uid": "other"})
    with pytest.raises(ValueError):
        pod_identity(pod, "job1")


@pytest.mark.parametrize("mutation", ["uid", "owner", "job_uid", "restart", "exit"])
def test_live_cannot_attribute_a_different_failure(mutation):
    original, report = live_fixture()
    pod = copy.deepcopy(original)
    if mutation == "uid":
        pod["metadata"]["uid"] = "another"
    elif mutation == "owner":
        pod["metadata"]["labels"]["owner"] = "another"
    elif mutation == "job_uid":
        pod["metadata"]["ownerReferences"][0]["uid"] = "another"
    elif mutation == "restart":
        pod["status"]["containerStatuses"][0]["restartCount"] = 1
    else:
        pod["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = 137
    assert failure_attribution(pod, [report])[0] == "UNKNOWN"


def check_saved_counterexamples(review_path, review_pin, output):
    """Recheck actual copied historical mutations with the repaired production gate."""
    if recovery.descriptor(review_path)["sha256"] != review_pin:
        raise ValueError("independent counterexample review pin mismatch")
    raw = recovery.read(review_path)
    reviewed = recovery.object_json(raw)
    expected = {
        "unchanged_copy": None,
        "replace_selection": "exact expected recovery case",
        "contradict_immutable_result": "coordinator/immutable",
        "contradict_job_template": "manifest/Job/Pod exact execution",
        "wrong_runner": "manifest/Job/Pod exact execution",
        "inflate_telemetry": "telemetry count",
    }
    rows = {row["mutation"]: row for row in reviewed["cases"]}
    if not set(expected).issubset(rows) or reviewed["verdict"] != "COUNTEREXAMPLES_VERIFIED":
        raise ValueError("missing independently reproduced counterexample membership")
    outcomes = []
    for name, first_cause in expected.items():
        row = rows[name]
        pin = row["report"]
        if recovery.descriptor(pin["path"]) != pin:
            raise ValueError("independent copied report pin mismatch")
        saved = recovery.object_json(recovery.read(pin["path"]))
        error = None
        try:
            report = recovery.build(saved["failed_run"], saved["recovered_run"])
            if first_cause is not None or report["verdict"] != "RECOVERY_VERIFIED":
                raise AssertionError("repaired gate accepted a retained negative control")
        except ValueError as cause:
            error = str(cause)
            if first_cause is None or first_cause not in error:
                raise AssertionError("repaired gate rejected for an unexpected first cause") from cause
        if recovery.descriptor(pin["path"]) != pin:
            raise ValueError("copied evidence changed during replay")
        outcomes.append({"mutation": name, "verdict": "PASS",
                         "gate": "ACCEPTED" if error is None else "REJECTED",
                         "first_cause": error, "report_sha256": pin["sha256"]})
    if recovery.read(review_path) != raw:
        raise ValueError("independent counterexample review changed during replay")
    output = Path(output)
    if not output.resolve().is_relative_to(recovery.SHARED):
        raise ValueError("counterexample receipt must be external shared evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"verdict": "COUNTEREXAMPLES_REPAIRED_PASS",
              "checked_at": datetime.now(timezone.utc).isoformat(), "cases": outcomes,
              "review_sha256": review_pin, "old_review_unchanged": True,
              "numerical_cuda_qualification": "NOT_GRANTED"}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if recovery.object_json(recovery.read(output)) != result:
        raise ValueError("counterexample receipt roundtrip mismatch")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay retained recovery mutations against the repaired gate")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check_saved_counterexamples(args.review, args.sha256, args.output)
    print(f"RECOVERY_COUNTEREXAMPLES verdict={report['verdict']} "
          f"at={report['checked_at']} cases={len(report['cases'])} evidence={args.output}")
