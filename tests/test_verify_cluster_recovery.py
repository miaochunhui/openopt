"""First-cause recovery checks do not grant numerical/performance qualification."""

import json
import copy

import pytest

from scripts import verify_cluster_recovery as recovery
from scripts.qp_campaign import failure_attribution


def xml(failed, cause="FileNotFoundError: docs/research/refinement.md"):
    payload = '<testsuite tests="7">'
    for i in range(7):
        payload += f'<testcase classname="tests.test_qp_refinement" name="case{i}">'
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
    (xml(False).replace(b'name="case6"', b'name="case5"'), False),
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
              "job_uid": "old", "image_id": "same"}
    fixed = {**failed, "source_sha": "new", "job_uid": "new", "core_sha256": {"qp": "b"}}
    monkeypatch.setattr(recovery, "leg", lambda path, old, inputs: failed if old else fixed)
    with pytest.raises(ValueError, match="solver/test changed"):
        recovery.build("old", "new")


def test_report_is_specific_recovery_not_gpu_qualification(monkeypatch):
    failed = {"core_sha256": {"qp": "a"}, "cases": ["x"], "source_sha": "old",
              "job_uid": "old", "image_id": "same"}
    fixed = {**failed, "source_sha": "new", "job_uid": "new"}
    monkeypatch.setattr(recovery, "leg", lambda path, old, inputs: failed if old else fixed)
    report = recovery.build("old", "new")
    assert report["verdict"] == "RECOVERY_VERIFIED"
    assert "qualification remains separate" in report["scope"]
    assert json.loads(json.dumps(report)) == report


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
