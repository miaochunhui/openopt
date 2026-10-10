"""Verify retained refinement snapshot failure and a changed-source recovery.

Reads coordinator-collected evidence only. This closes a specific packaging
failure, not numerical GPU qualification, performance or remote attestation.
No Jobs, credentials, source files or original evidence are changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree

SHARED = Path("/hf3fs-jd/data/tmp").resolve()
CORE = ("src/openopt/refinement.py", "src/openopt/qp.py",
        "src/openopt/certificates.py", "tests/test_qp_refinement.py")
DOC = "docs/research/refinement.md"
RUNNER = "scripts/gpu_release_check.py"
TARGET = "tests/test_qp_refinement.py"
SELECTION = "cuda_refinement"
EXPECTED_CASES = tuple(sorted(
    ("tests.test_qp_refinement", name) for name in (
        "test_cuda_refinement_actual_residual_correction",
        "test_cuda_refinement_frozen_model_accuracy_ab[dtype0-factor_dtype0]",
        "test_cuda_refinement_frozen_model_accuracy_ab[dtype1-factor_dtype1]",
        "test_cuda_refinement_frozen_model_accuracy_ab[dtype2-factor_dtype2]",
        "test_cuda_refinement_negative_controls[singular_kkt]",
        "test_cuda_refinement_negative_controls[wrong_active_set]",
        "test_cuda_refinement_negative_controls[wrong_sign]",
    )
))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(not path.is_symlink(), "symlink recovery evidence rejected")
    path = path.resolve(strict=True)
    require(path.is_relative_to(SHARED) and path.is_file(),
            "recovery evidence must be an external shared file")
    raw = path.read_bytes()
    require(raw, "empty recovery evidence")
    return raw


def descriptor(path):
    raw = read(path)
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def resources_match(actual, expected):
    if set(actual) != set(expected):
        return False
    for category, values in expected.items():
        if set(actual[category]) != set(values):
            return False
        for name, value in values.items():
            observed = actual[category][name]
            if name == "nvidia.com/gpu":
                if type(value) not in (int, str) or type(observed) not in (int, str):
                    return False
                if str(value) != str(observed):
                    return False
            elif observed != value:
                return False
    return True


def object_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate recovery JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite recovery JSON: {value}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    require(isinstance(value, dict), "recovery JSON must be an object")
    return value


def junit(raw, failed):
    require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(),
            "unsafe recovery JUnit")
    root = ElementTree.fromstring(raw)
    cases = list(root.iter("testcase"))
    require(len(cases) == 7, "refinement recovery requires all seven selected cases")
    identities = [(case.get("classname"), case.get("name")) for case in cases]
    require(tuple(sorted(identities)) == EXPECTED_CASES,
            "exact expected recovery case identities required")
    suites = [suite for suite in root.iter("testsuite") if suite.findall("testcase")]
    require(len(suites) == 1 and len(suites[0].findall("testcase")) == 7,
            "exact one complete recovery JUnit suite required")
    require({key: suites[0].get(key) for key in ("tests", "failures", "errors", "skipped")}
            == {"tests": "7", "failures": "7" if failed else "0", "errors": "0", "skipped": "0"},
            "recovery JUnit suite counts contradict cases")
    for case in cases:
        require(case.find("skipped") is None and case.find("error") is None,
                "recovery case skipped or errored")
        failures = case.findall("failure")
        require(len(failures) == (1 if failed else 0), "contradictory recovery failures")
        failure = failures[0] if failures else None
        if failed:
            require(failure is not None and "FileNotFoundError" in "".join(failure.itertext())
                    and DOC in "".join(failure.itertext()), "original first cause changed")
        else:
            require(failure is None, "recovery still has a failure")
    return [list(identity) for identity in sorted(identities)]


def environment(entries):
    require(isinstance(entries, list), "execution environment missing")
    names = [entry["name"] for entry in entries]
    require(len(names) == len(set(names)), "duplicate execution environment")
    normalized = {}
    for entry in entries:
        if "valueFrom" in entry:
            require(set(entry) == {"name", "valueFrom"}
                    and set(entry["valueFrom"]) == {"fieldRef"}, "unsupported execution environment")
            field = entry["valueFrom"]["fieldRef"]
            require(set(field) <= {"apiVersion", "fieldPath"}
                    and field.get("apiVersion", "v1") == "v1", "unsupported environment field reference")
            normalized[entry["name"]] = {"fieldPath": field["fieldPath"]}
        else:
            require(set(entry) == {"name", "value"}, "unsupported literal execution environment")
            normalized[entry["name"]] = entry["value"]
    return normalized


def execution_contract(manifest, job, pod, directory, output):
    """Bind the selected frozen runner to the submitted and observed templates."""
    specs = (manifest["spec"]["template"]["spec"],
             job["spec"]["template"]["spec"], pod["spec"])
    require(len(specs[0]["containers"]) == 1, "exact one manifest container required")
    expected = specs[0]["containers"][0]
    command = ["python3", str(directory / "source" / RUNNER),
               "--artifact-dir", str(output), "--target", TARGET, "--expression", SELECTION]
    env = environment(expected.get("env"))
    require(env.get("PYTHONPATH") == str(directory / "source/src")
            and env.get("TMPDIR") == str(output)
            and env.get("PYTHONDONTWRITEBYTECODE") == "1"
            and env.get("OPENOPT_NODE") == {"fieldPath": "spec.nodeName"},
            "frozen runner source/output environment mismatch")
    for obj in (manifest, job):
        require(type(obj["spec"]["backoffLimit"]) is int and obj["spec"]["backoffLimit"] == 0,
                "recovery Job retry policy mismatch")
    for spec in specs:
        require(spec.get("restartPolicy") == "Never" and len(spec["containers"]) == 1
                and not spec.get("initContainers"), "restart/multi-container execution mismatch")
        actual = spec["containers"][0]
        require(actual.get("name") == expected["name"]
                and actual.get("command") == command and not actual.get("args")
                and actual.get("image") == expected["image"]
                and resources_match(actual["resources"], expected["resources"])
                and environment(actual.get("env")) == env,
                "manifest/Job/Pod exact execution mismatch")
        require(all(str(actual["resources"][kind].get("nvidia.com/gpu")) == "1"
                    and type(actual["resources"][kind].get("nvidia.com/gpu")) in (str, int)
                    for kind in ("requests", "limits")), "recovery requires exactly one GPU")
    return command


def matching_results(coordinator_raw, immutable_raw):
    require(coordinator_raw == immutable_raw, "coordinator/immutable result contradiction")
    return object_json(coordinator_raw)


def check_telemetry(raw, count):
    samples = object_json(b'{"samples":' + raw + b"}")["samples"]
    require(type(count) is int and isinstance(samples, list) and 0 < len(samples) == count,
            "recovery telemetry count mismatch")
    require(all(isinstance(sample, dict) and type(sample.get("rc")) is int and sample["rc"] == 0
                and sample.get("error") is None and isinstance(sample.get("gpu"), str)
                and sample["gpu"] and sample.get("cpu", {}).get("usage_usec") is not None
                for sample in samples), "recovery telemetry sample failed")


def leg(directory, failed, inputs):
    directory = Path(directory).resolve(strict=True)

    def take(path):
        inputs.append(descriptor(path))
        return read(path)

    record = object_json(take(directory / "run.json"))
    collection = object_json(take(directory / "collection.json"))
    require(record["state"] == "SUBMITTED" and record["stage"] == "release-regression",
            "recovery run was not submitted as release regression")
    require(len(record["jobs"]) == len(collection["jobs"]) == 1,
            "recovery requires one isolated submitted Job")
    item, collected = record["jobs"][0], collection["jobs"][0]
    output = Path(item["output"]).resolve(strict=True)
    require(output == directory / "case-0", "recovery output escaped isolated case")
    require(collection["states"] == ["FAIL" if failed else "PASS"]
            and collection["source_sha"] == record["source_sha"]
            and collection["source_hashes"] == record["source_hashes"],
            "collection/run state or source mismatch")
    require(all(collected[key] == value for key, value in item.items()),
            "collection submission changed")
    require(record["source_identity"]["clean"] is True
            and record["source_identity"]["commit"] == record["source_sha"],
            "unclean/unbound submitted source")
    manifest = object_json(take(item["manifest"]))
    status = object_json(take(directory / f"{item['job']}-status.json"))
    job = status["job"]
    require(job["metadata"]["name"] == item["job"]
            and job["metadata"]["namespace"] == record["namespace"]
            and job["metadata"]["labels"].get("owner") == "infra-openopt",
            "Job ownership mismatch")
    pods = status["pods"]["items"]
    require(len(pods) == 1, "missing/duplicate recovery Pod")
    pod = pods[0]
    owners = pod["metadata"].get("ownerReferences", [])
    require(any(owner.get("controller") is True and owner.get("uid") == job["metadata"]["uid"]
                and owner.get("name") == item["job"] for owner in owners),
            "Pod Job UID mismatch")
    require(pod["metadata"]["labels"].get("owner") == "infra-openopt"
            and pod["metadata"]["labels"].get("repo") == "openopt",
            "Pod project owner mismatch")
    execution_contract(manifest, job, pod, directory, output)
    conditions = job["status"].get("conditions", [])
    require(any(item.get("type") == ("Failed" if failed else "Complete")
                and item.get("status") == "True" for item in conditions)
            and not any(item.get("type") == ("Complete" if failed else "Failed")
                        and item.get("status") == "True" for item in conditions),
            "recovery Job terminal condition mismatch")
    containers = pod["status"]["containerStatuses"]
    require(len(containers) == 1 and containers[0]["restartCount"] == 0,
            "restarted/duplicate recovery container")
    terminal = containers[0]["state"].get("terminated")
    require(terminal is not None and terminal["exitCode"] == (1 if failed else 0)
            and pod["status"]["phase"] == ("Failed" if failed else "Succeeded"),
            "recovery termination mismatch")
    log = take(directory / f"{pod['metadata']['name']}.log").decode()
    if failed:
        require("OPENOPT_RELEASE_CUDA error=" in log, "original completion lost first cause")
    else:
        require("OPENOPT_RELEASE_CUDA verdict=PASS" in log, "recovery lacks actual completion")
    result_path = output / "result.json"
    suites = list(output.glob("cuda-check-*/pytest.xml"))
    require(len(suites) == 1, "missing/duplicate recovery JUnit")
    result = matching_results(take(result_path), take(suites[0].with_name("result.json")))
    require(result["verdict"] == ("FAIL" if failed else "PASS")
            and result["returncode"] == (1 if failed else 0)
            and result["targets"] == ["tests/test_qp_refinement.py"]
            and result["selection"] == "cuda_refinement",
            "recovery result mismatch")
    if not failed:
        require(result["tests"] == {"tests": 7, "failures": 0, "errors": 0, "skipped": 0},
                "recovery count mismatch")
    require(result["hardware"]["node"] == pod["spec"]["nodeName"]
            and result["hardware"]["gpu"] == "NVIDIA RTX PRO 4500 Blackwell",
            "recovery hardware mismatch")
    if not failed:
        check_telemetry(take(suites[0].with_name("telemetry.json")), result["telemetry_samples"])
    if not failed:
        require(collected["result_path"] == str(result_path)
                and collected["result_sha256"] == descriptor(result_path)["sha256"],
                "collected result pin mismatch")
    executed = {name for name in record["source_hashes"]
                if name.startswith("src/openopt/") or name in (TARGET, RUNNER)}
    require(set(result["source_sha256"]) == executed, "executed source membership mismatch")
    for relative in sorted(executed):
        pin = descriptor(directory / "source" / relative)
        inputs.append(pin)
        require(pin["sha256"] == record["source_hashes"][relative]
                == result["source_sha256"][relative]
                == record["source_identity"]["sha256"][relative], "recovery executed source mismatch")
    cases = junit(take(suites[0]), failed)
    if failed:
        require(DOC not in record["source_hashes"]
                and not (directory / "source" / DOC).exists(), "missing-document diagnosis false")
        pytest_log = take(suites[0].with_name("pytest.log")).decode()
        require("FileNotFoundError" in pytest_log and DOC in pytest_log,
                "retained pytest first-cause log mismatch")
    else:
        pin = descriptor(directory / "source" / DOC)
        inputs.append(pin)
        require(pin["sha256"] == record["source_hashes"][DOC], "recovered document pin mismatch")
    return {"job": item["job"], "job_uid": job["metadata"]["uid"],
            "pod": pod["metadata"]["name"], "pod_uid": pod["metadata"]["uid"],
            "source_sha": record["source_sha"], "cases": cases,
            "core_sha256": {name: record["source_hashes"][name] for name in CORE},
            "executed_sha256": result["source_sha256"],
            "exit_code": terminal["exitCode"], "restart_count": 0,
            "image_id": containers[0]["imageID"], "finished_at": terminal["finishedAt"]}


def build(failed_run, recovered_run):
    inputs = []
    failed = leg(failed_run, True, inputs)
    recovered = leg(recovered_run, False, inputs)
    require(failed["core_sha256"] == recovered["core_sha256"]
            and failed["cases"] == recovered["cases"], "solver/test changed during packaging recovery")
    require(failed["executed_sha256"] == recovered["executed_sha256"],
            "executed source changed during packaging recovery")
    require(failed["source_sha"] != recovered["source_sha"]
            and failed["job_uid"] != recovered["job_uid"]
            and failed["image_id"] == recovered["image_id"], "recovery is not a new changed-source run")
    require(all(descriptor(item["path"]) == item for item in inputs), "recovery input changed during audit")
    return {"schema_version": 2, "verdict": "RECOVERY_VERIFIED",
            "owner": "infra-openopt", "cause": "refinement snapshot missing research document",
            "scope": "packaging failure recovery only; numerical CUDA qualification remains separate",
            "failed_run": str(Path(failed_run).resolve()),
            "recovered_run": str(Path(recovered_run).resolve()),
            "failed": failed, "recovered": recovered, "inputs": inputs}


def verify(path, pin):
    require(re.fullmatch(r"[0-9a-f]{64}", pin) is not None, "invalid caller recovery pin")
    require(descriptor(path)["sha256"] == pin, "caller recovery pin mismatch")
    report = object_json(read(path))
    require(report == build(report["failed_run"], report["recovered_run"]),
            "recovery report does not match retained inputs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "verify"))
    parser.add_argument("--failed-run", type=Path)
    parser.add_argument("--recovered-run", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.action == "record":
        require(args.failed_run and args.recovered_run and args.artifact_dir,
                "record requires both runs and output")
        report = build(args.failed_run, args.recovered_run)
        output = args.artifact_dir.resolve()
        require(output.is_relative_to(SHARED), "output must be external 3FS")
        output.mkdir(parents=True, exist_ok=False)
        path = output / "recovery.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        pin = descriptor(path)["sha256"]
        verify(path, pin)
    else:
        require(args.report and args.sha256, "verify requires report and caller pin")
        path, pin = args.report, args.sha256
        verify(path, pin)
    print(f"CLUSTER_RECOVERY verdict=RECOVERY_VERIFIED "
          f"at={datetime.now(timezone.utc).isoformat()} report={path} sha256={pin}")


if __name__ == "__main__":
    main()
