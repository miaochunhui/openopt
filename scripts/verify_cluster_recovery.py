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


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path).resolve(strict=True)
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
    require(len(set(identities)) == 7 and all(all(identity) for identity in identities),
            "duplicate/missing recovery case identities")
    for case in cases:
        require(case.find("skipped") is None and case.find("error") is None,
                "recovery case skipped or errored")
        failure = case.find("failure")
        if failed:
            require(failure is not None and "FileNotFoundError" in "".join(failure.itertext())
                    and DOC in "".join(failure.itertext()), "original first cause changed")
        else:
            require(failure is None, "recovery still has a failure")
    return [list(identity) for identity in sorted(identities)]


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
    expected = manifest["spec"]["template"]["spec"]["containers"][0]
    actual = pod["spec"]["containers"][0]
    require(actual["command"] == expected["command"]
            and actual["image"] == expected["image"]
            and resources_match(actual["resources"], expected["resources"]),
            "observed command/image/resources mismatch")
    require("--target" in actual["command"] and "tests/test_qp_refinement.py" in actual["command"]
            and actual["command"][-2:] == ["--expression", "cuda_refinement"],
            "unexpected recovery selection")
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
    output = Path(item["output"]).resolve(strict=True)
    require(output.is_relative_to(directory), "recovery output escaped run")
    result_path = output / "result.json"
    result = object_json(take(result_path))
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
        require(result["telemetry_samples"] > 0, "recovery telemetry missing")
    if not failed:
        require(collected["result_path"] == str(result_path)
                and collected["result_sha256"] == descriptor(result_path)["sha256"],
                "collected result pin mismatch")
    for relative in CORE:
        pin = descriptor(directory / "source" / relative)
        inputs.append(pin)
        require(pin["sha256"] == record["source_hashes"][relative]
                == result["source_sha256"][relative], "recovery core source mismatch")
    suites = list(output.glob("cuda-check-*/pytest.xml"))
    require(len(suites) == 1, "missing/duplicate recovery JUnit")
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
            "exit_code": terminal["exitCode"], "restart_count": 0,
            "image_id": containers[0]["imageID"], "finished_at": terminal["finishedAt"]}


def build(failed_run, recovered_run):
    inputs = []
    failed = leg(failed_run, True, inputs)
    recovered = leg(recovered_run, False, inputs)
    require(failed["core_sha256"] == recovered["core_sha256"]
            and failed["cases"] == recovered["cases"], "solver/test changed during packaging recovery")
    require(failed["source_sha"] != recovered["source_sha"]
            and failed["job_uid"] != recovered["job_uid"]
            and failed["image_id"] == recovered["image_id"], "recovery is not a new changed-source run")
    return {"schema_version": 1, "verdict": "RECOVERY_VERIFIED",
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
