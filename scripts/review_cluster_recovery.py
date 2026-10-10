"""Read-only independent review of the e618b68 packaging-recovery evidence.

Only a new external audit directory or explicit copied counterexample fixtures
are written. No kubectl, solver, installation, credentials or network is used.
RECOVERY_REVIEW_PASS is not numerical/commercial/publication qualification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_cluster_recovery as recovery  # noqa: E402

BASE = "e618b68c5209794c8cb1ec4cef727e7e13ef1ada"
REVIEWED = (
    "scripts/verify_cluster_recovery.py", "scripts/qp_campaign.py",
    "tests/test_verify_cluster_recovery.py", "docs/research/cluster-recovery.md",
)
SHARED = Path("/hf3fs-jd/data/tmp").resolve()
TARGET = "tests/test_qp_refinement.py"
SELECTION = "cuda_refinement"
SCOPE = "packaging failure recovery only; numerical CUDA qualification remains separate"
EXPECTED = tuple(sorted(
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
MUTANTS = (
    "replace_selection", "contradict_immutable_result", "contradict_job_template",
    "wrong_runner", "inflate_telemetry", "erase_retained_failure",
    "promote_qualification",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def external(path, *, exists=True):
    """Normalize the established mount alias, refusing other symlink components."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "absolute external path required")
    alias = Path("/hf3fs-jd")
    if path.is_relative_to(alias):
        path = alias.resolve() / path.relative_to(alias)
    require(path.is_relative_to(SHARED), "path outside shared temporary storage")
    for component in (path, *path.parents):
        if component == SHARED.parent:
            break
        require(not component.is_symlink(), "symlink evidence/output refused")
    if exists:
        require(path.exists(), "external evidence missing")
    return path


def read(path):
    path = external(path)
    require(path.is_file(), "nonregular evidence refused")
    raw = path.read_bytes()
    require(raw, "empty evidence refused")
    return raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def descriptor(path):
    path = external(path)
    raw = read(path)
    return {"path": str(path), "sha256": digest(raw), "bytes": len(raw)}


def pinned(path, pin):
    require(isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{64}", pin),
            "invalid caller SHA256")
    raw = read(path)
    require(digest(raw) == pin, "caller evidence pin mismatch")
    return recovery.object_json(raw)


def git(*args):
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=False,
    )
    require(completed.returncode == 0, "read-only Git source lookup failed")
    return completed.stdout


def reviewed_source():
    result = {}
    for relative in REVIEWED:
        frozen = git("show", f"{BASE}:{relative}")
        require((ROOT / relative).read_bytes() == frozen,
                f"reviewed e618b68 source drift: {relative}")
        result[relative] = digest(frozen)
    return {"commit": BASE, "sha256": result}


def strict_junit(raw, failed):
    require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(),
            "unsafe JUnit")
    root = ET.fromstring(raw)
    require(root.tag in ("testsuites", "testsuite"), "invalid JUnit root")
    suites = [root] if root.tag == "testsuite" else list(root)
    require(len(suites) == 1 and suites[0].tag == "testsuite", "exact one JUnit suite required")
    suite = suites[0]
    cases = suite.findall("testcase")
    require(len(cases) == len(list(root.iter("testcase"))) == 7,
            "JUnit complete selection denominator mismatch")
    identities = tuple(sorted((case.get("classname"), case.get("name")) for case in cases))
    require(identities == EXPECTED, "JUnit exact expected selection mismatch")
    counts = {"tests": 7, "failures": 7 if failed else 0, "errors": 0, "skipped": 0}
    require(all(suite.get(key) == str(value) for key, value in counts.items()),
            "JUnit declared counts mismatch")
    for case in cases:
        failures = case.findall("failure")
        require(not case.findall("error") and not case.findall("skipped"),
                "JUnit skip/error refused")
        require(len(failures) == (1 if failed else 0), "JUnit outcome mismatch")
        if failed:
            text = "".join(failures[0].itertext())
            require("FileNotFoundError" in text and recovery.DOC in text,
                    "original packaging first cause changed")
    return [list(identity) for identity in identities]


def strict_leg(directory, failed, inputs):
    directory = external(directory)

    def take(path, *, object_only=True):
        path = external(path)
        require(path.is_relative_to(directory), "leg input escaped pinned run")
        inputs.append(descriptor(path))
        raw = read(path)
        return recovery.object_json(raw) if object_only else raw

    record = take(directory / "run.json")
    collection = take(directory / "collection.json")
    item, collected = record["jobs"][0], collection["jobs"][0]
    output = external(item["output"])
    require(output == directory / "case-0", "unexpected isolated output path")
    require(external(item["manifest"]) == directory / "job-0.json", "manifest escaped run")
    manifest = take(item["manifest"])
    status = take(directory / f"{item['job']}-status.json")
    job, pod = status["job"], status["pods"]["items"][0]
    metadata = {"name": item["job"], "namespace": record["namespace"]}
    require(manifest.get("kind") == job.get("kind") == "Job", "not an actual Job")
    for key, value in metadata.items():
        require(manifest["metadata"].get(key) == job["metadata"].get(key) == value,
                "manifest/Job identity mismatch")
    require(pod["metadata"]["namespace"] == record["namespace"], "Pod namespace mismatch")
    for obj in (manifest["metadata"], job["metadata"], pod["metadata"]):
        require(obj["labels"].get("repo") == "openopt"
                and obj["labels"].get("owner") == "infra-openopt",
                "project identity mismatch")
    owners = [owner for owner in pod["metadata"]["ownerReferences"]
              if owner.get("controller") is True]
    require(len(owners) == 1 and owners[0].get("kind") == "Job"
            and owners[0].get("name") == item["job"]
            and owners[0].get("uid") == job["metadata"]["uid"], "exact Pod controller mismatch")
    require(pod["metadata"]["labels"].get("job-name") == item["job"],
            "Pod job label mismatch")
    expected_command = [
        "python3", str(directory / "source/scripts/gpu_release_check.py"),
        "--artifact-dir", str(output), "--target", TARGET, "--expression", SELECTION,
    ]
    expected = manifest["spec"]["template"]["spec"]["containers"][0]
    specs = (manifest["spec"]["template"]["spec"], job["spec"]["template"]["spec"], pod["spec"])
    for spec in specs:
        require(spec.get("restartPolicy") == "Never" and len(spec["containers"]) == 1
                and not spec.get("initContainers"), "restart/multi-container contract mismatch")
        actual = spec["containers"][0]
        require(actual.get("name") == expected["name"]
                and actual.get("command") == expected_command and not actual.get("args")
                and actual.get("image") == expected["image"]
                and recovery.resources_match(actual["resources"], expected["resources"]),
                "manifest/Job/Pod exact execution mismatch")
        require(all(type(actual["resources"][category].get("nvidia.com/gpu")) in (int, str)
                    and str(actual["resources"][category]["nvidia.com/gpu"]) == "1"
                    for category in ("requests", "limits")), "exact one GPU required")
        env = actual.get("env", [])
        names = [entry["name"] for entry in env]
        require(len(set(names)) == len(names), "duplicate execution environment")
        values = {entry["name"]: entry.get("value") for entry in env}
        require(values.get("PYTHONPATH") == str(directory / "source/src")
                and values.get("TMPDIR") == str(output)
                and values.get("PYTHONDONTWRITEBYTECODE") == "1",
                "execution source/output environment mismatch")
    for obj in (manifest, job):
        require(type(obj["spec"]["backoffLimit"]) is int and obj["spec"]["backoffLimit"] == 0,
                "Job retry policy mismatch")
    conditions = job["status"].get("conditions", [])
    require(any(c.get("type") == ("Failed" if failed else "Complete")
                and c.get("status") == "True" for c in conditions), "Job terminal condition missing")
    require(not any(c.get("type") == ("Complete" if failed else "Failed")
                    and c.get("status") == "True" for c in conditions),
            "contradictory Job terminal condition")
    container = pod["status"]["containerStatuses"][0]
    terminal = container["state"]["terminated"]
    require(type(container["restartCount"]) is int and container["restartCount"] == 0
            and type(terminal["exitCode"]) is int and terminal["exitCode"] == (1 if failed else 0),
            "actual exit/restart mismatch")
    result = take(output / "result.json")
    checks = list(output.glob("cuda-check-*/pytest.xml"))
    require(len(checks) == 1, "exact one immutable check required")
    check = external(checks[0].parent)
    immutable = take(check / "result.json")
    require(read(check / "result.json") == read(output / "result.json"),
            "coordinator/immutable result contradiction")
    cases = strict_junit(take(check / "pytest.xml", object_only=False), failed)
    take(check / "pytest.log", object_only=False)
    take(check / "execution.log", object_only=False)
    if not failed:
        telemetry_path = check / "telemetry.json"
        inputs.append(descriptor(telemetry_path))
        telemetry = recovery.object_json(b'{"samples":' + read(telemetry_path) + b"}")["samples"]
        require(isinstance(telemetry, list) and type(result["telemetry_samples"]) is int
                and 0 < len(telemetry) == result["telemetry_samples"], "telemetry count mismatch")
        require(all(isinstance(sample, dict) and type(sample.get("rc")) is int
                    and sample["rc"] == 0 and sample.get("error") is None
                    and isinstance(sample.get("gpu"), str) and sample["gpu"]
                    for sample in telemetry), "telemetry sample failed")
    require(record["source_identity"]["tree"]
            == git("rev-parse", f"{record['source_sha']}^{{tree}}").decode().strip(),
            "frozen source Git tree mismatch")
    hashes = record["source_hashes"]
    snapshot = directory / "source"
    members = {str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_file()}
    require(members == set(hashes) | {"source-hashes.json"}, "source snapshot membership mismatch")
    require(take(snapshot / "source-hashes.json") == hashes, "snapshot hash manifest mismatch")
    for relative, pin in hashes.items():
        require(isinstance(relative, str) and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts, "unsafe source membership")
        inputs.append(descriptor(snapshot / relative))
        require(digest(read(snapshot / relative)) == pin
                == digest(git("show", f"{record['source_sha']}:{relative}"))
                == record["source_identity"]["sha256"][relative],
                f"source snapshot/Git commit mismatch: {relative}")
    for relative, pin in immutable["source_sha256"].items():
        require(hashes.get(relative) == pin, "result executed-source mismatch")
    execution_members = {name for name in hashes
                         if name.startswith("src/openopt/") or name in (TARGET, "scripts/gpu_release_check.py")}
    require(set(immutable["source_sha256"]) == execution_members,
            "result executed-source membership mismatch")
    return {"cases": cases, "executed_sha256": immutable["source_sha256"],
            "job_uid": job["metadata"]["uid"], "pod_uid": pod["metadata"]["uid"],
            "finished_at": terminal["finishedAt"], "exit_code": terminal["exitCode"]}


def check_live(live, report, report_path, report_pin):
    require(live.get("verdict") == "PASS" and live.get("apiserver_ready") is True
            and live.get("ready") and all(value == "True" for value in live["ready"].values())
            and live.get("active") == [] and live.get("cordoned") == [],
            "historical live health summary inconsistent")
    require(len(live["recovery_reports"]) == 1
            and external(live["recovery_reports"][0]["path"]) == external(report_path)
            and live["recovery_reports"][0]["sha256"] == report_pin, "live recovery pin mismatch")
    require(datetime.fromisoformat(live["read_at"])
            > datetime.fromisoformat(report["recovered"]["finished_at"].replace("Z", "+00:00")),
            "live predates recovered termination")
    retained = live["retained_failures"]
    names = [item["job"] for item in retained]
    require(len(set(names)) == len(names), "duplicate retained failures")
    matched = [item for item in retained if item["job"] == report["failed"]["job"]]
    require(len(matched) == 1, "original failure not retained in live")
    row = matched[0]
    require(row["owner"] == report["owner"] and row["cause"] == report["cause"],
            "retained failure attribution mismatch")
    require(len(row["containers"]) == 1, "retained container denominator mismatch")
    state = row["containers"][0]
    require(type(state["restart_count"]) is int and state["restart_count"] == 0
            and type(state["termination"]["exitCode"]) is int
            and state["termination"]["exitCode"] == report["failed"]["exit_code"]
            and state["termination"]["finishedAt"] == report["failed"]["finished_at"],
            "retained actual first termination changed")
    require(all(item["owner"] != "UNKNOWN" and item["containers"] for item in retained),
            "unattributed retained failure")
    # The old live producer did not persist these identities. Never fill them in
    # from the earlier status capture and pretend they were observed at read_at.
    return {"summary": "CONSISTENT", "read_at": live["read_at"],
            "retained_failures": len(retained), "pod_job_uid_binding": "UNVERIFIED",
            "current_cluster_health": "NOT_READ",
            "historical_attribution_authentication": "UNVERIFIED"}


def review(report_path, report_pin, live_path, live_pin):
    source = reviewed_source()
    report = pinned(report_path, report_pin)
    live = pinned(live_path, live_pin)
    require(report.get("verdict") == "RECOVERY_VERIFIED" and report.get("scope") == SCOPE,
            "packaging report scope/qualification changed")
    recovery.verify(external(report_path), report_pin)
    inputs = [descriptor(report_path), descriptor(live_path)]
    failed = strict_leg(report["failed_run"], True, inputs)
    recovered = strict_leg(report["recovered_run"], False, inputs)
    require(failed["executed_sha256"] == recovered["executed_sha256"],
            "imported solver/checker/test changed during packaging recovery")
    require(failed["pod_uid"] != recovered["pod_uid"], "reused failed Pod")
    live_result = check_live(live, report, report_path, report_pin)
    for entry in report["inputs"]:
        require(entry == descriptor(entry["path"]), "original recovery input changed")
    for entry in inputs:
        require(entry == descriptor(entry["path"]), "independent review input changed during read")
    return {
        "schema_version": 1, "verdict": "RECOVERY_REVIEW_PASS",
        "checked_at": datetime.now(timezone.utc).isoformat(), "reviewed_source": source,
        "packaging_recovery": "PASS", "old_failure": "PRESERVED_FAIL",
        "denominator": {"failed": 7, "recovered": 7, "negative_cases": 3},
        "live": live_result, "numerical_cuda_qualification": "NOT_GRANTED",
        "commercial_qualification": "UNVERIFIED", "publication": "UNVERIFIED",
        "remote_attestation": "NOT_PROVIDED", "inputs": inputs,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relocated(value, paths):
    if isinstance(value, dict):
        return {key: relocated(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [relocated(item, paths) for item in value]
    if isinstance(value, str):
        for old, new in paths.items():
            value = value.replace(old, new)
    return value


def clone_fixture(report, live, destination):
    """Copy real frozen evidence, including sidecars, never edit its original."""
    destination = external(destination, exists=False)
    destination.mkdir(parents=True, exist_ok=False)
    paths = {}
    for key in ("failed_run", "recovered_run"):
        old = external(report[key])
        new = destination / key
        paths[str(old)] = str(new)
        paths[str(Path(report[key]))] = str(new)
        for path in old.rglob("*"):
            if {"pytest-tmp", "pytest-cache"} & set(path.relative_to(old).parts):
                continue
            require(not path.is_symlink(), "fixture source symlink refused")
        shutil.copytree(old, new, ignore=shutil.ignore_patterns("pytest-tmp", "pytest-cache"))
    for new in set(paths.values()):
        for path in Path(new).rglob("*.json"):
            if path.is_file():
                value = json.loads(read(path))
                write_json(path, relocated(value, paths))
        for path in Path(new).rglob("*.xml"):
            path.write_text(relocated(path.read_text(), paths))
        for path in Path(new).rglob("*.log"):
            path.write_text(relocated(path.read_text(), paths))
    return relocated(live, paths)


def mutate_fixture(name, directory, live):
    runs = [directory / "failed_run", directory / "recovered_run"]
    if name == "replace_selection":
        for run in runs:
            xml = next((run / "case-0").glob("cuda-check-*/pytest.xml"))
            root = ET.fromstring(read(xml))
            for index, case in enumerate(root.iter("testcase")):
                case.set("name", f"unrelated_case_{index}")
            xml.write_bytes(ET.tostring(root))
    elif name == "contradict_immutable_result":
        path = next((runs[1] / "case-0").glob("cuda-check-*/result.json"))
        result = recovery.object_json(read(path))
        result["verdict"], result["returncode"] = "FAIL", 1
        write_json(path, result)
    elif name == "contradict_job_template":
        path = next(runs[1].glob("*-status.json"))
        status = recovery.object_json(read(path))
        status["job"]["spec"]["template"]["spec"]["containers"][0]["command"] = ["false"]
        write_json(path, status)
    elif name == "wrong_runner":
        for run in runs:
            path = next(run.glob("*-status.json"))
            status = recovery.object_json(read(path))
            status["pods"]["items"][0]["spec"]["containers"][0]["command"][1] = "/not-frozen/runner.py"
            status["job"]["spec"]["template"]["spec"]["containers"][0]["command"][1] = "/not-frozen/runner.py"
            write_json(path, status)
            path = run / "job-0.json"
            manifest = recovery.object_json(read(path))
            manifest["spec"]["template"]["spec"]["containers"][0]["command"][1] = "/not-frozen/runner.py"
            write_json(path, manifest)
    elif name == "inflate_telemetry":
        for path in (runs[1] / "case-0/result.json",
                     next((runs[1] / "case-0").glob("cuda-check-*/result.json"))):
            result = recovery.object_json(read(path))
            result["telemetry_samples"] += 1
            write_json(path, result)
    elif name == "erase_retained_failure":
        live["retained_failures"] = [
            row for row in live["retained_failures"] if row["owner"] != "infra-openopt"]
    elif name != "promote_qualification":
        raise ValueError("unknown coordinated counterexample")
    # Rebind the coordinator's result pin when it is the mutated surface.
    path = runs[1] / "collection.json"
    collection = recovery.object_json(read(path))
    collection["jobs"][0]["result_sha256"] = descriptor(runs[1] / "case-0/result.json")["sha256"]
    write_json(path, collection)


def counterexamples(report_path, report_pin, live_path, live_pin, directory):
    original = review(report_path, report_pin, live_path, live_pin)
    report, live = pinned(report_path, report_pin), pinned(live_path, live_pin)
    results = []
    for name in ("unchanged_copy", *MUTANTS):
        fixture = directory / name
        copied_live = clone_fixture(report, live, fixture)
        if name != "unchanged_copy":
            mutate_fixture(name, fixture, copied_live)
        forged = recovery.build(fixture / "failed_run", fixture / "recovered_run")
        if name == "promote_qualification":
            forged["scope"] = "certified commercial CUDA qualification"
        report_copy = fixture / "recovery.json"
        write_json(report_copy, forged)
        pin = descriptor(report_copy)["sha256"]
        copied_live["recovery_reports"] = [{"path": str(report_copy), "sha256": pin}]
        live_copy = fixture / "live.json"
        write_json(live_copy, copied_live)
        live_hash = descriptor(live_copy)["sha256"]
        upstream = "ACCEPTED"
        try:
            recovery.verify(report_copy, pin)
        except ValueError:
            upstream = "REJECTED"
        independent, reason = "ACCEPTED", None
        try:
            review(report_copy, pin, live_copy, live_hash)
        except ValueError as exc:
            independent, reason = "REJECTED", str(exc)
        require(independent == ("ACCEPTED" if name == "unchanged_copy" else "REJECTED"),
                f"counterexample oracle failed: {name}")
        results.append({"mutation": name, "upstream_e618b68": upstream,
                        "independent_review": independent, "reason": reason,
                        "report": descriptor(report_copy), "live": descriptor(live_copy)})
        write_json(directory / "counterexamples-progress.json",
                   {"verdict": "RUNNING", "cases": results})
        print(f"RECOVERY_COUNTEREXAMPLE case={name} upstream={upstream} independent={independent}",
              flush=True)
    require(descriptor(report_path)["sha256"] == report_pin
            and descriptor(live_path)["sha256"] == live_pin, "original fixture input mutated")
    review(report_path, report_pin, live_path, live_pin)
    return {"verdict": "COUNTEREXAMPLES_VERIFIED", "original_review": original,
            "negative_controls": len(MUTANTS), "positive_copied_control": 1,
            "originals_unchanged": True, "cases": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("review", "counterexamples"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--live-sha256", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = external(args.artifact_dir, exists=False)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    try:
        call = (args.report, args.report_sha256, args.live, args.live_sha256)
        result = review(*call) if args.action == "review" else counterexamples(*call, output)
        result["exit_code"] = 0
        write_json(output / "review.json", result)
        marker = f"CLUSTER_RECOVERY_REVIEW verdict={result['verdict']} at={started} evidence={output / 'review.json'}"
        (output / "execution.log").write_text(marker + "\n")
        print(marker, flush=True)
        return 0
    except Exception as exc:
        failure = {"verdict": "FAIL", "exit_code": 1, "started_at": started,
                   "first_cause": f"{type(exc).__name__}: {exc}",
                   "report": str(args.report), "live": str(args.live)}
        write_json(output / "failure.json", failure)
        marker = f"CLUSTER_RECOVERY_REVIEW verdict=FAIL at={started} evidence={output / 'failure.json'}"
        (output / "execution.log").write_text(failure["first_cause"] + "\n" + marker + "\n")
        print(marker, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
