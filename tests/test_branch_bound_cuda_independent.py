"""Read-only independent original-model replay of the bounded CUDA smoke.

This is a research evidence reader, not a general MIQP certificate or remote
attestation. The oracle is the independently authored Fraction checker, not
the producer's checker. Actual archived evidence is explicitly caller-pinned.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_branch_bound_independent import (
    independent_review, make_problem, specs,
)

SOURCE_COMMIT = "af1ef6d9993357d3647170246fe3d8192392a633"
SOURCE_TREE = "0dee304d077d76e0cc3b65cb037b5ee993365681"
TARGET = "tests/test_branch_bound.py::test_cuda_frozen_bound_tree"
SOURCES = {
    "src/openopt/branch_bound.py":
        "424a6eecfeafee6d114d3f6a5c37189b8af6b774d9c590360f39f440b4040ad6",
    "src/openopt/qp.py":
        "955d741317e5de1c5743f6e6bc65406ba14560f9a8b4c989afe6d11a1fc58e40",
}
SHARED = Path("/hf3fs-jd/data/tmp").resolve()
SELECTED_SOURCES = {**SOURCES, "tests/test_branch_bound.py":
                    "47af150f3c3bdd44cdc2d59b3201fc311ba0a7876b0c6b8f1d8c6f057ed003cf"}
ROLES = {"run", "collection", "job", "status", "result", "side_result",
         "junit", "proof", "trace"}
COUNTS = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
MAX_BYTES = 64 * 1024 * 1024
ORACLE_SHA256 = "592dc7ed9d467da240d2f098e45e7fdac6a2f8e134ae60cdb8023910bf5a8e0d"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def read_json(path):
    return decode_json(Path(path).read_bytes())


def pin_string(value, length=64):
    require(type(value) is str and len(value) == length
            and all(char in "0123456789abcdef" for char in value), "invalid caller pin")
    return value


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys), label + " schema differs")


def same(first, second):
    # JSON distinguishes bool from int, unlike ordinary Python equality.
    return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False)


def relative_name(value):
    require(type(value) is str and value and len(value) <= 4096
            and not value.startswith("/") and "\\" not in value
            and all(part not in ("", ".", "..") for part in value.split("/")),
            "noncanonical relative evidence path")
    return value


def bounded_read(path, *, root=None, relative=None):
    """Open every child with no-follow; never execute archived source."""
    if root is None:
        path = Path(path)
        require(path.is_absolute() and not path.is_symlink(), "absolute regular file required")
        root, relative = path.parent.resolve(strict=True), path.name
    parts = relative_name(relative).split("/")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(file_fd)
            require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_BYTES,
                    "empty/nonregular/oversized evidence")
            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                raw = stream.read(MAX_BYTES + 1)
            after = os.fstat(file_fd)
            require(before == after and len(raw) == before.st_size, "evidence changed while reading")
            return raw
        finally:
            os.close(file_fd)
    finally:
        os.close(fd)


def validate_enclosures(report):
    summary = report["summary"]
    names = ("objective", "lower_bound", "gap_abs", "gap_rel")
    exact_keys(summary["enclosures"], names, "enclosures")
    for key in names:
        value, ends = summary[key], summary["enclosures"][key]
        require(type(value) is str and len(value) <= 5000, "bounded rational summary required")
        exact = Fraction(value)
        require(value == f"{exact.numerator}/{exact.denominator}"
                and max(abs(exact.numerator).bit_length(), exact.denominator.bit_length()) <= 8192,
                "canonical bounded rational required")
        require(type(ends) is list and len(ends) == 2
                and all(type(end) in (int, float) and math.isfinite(end) for end in ends),
                "finite nonbool numerical enclosure required")
        require(Fraction(ends[0]) <= exact <= Fraction(ends[1]), "nonenclosing original value")
        if key in ("gap_abs", "gap_rel"):
            require(exact >= 0, "negative signed gap")


def original_replay(report):
    # The original polynomial and bounds come from independent fixtures,
    # never from the saved model descriptor.
    validate_enclosures(report)
    require(digest(ROOT / "tests/test_branch_bound_independent.py") == ORACLE_SHA256,
            "readonly mathematical oracle bytes differ")
    problem = make_problem(
        SimpleNamespace(BoundedMIQPProblem=SimpleNamespace), specs()["pure"])
    config = SimpleNamespace(
        enabled=True, device="cuda:0", dtype=torch.float64, max_nodes=128,
        qp_max_iterations=2000, qp_check_interval=25, abs_gap=0.,
        rel_gap=0., wall_seconds=None)
    checked = independent_review(problem, config, report, trusted_source=SOURCES)
    require(checked["status"] == "solved_bounded"
            and report["summary"]["objective"] == "0/1"
            and report["summary"]["gap_abs"] == "0/1"
            and report["actual_qp_calls"] == 2
            and report["visited_nodes"] == 3,
            "selected analytic original model differs")
    return checked


def trace_summary(trace):
    require(isinstance(trace, dict), "trace object required")
    devices = trace.get("deviceProperties")
    require(isinstance(devices, list) and len(devices) == 1
            and devices[0].get("name") == "NVIDIA RTX PRO 4500 Blackwell",
            "one actual RTX PRO 4500 trace device required")
    events = trace.get("traceEvents")
    require(isinstance(events, list) and events, "trace events absent")
    require(all(type(event) is dict for event in events), "trace event object required")
    kernels = [event for event in events if event.get("cat") == "kernel"
               and event.get("ph") == "X"]
    require(kernels, "no actual CUDA kernel events")
    for event in kernels:
        require(type(event.get("dur")) in (int, float)
                and math.isfinite(event["dur"]) and event["dur"] > 0
                and isinstance(event.get("name"), str) and event["name"],
                "invalid CUDA kernel duration/name")
    total = sum(event["dur"] for event in kernels)
    require(math.isfinite(total) and total > 0, "trace duration sum overflow")
    return {"kernels": len(kernels),
            "kernel_sum_us": total,
            "scope": "CUPTI archived trace consistency only; no Nsight or speedup qualification"}


def audit(run, proof_pin):
    """Legacy proof-only replay; explicitly NOT v2 archive acceptance.

    Retained for historical compatibility. Its locally observed hashes do not
    establish caller authority for the archive, even if mathematics passes.
    """
    run = Path(run).resolve(strict=True)
    require(run.is_relative_to(SHARED), "external shared run required")
    pin_string(proof_pin)
    before = {}

    def take(path):
        path = Path(path)
        require(not path.is_symlink(), "symlink evidence file rejected")
        path = path.resolve(strict=True)
        require(path.is_relative_to(run) and path.is_file(), "evidence escaped run")
        before[str(path)] = digest(path)
        return path

    record = read_json(take(run / "run.json"))
    collection = read_json(take(run / "collection.json"))
    require(record["source_sha"] == SOURCE_COMMIT
            and record["source_identity"]["clean"] is True
            and record["source_identity"]["commit"] == SOURCE_COMMIT
            and collection["states"] == ["PASS"], "selected clean run differs")
    require(len(record["jobs"]) == len(collection["jobs"]) == 1, "one Job required")
    item = record["jobs"][0]
    output = Path(item["output"]).resolve(strict=True)
    require(output == run / "case-0", "isolated result output differs")
    result = read_json(take(output / "result.json"))
    require(result["verdict"] == "PASS" and type(result["returncode"]) is int
            and result["returncode"] == 0 and result["targets"] == [TARGET],
            "actual CUDA test completion differs")
    require(result["tests"] == COUNTS and all(type(value) is int for value in result["tests"].values()),
            "actual counts differ")
    require(result["hardware"]["gpu"] == "NVIDIA RTX PRO 4500 Blackwell", "result hardware differs")
    for name, pin in SOURCES.items():
        require(digest(take(run / "source" / name)) == pin
                == record["source_hashes"][name] == result["source_sha256"][name],
                "frozen numerical source differs")
    suites = list(output.glob("cuda-check-*/pytest.xml"))
    require(len(suites) == 1, "one JUnit sidecar required")
    junit = take(suites[0])
    require((output / "result.json").read_bytes() == take(junit.with_name("result.json")).read_bytes(),
            "immutable result differs")
    raw = junit.read_bytes()
    require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(), "unsafe JUnit")
    cases = list(ET.fromstring(raw).iter("testcase"))
    require(len(cases) == 1 and cases[0].get("classname") == "tests.test_branch_bound"
            and cases[0].get("name") == "test_cuda_frozen_bound_tree"
            and not any(cases[0].find(tag) is not None for tag in ("failure", "error", "skipped")),
            "actual JUnit test differs")
    properties = cases[0].findall("properties/property")
    names = [entry.get("name") for entry in properties]
    require(len(names) == len(set(names)), "duplicate proof properties")
    props = {entry.get("name"): entry.get("value") for entry in properties}
    proof = take(props["receipt"])
    require(digest(proof) == proof_pin == props["receipt_sha256"], "caller proof pin differs")
    mathematics = original_replay(read_json(proof))
    trace = trace_summary(read_json(take(props["trace"])))
    require(all(digest(path) == pin for path, pin in before.items()), "input drift")
    return {"schema_version": 1, "verdict": "LEGACY_NUMERICAL_ONLY",
            "numerical_verdict": "PASS", "archive_binding_verdict": "UNVERIFIED",
            "read_at": datetime.now(timezone.utc).isoformat(), "source_commit": SOURCE_COMMIT,
            "inputs": before, "original_model": mathematics, "trace": trace,
            "execution_verdict": "UNVERIFIED", "performance_verdict": "UNVERIFIED",
            "commercial_verdict": "UNVERIFIED",
            "qualification": "legacy proof-only replay; unsupported for full caller-pinned archive acceptance"}


def audit_v2(run, manifest, expected_pin):
    """Check caller-selected archived bytes and mathematics, never execution origin."""
    pin_string(expected_pin)
    run = Path(run).resolve(strict=True)
    require(run.is_dir(), "run directory required")
    manifest = Path(manifest).absolute()
    manifest_raw = bounded_read(manifest)
    require(hashlib.sha256(manifest_raw).hexdigest() == expected_pin, "caller manifest pin differs")
    selected = decode_json(manifest_raw)
    exact_keys(selected, {"schema_version", "authority", "run_root", "target", "source_commit",
                          "source_tree", "job_name", "node_name", "pod_name", "files",
                          "selected_source"}, "manifest")
    require(type(selected["schema_version"]) is int and selected["schema_version"] == 2,
            "explicit v2 manifest required")
    require(selected["authority"] == "caller-selected-archive-only"
            and selected["run_root"] == str(run) and selected["target"] == TARGET
            and selected["source_commit"] == SOURCE_COMMIT
            and selected["source_tree"] == SOURCE_TREE, "caller archive identity differs")
    for key in ("job_name", "node_name", "pod_name"):
        require(type(selected[key]) is str and 0 < len(selected[key]) <= 253,
                "nonempty caller identity required")
    exact_keys(selected["files"], ROLES, "file membership")
    exact_keys(selected["selected_source"], SELECTED_SOURCES, "selected source membership")
    descriptors = {**selected["files"], **selected["selected_source"]}
    names, saved = {}, {}
    for role, descriptor in descriptors.items():
        exact_keys(descriptor, {"path", "sha256", "bytes"}, "descriptor")
        name = relative_name(descriptor["path"])
        pin_string(descriptor["sha256"])
        require(type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] <= MAX_BYTES,
                "positive nonbool bounded evidence bytes required")
        require(name not in names.values(), "duplicate evidence member")
        raw = bounded_read(None, root=run, relative=name)
        require(len(raw) == descriptor["bytes"]
                and hashlib.sha256(raw).hexdigest() == descriptor["sha256"],
                "caller pinned evidence bytes differ: " + role)
        names[role], saved[role] = name, raw
    require(names["run"] == "run.json" and names["collection"] == "collection.json"
            and names["job"] == "job-0.json" and names["result"] == "case-0/result.json"
            and names["status"] == selected["job_name"] + "-status.json",
            "canonical run membership differs")
    junit_path = Path(names["junit"])
    side = junit_path.parent
    require(junit_path.name == "pytest.xml" and side.parent == Path("case-0")
            and side.name.startswith("cuda-check-") and len(side.name) > len("cuda-check-")
            and names["side_result"] == str(side / "result.json")
            and names["proof"] == str(side / "pytest-tmp/test_cuda_frozen_bound_tree0/bounded-miqp-cuda.json")
            and names["trace"] == str(side / "pytest-tmp/test_cuda_frozen_bound_tree0/cuda-trace.json"),
            "selected sidecar membership differs")

    def path_match(value, name):
        require(type(value) is str and Path(value).is_absolute(), "absolute archived path required")
        require(Path(value).resolve(strict=True) == run / name, "archived path identity differs")

    record, collection = (decode_json(saved[key]) for key in ("run", "collection"))
    for document in (record, collection):
        identity = document["source_identity"]
        require(document["source_sha"] == SOURCE_COMMIT and identity["commit"] == SOURCE_COMMIT
                and identity["tree"] == SOURCE_TREE and identity["clean"] is True,
                "clean source commit/tree differs")
        hashes = document["source_hashes"]
        require(type(hashes) is dict and hashes and type(identity["sha256"]) is dict,
                "nonempty source maps required")
        for name, pin in hashes.items():
            relative_name(name)
            pin_string(pin)
            require(identity["sha256"].get(name) == pin, "selected source hash map differs")
        require(type(document["jobs"]) is list and len(document["jobs"]) == 1,
                "exactly one selected Job required")
    require(same(record["source_identity"], collection["source_identity"])
            and same(record["source_hashes"], collection["source_hashes"])
            and collection["states"] == ["PASS"]
            and record["namespace"] == collection["namespace"]
            and record["stage"] == collection["stage"] == "release-regression",
            "run/collection source or stage differs")
    item, collected = record["jobs"][0], collection["jobs"][0]
    exact_keys(item, {"job", "size", "output", "manifest"}, "run Job")
    exact_keys(collected, set(item) | {"result_path", "result_sha256"}, "collection Job")
    require(type(item["size"]) is int and item["size"] == 1
            and item["job"] == selected["job_name"]
            and same(item, {key: collected[key] for key in item}), "selected Job identity differs")
    path_match(item["output"], "case-0")
    path_match(item["manifest"], names["job"])
    path_match(collected["result_path"], names["result"])
    require(collected["result_sha256"] == selected["files"]["result"]["sha256"],
            "collection result SHA differs")
    result = decode_json(saved["result"])
    require(saved["result"] == saved["side_result"], "result copies differ")
    require(result["verdict"] == "PASS" and type(result["returncode"]) is int
            and result["returncode"] == 0 and result["targets"] == [TARGET]
            and same(result["tests"], COUNTS), "selected result completion/counts differ")
    require(result["hardware"]["gpu"] == "NVIDIA RTX PRO 4500 Blackwell"
            and result["hardware"]["node"] == selected["node_name"], "result node/GPU differs")
    for name, pin in SELECTED_SOURCES.items():
        require(names[name] == "source/" + name
                and selected["selected_source"][name]["sha256"] == pin
                == record["source_hashes"][name] == result["source_sha256"][name],
                "frozen selected source differs: " + name)
    job, status = (decode_json(saved[key]) for key in ("job", "status"))
    observed = status["job"]
    for document in (job, observed):
        require(document["apiVersion"] == "batch/v1" and document["kind"] == "Job"
                and document["metadata"]["name"] == selected["job_name"]
                and document["metadata"]["namespace"] == record["namespace"],
                "archived Job identity differs")
    require(type(observed["status"]["succeeded"]) is int and observed["status"]["succeeded"] == 1
            and any(c["type"] == "Complete" and c["status"] == "True"
                    for c in observed["status"]["conditions"])
            and not any(c["type"] == "Failed" and c["status"] == "True"
                        for c in observed["status"]["conditions"]), "archived Job not complete")
    pods = status["pods"]["items"]
    require(type(pods) is list and len(pods) == 1, "exactly one archived Pod required")
    pod = pods[0]
    require(pod["apiVersion"] == "v1" and pod["kind"] == "Pod"
            and pod["metadata"]["name"] == selected["pod_name"]
            and pod["metadata"]["namespace"] == record["namespace"]
            and pod["metadata"]["labels"]["job-name"] == selected["job_name"]
            and pod["spec"]["nodeName"] == selected["node_name"]
            and pod["status"]["phase"] == "Succeeded", "archived Pod/node differs")
    owners = [owner for owner in pod["metadata"]["ownerReferences"] if owner["controller"] is True]
    require(len(owners) == 1 and owners[0]["apiVersion"] == "batch/v1"
            and owners[0]["kind"] == "Job" and owners[0]["name"] == selected["job_name"]
            and type(observed["metadata"]["uid"]) is str and observed["metadata"]["uid"]
            and owners[0]["uid"] == observed["metadata"]["uid"], "archived controller binding differs")
    containers = pod["status"]["containerStatuses"]
    require(type(containers) is list and len(containers) == 1, "one completed container required")
    container = containers[0]
    require(type(container["restartCount"]) is int and container["restartCount"] == 0
            and type(container["state"]["terminated"]["exitCode"]) is int
            and container["state"]["terminated"]["exitCode"] == 0, "archived exit/restart differs")
    raw = saved["junit"]
    require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(), "unsafe JUnit")
    xml = ET.fromstring(raw)
    require(xml.tag == "testsuites" and xml.get("name") == "pytest tests"
            and len(xml) == 1 and xml[0].tag == "testsuite", "JUnit suite identity differs")
    suite = xml[0]
    require(suite.get("name") == "pytest" and suite.get("hostname") == selected["pod_name"]
            and all(suite.get(key) == str(value) for key, value in COUNTS.items())
            and len(suite) == 1 and suite[0].tag == "testcase", "JUnit suite counts/host differ")
    case = suite[0]
    require(case.get("classname") == "tests.test_branch_bound"
            and case.get("name") == "test_cuda_frozen_bound_tree"
            and len(case) == 1 and case[0].tag == "properties", "JUnit exact test membership differs")
    entries = list(case[0])
    require(len(entries) == 3 and all(entry.tag == "property" for entry in entries)
            and len({entry.get("name") for entry in entries}) == 3, "duplicate JUnit properties")
    props = {entry.get("name"): entry.get("value") for entry in entries}
    exact_keys(props, {"receipt", "receipt_sha256", "trace"}, "JUnit properties")
    path_match(props["receipt"], names["proof"])
    path_match(props["trace"], names["trace"])
    require(props["receipt_sha256"] == selected["files"]["proof"]["sha256"], "JUnit proof SHA differs")
    mathematics = original_replay(decode_json(saved["proof"]))
    trace = trace_summary(decode_json(saved["trace"]))
    require(all(bounded_read(None, root=run, relative=names[key]) == raw
                for key, raw in saved.items()) and bounded_read(manifest) == manifest_raw, "input drift")
    return {"schema_version": 2, "verdict": "ARCHIVE_NUMERICAL_PASS",
            "numerical_verdict": "PASS", "archive_binding_verdict": "PASS",
            "read_at": datetime.now(timezone.utc).isoformat(), "source_commit": SOURCE_COMMIT,
            "source_tree": SOURCE_TREE, "manifest_sha256": expected_pin,
            "inputs": descriptors, "original_model": mathematics, "trace": trace,
            "execution_verdict": "UNVERIFIED", "performance_verdict": "UNVERIFIED",
            "commercial_verdict": "UNVERIFIED",
            "qualification": "caller-selected archive and tiny host proof only; no remote origin attestation"}


def test_trace_missing_kernel_and_invalid_duration_reject():
    trace = {"deviceProperties": [{"name": "NVIDIA RTX PRO 4500 Blackwell"}],
             "traceEvents": [{"cat": "kernel", "ph": "X", "name": "real", "dur": 1.}]}
    assert trace_summary(trace)["kernels"] == 1
    for duration in (True, 0., -1., math.inf):
        changed = deepcopy(trace)
        changed["traceEvents"][0]["dur"] = duration
        with pytest.raises(ValueError):
            trace_summary(changed)
    trace["traceEvents"] = [{"cat": "cpu_op", "ph": "X"}]
    with pytest.raises(ValueError, match="no actual CUDA"):
        trace_summary(trace)


def test_duplicate_json_is_not_accepted(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"passed":true,"passed":false}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest-sha256", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only replay required")
    root = args.artifact_dir.resolve()
    require(root.is_relative_to(SHARED) and not root.is_relative_to(ROOT),
            "external shared output required")
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="replay-", dir=root))
    report = {"verdict": "RUNNING"}
    try:
        report = audit_v2(args.run, args.evidence_manifest, args.evidence_manifest_sha256)
    except Exception as error:
        report = {"verdict": "FAIL", "read_at": datetime.now(timezone.utc).isoformat(),
                  "execution_verdict": "UNVERIFIED", "performance_verdict": "UNVERIFIED",
                  "commercial_verdict": "UNVERIFIED",
                  "first_failure": f"{type(error).__name__}: {error}"}
        raise
    finally:
        path = output / "independent.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        line = f"BOUNDED_MIQP_CUDA_REPLAY verdict={report['verdict']} report={path}"
        (output / "execution.log").write_text(line + "\n")
        print(line, flush=True)


if __name__ == "__main__":
    main()
