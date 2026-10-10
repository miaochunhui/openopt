"""Read-only independent original-model replay of the bounded CUDA smoke.

This is a research evidence reader, not a general MIQP certificate or remote
attestation. The oracle is the independently authored Fraction checker, not
the producer's checker. Actual archived evidence is explicitly caller-pinned.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
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
TARGET = "tests/test_branch_bound.py::test_cuda_frozen_bound_tree"
SOURCES = {
    "src/openopt/branch_bound.py":
        "424a6eecfeafee6d114d3f6a5c37189b8af6b774d9c590360f39f440b4040ad6",
    "src/openopt/qp.py":
        "955d741317e5de1c5743f6e6bc65406ba14560f9a8b4c989afe6d11a1fc58e40",
}
SHARED = Path("/hf3fs-jd/data/tmp").resolve()


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def original_replay(report):
    # The original polynomial and bounds come from independent fixtures,
    # never from the saved model descriptor.
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
    kernels = [event for event in events if event.get("cat") == "kernel"
               and event.get("ph") == "X"]
    require(kernels, "no actual CUDA kernel events")
    for event in kernels:
        require(type(event.get("dur")) in (int, float)
                and math.isfinite(event["dur"]) and event["dur"] > 0
                and isinstance(event.get("name"), str) and event["name"],
                "invalid CUDA kernel duration/name")
    return {"kernels": len(kernels),
            "kernel_sum_us": sum(event["dur"] for event in kernels),
            "scope": "CUPTI archived trace consistency only; no Nsight or speedup qualification"}


def audit(run, proof_pin):
    run = Path(run).resolve(strict=True)
    require(run.is_relative_to(SHARED), "external shared run required")
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
    require(result["tests"] == {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
            and all(type(value) is int for value in result["tests"].values()),
            "actual counts differ")
    require(result["hardware"]["gpu"] == "NVIDIA RTX PRO 4500 Blackwell",
            "result hardware differs")
    for name, pin in SOURCES.items():
        require(digest(take(run / "source" / name)) == pin
                == record["source_hashes"][name] == result["source_sha256"][name],
                "frozen numerical source differs")
    suites = list(output.glob("cuda-check-*/pytest.xml"))
    require(len(suites) == 1, "one JUnit sidecar required")
    junit = take(suites[0])
    require((output / "result.json").read_bytes() ==
            take(junit.with_name("result.json")).read_bytes(),
            "immutable result differs")
    raw = junit.read_bytes()
    require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(),
            "unsafe JUnit")
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
    return {"verdict": "PASS", "read_at": datetime.now(timezone.utc).isoformat(),
            "source_commit": SOURCE_COMMIT, "inputs": before, "original_model": mathematics,
            "trace": trace, "execution_verdict": "UNVERIFIED",
            "performance_verdict": "UNVERIFIED",
            "qualification": "one tiny analytic original MIQP tree replay, not general/commercial qualification"}


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
    parser.add_argument("--proof-sha256", required=True)
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
        report = audit(args.run, args.proof_sha256)
    except Exception as error:
        report = {"verdict": "FAIL", "read_at": datetime.now(timezone.utc).isoformat(),
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
