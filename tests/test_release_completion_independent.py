"""Independent offline precheck review, not actual optimizer/release qualification.

All invented records are explicit adversarial test bytes. Nothing solves,
imports an SDK, contacts a server, executes input source, or submits a GPU.
Use this file's --run/--replay CLI with an explicit frozen owner selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = "afb1f9bc19acd40fef9c656c7e7ce2c45db43511"
TEST = "tests/test_release_completion_independent.py"
DOC = "docs/research/release-completion-independent.md"
TOOL = "scripts/check_release_completion.py"
ENV = ("OPENOPT_COMPLETION_OWNER_ROOT", "OPENOPT_COMPLETION_OWNER_COMMIT",
       "OPENOPT_COMPLETION_TOOL_SHA256")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
CAPABILITIES = (
    "lp", "convex_qp", "sparse_matrix_free", "scaling", "infeasibility_recession",
    "signed_gap_status", "soc_rsoc", "sdp", "exponential_power", "general_milp",
    "general_convex_miqp", "nonlinear_global", "presolve_postsolve", "refinement",
    "held_out_public_robustness", "perturbation_long_run", "failure_tail_controls",
    "cold_prepared_changing_data", "release_runtime_compatibility",
)
SOLVERS = ("gurobi", "cplex", "mosek")
SCOPE = "insufficient-release-evidence-v1"
EVIDENCE_ROOT = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/release-completion-independent")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(path, value):
    payload = value if type(value) is bytes else encoded(value)
    with Path(path).open("xb") as stream:
        stream.write(payload)
    assert Path(path).read_bytes() == payload
    return {"path": str(path), "sha256": digest(payload), "bytes": len(payload)}


def git(root, *args, input_bytes=None):
    process = subprocess.run(["git", "-C", str(root), *args], input=input_bytes,
                             capture_output=True, timeout=25, check=False)
    if process.returncode:
        raise ValueError(f"Git failed rc={process.returncode}")
    return process.stdout


def snapshot(root):
    root = Path(root)
    paths = sorted(set(git(root, "ls-files", "--cached", "--others",
                           "--exclude-standard", "-z").decode().split("\0")) - {""})
    hashes = {}
    for name in paths:
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"unsafe source membership: {name}")
        hashes[name] = digest(path.read_bytes())
    if not hashes:
        raise ValueError("empty source membership")
    return {"commit": git(root, "rev-parse", "HEAD").decode().strip(),
            "tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "clean": git(root, "status", "--porcelain", "--untracked-files=all") == b"",
            "sha256": hashes}


def committed(root, value):
    names = sorted(set(git(root, "ls-files", "-z").decode().split("\0")) - {""})
    if set(names) != set(value["sha256"]):
        raise ValueError("current/committed source membership differs")
    query = "".join(f"{value['commit']}:{name}\n" for name in names).encode()
    stream = io.BytesIO(git(root, "cat-file", "--batch", input_bytes=query))
    for name in names:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("source Git blob unavailable")
        payload = stream.read(int(header[2]))
        if stream.read(1) != b"\n" or digest(payload) != value["sha256"][name]:
            raise ValueError(f"committed/current source differs: {name}")
    if stream.read():
        raise ValueError("extra Git blob output")


def owner_selection():
    values = [os.environ.get(key) for key in ENV]
    if not all(values):
        raise ValueError("explicit owner root/full commit/tool SHA required together")
    root, commit, pin = values
    path = Path(root)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("owner root must be canonical absolute")
    if not re.fullmatch("[0-9a-f]{40}", commit) or not re.fullmatch("[0-9a-f]{64}", pin):
        raise ValueError("full lowercase owner commit/tool SHA required")
    source = snapshot(path)
    if source["commit"] != commit or not source["clean"]:
        raise ValueError("owner caller commit/clean mismatch")
    committed(path, source)
    if source["sha256"].get(TOOL) != pin:
        raise ValueError("caller-pinned tool bytes mismatch")
    return path, source, pin


def external(path):
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError("external evidence requires absolute path without traversal")
    path = raw.resolve()
    owner = os.environ.get(ENV[0])
    roots = [ROOT] + ([Path(owner).resolve()] if owner else [])
    if (not path.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve())
            or any(path.is_relative_to(root) for root in roots)):
        raise ValueError("evidence must be external to all source trees on 3FS tmp")
    return path


@pytest.fixture(scope="session")
def subject():
    root, source, pin = owner_selection()
    path = root / TOOL
    spec = importlib.util.spec_from_file_location("independent_completion_subject", path)
    module = importlib.util.module_from_spec(spec)
    # Compile the selected immutable bytes, not input descriptors or .pyc.
    payload = path.read_bytes()
    assert digest(payload) == pin
    exec(compile(payload, str(path), "exec"), module.__dict__)
    assert module.precheck.__code__.co_filename == str(path)
    yield module
    assert snapshot(root) == source, "owner source changed during independent tests"


class AdversarialBytes:
    """Our own Git/byte graph; fabricated assertions never become real evidence."""

    def __init__(self, root):
        self.root = root
        root.mkdir()
        self.repo = root / "invented-source"
        self.repo.mkdir()
        (self.repo / "src").mkdir()
        self.producer_bytes = b"# Independent adversarial producer: no solver implementation.\n"
        self.license_bytes = b"BSD Zero Clause License - invented test only, no legal authority.\n"
        write(self.repo / "src/marker.py", self.producer_bytes)
        write(self.repo / "LICENSE", self.license_bytes)
        write(self.repo / "pyproject.toml", b"# Independent adversarial metadata only.\n")
        git(self.repo, "init")
        git(self.repo, "add", "--", "src/marker.py", "LICENSE", "pyproject.toml")
        git(self.repo, "-c", "user.name=IndependentTest", "-c", "user.email=audit@example.invalid",
            "commit", "-m", "Invented source without any solver or authorization")
        self.source = snapshot(self.repo)
        assert self.source["clean"]
        self.index = 0
        self.files = {}
        self.manifest = {
            "schema_version": 1, "scope": SCOPE, "source_commit": self.source["commit"],
            "source_sha256": self.source["sha256"], "capability_scope": list(CAPABILITIES),
            "capabilities": {}, "commercial": {}, "gpu_scaling": {},
            "publication": None, "licensing": None, "historical_observations": [],
        }
        self.model = self.put({"adversarial_not_executed": "one invented LP",
                               "P": [[0]], "q": [1], "A": [[1]], "l": [0], "u": [1]})
        self.resource = digest(b"fabricated resource identity; no GPU or CPU benchmark")
        self.producer = self.put(self.producer_bytes)
        self.auditor = self.put(b"raise RuntimeError('input auditor bytes must never execute')\n")
        self.known_pairs = {solver: self.pairs(solver) for solver in ("openopt", *SOLVERS)}

    def put(self, value):
        self.index += 1
        desc = write(self.root / f"adversarial-{self.index:04d}.json", value)
        self.files[desc["path"]] = desc
        return desc

    def load(self, desc):
        return json.loads(Path(desc["path"]).read_bytes())

    @staticmethod
    def counts(n):
        return {"attempted": n, "accepted": n, "failed": 0, "unsupported": 0, "skipped": 0}

    def pairs(self, solver, *, model=None, resource=None):
        records = []
        model = model or self.model
        resource = resource or self.resource
        for mode in ("cold", "prepared"):
            result = self.put({
                "source_sha256": self.source["sha256"], "model_sha256": model["sha256"],
                "sample_id": mode, "mode": mode, "resource_sha256": resource,
                "solver": solver, "execution": "ACTUAL_NATIVE_SOLVE", "status": "solved", "wall_s": 1,
            })
            check = self.put({
                "source_sha256": self.source["sha256"], "model_sha256": model["sha256"],
                "result_sha256": result["sha256"], "verdict": "PASS", "original_coordinates": "PASS",
                "signed_gap": "PASS", "status_certificates": "PASS",
            })
            records.append({"id": mode, "model": model, "result": result, "check": check,
                            "mode": mode, "resource_sha256": resource})
        return records

    def envelope(self, section, data):
        actual = self.put({"source_sha256": self.source["sha256"], "section": section, "data": data})
        acceptance = self.put({
            "source_sha256": self.source["sha256"], "section": section,
            "actual_sha256": actual["sha256"], "auditor_sha256": self.auditor["sha256"],
            "verdict": "PASS",
        })
        return self.put({"schema_version": 1, "section": section, "source_sha256": self.source["sha256"],
                         "producer": self.producer, "auditor": self.auditor,
                         "actual": actual, "acceptance": acceptance})

    def section(self, field, key=None):
        desc = self.manifest[field] if key is None else self.manifest[field][key]
        receipt = self.load(desc)
        return receipt, self.load(receipt["actual"])["data"]

    def replace(self, field, key, operation, *, reseal=True):
        receipt, data = self.section(field, key)
        operation(data)
        actual = self.load(receipt["actual"])
        actual["data"] = data
        receipt["actual"] = self.put(actual)
        if reseal:
            acceptance = self.load(receipt["acceptance"])
            acceptance["actual_sha256"] = receipt["actual"]["sha256"]
            receipt["acceptance"] = self.put(acceptance)
        desc = self.put(receipt)
        if key is None:
            self.manifest[field] = desc
        else:
            self.manifest[field][key] = desc

    def gpu_data(self, count, *, pairs=None):
        pairs = pairs or self.known_pairs["openopt"]
        ids = [f"GPU-INVENTED-{i}" for i in range(count)]
        results = [row["result"]["sha256"] for row in pairs]
        return {
            "gpu_count": count, "devices": [{"id": name, "model": "RTX PRO 4500"} for name in ids],
            "timing_scope": "transfer_inclusive_end_to_end", "counts": self.counts(2), "samples": pairs,
            "trace": self.put({"gpu_count": count, "result_sha256": results,
                               "kernel_events": {name: 1 for name in ids}}),
            "coordinator": self.put({"source_sha256": self.source["sha256"], "gpu_count": count,
                                     "result_sha256": results, "execution_verdict": "PASS"}),
        }

    def complete_looking(self):
        for capability in CAPABILITIES:
            self.manifest["capabilities"][capability] = self.envelope("capability:" + capability, {
                "capability": capability, "status": "PASS", "negative_controls": 1,
                "counts": self.counts(2), "samples": self.known_pairs["openopt"],
            })
        proof = {"status": "AUTHORIZED", "pin": digest(b"invented approval pointer, not authority")}
        for solver in SOLVERS:
            self.manifest["commercial"][solver] = self.envelope("commercial:" + solver, {
                "authorization": proof, "actual_native_calls": 2, "completed": 2,
                "counts": self.counts(2), "samples": self.known_pairs[solver],
                "openopt": {"counts": self.counts(2), "samples": self.known_pairs["openopt"]},
            })
        for count in (1, 2, 3):
            self.manifest["gpu_scaling"][str(count)] = self.envelope(f"gpu:{count}", self.gpu_data(count))
        page = self.put(b"<html>Invented local bytes only; never published.</html>\n")
        remote = self.put({"commit": self.source["commit"], "source_sha256": self.source["sha256"]})
        self.manifest["publication"] = self.envelope("publication", {
            "status": "PUBLISHED", "source_commit": self.source["commit"], "published": page,
            "readback": self.put(Path(page["path"]).read_bytes()), "remote_source": remote,
            "transport": self.put({"published_sha256": page["sha256"], "readback_sha256": page["sha256"],
                                   "remote_source_sha256": remote["sha256"], "http_status": 200,
                                   "status": "PUBLISHED", "method": "AUTHORIZED_MANAGED_READBACK"}),
        })
        scope = "runtime_build_optional_sdk"
        self.manifest["licensing"] = self.envelope("licensing", {
            "project_license": "0BSD", "license": self.put(self.license_bytes),
            "origins": [{"paths": list(self.source["sha256"]), "status": "REDISTRIBUTION_AUTHORIZED",
                         "proof": proof}],
            "dependency_scope": scope,
            "sbom": self.put({"source_sha256": self.source["sha256"], "scope": scope, "dependencies": [
                {"name": "invented", "version": "1", "license": "0BSD", "distribution": "NOT_BUNDLED",
                 "proof": proof}]}),
            "inventory": self.put({"source_sha256": self.source["sha256"], "scope": scope,
                                   "dependencies": [{"name": "invented", "version": "1"}]}),
        })
        return self


@pytest.fixture(scope="session")
def invented(tmp_path_factory):
    return AdversarialBytes(tmp_path_factory.mktemp("invented-parent") / "bytes").complete_looking()


@pytest.fixture
def graph(invented, tmp_path):
    instance = copy.copy(invented)
    instance.manifest = copy.deepcopy(invented.manifest)
    instance.files = copy.deepcopy(invented.files)
    instance.root = tmp_path / "new-mutation-bytes"
    instance.root.mkdir()
    return instance


def never_complete(report):
    assert report["verdict"] in ("INCOMPLETE", "REVIEW_REQUIRED")
    assert report["release_complete"] is False
    assert report["execution_attestation"] == "UNVERIFIED"
    for key in ("legal_authority", "general_capability"):
        if key in report:
            assert report[key] == "UNVERIFIED"


def observe(subject, graph, request, tmp_path, expected, *, cli=False, raw=None,
            pin=None, root=None):
    desc = graph.put(raw if raw is not None else graph.manifest)
    selected_pin = pin or desc["sha256"]
    source_root = root or graph.repo
    before = snapshot(source_root)
    source_files = copy.deepcopy(graph.files)
    assert all(digest(Path(path).read_bytes()) == data["sha256"] for path, data in source_files.items())
    outcome, report, error = None, None, None
    if cli:
        owner_root = Path(os.environ[ENV[0]])
        output = tmp_path / "actual-cli-output"
        command = [sys.executable, "-B", str(owner_root / TOOL), "--manifest", desc["path"],
                   "--manifest-sha256", selected_pin, "--current-root", str(source_root),
                   "--artifact-dir", str(output)]
        process = subprocess.run(command, cwd=ROOT, env=environment(tmp_path), capture_output=True,
                                 text=True, timeout=30, check=False)
        log = tmp_path / "actual-cli.log"
        write(log, (process.stdout + process.stderr + f"\nreturncode={process.returncode}\n").encode())
        paths = list(output.glob("completion-*/completion.json"))
        assert len(paths) == 1, "CLI terminal receipt missing/ambiguous"
        report = json.loads(paths[0].read_bytes())
        outcome = "CLI"
        assert process.returncode == report["returncode"] == (1 if report["verdict"] == "INCOMPLETE" else 2)
        assert report["gate_source_sha256"] == os.environ[ENV[2]]
        command_evidence = {"command": command, "returncode": process.returncode,
                            "report": str(paths[0]), "report_sha256": digest(paths[0].read_bytes()),
                            "log": str(log), "log_sha256": digest(log.read_bytes())}
    else:
        command_evidence = None
        try:
            report = subject.precheck(desc["path"], selected_pin, current_root=source_root)
            outcome = "RETURNED"
        except Exception as failure:
            outcome = "REJECTED"
            error = {"type": type(failure).__name__, "code": getattr(failure, "code", None),
                     "coordinate": getattr(failure, "at", None)}
    if report is not None:
        never_complete(report)
    codes = sorted({row["code"] for row in report["incomplete"]}) if report else []
    value = {"nodeid": request.node.nodeid, "outcome": outcome, "report": report, "error": error,
             "expected": expected, "codes": codes, "manifest": desc, "caller_manifest_pin": selected_pin,
             "source_root": str(source_root), "source_before": before, "source_after": snapshot(source_root),
             "files": source_files, "cli": command_evidence, "actual_solver_calls": 0,
             "actual_gpu_calls": 0, "actual_network_calls": 0, "input_execution_calls": 0}
    path = tmp_path / "independent-observation.json"
    write(path, value)
    request.node.user_properties.extend([
        ("independent_observation", str(path)), ("independent_observation_sha256", digest(path.read_bytes())),
    ])
    assert value["source_before"] == value["source_after"], "assessed source changed"
    assert all(digest(Path(p).read_bytes()) == d["sha256"] for p, d in source_files.items())
    if expected == "REVIEW_REQUIRED":
        assert report is not None and report["verdict"] == expected and report["incomplete"] == []
    elif expected == "INCOMPLETE":
        assert report is not None and report["verdict"] == expected
    elif outcome == "REJECTED":
        assert error == {"type": "Rejected", "code": expected, "coordinate": error["coordinate"]}
    else:
        assert expected in codes and report["verdict"] == "INCOMPLETE"
    return value


def test_all_coordinated_invented_flags_never_complete(subject, graph, request, tmp_path):
    assert not (graph.repo / "src/solver.py").exists()
    assert b"no solver implementation" in (graph.repo / "src/marker.py").read_bytes()
    observe(subject, graph, request, tmp_path, "REVIEW_REQUIRED", cli=True)


def test_empty_enumerated_evidence_reports_all_missing(subject, graph, request, tmp_path):
    graph.manifest.update(capabilities={}, commercial={}, gpu_scaling={}, publication=None, licensing=None)
    value = observe(subject, graph, request, tmp_path, "INCOMPLETE")
    missing = {row["section"] for row in value["report"]["incomplete"]}
    assert missing == ({f"capability:{name}" for name in CAPABILITIES}
                       | {f"commercial:{name}" for name in SOLVERS}
                       | {"gpu:1", "gpu:2", "gpu:3", "gpu_pairing", "publication", "licensing"})


@pytest.mark.parametrize("field", ("release_complete", "verdict", "authorized", "fixture_is_actual"))
def test_added_manual_approval_fields_reject(subject, graph, request, tmp_path, field):
    graph.manifest[field] = "PASS"
    observe(subject, graph, request, tmp_path, "schema_keys")


@pytest.mark.parametrize("case", ("empty-sources", "drop-capability", "unknown-capability",
                                  "short-commit", "wrong-commit", "drop-source", "wrong-source"))
def test_scope_and_source_membership_negatives(subject, graph, request, tmp_path, case):
    expected = "current_source_mismatch"
    if case == "empty-sources":
        graph.manifest["source_sha256"] = {}
        expected = "empty_source_map"
    elif case == "drop-capability":
        graph.manifest["capability_scope"].remove("general_milp")
        expected = "incomplete_capability_scope"
    elif case == "unknown-capability":
        graph.manifest["capabilities"]["universal_optimizer"] = {"verdict": "PASS"}
        expected = "unknown_selection"
    elif case == "short-commit":
        graph.manifest["source_commit"] = graph.manifest["source_commit"][:7]
        expected = "source_commit"
    elif case == "wrong-commit":
        graph.manifest["source_commit"] = "0" * 40
    elif case == "drop-source":
        graph.manifest["source_sha256"].pop("pyproject.toml")
    else:
        graph.manifest["source_sha256"]["src/marker.py"] = "0" * 64
    observe(subject, graph, request, tmp_path, expected)


@pytest.mark.parametrize("raw,expected", (
    (b"", "empty_or_large_input"),
    (b"{}", "schema_keys"),
    (b'{"schema_version":1,"schema_version":1}', "duplicate_json_key"),
    (b'{"schema_version":true}', "boolean_not_evidence"),
    (b'{"schema_version":1e400}', "nonfinite_json"),
    (b'{"schema_version":NaN}', "nonfinite_json"),
))
def test_raw_empty_duplicate_bool_nonfinite(subject, graph, request, tmp_path, raw, expected):
    observe(subject, graph, request, tmp_path, expected, raw=raw)


@pytest.mark.parametrize("case,expected", (
    ("old-acceptance", "independent_acceptance_binding"),
    ("zero-samples", "no_actual_solves"),
    ("dropped-failure-denominator", "incomplete_or_contradictory_counts"),
    ("skip", "incomplete_or_contradictory_counts"),
    ("zero-negative-controls", "invalid_integer"),
    ("bool-count", "boolean_not_evidence"),
    ("duplicate-pair", "duplicate_samples"),
    ("signed-gap-unverified", "original_check_binding"),
    ("planned-not-actual", "actual_result_binding"),
    ("bare-pass", "schema_keys"),
))
def test_coordinated_capability_controls(subject, graph, request, tmp_path, case, expected):
    def mutation(data):
        if case == "old-acceptance":
            data["negative_controls"] = 2
        elif case == "zero-samples":
            data.update(samples=[], counts=graph.counts(0))
        elif case == "dropped-failure-denominator":
            data["counts"].update(attempted=3, failed=1)
        elif case == "skip":
            data["counts"]["skipped"] = 1
        elif case == "zero-negative-controls":
            data["negative_controls"] = 0
        elif case == "bool-count":
            data["counts"]["attempted"] = True
        elif case == "duplicate-pair":
            data["samples"] = [data["samples"][0], data["samples"][0]]
        elif case == "signed-gap-unverified":
            check = graph.load(data["samples"][0]["check"])
            check["signed_gap"] = "UNVERIFIED"
            data["samples"][0]["check"] = graph.put(check)
        elif case == "planned-not-actual":
            result = graph.load(data["samples"][0]["result"])
            result["execution"] = "PLANNED"
            data["samples"][0]["result"] = graph.put(result)
    if case == "bare-pass":
        graph.manifest["capabilities"]["lp"] = graph.put({"verdict": "PASS"})
    else:
        graph.replace("capabilities", "lp", mutation, reseal=case != "old-acceptance")
    observe(subject, graph, request, tmp_path, expected)


@pytest.mark.parametrize("case,expected", (
    ("zero-sdk-calls", "invalid_integer"),
    ("license-unverified", "authorization_not_established"),
    ("only-cold", "unpaired_commercial_models"),
    ("different-model", "unpaired_commercial_models"),
    ("extra-authorization-material", "schema_keys"),
))
def test_commercial_discovery_or_flags_not_qualification(subject, graph, request, tmp_path, case, expected):
    def mutation(data):
        if case == "zero-sdk-calls":
            data["actual_native_calls"] = 0
        elif case == "license-unverified":
            data["authorization"]["status"] = "UNVERIFIED"
        elif case == "only-cold":
            data.update(actual_native_calls=1, completed=1, counts=graph.counts(1))
            data["samples"] = data["samples"][:1]
            data["openopt"] = {"counts": graph.counts(1), "samples": data["openopt"]["samples"][:1]}
        elif case == "different-model":
            model = graph.put({"different_unexecuted_model": "not paired"})
            data["samples"] = graph.pairs("gurobi", model=model)
        else:
            data["authorization"]["self_approved"] = "PASS"
    graph.replace("commercial", "gurobi", mutation)
    observe(subject, graph, request, tmp_path, expected)


@pytest.mark.parametrize("case,expected", (
    ("three-one-card-jobs", "gpu_count_or_timing"),
    ("duplicate-ids", "duplicate_gpu_devices"),
    ("kernel-only", "gpu_count_or_timing"),
    ("zero-events", "invalid_integer"),
    ("wrong-coordinator", "gpu_coordinator_binding"),
    ("wrong-trace-results", "gpu_trace_binding"),
    ("different-workload", "missing_or_different_scaling_problem"),
))
def test_multigpu_scaling_trace_controls(subject, graph, request, tmp_path, case, expected):
    def mutation(data):
        if case == "three-one-card-jobs":
            data.update(gpu_count=1, devices=data["devices"][:1])
        elif case == "duplicate-ids":
            data["devices"][1]["id"] = data["devices"][0]["id"]
        elif case == "kernel-only":
            data["timing_scope"] = "kernel_only"
        elif case == "zero-events":
            trace = graph.load(data["trace"])
            trace["kernel_events"][data["devices"][0]["id"]] = 0
            data["trace"] = graph.put(trace)
        elif case == "wrong-coordinator":
            record = graph.load(data["coordinator"])
            record["source_sha256"] = {"src/marker.py": "0" * 64}
            data["coordinator"] = graph.put(record)
        elif case == "wrong-trace-results":
            record = graph.load(data["trace"])
            record["result_sha256"] = ["0" * 64]
            data["trace"] = graph.put(record)
        else:
            model = graph.put({"invented_other_workload": "not strong scaling"})
            data.update(graph.gpu_data(2, pairs=graph.pairs("openopt", model=model)))
    graph.replace("gpu_scaling", "2", mutation)
    observe(subject, graph, request, tmp_path, expected)


@pytest.mark.parametrize("case,expected", (
    ("prepared-only", "not_published_current_source"),
    ("readback-diff", "online_readback_bytes"),
    ("old-remote-source", "remote_source_mismatch"),
    ("local-transport", "publication_transport_binding"),
    ("http-error", "publication_transport_binding"),
))
def test_local_publication_flags_and_changed_readback(subject, graph, request, tmp_path, case, expected):
    def mutation(data):
        if case == "prepared-only":
            data["status"] = "PREPARED"
        elif case == "readback-diff":
            data["readback"] = graph.put(b"Different local HTML, never online.\n")
        elif case == "old-remote-source":
            record = graph.load(data["remote_source"])
            record["commit"] = "0" * 40
            data["remote_source"] = graph.put(record)
        else:
            record = graph.load(data["transport"])
            record["method" if case == "local-transport" else "http_status"] = (
                "LOCAL_PREPARE" if case == "local-transport" else 404)
            data["transport"] = graph.put(record)
    graph.replace("publication", None, mutation)
    observe(subject, graph, request, tmp_path, expected)


@pytest.mark.parametrize("case,expected", (
    ("unknown-origin", "origin_rights_unverified"),
    ("missing-origin", "origin_source_membership"),
    ("project-license-covers-sdk", "license_scope_missing"),
    ("unknown-dependency-license", "dependency_rights_unknown"),
    ("drop-inventory", "sbom_inventory_mismatch"),
))
def test_project_0bsd_cannot_prove_other_rights(subject, graph, request, tmp_path, case, expected):
    def mutation(data):
        if case == "unknown-origin":
            data["origins"][0]["status"] = "UNKNOWN"
        elif case == "missing-origin":
            data["origins"][0]["paths"].remove("src/marker.py")
        elif case == "project-license-covers-sdk":
            data["dependency_scope"] = "project_0bsd_applies_to_everything"
        elif case == "unknown-dependency-license":
            record = graph.load(data["sbom"])
            record["dependencies"][0]["license"] = "UNKNOWN"
            data["sbom"] = graph.put(record)
        else:
            record = graph.load(data["inventory"])
            record["dependencies"] = []
            data["inventory"] = graph.put(record)
    graph.replace("licensing", None, mutation)
    observe(subject, graph, request, tmp_path, expected)


def test_wrong_caller_manifest_pin_safe_cli_terminal(subject, graph, request, tmp_path):
    observe(subject, graph, request, tmp_path, "byte_pin_mismatch", cli=True, pin="0" * 64)


def test_same_producer_auditor_cannot_claim_independence(subject, graph, request, tmp_path):
    desc = graph.manifest["capabilities"]["lp"]
    receipt = graph.load(desc)
    receipt["auditor"] = receipt["producer"]
    graph.manifest["capabilities"]["lp"] = graph.put(receipt)
    observe(subject, graph, request, tmp_path, "nonindependent_source")


def test_real_owner_source_without_evidence_is_incomplete(subject, graph, request, tmp_path):
    root, source, _ = owner_selection()
    selected = {name: pin for name, pin in source["sha256"].items()
                if name.startswith(("src/", "scripts/")) or name in ("LICENSE", "pyproject.toml")}
    assert len(selected) > 100 and TOOL in selected
    graph.manifest.update(source_commit=source["commit"], source_sha256=selected,
                          capabilities={}, commercial={}, gpu_scaling={}, publication=None, licensing=None)
    value = observe(subject, graph, request, tmp_path, "INCOMPLETE", root=root)
    assert "source" in value["report"]["checked"]
    assert all(row["section"] != "source" for row in value["report"]["incomplete"])


def test_historical_zero_sdk_calls_never_fill_current_missing_scope(subject, graph, request, tmp_path):
    history = {"schema_version": 1, "qualification": "UNVERIFIED",
               "solvers": {name: {"modes": {mode: {"samples": [], "available": False}
                                          for mode in ("cold", "prepared")}} for name in SOLVERS},
               "counts": {"attempted": 0, "solve_attempted": 0, "passed": 0, "failed": 0}}
    graph.manifest["historical_observations"] = [
        {"kind": "commercial_runner_discovery_v1", "receipt": graph.put(history)}]
    value = observe(subject, graph, request, tmp_path, "pinned_historical_zero_commercial_solves")
    assert value["report"]["historical_observations"][0]["actual_commercial_solves"] == 0


def environment(directory):
    value = os.environ.copy()
    value.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
                 PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", TMPDIR=str(directory), PYTHONPATH=str(ROOT))
    value.update({key: "1" for key in THREADS})
    return value


def collect(directory):
    command = [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q",
               "-o", f"cache_dir={directory / 'collect-cache'}", TEST]
    process = subprocess.run(command, cwd=ROOT, env=environment(directory),
                             capture_output=True, text=True, timeout=30, check=False)
    log = directory / f"collection-{time.time_ns()}.log"
    write(log, (process.stdout + process.stderr + f"\nreturncode={process.returncode}\n").encode())
    if process.returncode:
        raise ValueError(f"collection failed rc={process.returncode}")
    nodes = [line.strip() for line in process.stdout.splitlines() if line.startswith(TEST + "::")]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("empty/duplicate collection")
    return nodes


def replay(directory, name="replay.json", development=False):
    directory = external(directory)
    execution = json.loads((directory / "execution.json").read_bytes())
    owner_root, owner, pin = owner_selection()
    reviewer = snapshot(ROOT)
    assert execution["owner_before"] == execution["owner_after"] == owner
    assert execution["reviewer_before"] == execution["reviewer_after"] == reviewer
    assert reviewer["clean"] or development
    if not development:
        committed(ROOT, reviewer)
        changed = set(git(ROOT, "diff", "--name-only", BASE, "HEAD").decode().splitlines())
        assert changed == {TEST, DOC}, "reviewer write scope differs"
    selected = collect(directory)
    old = json.loads((directory / "collection.json").read_bytes())
    assert selected == old["nodeids"]
    tree = ET.parse(directory / "pytest.xml").getroot()
    cases = list(tree.iter("testcase"))
    actual = [case.get("classname").replace(".", "/") + ".py::" + case.get("name") for case in cases]
    assert len(set(actual)) == len(actual) and set(actual) == set(selected)
    counts = {"tests": len(cases), **{
        key: sum(case.find(tag) is not None for case in cases)
        for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))
    }}
    for suite in tree.iter("testsuite"):
        leaves = suite.findall("testcase")
        if not suite.findall("testsuite"):
            assert int(suite.get("tests")) == len(leaves)
            assert all(int(suite.get(key)) == sum(case.find(tag) is not None for case in leaves)
                       for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")))
    observations = []
    assessed_sources = {}
    for case in cases:
        nodeid = case.get("classname").replace(".", "/") + ".py::" + case.get("name")
        props = {p.get("name"): p.get("value") for p in case.findall("properties/property")}
        path = Path(props["independent_observation"]).resolve(strict=True)
        assert path.is_relative_to(directory) and digest(path.read_bytes()) == props["independent_observation_sha256"]
        value = json.loads(path.read_bytes())
        assert value["nodeid"] == nodeid, "JUnit/observation node identity differs"
        assert value["source_before"] == value["source_after"], "assessed source changed during test"
        assessed_root = Path(value["source_root"]).resolve(strict=True)
        if assessed_root not in assessed_sources:
            assessed_sources[assessed_root] = snapshot(assessed_root)
            committed(assessed_root, assessed_sources[assessed_root])
        assert assessed_sources[assessed_root] == value["source_before"], "assessed source bytes changed"
        for input_path, desc in value["files"].items():
            input_file = Path(input_path).resolve(strict=True)
            assert input_file.is_relative_to(directory), "case input escaped external run"
            payload = input_file.read_bytes()
            assert desc["path"] == input_path and desc["bytes"] == len(payload) and digest(payload) == desc["sha256"]
        if value["report"] is not None:
            never_complete(value["report"])
        assert all(value[key] == 0 for key in (
            "actual_solver_calls", "actual_gpu_calls", "actual_network_calls", "input_execution_calls"))
        if value["cli"] is not None:
            record = value["cli"]
            assert digest(Path(record["report"]).read_bytes()) == record["report_sha256"]
            assert json.loads(Path(record["report"]).read_bytes()) == value["report"]
            assert record["returncode"] in (1, 2)
            assert digest(Path(record["log"]).read_bytes()) == record["log_sha256"]
        observations.append({"path": str(path), "sha256": digest(path.read_bytes()),
                             "expected": value["expected"], "outcome": value["outcome"]})
    assert execution["returncode"] == (1 if counts["failures"] or counts["errors"] else 0)
    assert not counts["errors"] and not counts["skipped"]
    report = {
        "schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
        "evidence_verdict": "PASS", "review_verdict": "FAIL" if counts["failures"] else "PASS",
        "bounded_precheck_signoff": not bool(counts["failures"]), "release_complete": False,
        "release_qualification": "UNVERIFIED", "tests": counts,
        "owner_root": str(owner_root), "owner": owner, "tool_sha256": pin, "reviewer": reviewer,
        "membership": {"missing": [], "extra": [], "duplicates": []},
        "collection_nodeids": selected, "junit_nodeids": actual,
        "collection_sha256": digest((directory / "collection.json").read_bytes()),
        "junit_sha256": digest((directory / "pytest.xml").read_bytes()),
        "execution_sha256": digest((directory / "execution.json").read_bytes()),
        "wall_seconds": execution["wall_seconds"], "actual_solver_calls": 0,
        "actual_gpu_calls": 0, "actual_network_calls": 0, "observations": observations,
        "observation_outcomes": dict(Counter(row["outcome"] for row in observations)),
        "source_consistency": {"verdict": "PASS", "owner": True, "reviewer": True,
                               "assessed_roots": len(assessed_sources), "committed_blob_bytes": True},
        "scope": [TEST, DOC], "development": development,
    }
    write(directory / name, report)
    print(f"RELEASE_INDEPENDENT evidence=PASS review={report['review_verdict']} "
          f"tests={counts} release_complete=false receipt={directory / name}", flush=True)
    return 1 if counts["failures"] else 0


def run(artifact_root, development=False):
    root = external(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="review-", dir=root))
    owner_root, owner, _ = owner_selection()
    reviewer = snapshot(ROOT)
    assert reviewer["clean"] or development
    nodes = collect(directory)
    write(directory / "collection.json", {"nodeids": nodes, "reviewer": reviewer, "owner": owner})
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-ra",
               "-o", f"cache_dir={directory / 'pytest-cache'}", "--basetemp", str(directory / "pytest-tmp"),
               f"--junitxml={directory / 'pytest.xml'}", TEST]
    started = time.monotonic()
    with (directory / "pytest.log").open("x") as stream:
        process = subprocess.run(command, cwd=ROOT, env=environment(directory), stdout=stream,
                                 stderr=subprocess.STDOUT, timeout=165, check=False)
        stream.write(f"\nreturncode={process.returncode}\n")
    write(directory / "execution.json", {
        "command": command, "returncode": process.returncode, "wall_seconds": time.monotonic() - started,
        "owner_before": owner, "owner_after": snapshot(owner_root),
        "reviewer_before": reviewer, "reviewer_after": snapshot(ROOT),
    })
    return replay(directory, "receipt.json", development)


def scan(artifact_root):
    root = external(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "completion_safe_scanner", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(
        secret_hex=None, secrets_file=None, no_needle=False,
        needle_from=str(Path.home() / ".config/dsh-n600-token")))
    if not needles:
        raise ValueError("authorized credential scan needle unavailable")
    paths = [str(ROOT / TEST), str(ROOT / DOC)]
    literal, _ = scanner.scan_literal(paths, needles)
    pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                         r"(?i:password|token|secret)\s*[\"\x27]?\s*[:=]\s*[\"\x27][^\"\x27\s]{6,}[\"\x27]")
    shape = sum(bool(pattern.search(Path(p).read_text())) for p in paths)
    write(root / "source-security-scan.json", {
        "verdict": "PASS" if literal == shape == 0 else "FAIL",
        "literal_hits": literal, "shape_hits": shape, "files": {p: digest(Path(p).read_bytes()) for p in paths},
    })
    print(f"RELEASE_REVIEW_SCAN literal={literal} shape={shape}", flush=True)
    return 0 if literal == shape == 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--replay", type=Path)
    modes.add_argument("--scan-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--owner-root")
    parser.add_argument("--owner-commit")
    parser.add_argument("--tool-sha256")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--receipt-name", default="replay.json")
    args = parser.parse_args(argv)
    for key, value in zip(ENV, (args.owner_root, args.owner_commit, args.tool_sha256)):
        if value is not None:
            os.environ[key] = value
    try:
        if args.scan_only:
            return scan(args.artifact_dir)
        if args.run:
            return run(args.artifact_dir, args.development)
        if Path(args.receipt_name).name != args.receipt_name or not args.receipt_name.endswith(".json"):
            raise ValueError("simple new JSON receipt name required")
        return replay(args.replay, args.receipt_name, args.development)
    except Exception as error:
        root = external(args.artifact_dir or EVIDENCE_ROOT / "command-failures")
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="failed-", dir=root))
        write(directory / "receipt.json", {
            "schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
            "evidence_verdict": "FAIL", "review_verdict": "UNVERIFIED", "release_complete": False,
            "returncode": 2, "first_cause": {"type": type(error).__name__, "message": str(error)},
            "requested_replay": str(args.replay) if args.replay else None,
            "requested_artifact_dir": str(args.artifact_dir) if args.artifact_dir else None,
            "actual_solver_calls": 0,
        })
        print(f"RELEASE_INDEPENDENT evidence=FAIL cause={type(error).__name__} "
              f"receipt={directory / 'receipt.json'}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
