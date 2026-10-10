"""Independent CPU-only review of the frozen caller-pinned bounded reader.

Construct our own tiny polynomial/tree/archive, not an owner test fixture.
Never issue authority for the historical run or execute archived Python.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELF = "tests/test_bounded_cuda_v2_independent.py"
SUBJECT = "50e9bf111960a7e09f3009c4f6202831e7975b5e"
OWNER = {
    "tests/test_branch_bound_cuda_independent.py":
        "5fe94b0b40134cf72da7defe35a839855da023285b6836ff90b79f9125459c79",
    "tests/test_bounded_cuda_reader_contract.py":
        "21b8ceb8fa3aacaa56a9490f30bed0bd439ac354715dbf3396af698c122fa618",
    "docs/research/bounded-miqp-cuda-review.md":
        "753fe6bde93ec1e88f37cdc764385851c12d2b50a14a3b2aa620b24e5d748138",
    "tests/test_branch_bound_independent.py":
        "592dc7ed9d467da240d2f098e45e7fdac6a2f8e134ae60cdb8023910bf5a8e0d",
}
OLD_ROOT = Path("/data/dev/openopt-wt-release4-20261010-cuda-replay-review")
OLD_COMMIT = "b55aa309823aa9b9d911da9b9e2e37074f740eb2"
OLD_TEST = "tests/test_bounded_cuda_reader_independent.py"
OLD_PIN = "213e01687a5386073c76fd21fb0a671689afb25626d7888439754fc532b1a63f"
RELEASE = Path("/hf3fs-jd/data/tmp/openopt-release-20261010")
OLD_RECEIPT = RELEASE / "bounded-cuda-reader-independent/clean-final-v1/tests-jaqrw8pq/independent-review.json"
OLD_RECEIPT_PIN = "b499296c59e02227edbfb96d7a8ee701e6aae4a07cc0b0fd19853ba28655b84e"
SOURCE_COMMIT = "af1ef6d9993357d3647170246fe3d8192392a633"
SOURCE_TREE = "0dee304d077d76e0cc3b65cb037b5ee993365681"
SOURCES = {
    "src/openopt/branch_bound.py": "424a6eecfeafee6d114d3f6a5c37189b8af6b774d9c590360f39f440b4040ad6",
    "src/openopt/qp.py": "955d741317e5de1c5743f6e6bc65406ba14560f9a8b4c989afe6d11a1fc58e40",
    "tests/test_branch_bound.py": "47af150f3c3bdd44cdc2d59b3201fc311ba0a7876b0c6b8f1d8c6f057ed003cf",
}
TARGET = "tests/test_branch_bound.py::test_cuda_frozen_bound_tree"
COUNTS = dict(tests=1, failures=0, errors=0, skipped=0)
JOB, NODE, POD = "independent-synthetic-job", "independent-node", "independent-pod"
SIDE = "case-0/cuda-check-independent"
ROLES = {
    "run": "run.json", "collection": "collection.json", "job": "job-0.json",
    "status": JOB + "-status.json", "result": "case-0/result.json",
    "side_result": SIDE + "/result.json", "junit": SIDE + "/pytest.xml",
    "proof": SIDE + "/pytest-tmp/test_cuda_frozen_bound_tree0/bounded-miqp-cuda.json",
    "trace": SIDE + "/pytest-tmp/test_cuda_frozen_bound_tree0/cuda-trace.json",
}
MUTATIONS = (
    "model-q", "inflated-bound", "deleted-child", "branch-gap", "fractional-incumbent",
    "wrong-gap", "bool-schema", "bool-node-id", "bool-parent", "bool-visit", "bool-count",
    "bool-config", "bool-enclosure", "core-source", "result-count", "result-bool-count",
    "result-bool-rc", "result-node", "result-source-test", "collection-job",
    "collection-source", "collection-result-pin", "run-job", "run-tree", "job-name",
    "job-kind", "junit-count", "junit-missing", "junit-duplicate", "receipt-pin",
    "trace-no-kernel", "trace-bool-duration", "trace-negative-duration", "trace-device",
    "trace-unrelated-kernel", "trace-changed-duration", "status-owner", "status-node",
    "status-pod", "status-exit", "status-restart", "status-failed", "junit-suite-name",
    "junit-host", "junit-test", "junit-source-class", "run-bool-size",
    "source-map-subset-mismatch",
)
ORACLE_MUTATIONS = (
    "model-q", "inflated-bound", "deleted-child", "branch-gap", "fractional-incumbent",
    "wrong-gap", "bool-schema", "bool-node-id", "bool-parent", "bool-visit", "bool-enclosure",
)
LEGAL_RESELECTION = ("trace-unrelated-kernel", "trace-changed-duration")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else encoded(value)
    path.write_bytes(raw)
    return sha(raw)


def load(path):
    return json.loads(path.read_bytes())


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, timeout=20)
    assert result.returncode == 0, f"Git query rc={result.returncode}"
    return result.stdout


def pinned_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sources():
    for name, pin in OWNER.items():
        assert sha((ROOT / name).read_bytes()) == sha(git("show", SUBJECT + ":" + name)) == pin
    assert git("rev-parse", SOURCE_COMMIT + "^{tree}").decode().strip() == SOURCE_TREE
    for name, pin in SOURCES.items():
        assert sha((ROOT / name).read_bytes()) == sha(git("show", SOURCE_COMMIT + ":" + name)) == pin
    raw = (OLD_ROOT / OLD_TEST).read_bytes()
    assert sha(raw) == OLD_PIN and git("show", OLD_COMMIT + ":" + OLD_TEST) == raw
    assert sha(OLD_RECEIPT.read_bytes()) == OLD_RECEIPT_PIN


@pytest.fixture(scope="module")
def subject():
    verify_sources()
    sys.path.insert(0, str(ROOT))
    module = pinned_module("_independent_bounded_reader_v2", ROOT / "tests/test_branch_bound_cuda_independent.py")
    assert module.TARGET == TARGET and module.SELECTED_SOURCES == SOURCES
    yield module
    verify_sources()


@pytest.fixture(scope="module")
def original_oracle():
    verify_sources()
    # Only the original independent scalar oracle; not its old reader/fixtures.
    module = pinned_module("_original_b55aa30_polynomial_oracle", OLD_ROOT / OLD_TEST)
    return module.scalar_tree_oracle


def analytic_descriptor():
    """Our represented polynomial is exactly x*x-x on integer [0,2]."""
    definitions = {
        "P": ([1, 1], [2.]), "q": ([1], [-1.]), "A": ([0, 1], []),
        "l": ([0], []), "u": ([0], []), "lower": ([1], [0.]),
        "upper": ([1], [2.]), "integrality": ([1], [1]),
    }
    arrays = {}
    for name, (shape, values) in definitions.items():
        payload = b"".join(struct.pack("<q" if name == "integrality" else "<d", x) for x in values)
        arrays[name] = dict(dtype="torch.int64" if name == "integrality" else "torch.float64",
                            shape=shape, bytes=len(payload), sha256=sha(payload),
                            values=[float(x).hex() for x in values])
    return dict(arrays=arrays, sha256=sha(json.dumps(arrays, sort_keys=True, separators=(",", ":")).encode()))


def analytic_proof():
    """Hand-derived tangents -z*z+(2*z-1)*x; no factory or solve."""
    witness = lambda z: dict(kind="qp_tangent", z=[z], y=[])
    proposal = lambda z: dict(status="solved", iterations=25,
                              actual_device="cuda:0", dtype="torch.float64", z=[z], y=[])
    root = dict(id=0, parent=None, box=[["0/1", "2/1"]], bound="-1/4",
                witnesses=[witness(.5)], disposition="branch",
                split=dict(index=0, cut=0, children=[1, 2]), proposal=proposal(.5),
                candidates=[dict(kind="independent_endpoint", x=[0.], objective="0/1")], visited=True)
    left = dict(id=1, parent=0, box=[["0/1", "0/1"]], bound="0/1",
                witnesses=[dict(kind="zero_tangent", z=[0.], y=[])],
                disposition="pruned", split=None, proposal=None, candidates=[], visited=True)
    right = dict(id=2, parent=0, box=[["1/1", "2/1"]], bound="0/1",
                 witnesses=[witness(1.)], disposition="pruned", split=None,
                 proposal=proposal(1.), candidates=[], visited=True)
    return dict(schema_version=1, model=analytic_descriptor(), source_sha256={
        name: SOURCES[name] for name in ("src/openopt/branch_bound.py", "src/openopt/qp.py")},
        config=dict(enabled=True, device="cuda:0", dtype="torch.float64", max_nodes=128,
                    qp_max_iterations=2000, qp_check_interval=25, abs_gap=0.,
                    rel_gap=0., wall_seconds=None), psd_dd_margins=["2/1"],
        nodes=[root, left, right], incumbent=[0.],
        summary=dict(status="solved_bounded", objective="0/1", lower_bound="0/1",
                     gap_abs="0/1", gap_rel="0/1",
                     enclosures={key: [0., 0.] for key in ("objective", "lower_bound", "gap_abs", "gap_rel")}),
        stop_reason="tree_exhausted", visited_nodes=3, actual_qp_calls=2,
        qp_iteration_limit_count=0,
        cost_seconds=dict(total_before_replay=.02, qp_proposal=.01,
                          host_coordination_and_scalar_proof=.01),
        proof_arithmetic="host_bounded_fraction_8192_v1",
        execution_verdict="UNVERIFIED", performance_verdict="UNVERIFIED")


class Archive:
    """Test caller creates fresh synthetic authority; never backfills a real run."""

    def __init__(self, parent):
        self.root = (parent / "synthetic-archive").resolve()
        self.root.mkdir()
        self.manifest = parent / "independent-caller-manifest.json"
        self.put("proof", analytic_proof())
        self.put("trace", dict(deviceProperties=[dict(name="NVIDIA RTX PRO 4500 Blackwell")],
                              traceEvents=[dict(cat="kernel", ph="X", name="TEST_ONLY_KERNEL", dur=1.)]))
        for name, pin in SOURCES.items():
            raw = (ROOT / name).read_bytes()
            assert sha(raw) == pin
            save(self.root / "source" / name, raw)
        identity = dict(commit=SOURCE_COMMIT, tree=SOURCE_TREE, clean=True, sha256={
            **SOURCES, "README.md": sha(b"TEST-ONLY SNAPSHOT METADATA")})
        item = dict(job=JOB, size=1, output=str(self.root / "case-0"), manifest=str(self.path("job")))
        record = dict(namespace="independent-test", stage="release-regression",
                      source_sha=SOURCE_COMMIT, source_identity=identity,
                      source_hashes=deepcopy(SOURCES), jobs=[item])
        self.put("run", record)
        self.result(dict(verdict="PASS", returncode=0, targets=[TARGET], tests=COUNTS,
                         hardware=dict(node=NODE, gpu="NVIDIA RTX PRO 4500 Blackwell"),
                         source_sha256=SOURCES))
        collection = deepcopy(record)
        collection.update(states=["PASS"])
        collection["jobs"][0].update(result_path=str(self.path("result")), result_sha256=sha(self.path("result").read_bytes()))
        self.put("collection", collection)
        job = dict(apiVersion="batch/v1", kind="Job", metadata=dict(name=JOB, namespace="independent-test"))
        self.put("job", job)
        observed = deepcopy(job)
        observed["metadata"]["uid"] = "independent-job-uid"
        observed["status"] = dict(succeeded=1, conditions=[dict(type="Complete", status="True")])
        pod = dict(apiVersion="v1", kind="Pod", metadata=dict(name=POD, namespace="independent-test",
            labels={"job-name": JOB}, ownerReferences=[dict(controller=True, apiVersion="batch/v1",
                kind="Job", name=JOB, uid="independent-job-uid")]), spec=dict(nodeName=NODE),
            status=dict(phase="Succeeded", containerStatuses=[dict(restartCount=0,
                state=dict(terminated=dict(exitCode=0)))]))
        self.put("status", dict(job=observed, pods=dict(items=[pod])))
        xml = ET.Element("testsuites", name="pytest tests")
        suite = ET.SubElement(xml, "testsuite", name="pytest", hostname=POD,
                              **{key: str(value) for key, value in COUNTS.items()})
        case = ET.SubElement(suite, "testcase", classname="tests.test_branch_bound", name="test_cuda_frozen_bound_tree")
        props = ET.SubElement(case, "properties")
        for name, value in dict(receipt=str(self.path("proof")), trace=str(self.path("trace")),
                                receipt_sha256=sha(self.path("proof").read_bytes())).items():
            ET.SubElement(props, "property", name=name, value=value)
        self.put("junit", ET.tostring(xml))
        self.pin = self.freeze()

    def path(self, role):
        return self.root / ROLES[role]

    def put(self, role, value):
        return save(self.path(role), value)

    def get(self, role):
        return load(self.path(role))

    def result(self, value):
        self.put("result", value)
        self.put("side_result", value)

    def descriptor(self, name):
        raw = (self.root / name).read_bytes()
        return dict(path=name, sha256=sha(raw), bytes=len(raw))

    def selection(self):
        return dict(schema_version=2, authority="caller-selected-archive-only", run_root=str(self.root),
                    target=TARGET, source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE,
                    job_name=JOB, node_name=NODE, pod_name=POD,
                    files={role: self.descriptor(name) for role, name in ROLES.items()},
                    selected_source={name: self.descriptor("source/" + name) for name in SOURCES})

    def freeze(self):
        return save(self.manifest, self.selection())

    def call(self, subject, pin=None):
        return subject.audit_v2(self.root, self.manifest, self.pin if pin is None else pin)

    def coordinate(self):
        tree = ET.fromstring(self.path("junit").read_bytes())
        for prop in tree.findall(".//property[@name='receipt_sha256']"):
            prop.set("value", sha(self.path("proof").read_bytes()))
        self.put("junit", ET.tostring(tree))
        value = self.get("collection")
        value["jobs"][0]["result_sha256"] = sha(self.path("result").read_bytes())
        self.put("collection", value)


@pytest.fixture
def archive(tmp_path):
    return Archive(tmp_path)


def mutate(archive, kind):
    """Independently authored attacks with coordinated attacker-owned digests."""
    proof = archive.get("proof")
    marker = archive.root.parent / "ARCHIVED_CODE_EXECUTED"
    if kind in ("model-q", "model-upper"):
        array = proof["model"]["arrays"]["q" if kind == "model-q" else "upper"]
        array.update(values=[float(1.).hex()], sha256=sha(struct.pack("<d", 1.)))
        proof["model"]["sha256"] = sha(json.dumps(proof["model"]["arrays"], sort_keys=True, separators=(",", ":")).encode())
    elif kind in ("inflated-bound", "deleted-child", "branch-gap", "fractional-incumbent", "wrong-gap"):
        if kind == "inflated-bound":
            proof["nodes"][2]["bound"] = "1/1"
        elif kind == "deleted-child":
            proof["nodes"][0]["split"]["children"] = [1, 1]
        elif kind == "branch-gap":
            proof["nodes"][2]["box"][0][0] = "2/1"
        elif kind == "fractional-incumbent":
            proof["incumbent"] = [.5]
        else:
            proof["summary"]["gap_abs"] = "-1/1"
    elif kind.startswith("bool-"):
        if kind == "bool-schema":
            proof["schema_version"] = True
        elif kind == "bool-node-id":
            proof["nodes"][1]["id"] = True
        elif kind == "bool-parent":
            proof["nodes"][1]["parent"] = False
        elif kind == "bool-visit":
            proof["nodes"][0]["visited"] = 1
        elif kind == "bool-count":
            proof["qp_iteration_limit_count"] = False
        elif kind == "bool-config":
            proof["config"]["enabled"] = 1
        else:
            proof["summary"]["enclosures"]["gap_abs"] = [False, False]
    elif kind == "core-source":
        source = archive.root / "source/src/openopt/qp.py"
        raw = source.read_bytes() + f"\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('BAD')\n".encode()
        pin = save(source, raw)
        proof["source_sha256"]["src/openopt/qp.py"] = pin
        for role in ("run", "collection"):
            value = archive.get(role)
            value["source_hashes"]["src/openopt/qp.py"] = pin
            value["source_identity"]["sha256"]["src/openopt/qp.py"] = pin
            archive.put(role, value)
        result = archive.get("result")
        result["source_sha256"]["src/openopt/qp.py"] = pin
        archive.result(result)
    elif kind.startswith("result-"):
        value = archive.get("result")
        if kind in ("result-count", "result-bool-count"):
            value["tests"]["tests"] = 2 if kind == "result-count" else True
        elif kind == "result-bool-rc":
            value["returncode"] = False
        elif kind == "result-node":
            value["hardware"]["node"] = "foreign-node"
        else:
            value["source_sha256"]["tests/test_branch_bound.py"] = "0" * 64
        archive.result(value)
    elif kind.startswith("collection-"):
        value = archive.get("collection")
        if kind == "collection-job":
            value["jobs"][0]["job"] = "foreign-job"
        elif kind == "collection-source":
            value["source_sha"] = "0" * 40
        else:
            value["jobs"][0]["result_sha256"] = "0" * 64
        archive.put("collection", value)
    elif kind.startswith("run-"):
        value = archive.get("run")
        if kind == "run-job":
            value["jobs"][0]["job"] = "foreign-job"
        elif kind == "run-bool-size":
            value["jobs"][0]["size"] = True
        else:
            value["source_identity"]["tree"] = "0" * 40
        archive.put("run", value)
    elif kind == "source-map-subset-mismatch":
        for role in ("run", "collection"):
            value = archive.get(role)
            value["source_identity"]["sha256"]["tests/test_branch_bound.py"] = "0" * 64
            archive.put(role, value)
    elif kind.startswith("job-"):
        value = archive.get("job")
        if kind == "job-name":
            value["metadata"]["name"] = "foreign-job"
        else:
            value["kind"] = "Deployment"
        archive.put("job", value)
    elif kind.startswith("status-"):
        value = archive.get("status")
        pod = value["pods"]["items"][0]
        container = pod["status"]["containerStatuses"][0]
        if kind == "status-owner":
            pod["metadata"]["ownerReferences"][0]["uid"] = "foreign-controller"
        elif kind == "status-node":
            pod["spec"]["nodeName"] = "foreign-node"
        elif kind == "status-pod":
            pod["metadata"]["name"] = "foreign-pod"
        elif kind == "status-exit":
            container["state"]["terminated"]["exitCode"] = False
        elif kind == "status-restart":
            container["restartCount"] = True
        else:
            value["job"]["status"]["conditions"].append(dict(type="Failed", status="True"))
        archive.put("status", value)
    elif kind.startswith("junit-"):
        tree = ET.fromstring(archive.path("junit").read_bytes())
        suite, case = tree[0], tree[0][0]
        if kind == "junit-count":
            suite.set("tests", "2")
        elif kind == "junit-missing":
            suite.remove(case)
        elif kind == "junit-duplicate":
            suite.append(deepcopy(case))
        elif kind == "junit-suite-name":
            suite.set("name", "foreign-suite")
        elif kind == "junit-host":
            suite.set("hostname", "foreign-pod")
        elif kind == "junit-test":
            case.set("name", "foreign-test")
        else:
            case.set("classname", "tests.foreign_source")
        archive.put("junit", ET.tostring(tree))
    elif kind.startswith("trace-"):
        value = archive.get("trace")
        if kind == "trace-no-kernel":
            value["traceEvents"] = [dict(cat="cpu_op", ph="X")]
        elif kind == "trace-device":
            value["deviceProperties"][0]["name"] = "foreign-device"
        elif kind == "trace-unrelated-kernel":
            value["traceEvents"][0].update(name="UNRELATED_TEST_KERNEL", pid=99999)
        else:
            value["traceEvents"][0]["dur"] = True if kind == "trace-bool-duration" else (
                -1. if kind == "trace-negative-duration" else 2.)
        archive.put("trace", value)
    else:
        assert kind == "receipt-pin"
    archive.put("proof", proof)
    archive.coordinate()
    if kind == "collection-result-pin":
        value = archive.get("collection")
        value["jobs"][0]["result_sha256"] = "0" * 64
        archive.put("collection", value)
    if kind == "receipt-pin":
        tree = ET.fromstring(archive.path("junit").read_bytes())
        tree.find(".//property[@name='receipt_sha256']").set("value", "0" * 64)
        archive.put("junit", ET.tostring(tree))
    return marker


def observe(archive, request, label, call, rejected):
    try:
        outcome = dict(outcome="RETURNED", result=call())
    except (ValueError, KeyError, TypeError, ArithmeticError, OSError, AssertionError) as error:
        outcome = dict(outcome="REJECTED", reason=f"{type(error).__name__}: {error}")
    selection = archive.selection()
    path = archive.root.parent / "independent-observation.json"
    save(path, dict(case=label, expected_rejected=rejected, subject_commit=SUBJECT,
                   original_caller_pin=archive.pin, current_manifest_sha256=sha(archive.manifest.read_bytes()),
                   run_root=str(archive.root), manifest_path=str(archive.manifest),
                   files=selection["files"], selected_source=selection["selected_source"], **outcome))
    request.node.user_properties.extend([("bounded_v2_observation", str(path)),
                                         ("bounded_v2_observation_sha256", sha(path.read_bytes()))])
    assert (outcome["outcome"] == "REJECTED") is rejected, "FALSE ACCEPT/REJECT: " + label
    return outcome


def test_frozen_owner_original56_red_and_independent_domain():
    verify_sources()
    old = load(OLD_RECEIPT)
    assert old["verdict"] == "FAIL" and old["tests"] == dict(tests=56, failures=13, errors=0, skipped=0)
    assert len(MUTATIONS) == len(set(MUTATIONS)) == 48
    old_module = pinned_module("_old_domain_only", OLD_ROOT / OLD_TEST)
    assert set(old_module.MUTATIONS) <= set(MUTATIONS) | {"model-upper"}


def test_v2_positive_reaches_original_polynomial_and_never_solves(subject, original_oracle, archive, request, monkeypatch):
    assert original_oracle(archive.get("proof"))["numerical"] == "PASS"
    assert min(Fraction(x * x - x) for x in (0, 1, 2)) == 0
    calls = []
    original = subject.original_replay

    def observed_replay(proof):
        calls.append(proof)
        return original(proof)

    monkeypatch.setattr(subject, "original_replay", observed_replay)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("reader launched external process"))
    value = observe(archive, request, "v2-independent-positive", lambda: archive.call(subject), False)["result"]
    assert len(calls) == 1 and calls[0] == archive.get("proof")
    assert value["verdict"] == "ARCHIVE_NUMERICAL_PASS"
    assert value["numerical_verdict"] == value["archive_binding_verdict"] == "PASS"
    assert value["execution_verdict"] == value["performance_verdict"] == value["commercial_verdict"] == "UNVERIFIED"
    assert value["original_model"]["coverage"] == "PASS" and value["manifest_sha256"] == archive.pin


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_original_caller_pin_rejects_all48_coordinated_rehashes(subject, archive, request, mutation):
    assert archive.call(subject)["numerical_verdict"] == "PASS"
    before = archive.manifest.read_bytes()
    marker = mutate(archive, mutation)
    assert archive.manifest.read_bytes() == before and sha(before) == archive.pin
    changed = archive.selection()
    assert changed != load(archive.manifest), "mutation must reach actual selected bytes"
    result = observe(archive, request, "original-pin:" + mutation, lambda: archive.call(subject), True)
    assert "caller pinned evidence bytes differ:" in result["reason"]
    assert not marker.exists()


@pytest.mark.parametrize("mutation", tuple(x for x in MUTATIONS if x not in LEGAL_RESELECTION))
def test_explicit_repinned_invalid_bytes_reach_semantic_gates(subject, archive, request, mutation):
    assert archive.call(subject)["numerical_verdict"] == "PASS"
    marker = mutate(archive, mutation)
    new_pin = archive.freeze()
    assert new_pin != archive.pin
    result = observe(archive, request, "semantic:" + mutation, lambda: archive.call(subject, new_pin), True)
    assert "caller pinned evidence bytes differ" not in result["reason"], result
    if mutation == "bool-enclosure":
        assert "finite nonbool numerical enclosure required" in result["reason"]
    assert not marker.exists()


@pytest.mark.parametrize("mode", ("original-pin", "semantic", "oracle"))
def test_original_model_upper_control_is_not_dropped(subject, original_oracle, archive, request, mode):
    assert archive.call(subject)["numerical_verdict"] == "PASS"
    assert original_oracle(archive.get("proof"))["numerical"] == "PASS"
    mutate(archive, "model-upper")
    if mode == "oracle":
        call = lambda: original_oracle(archive.get("proof"))
    else:
        pin = archive.freeze() if mode == "semantic" else archive.pin
        call = lambda: archive.call(subject, pin)
    result = observe(archive, request, "legacy-model-upper:" + mode, call, True)
    if mode != "oracle":
        assert ("caller pinned evidence bytes differ:" in result["reason"]) is (mode == "original-pin")


@pytest.mark.parametrize("mutation", LEGAL_RESELECTION)
def test_legal_reselected_trace_is_not_origin_attestation(subject, archive, request, mutation):
    mutate(archive, mutation)
    result = observe(archive, request, "nonattestation:" + mutation,
                     lambda: archive.call(subject, archive.freeze()), False)["result"]
    assert result["execution_verdict"] == result["performance_verdict"] == "UNVERIFIED"
    assert result["trace"]["kernels"] == 1


@pytest.mark.parametrize("mutation", ORACLE_MUTATIONS)
def test_unchanged_original_fraction_oracle_sees_mathematical_red(original_oracle, archive, mutation):
    assert original_oracle(archive.get("proof"))["numerical"] == "PASS"
    mutate(archive, mutation)
    with pytest.raises((ValueError, KeyError, TypeError, ArithmeticError)):
        original_oracle(archive.get("proof"))


@pytest.mark.parametrize("mutation", ("version", "bool-version", "float-version", "missing-trace",
    "missing-source", "extra-role", "duplicate-path", "bool-bytes", "wrong-root",
    "execution-authority", "traversal", "absolute", "zero-bytes", "uppercase-pin"))
def test_reselected_manifest_schema_is_strict(subject, archive, request, mutation):
    value = load(archive.manifest)
    if mutation.endswith("version"):
        value["schema_version"] = {"version": 1, "bool-version": True, "float-version": 2.}[mutation]
    elif mutation == "missing-trace":
        del value["files"]["trace"]
    elif mutation == "missing-source":
        del value["selected_source"]["tests/test_branch_bound.py"]
    elif mutation == "extra-role":
        value["files"]["extra"] = value["files"]["trace"]
    elif mutation == "duplicate-path":
        value["files"]["trace"] = value["files"]["proof"]
    elif mutation in ("bool-bytes", "zero-bytes"):
        value["files"]["trace"]["bytes"] = True if mutation == "bool-bytes" else 0
    elif mutation == "wrong-root":
        value["run_root"] = str(archive.root.parent)
    elif mutation == "execution-authority":
        value["authority"] = "EXECUTION_ATTESTED"
    elif mutation == "traversal":
        value["files"]["trace"]["path"] = "../foreign-trace.json"
    elif mutation == "absolute":
        value["files"]["trace"]["path"] = str(archive.path("trace"))
    else:
        value["files"]["trace"]["sha256"] = "A" * 64
    pin = save(archive.manifest, value)
    observe(archive, request, "schema:" + mutation, lambda: archive.call(subject, pin), True)


@pytest.mark.parametrize("mutation", ("duplicate-json", "nan-json", "empty-json", "symlink-file",
    "symlink-directory", "symlink-manifest", "fifo", "oversized-file", "changed-manifest", "changed-after-replay"))
def test_safe_reads_and_live_input_drift_fail_loud(subject, archive, request, mutation, monkeypatch):
    pin = archive.pin
    trace = archive.path("trace")
    if mutation in ("duplicate-json", "nan-json", "empty-json"):
        raw = {"duplicate-json": b'{"traceEvents":[],"traceEvents":[]}',
               "nan-json": b'{"traceEvents":NaN}', "empty-json": b""}[mutation]
        archive.put("trace", raw)
        pin = archive.freeze()
    elif mutation == "symlink-manifest":
        alias = archive.manifest.with_name("manifest-alias.json")
        alias.symlink_to(archive.manifest)
        archive.manifest = alias
    elif mutation == "symlink-file":
        target = archive.root.parent / "foreign-trace.json"
        save(target, trace.read_bytes())
        trace.unlink()
        trace.symlink_to(target)
    elif mutation == "symlink-directory":
        target = archive.root.parent / "sidecar-real"
        trace.parent.rename(target)
        trace.parent.symlink_to(target, target_is_directory=True)
    elif mutation == "fifo":
        # 3FS returns ENOSYS for mkfifo. Inject a real FIFO descriptor only at
        # the final open; actual fstat exercises the nonregular-file guard.
        read_fd, write_fd = os.pipe()
        original_open = subject.os.open

        def fifo_open(path, flags, *args, **kwargs):
            if path == trace.name and kwargs.get("dir_fd") is not None:
                return os.dup(read_fd)
            return original_open(path, flags, *args, **kwargs)

        with monkeypatch.context() as injected:
            injected.setattr(subject.os, "open", fifo_open)
            try:
                result = observe(archive, request, "safe-read:fifo",
                                 lambda: archive.call(subject, pin), True)
                assert "empty/nonregular/oversized evidence" in result["reason"]
            finally:
                os.close(read_fd)
                os.close(write_fd)
        return
    elif mutation == "oversized-file":
        with trace.open("r+b") as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
    elif mutation == "changed-manifest":
        selected = load(archive.manifest)
        selected["job_name"] = "different-selected-job"
        save(archive.manifest, selected)
    else:
        original = subject.original_replay

        def drift_after_read(proof):
            result = original(proof)
            value = archive.get("trace")
            value["traceEvents"][0]["dur"] = 2.
            archive.put("trace", value)
            return result

        monkeypatch.setattr(subject, "original_replay", drift_after_read)
    # Sparse oversized files must reject without reading their entire payload.
    if mutation == "oversized-file":
        selection = load(archive.manifest)
        path = archive.root.parent / "read-observation.json"
        try:
            archive.call(subject, pin)
        except (ValueError, OSError) as error:
            save(path, dict(outcome="REJECTED", reason=str(error), mutation=mutation, manifest=selection))
        else:
            pytest.fail("nonregular/oversized file accepted")
        request.node.user_properties.extend([("bounded_v2_read_observation", str(path)),
                                             ("bounded_v2_read_observation_sha256", sha(path.read_bytes()))])
    else:
        observe(archive, request, "safe-read:" + mutation, lambda: archive.call(subject, pin), True)


def cpu_env(output):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTEST_ADDOPTS="", PYTEST_PLUGINS="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", TMPDIR=str(output),
               PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))))
    for key in tuple(env):
        if key.startswith(("OPENOPT_SHARD_", "OPENOPT_BOUNDED_READER_")):
            env.pop(key)
    return env


@pytest.mark.parametrize("mode", ("positive", "negative"))
def test_real_cli_local_archive_retains_completion_or_first_failure(subject, archive, request, mode):
    if mode == "negative":
        mutate(archive, "bool-enclosure")
        pin = archive.freeze()
    else:
        pin = archive.pin
    output = archive.root.parent / "real-cli"
    command = [sys.executable, "-B", str(ROOT / "tests/test_branch_bound_cuda_independent.py"),
               "--run", str(archive.root), "--evidence-manifest", str(archive.manifest),
               "--evidence-manifest-sha256", pin, "--artifact-dir", str(output)]
    result = subprocess.run(command, cwd=ROOT, env=cpu_env(archive.root.parent),
                            capture_output=True, timeout=30)
    save(archive.root.parent / "cli.log", result.stdout + result.stderr)
    receipts = list(output.glob("replay-*/independent.json"))
    assert len(receipts) == 1
    value = load(receipts[0])
    assert result.returncode == (0 if mode == "positive" else 1)
    assert value["verdict"] == ("ARCHIVE_NUMERICAL_PASS" if mode == "positive" else "FAIL")
    assert value["execution_verdict"] == value["performance_verdict"] == value["commercial_verdict"] == "UNVERIFIED"
    if mode == "negative":
        assert "finite nonbool numerical enclosure required" in value["first_failure"]
    request.node.user_properties.extend([("bounded_v2_cli", str(receipts[0])),
                                         ("bounded_v2_cli_sha256", sha(receipts[0].read_bytes()))])


def test_legacy_v1_never_becomes_v2_or_gpu_qualification(subject, archive, request):
    value = observe(archive, request, "legacy-numerical-only",
                    lambda: subject.audit(archive.root, sha(archive.path("proof").read_bytes())), False)["result"]
    assert value["verdict"] == "LEGACY_NUMERICAL_ONLY" and value["schema_version"] == 1
    assert value["numerical_verdict"] == "PASS" and value["archive_binding_verdict"] == "UNVERIFIED"
    assert value["execution_verdict"] == value["performance_verdict"] == value["commercial_verdict"] == "UNVERIFIED"


def junit_membership(path, nodes):
    tree = ET.parse(path)
    cases = list(tree.getroot().iter("testcase"))
    actual = [case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases]
    expected, observed = Counter(nodes), Counter(actual)
    detail = dict(expected=nodes, observed=actual, missing=list((expected - observed).elements()),
                  extra=list((observed - expected).elements()),
                  duplicates=[name for name, count in observed.items() if count != 1])
    assert expected and expected == observed and not detail["duplicates"], detail
    counts = dict(tests=len(cases), **{
        name: sum(case.find(tag) is not None for case in cases)
        for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))})
    leaves = [suite for suite in tree.getroot().iter("testsuite") if not suite.findall("testsuite")]
    assert {key: sum(int(suite.attrib[key]) for suite in leaves) for key in counts} == counts
    return counts, detail, cases


@pytest.mark.parametrize("mutation", ("wrong-class", "duplicate", "missing", "extra", "wrong-count"))
def test_junit_membership_oracle_itself_rejects_forged_leaves(tmp_path, mutation):
    path = tmp_path / "oracle.xml"
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", tests="1", failures="0", errors="0", skipped="0")
    case = ET.SubElement(suite, "testcase", classname="tests.independent", name="selected")
    node = "tests/independent.py::selected"
    save(path, ET.tostring(xml))
    assert junit_membership(path, [node])[0] == COUNTS
    if mutation == "wrong-class":
        case.set("classname", "tests.foreign")
    elif mutation == "missing":
        suite.remove(case)
        suite.set("tests", "0")
    elif mutation == "wrong-count":
        suite.set("tests", "2")
    else:
        suite.append(deepcopy(case))
        suite.set("tests", "2")
        if mutation == "extra":
            suite[-1].set("name", "unselected")
    save(path, ET.tostring(xml))
    with pytest.raises(AssertionError):
        junit_membership(path, [node])


def artifact_observations(output, cases):
    artifacts = []
    for case in cases:
        properties = case.findall("properties/property")
        props = {prop.attrib["name"]: prop.attrib["value"] for prop in properties}
        assert len(props) == len(properties), "duplicate observation properties"
        for key, path_value in props.items():
            if key.startswith("bounded_v2_") and not key.endswith("_sha256"):
                path = Path(path_value).resolve(strict=True)
                assert path.is_relative_to(output.resolve())
                raw = path.read_bytes()
                assert sha(raw) == props[key + "_sha256"]
                artifacts.append(dict(kind=key, nodeid=case.attrib["classname"].replace(".", "/") +
                    ".py::" + case.attrib["name"], path=str(path), sha256=sha(raw), bytes=len(raw)))
    assert len({item["path"] for item in artifacts}) == len(artifacts)
    return artifacts


def observation_membership(nodes, artifacts, observations):
    neutral_prefixes = (
        "test_frozen_owner_original56_red_and_independent_domain",
        "test_unchanged_original_fraction_oracle_sees_mathematical_red[",
        "test_junit_membership_oracle_itself_rejects_forged_leaves[",
    )
    expected = {node for node in nodes if not node.split("::", 1)[1].startswith(neutral_prefixes)}
    actual = Counter(item["nodeid"] for item in artifacts)
    assert set(actual) == expected and all(count == 1 for count in actual.values())
    original_pin = [item for item in observations if item["case"].startswith("original-pin:")]
    semantic = [item for item in observations if item["case"].startswith("semantic:")]
    assert len(original_pin) == 48 and {x["case"].split(":", 1)[1] for x in original_pin} == set(MUTATIONS)
    assert len(semantic) == 46 and {x["case"].split(":", 1)[1] for x in semantic} == set(MUTATIONS) - set(LEGAL_RESELECTION)
    legacy = [item for item in observations if item["case"].startswith("legacy-model-upper:")]
    assert len(legacy) == 3 and {item["case"] for item in legacy} == {
        "legacy-model-upper:" + mode for mode in ("original-pin", "semantic", "oracle")}
    assert any(item["case"] == "v2-independent-positive" and item["outcome"] == "RETURNED" for item in observations)
    assert all((item["outcome"] == "REJECTED") == item["expected_rejected"] for item in observations)


def review(directory):
    from scripts.verify import source_snapshot, check_source_consistency

    verify_sources()
    assert directory.resolve().is_relative_to(RELEASE.resolve()) and not directory.resolve().is_relative_to(ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="review-", dir=directory))
    before = source_snapshot(ROOT)
    started = time.monotonic()
    value = dict(verdict="RUNNING", subject_commit=SUBJECT, reviewer=before,
                 owner_pins=OWNER, original_test_pin=OLD_PIN, old_failure_pin=OLD_RECEIPT_PIN,
                 started_at=datetime.now(timezone.utc).isoformat(),
                 scope="synthetic v2 reader code review only; no historical caller authority or attestation")
    try:
        common = [sys.executable, "-B", "-m", "pytest", "-o", "addopts=",
                  "-o", f"cache_dir={output / 'cache'}"]
        env = cpu_env(output)
        collect = subprocess.run([*common, "--collect-only", "-q", SELF], cwd=ROOT, env=env,
                                 capture_output=True, timeout=30)
        save(output / "collection.log", collect.stdout + collect.stderr)
        assert collect.returncode == 0
        nodes = [line for line in collect.stdout.decode().splitlines() if line.startswith(SELF + "::")]
        with (output / "pytest.log").open("xb") as log:
            result = subprocess.run([*common, "-q", "-ra", f"--junitxml={output / 'pytest.xml'}",
                                     "--basetemp", str(output / "pytest-tmp"), SELF], cwd=ROOT,
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=170)
        counts, membership, cases = junit_membership(output / "pytest.xml", nodes)
        assert counts["skipped"] == 0
        artifacts = artifact_observations(output, cases)
        observations = [load(Path(item["path"])) for item in artifacts if item["kind"] == "bounded_v2_observation"]
        original_pin = [item for item in observations if item["case"].startswith("original-pin:")]
        semantic = [item for item in observations if item["case"].startswith("semantic:")]
        if not counts["failures"] and not counts["errors"]:
            observation_membership(nodes, artifacts, observations)
        consistency = check_source_consistency(before, source_snapshot(ROOT))
        assert consistency["verdict"] == "PASS"
        verify_sources()
        failures = {case.attrib["name"]: case.find(tag).attrib.get("message", "")
                    for case in cases for tag in ("failure", "error") if case.find(tag) is not None}
        assert result.returncode == (1 if failures else 0)
        value.update(verdict="FAIL" if failures else "PASS", tests=counts,
                     passed=counts["tests"] - counts["failures"] - counts["errors"],
                     pytest_returncode=result.returncode, failures=failures, membership=membership,
                     source_consistency=consistency, artifacts=artifacts, original_pin_negatives=len(original_pin),
                     semantic_negatives=len(semantic),
                     false_accept_count=sum(item["expected_rejected"] and item["outcome"] == "RETURNED"
                                            for item in observations),
                     hashes={name: sha((output / name).read_bytes())
                             for name in ("collection.log", "pytest.xml", "pytest.log")})
    except Exception as error:
        value.update(verdict="ERROR", first_failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        value.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
        path = output / "independent-review.json"
        save(path, value)
        print(f"BOUNDED_V2_INDEPENDENT verdict={value['verdict']} receipt={path} sha256={sha(path.read_bytes())}", flush=True)
    return 0 if value["verdict"] == "PASS" else 1


def inspect_receipt(path):
    """Recheck frozen terminal source, exact leaves, and actual observation files."""
    from scripts.verify import source_snapshot, check_source_consistency

    value = load(path)
    assert value["verdict"] == "PASS" and value["reviewer"]["clean"] is True
    assert value["reviewer"]["commit"] == git("rev-parse", "HEAD").decode().strip()
    assert value["subject_commit"] == SUBJECT and value["owner_pins"] == OWNER
    assert value["original_test_pin"] == OLD_PIN and value["old_failure_pin"] == OLD_RECEIPT_PIN
    assert value["source_consistency"]["verdict"] == "PASS"
    assert check_source_consistency(value["reviewer"], source_snapshot(ROOT))["verdict"] == "PASS"
    verify_sources()
    for name, pin in value["hashes"].items():
        assert sha((path.parent / name).read_bytes()) == pin
    nodes = [line for line in (path.parent / "collection.log").read_text().splitlines()
             if line.startswith(SELF + "::")]
    counts, membership, cases = junit_membership(path.parent / "pytest.xml", nodes)
    assert counts == value["tests"] and membership == value["membership"]
    assert counts["tests"] == value["passed"] > 0
    assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
    artifacts = artifact_observations(path.parent, cases)
    assert artifacts == value["artifacts"]
    observations = [load(Path(item["path"])) for item in artifacts if item["kind"] == "bounded_v2_observation"]
    observation_membership(nodes, artifacts, observations)
    assert value["original_pin_negatives"] == 48 and value["semantic_negatives"] == 46
    assert value["false_accept_count"] == 0
    for observation in observations:
        manifest = Path(observation["manifest_path"]).resolve(strict=True)
        root = Path(observation["run_root"]).resolve(strict=True)
        assert manifest.is_relative_to(path.parent.resolve()) and root.is_relative_to(path.parent.resolve())
        assert sha(manifest.read_bytes()) == observation["current_manifest_sha256"]
        for descriptor in [*observation["files"].values(), *observation["selected_source"].values()]:
            selected = (root / descriptor["path"]).resolve(strict=True)
            # A symlink-negative snapshot can resolve within this test output
            # but outside its archive; no reader acceptance is implied.
            assert selected.is_relative_to(path.parent.resolve())
            raw = selected.read_bytes()
            assert sha(raw) == descriptor["sha256"] and len(raw) == descriptor["bytes"]
        if observation["case"].startswith("original-pin:"):
            assert observation["current_manifest_sha256"] == observation["original_caller_pin"]
            assert "caller pinned evidence bytes differ:" in observation["reason"]
        if observation["case"].startswith("semantic:"):
            assert "caller pinned evidence bytes differ" not in observation["reason"]
    print(f"BOUNDED_V2_TERMINAL_READ verdict=PASS read_at={datetime.now(timezone.utc).isoformat()} "
          f"reviewer={value['reviewer']['commit']} tests={counts['tests']} original_pin=48 semantic=46 "
          f"false_accept=0 skip=0 receipt={path} sha256={sha(path.read_bytes())}")
    return 0


def scan(directory):
    spec = importlib.util.spec_from_file_location("_bounded_v2_safe_scan", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(secret_hex=None, secrets_file=None,
        no_needle=False, needle_from=str(Path.home() / ".config/dsh-n600-token")))
    assert needles
    paths = [ROOT / SELF, ROOT / "docs/research/bounded-cuda-v2-independent.md"]
    literal, _ = scanner.scan_literal([str(path) for path in paths], needles)
    shape = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                       r"""(?i:password|token|secret)\s*["']?\s*[:=]\s*["'][^"'\s]{6,}["']""")
    count = sum(bool(shape.search(path.read_text())) for path in paths)
    path = directory / "scan.json"
    save(path, dict(verdict="PASS" if literal == count == 0 else "FAIL", literal_hits=literal,
                    shape_hits=count, files={str(item): sha(item.read_bytes()) for item in paths}))
    print(f"BOUNDED_V2_SCAN receipt={path} sha256={sha(path.read_bytes())}")
    return 0 if literal == count == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--inspect-receipt", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    assert sum((args.review, args.scan_only, args.inspect_receipt is not None)) == 1
    sys.path.insert(0, str(ROOT))
    if args.inspect_receipt is not None:
        raise SystemExit(inspect_receipt(args.inspect_receipt))
    raise SystemExit(review(args.artifact_dir) if args.review else scan(args.artifact_dir))
