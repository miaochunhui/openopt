"""Self-contained v2 archive contract controls, not new CUDA execution.

The test caller seals the complete synthetic archive BEFORE mutations. A second
set intentionally pins invalid archives to exercise semantics rather than just
hash rejection. No archived source is imported and no optimizer is invoked.
--audit-artifacts independently records actual collection/JUnit membership.
"""

from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from tests import test_branch_bound_cuda_independent as reader
from tests.test_branch_bound_independent import (
    handcrafted_report, lower_bound, original_model, pack,
)

ROOT = Path(__file__).resolve().parents[1]
TEST = "tests/test_bounded_cuda_reader_contract.py"
OWNER = "tests/test_branch_bound_cuda_independent.py"
SIDE = "case-0/cuda-check-synthetic"
PROOF = SIDE + "/pytest-tmp/test_cuda_frozen_bound_tree0/bounded-miqp-cuda.json"
TRACE = SIDE + "/pytest-tmp/test_cuda_frozen_bound_tree0/cuda-trace.json"
JUNIT = SIDE + "/pytest.xml"
JOB = "synthetic-archive-job"
STATUS = JOB + "-status.json"
ROLE_PATHS = {"run": "run.json", "collection": "collection.json", "job": "job-0.json",
              "status": STATUS, "result": "case-0/result.json",
              "side_result": SIDE + "/result.json", "junit": JUNIT,
              "proof": PROOF, "trace": TRACE}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(encoded(value))


def read(path):
    return reader.read_json(path)


def caller_manifest(root):
    """Test-only caller selection. Never called inside audit_v2."""
    def descriptor(name):
        raw = (root / name).read_bytes()
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    return {"schema_version": 2, "authority": "caller-selected-archive-only",
            "run_root": str(root.resolve()), "target": reader.TARGET,
            "source_commit": reader.SOURCE_COMMIT, "source_tree": reader.SOURCE_TREE,
            "job_name": JOB, "node_name": "synthetic-node", "pod_name": "synthetic-pod",
            "files": {role: descriptor(name) for role, name in ROLE_PATHS.items()},
            "selected_source": {name: descriptor("source/" + name)
                                for name in reader.SELECTED_SOURCES}}


def freeze(root):
    path = root.parent / "caller-manifest-v2.json"
    write(path, caller_manifest(root))
    return path, sha(path)


def synchronize_untrusted(root):
    """Coordinate attacker-owned proof/JUnit/result hashes, not the caller pin."""
    tree = ET.parse(root / JUNIT)
    for entry in tree.findall(".//property[@name=\"receipt_sha256\"]"):
        entry.set("value", sha(root / PROOF))
    tree.write(root / JUNIT, encoding="utf-8", xml_declaration=True)
    collection = read(root / "collection.json")
    collection["jobs"][0]["result_sha256"] = sha(root / "case-0/result.json")
    write(root / "collection.json", collection)


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "synthetic-archive"
    root.mkdir()
    problem, _, proof = handcrafted_report("pure")
    proof["config"]["device"] = "cuda:0"
    proof["source_sha256"] = deepcopy(reader.SOURCES)
    m = original_model(problem)
    for index, z in ((0, .5), (2, 1.)):
        node = proof["nodes"][index]
        proposal = {"status": "solved", "iterations": 25,
                    "actual_device": "cuda:0", "dtype": "torch.float64", "z": [z], "y": []}
        node["proposal"] = proposal
        node["witnesses"].append({"kind": "qp_tangent", "z": [z], "y": []})
        box = tuple((Fraction(left), Fraction(right)) for left, right in node["box"])
        node["bound"] = pack(max(lower_bound(m, box, w["z"], w["y"])
                                 for w in node["witnesses"]))
    proof["nodes"][1]["disposition"] = "pruned"
    proof["actual_qp_calls"] = 2
    write(root / PROOF, proof)
    write(root / TRACE, {"deviceProperties": [{"name": "NVIDIA RTX PRO 4500 Blackwell"}],
                        "traceEvents": [{"cat": "kernel", "ph": "X", "name": "synthetic_kernel",
                                         "dur": 1., "pid": 1, "tid": 1, "ts": 1.}]})
    for name, pin in reader.SELECTED_SOURCES.items():
        raw = (ROOT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin, "self-contained selected fixture bytes drift"
        path = root / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    identity = {"commit": reader.SOURCE_COMMIT, "tree": reader.SOURCE_TREE,
                "clean": True, "sha256": deepcopy(reader.SELECTED_SOURCES)}
    identity["sha256"]["README.md"] = hashlib.sha256(b"synthetic caller-selected metadata").hexdigest()
    item = {"job": JOB, "size": 1, "output": str(root / "case-0"),
            "manifest": str(root / "job-0.json")}
    record = {"namespace": "synthetic-namespace", "stage": "release-regression",
              "source_sha": reader.SOURCE_COMMIT, "source_identity": identity,
              "source_hashes": deepcopy(reader.SELECTED_SOURCES), "jobs": [item]}
    write(root / "run.json", record)
    result = {"verdict": "PASS", "returncode": 0, "targets": [reader.TARGET],
              "tests": deepcopy(reader.COUNTS), "source_sha256": deepcopy(reader.SELECTED_SOURCES),
              "hardware": {"gpu": "NVIDIA RTX PRO 4500 Blackwell", "node": "synthetic-node"}}
    write(root / "case-0/result.json", result)
    write(root / SIDE / "result.json", result)
    collection = deepcopy(record)
    collection["jobs"][0].update(result_path=str(root / "case-0/result.json"),
                                  result_sha256=sha(root / "case-0/result.json"))
    collection["states"] = ["PASS"]
    write(root / "collection.json", collection)
    job = {"apiVersion": "batch/v1", "kind": "Job",
           "metadata": {"name": JOB, "namespace": "synthetic-namespace"}}
    write(root / "job-0.json", job)
    observed = deepcopy(job)
    observed["metadata"]["uid"] = "synthetic-controller-uid"
    observed["status"] = {"succeeded": 1, "conditions": [{"type": "Complete", "status": "True"}]}
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {
        "name": "synthetic-pod", "namespace": "synthetic-namespace", "labels": {"job-name": JOB},
        "ownerReferences": [{"controller": True, "apiVersion": "batch/v1", "kind": "Job",
                             "name": JOB, "uid": "synthetic-controller-uid"}]},
        "spec": {"nodeName": "synthetic-node"}, "status": {"phase": "Succeeded",
            "containerStatuses": [{"restartCount": 0, "state": {"terminated": {"exitCode": 0}}}]}}
    write(root / STATUS, {"job": observed, "pods": {"items": [pod]}})
    xml = ET.Element("testsuites", name="pytest tests")
    suite = ET.SubElement(xml, "testsuite", name="pytest", hostname="synthetic-pod",
                          **{key: str(value) for key, value in reader.COUNTS.items()})
    case = ET.SubElement(suite, "testcase", classname="tests.test_branch_bound",
                         name="test_cuda_frozen_bound_tree")
    props = ET.SubElement(case, "properties")
    for name, value in {"receipt": str(root / PROOF), "trace": str(root / TRACE),
                        "receipt_sha256": sha(root / PROOF)}.items():
        ET.SubElement(props, "property", name=name, value=value)
    ET.ElementTree(xml).write(root / JUNIT, encoding="utf-8", xml_declaration=True)
    manifest, pin = freeze(root)
    return root, manifest, pin


MUTATIONS = [
    "model-q", "inflated-bound", "deleted-child", "branch-gap", "fractional-incumbent",
    "wrong-gap", "bool-schema", "bool-node-id", "bool-parent", "bool-visit", "bool-count",
    "bool-config", "bool-enclosure", "core-source", "result-count", "result-bool-count",
    "result-bool-rc", "result-node", "result-source-test", "collection-job", "collection-source",
    "collection-result-pin", "run-job", "run-tree", "job-name", "job-kind", "junit-count",
    "junit-missing", "junit-duplicate", "receipt-pin", "trace-no-kernel", "trace-bool-duration",
    "trace-negative-duration", "trace-device", "trace-unrelated-kernel", "trace-changed-duration",
    "status-owner", "status-node", "status-pod", "status-exit", "status-restart", "status-failed",
    "junit-suite-name", "junit-host", "junit-test", "junit-source-class",
    "run-bool-size", "source-map-subset-mismatch",
]


def mutate(root, name):
    proof = read(root / PROOF)
    if name == "model-q":
        proof["model"]["arrays"]["q"]["values"][0] = float(1.).hex()
    elif name == "inflated-bound":
        proof["nodes"][2]["bound"] = "1/1"
    elif name == "deleted-child":
        proof["nodes"][0]["split"]["children"] = [1, 1]
    elif name == "branch-gap":
        proof["nodes"][2]["box"][0][0] = "2/1"
    elif name == "fractional-incumbent":
        proof["incumbent"] = [.5]
    elif name == "wrong-gap":
        proof["summary"]["gap_abs"] = "-1/1"
    elif name == "bool-schema":
        proof["schema_version"] = True
    elif name == "bool-node-id":
        proof["nodes"][1]["id"] = True
    elif name == "bool-parent":
        proof["nodes"][1]["parent"] = False
    elif name == "bool-visit":
        proof["nodes"][0]["visited"] = 1
    elif name == "bool-count":
        proof["qp_iteration_limit_count"] = False
    elif name == "bool-config":
        proof["config"]["enabled"] = 1
    elif name == "bool-enclosure":
        proof["summary"]["enclosures"]["gap_abs"] = [False, False]
    elif name == "core-source":
        source = root / "source/src/openopt/qp.py"
        source.write_bytes(source.read_bytes() + b"\n# changed archived bytes\n")
        pin = sha(source)
        proof["source_sha256"]["src/openopt/qp.py"] = pin
        for path in ("run.json", "collection.json"):
            value = read(root / path)
            value["source_hashes"]["src/openopt/qp.py"] = pin
            value["source_identity"]["sha256"]["src/openopt/qp.py"] = pin
            write(root / path, value)
        result = read(root / "case-0/result.json")
        result["source_sha256"]["src/openopt/qp.py"] = pin
        write(root / "case-0/result.json", result)
        write(root / SIDE / "result.json", result)
    elif name.startswith("result-"):
        result = read(root / "case-0/result.json")
        if name in ("result-count", "result-bool-count"):
            result["tests"]["tests"] = 2 if name == "result-count" else True
        elif name == "result-bool-rc":
            result["returncode"] = False
        elif name == "result-node":
            result["hardware"]["node"] = "different-node"
        else:
            result["source_sha256"]["tests/test_branch_bound.py"] = "0" * 64
        write(root / "case-0/result.json", result)
        write(root / SIDE / "result.json", result)
    elif name.startswith("collection-"):
        value = read(root / "collection.json")
        if name == "collection-job":
            value["jobs"][0]["job"] = "different-job"
        elif name == "collection-source":
            value["source_sha"] = "0" * 40
        else:
            value["jobs"][0]["result_sha256"] = "0" * 64
        write(root / "collection.json", value)
    elif name.startswith("run-"):
        value = read(root / "run.json")
        if name == "run-job":
            value["jobs"][0]["job"] = "different-job"
        elif name == "run-bool-size":
            value["jobs"][0]["size"] = True
        else:
            value["source_identity"]["tree"] = "0" * 40
        write(root / "run.json", value)
    elif name == "source-map-subset-mismatch":
        for path in ("run.json", "collection.json"):
            value = read(root / path)
            value["source_identity"]["sha256"]["tests/test_branch_bound.py"] = "0" * 64
            write(root / path, value)
    elif name.startswith("job-"):
        value = read(root / "job-0.json")
        if name == "job-name":
            value["metadata"]["name"] = "different-job"
        else:
            value["kind"] = "Deployment"
        write(root / "job-0.json", value)
    elif name.startswith("status-"):
        value = read(root / STATUS)
        pod = value["pods"]["items"][0]
        container = pod["status"]["containerStatuses"][0]
        if name == "status-owner":
            pod["metadata"]["ownerReferences"][0]["uid"] = "different-uid"
        elif name == "status-node":
            pod["spec"]["nodeName"] = "different-node"
        elif name == "status-pod":
            pod["metadata"]["name"] = "different-pod"
        elif name == "status-exit":
            container["state"]["terminated"]["exitCode"] = False
        elif name == "status-restart":
            container["restartCount"] = True
        else:
            value["job"]["status"]["conditions"].append({"type": "Failed", "status": "True"})
        write(root / STATUS, value)
    elif name.startswith("junit-"):
        tree = ET.parse(root / JUNIT)
        suite = tree.find("testsuite")
        if name == "junit-count":
            suite.set("tests", "2")
        elif name == "junit-missing":
            suite.remove(suite.find("testcase"))
        elif name == "junit-duplicate":
            suite.append(deepcopy(suite.find("testcase")))
        elif name == "junit-suite-name":
            suite.set("name", "unrelated-suite")
        elif name == "junit-host":
            suite.set("hostname", "different-pod")
        elif name == "junit-test":
            suite.find("testcase").set("name", "different-test")
        else:
            suite.find("testcase").set("classname", "tests.different_source")
        tree.write(root / JUNIT, encoding="utf-8", xml_declaration=True)
    elif name.startswith("trace-"):
        value = read(root / TRACE)
        if name == "trace-no-kernel":
            value["traceEvents"] = [{"cat": "cpu_op"}]
        elif name == "trace-device":
            value["deviceProperties"][0]["name"] = "different GPU"
        elif name == "trace-unrelated-kernel":
            value["traceEvents"][0]["name"] = "unrelated_kernel"
        else:
            value["traceEvents"][0]["dur"] = (True if name == "trace-bool-duration" else
                                               -1. if name == "trace-negative-duration" else 2.)
        write(root / TRACE, value)
    elif name != "receipt-pin":
        raise ValueError("unknown mutation")
    write(root / PROOF, proof)
    synchronize_untrusted(root)
    if name == "collection-result-pin":
        value = read(root / "collection.json")
        value["jobs"][0]["result_sha256"] = "0" * 64
        write(root / "collection.json", value)
    if name == "receipt-pin":
        tree = ET.parse(root / JUNIT)
        tree.find(".//property[@name=\"receipt_sha256\"]").set("value", "0" * 64)
        tree.write(root / JUNIT, encoding="utf-8", xml_declaration=True)


def observed(request, tmp_path, label, call, rejected):
    try:
        observation = {"outcome": "RETURNED", "result": call()}
    except (ValueError, KeyError, TypeError, ArithmeticError, OSError) as error:
        observation = {"outcome": "REJECTED", "reason": f"{type(error).__name__}: {error}"}
    observation["case"] = label
    path = tmp_path / "contract-observation.json"
    write(path, observation)
    request.node.user_properties.extend([
        ("contract_observation", str(path)), ("contract_observation_sha256", sha(path))])
    assert (observation["outcome"] == "REJECTED") is rejected, observation
    return observation


def test_explicit_v2_positive_not_execution(archive, request, tmp_path, monkeypatch):
    root, manifest, pin = archive
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("reader launched subprocess"))
    value = observed(request, tmp_path, "synthetic-v2-positive",
                     lambda: reader.audit_v2(root, manifest, pin), False)["result"]
    assert value["verdict"] == "ARCHIVE_NUMERICAL_PASS"
    assert value["numerical_verdict"] == value["archive_binding_verdict"] == "PASS"
    assert value["execution_verdict"] == value["performance_verdict"] == "UNVERIFIED"
    assert value["commercial_verdict"] == "UNVERIFIED" and value["manifest_sha256"] == pin
    assert value["original_model"]["coverage"] == "PASS"


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_coordinated_rehash_old_caller_manifest_unchanged(archive, tmp_path, request, mutation):
    root, manifest, pin = archive
    assert reader.audit_v2(root, manifest, pin)["numerical_verdict"] == "PASS"
    baseline = manifest.read_bytes()
    mutate(root, mutation)
    assert manifest.read_bytes() == baseline and sha(manifest) == pin
    observed(request, tmp_path, "unchanged-caller-pin:" + mutation,
             lambda: reader.audit_v2(root, manifest, pin), True)


@pytest.mark.parametrize("mutation", [name for name in MUTATIONS
                                      if name not in ("trace-unrelated-kernel", "trace-changed-duration")])
def test_repinned_invalid_bytes_still_fail_semantics(archive, tmp_path, request, mutation):
    root, _, _ = archive
    mutate(root, mutation)
    manifest, pin = freeze(root)
    observed(request, tmp_path, "repinned-semantic:" + mutation,
             lambda: reader.audit_v2(root, manifest, pin), True)


@pytest.mark.parametrize("mutation", ("trace-unrelated-kernel", "trace-changed-duration"))
def test_caller_explicitly_reselecting_legal_trace_does_not_attest_origin(archive, mutation):
    root, _, _ = archive
    mutate(root, mutation)
    manifest, pin = freeze(root)
    result = reader.audit_v2(root, manifest, pin)
    assert result["execution_verdict"] == result["performance_verdict"] == "UNVERIFIED"
    assert result["archive_binding_verdict"] == "PASS"


@pytest.mark.parametrize("mutation", ("version", "bool-version", "missing-trace", "missing-source",
                                      "extra-role", "duplicate-path", "bool-bytes", "wrong-pin",
                                      "wrong-root", "execution-authority", "traversal", "absolute"))
def test_strict_caller_manifest_schema(archive, mutation):
    root, manifest, pin = archive
    value = read(manifest)
    if mutation == "version":
        value["schema_version"] = 1
    elif mutation == "bool-version":
        value["schema_version"] = True
    elif mutation == "missing-trace":
        del value["files"]["trace"]
    elif mutation == "missing-source":
        del value["selected_source"]["tests/test_branch_bound.py"]
    elif mutation == "extra-role":
        value["files"]["unreviewed"] = value["files"]["proof"]
    elif mutation == "duplicate-path":
        value["files"]["trace"] = value["files"]["proof"]
    elif mutation == "bool-bytes":
        value["files"]["proof"]["bytes"] = True
    elif mutation == "wrong-root":
        value["run_root"] = str(root.parent)
    elif mutation == "execution-authority":
        value["authority"] = "GPU-verified"
    elif mutation == "traversal":
        value["files"]["trace"]["path"] = "../escape.json"
    elif mutation == "absolute":
        value["files"]["trace"]["path"] = str(root / TRACE)
    if mutation != "wrong-pin":
        write(manifest, value)
        pin = sha(manifest)
    else:
        pin = "0" * 64
    with pytest.raises((ValueError, OSError)):
        reader.audit_v2(root, manifest, pin)


@pytest.mark.parametrize("value", (False, True, "0", None, float("inf")))
def test_enclosure_is_finite_numerical_not_bool_or_coercion(archive, value):
    root, _, _ = archive
    proof = read(root / PROOF)
    proof["summary"]["enclosures"]["gap_abs"] = [value, value]
    with pytest.raises(ValueError):
        reader.original_replay(proof)


def test_v1_not_silently_promoted(archive):
    root, _, _ = archive
    value = reader.audit(root, sha(root / PROOF))
    assert value["verdict"] == "LEGACY_NUMERICAL_ONLY"
    assert value["numerical_verdict"] == "PASS"
    assert value["archive_binding_verdict"] == value["execution_verdict"] == "UNVERIFIED"
    mutate(root, "result-node")
    value = reader.audit(root, sha(root / PROOF))
    assert value["archive_binding_verdict"] == "UNVERIFIED"


@pytest.mark.parametrize("kind", ("file", "parent", "manifest"))
def test_evidence_symlinks_rejected(archive, kind):
    root, manifest, pin = archive
    path = root / TRACE if kind == "file" else root / "source/src" if kind == "parent" else manifest
    target = path.with_name(path.name + "-real")
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises((ValueError, OSError)):
        reader.audit_v2(root, manifest, pin)


def test_changed_manifest_not_allowed_by_new_self_reported_pin(archive):
    root, manifest, pin = archive
    mutate(root, "trace-unrelated-kernel")
    freeze(root)
    assert sha(manifest) != pin
    with pytest.raises(ValueError, match="caller manifest pin"):
        reader.audit_v2(root, manifest, pin)


def test_readonly_oracle_source_not_replaced_by_artifact(archive, monkeypatch):
    root, manifest, pin = archive
    original_digest = reader.digest
    monkeypatch.setattr(reader, "digest", lambda path: "0" * 64
                        if Path(path).name == "test_branch_bound_independent.py" else original_digest(path))
    with pytest.raises(ValueError, match="readonly mathematical oracle bytes"):
        reader.audit_v2(root, manifest, pin)


def test_reader_cli_requires_v2_and_writes_real_completion(archive, tmp_path, request):
    root, manifest, pin = archive
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])}
    artifacts = tmp_path / "cli"
    result = subprocess.run([sys.executable, "-B", str(ROOT / OWNER), "--run", str(root),
                             "--evidence-manifest", str(manifest), "--evidence-manifest-sha256", pin,
                             "--artifact-dir", str(artifacts)], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30, check=False)
    (tmp_path / "cli-output.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    reports = list(artifacts.glob("replay-*/independent.json"))
    assert len(reports) == 1 and read(reports[0])["execution_verdict"] == "UNVERIFIED"
    assert result.stdout.rstrip().endswith("report=" + str(reports[0]))
    request.node.user_properties.extend([("cli_report", str(reports[0])), ("cli_report_sha256", sha(reports[0]))])


def test_reader_cli_invalid_pin_has_failed_receipt_no_execution(archive, tmp_path):
    root, manifest, _ = archive
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])}
    artifacts = tmp_path / "failed-cli"
    process = subprocess.run([sys.executable, "-B", str(ROOT / OWNER), "--run", str(root),
                              "--evidence-manifest", str(manifest), "--evidence-manifest-sha256", "0" * 64,
                              "--artifact-dir", str(artifacts)], env=env, cwd=ROOT,
                             capture_output=True, text=True, timeout=30, check=False)
    (tmp_path / "failed-cli-output.log").write_text(process.stdout + process.stderr)
    assert process.returncode != 0
    reports = list(artifacts.glob("replay-*/independent.json"))
    assert len(reports) == 1
    report = read(reports[0])
    assert report["verdict"] == "FAIL" and "caller manifest pin differs" in report["first_failure"]
    assert report["execution_verdict"] == report["performance_verdict"] == "UNVERIFIED"
    assert "verdict=FAIL" in process.stdout


def test_trace_kernel_sum_overflow_fails_loud():
    trace = {"deviceProperties": [{"name": "NVIDIA RTX PRO 4500 Blackwell"}],
             "traceEvents": [{"cat": "kernel", "ph": "X", "name": "kernel", "dur": 1e308}] * 2}
    with pytest.raises(ValueError, match="sum overflow"):
        reader.trace_summary(trace)


def test_actual_archive_only_with_external_caller_selected_v2(request):
    run = os.environ.get("OPENOPT_BOUNDED_READER_RUN")
    manifest = os.environ.get("OPENOPT_BOUNDED_READER_MANIFEST")
    pin = os.environ.get("OPENOPT_BOUNDED_READER_MANIFEST_SHA256")
    if not any((run, manifest, pin)):
        pytest.skip("actual archive not selected by full caller v2 pin; no historical qualification")
    if not all((run, manifest, pin)):
        raise ValueError("all three external caller selection fields required")
    result = reader.audit_v2(run, manifest, pin)
    assert result["execution_verdict"] == "UNVERIFIED" and result["numerical_verdict"] == "PASS"
    request.node.user_properties.extend([("caller_manifest_sha256", pin),
                                         ("archive_selection", str(Path(run).resolve()))])


def audit_artifacts(directory):
    """Seal true counts, exact membership, observed controls and source after freeze."""
    from scripts.verify import check_source_consistency, source_snapshot

    directory = Path(directory).resolve(strict=True)
    reader.require(directory.is_relative_to(reader.SHARED), "external receipt required")
    report = read(directory / "verification.json")
    before = source_snapshot(ROOT)
    reader.require(before["clean"] and report["source"]["clean"], "clean source required")
    consistency = check_source_consistency(report["source"], before)
    reader.require(consistency["verdict"] == "PASS", "source differs")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
           "CUDA_VISIBLE_DEVICES": "-1"}
    cache = directory / "collection-cache"
    process = subprocess.run([sys.executable, "-B", "-m", "pytest", "--collect-only", "-q",
                              "-o", f"cache_dir={cache}", OWNER, TEST],
                             env=env, cwd=ROOT, capture_output=True, text=True, timeout=45, check=False)
    with (directory / "contract-collection.log").open("x") as stream:
        stream.write(process.stdout + process.stderr + f"\nreturncode={process.returncode}\n")
    reader.require(process.returncode == 0, "collection failed")
    selected = [line for line in process.stdout.splitlines()
                if line.startswith((OWNER + "::", TEST + "::"))]
    cases = list(ET.parse(directory / "pytest.xml").getroot().iter("testcase"))
    actual = [case.get("classname").replace(".", "/") + ".py::" + case.get("name") for case in cases]
    reader.require(selected and len(selected) == len(set(selected))
                   and len(actual) == len(set(actual)) and set(selected) == set(actual), "exact membership differs")
    counts = {"tests": len(cases), **{name: sum(case.find(tag) is not None for case in cases)
              for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}}
    for suite in ET.parse(directory / "pytest.xml").getroot().iter("testsuite"):
        leaves = list(suite.findall("testcase"))
        reader.require(suite.get("tests") == str(len(leaves)), "JUnit count differs")
        for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            reader.require(suite.get(name) == str(sum(case.find(tag) is not None for case in leaves)),
                           "JUnit outcome count differs")
    if report["verdict"] == "PASS":
        reader.require(report["tests"] == counts, "verification counts differ")
    for nodeid, case in zip(actual, cases):
        reader.require(case.find("skipped") is None
                       or nodeid == TEST + "::test_actual_archive_only_with_external_caller_selected_v2",
                       "unexpected contract skip")
    observations = []
    for case in cases:
        props = {entry.get("name"): entry.get("value") for entry in case.findall("properties/property")}
        if "contract_observation" in props:
            path = Path(props["contract_observation"]).resolve(strict=True)
            reader.require(path.is_relative_to(directory) and sha(path) == props["contract_observation_sha256"],
                           "observation bytes differ")
            observations.append({"path": str(path), "sha256": sha(path), **read(path)})
    expected_observations = {"synthetic-v2-positive", *["unchanged-caller-pin:" + name for name in MUTATIONS],
                             *["repinned-semantic:" + name for name in MUTATIONS
                               if name not in ("trace-unrelated-kernel", "trace-changed-duration")]}
    reader.require(len(observations) == len(expected_observations)
                   and {item["case"] for item in observations} == expected_observations, "control membership differs")
    false_accepts = [item for item in observations
                     if item["case"] != "synthetic-v2-positive" and item["outcome"] != "REJECTED"]
    failures = [{"nodeid": nodeid, "reason": case.find(tag).get("message", "")}
                for nodeid, case in zip(actual, cases) for tag in ("failure", "error")
                if case.find(tag) is not None]
    positive = next(item for item in observations if item["case"] == "synthetic-v2-positive")
    source_after = source_snapshot(ROOT)
    reader.require(source_after == before and report["targets"] == [OWNER, TEST], "source/scope differs")
    accepted = (report["verdict"] == "PASS" and counts["failures"] == counts["errors"] == 0
                and not false_accepts and positive["outcome"] == "RETURNED")
    receipt = {"schema_version": 2, "verdict": "PASS" if accepted else "FAIL",
               "read_at": datetime.now(timezone.utc).isoformat(), "source": source_after,
               "source_consistency": consistency, "tests": counts,
               "collection_nodeids": selected, "junit_nodeids": actual,
               "membership_missing": [], "membership_extra": [], "duplicates": [],
               "false_accepts": false_accepts, "false_accept_count": len(false_accepts),
               "first_failure": failures[0] if failures else None, "failures": failures,
               "wall_seconds": (datetime.fromisoformat(report["finished_at"])
                                - datetime.fromisoformat(report["started_at"])).total_seconds(),
               "observations": observations, "verification_sha256": sha(directory / "verification.json"),
               "junit_sha256": sha(directory / "pytest.xml"),
               "collection_log_sha256": sha(directory / "contract-collection.log"),
               "execution_verdict": "UNVERIFIED", "performance_verdict": "UNVERIFIED",
               "commercial_verdict": "UNVERIFIED", "gpu_jobs_submitted": 0,
               "historical_v1_verdict": "FAIL; never reclassified",
               "historical_v1_receipt_sha256": "b499296c59e02227edbfb96d7a8ee701e6aae4a07cc0b0fd19853ba28655b84e",
               "qualification": "author contract tests only; independent review still required"}
    path = directory / "reader-contract.json"
    with path.open("x") as stream:
        stream.write(encoded(receipt).decode())
    reader.require(read(path) == receipt, "receipt readback differs")
    verdict = receipt["verdict"]
    print(f"BOUNDED_CUDA_READER_CONTRACT verdict={verdict} "
          f"tests={counts} false_accepts={len(false_accepts)} source=PASS receipt={path}", flush=True)
    return 0 if accepted else 1


def scan_sources(directory):
    """Use the existing local credential scanner without rendering any value.

    Infra tool is readonly, never loaded from input evidence. Known local
    credentials are compared only in memory; output contains counts and pins.
    """
    import re

    directory = Path(directory).resolve()
    reader.require(directory.is_relative_to(reader.SHARED), "external scan output required")
    directory.mkdir(parents=True, exist_ok=True)
    scanner_path = Path("/data/dev/infra/ops/dsh/scan-secrets-precommit.py").resolve(strict=True)
    specification = importlib.util.spec_from_file_location("_readonly_local_secret_scanner", scanner_path)
    scanner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(scanner)
    needle_args = SimpleNamespace(secret_hex=None, secrets_file=None, no_needle=False,
                                  needle_from=str(Path.home() / ".config/dsh-n600-token"))
    needles = scanner.read_needles(needle_args)
    reader.require(needles, "no literal scan needle available")
    paths = [ROOT / OWNER, ROOT / TEST, ROOT / "docs/research/bounded-miqp-cuda-review.md"]
    hits, _ = scanner.scan_literal([str(path) for path in paths], needles)
    shape = re.compile(r"(?i:IDENTIFIED BY|password)\s*[\"\x27]?\s*[:=]\s*[\"\x27][^\"\x27]{6,}")
    shape_hits = sum(bool(shape.search(path.read_text())) for path in paths)
    result = {"verdict": "PASS" if hits == shape_hits == 0 else "FAIL",
              "read_at": datetime.now(timezone.utc).isoformat(), "files": len(paths),
              "literal_hits": hits, "shape_hits": shape_hits, "needles": len(needles),
              "scanner_sha256": sha(scanner_path),
              "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths}}
    path = directory / "source-security-scan.json"
    with path.open("x") as stream:
        stream.write(encoded(result).decode())
    print(f"BOUNDED_READER_SOURCE_SCAN literal_hits={hits} shape_hits={shape_hits} files={len(paths)} receipt={path}",
          flush=True)
    reader.require(result["verdict"] == "PASS", "source security scan failed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--audit-artifacts":
        raise SystemExit(audit_artifacts(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--scan-sources":
        raise SystemExit(scan_sources(sys.argv[2]))
    raise SystemExit("use explicit pytest targets or --audit-artifacts EXTERNAL_VERIFICATION_DIRECTORY")
