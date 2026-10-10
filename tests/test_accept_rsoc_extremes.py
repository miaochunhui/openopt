"""Self-contained CPU audit fixtures; synthetic archives are not GPU evidence."""

import copy
from decimal import localcontext
import hashlib
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET

import pytest

from scripts import accept_rsoc_extremes as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


def pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fabricated_sidecar(case, dual, sources):
    """Only synthesize test fixtures, with explicit fake operation observations."""
    values = list(audit.cases()[case])
    dtype = "torch.float64" if case.startswith("f64-") else "torch.float32"
    with localcontext() as context:
        context.prec = 1100
        reference, opposite = audit.projection(values), audit.projection([-x for x in values])
        positive = [float(x) if dtype == "torch.float64" else audit.f32(float(x)) for x in reference]
        negative = [float(x) if dtype == "torch.float64" else audit.f32(float(x)) for x in opposite]
    a, b, *tail = map(audit.Fraction.from_float, values)
    excess = sum(x * x for x in tail) - 2 * a * b

    def trace(original):
        aa, bb, *tt = map(audit.Fraction.from_float, original)
        delta = sum(x * x for x in tt) - 2 * aa * bb
        early = (aa >= 0 and bb >= 0 or aa <= 0 and bb <= 0) and delta <= 0
        return {"device": "cuda:0", "tensor_count": 1,
                "dtypes": sorted({dtype, "torch.float64"}),
                "operations": {"aten.linalg_vector_norm.default": 1 if early else 2,
                               "aten.frexp.Tensor": 1}}

    return {
        "schema_version": 1, "case": case,
        "nodeid": f"{audit.TEST}::{audit.FUNCTION}[{case}-{'dual' if dual else 'primal'}]",
        "verdict": "CUDA_NUMERICAL_REGRESSION_PASS",
        "qualification": "projection regression only; no commercial, performance or remote attestation",
        "requested_device": "cuda:0", "observed_device": "cuda:0", "dtype": dtype,
        "gpu_name": "NVIDIA RTX PRO 4500 Blackwell",
        "input_hex": [x.hex() for x in values], "positive_hex": [x.hex() for x in positive],
        "negative_hex": [x.hex() for x in negative],
        "expected_decimal": list(map(str, reference)),
        "negative_expected_decimal": list(map(str, opposite)),
        "exact_excess": {"numerator": str(excess.numerator), "denominator": str(excess.denominator)},
        "dual": dual, "source_sha256": {name: sources[name] for name in audit.SOURCE_PATHS},
        "trace": {"positive": trace(values), "negative": trace([-x for x in values])},
        "original_coordinate_check": True, "scaled_moreau_check": True,
    }


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.execution, "ARTIFACT_ROOT", tmp_path)
    root, current = tmp_path / "run", tmp_path / "trusted"
    source = root / "source"
    names = set(audit.SOURCE_PATHS) | {audit.execution.RUNNER}
    names |= {path.relative_to(audit.ROOT).as_posix()
              for path in (audit.ROOT / "src/openopt").rglob("*.py")}
    for name in sorted(names):
        for destination in (source, current):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audit.ROOT / name, target)
    sources = audit.execution.source_files(source)
    write(source / "source-hashes.json", sources)
    output = root / "case-0"
    result_dir = output / "cuda-check-synthetic"
    result_dir.mkdir(parents=True)
    job_name, pod_name = "openopt-synthetic", "openopt-synthetic-pod"
    env = [
        {"name": "PYTHONPATH", "value": str(source / "src")},
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "OPENOPT_NODE", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        {"name": "TMPDIR", "value": str(output)},
        *[{"name": name, "value": "4"} for name in
          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")],
    ]
    resources = {"cpu": "4", "memory": "12Gi", "nvidia.com/gpu": 1}
    container = {
        "name": "benchmark", "image": "synthetic-only:never-executed",
        "command": ["python3", str(source / audit.execution.RUNNER), "--artifact-dir",
                    str(output), "--target", f"{audit.TEST}::{audit.FUNCTION}"],
        "env": env, "resources": {"requests": resources.copy(), "limits": resources.copy()},
    }
    pod_spec = {"restartPolicy": "Never", "containers": [container]}
    labels = {"repo": "openopt", "owner": "infra-openopt"}
    manifest = {
        "kind": "Job", "metadata": {"name": job_name, "namespace": "synthetic", "labels": labels},
        "spec": {"backoffLimit": 0, "template": {"spec": pod_spec}},
    }
    job = copy.deepcopy(manifest)
    job["metadata"]["uid"] = "synthetic-job-uid"
    job["status"] = {"succeeded": 1, "conditions": [{"type": "Complete", "status": "True"}]}
    pod = {
        "metadata": {"name": pod_name, "uid": "synthetic-pod-uid", "labels": labels,
                     "ownerReferences": [{"name": job_name, "uid": "synthetic-job-uid", "controller": True}]},
        "spec": {**copy.deepcopy(pod_spec), "nodeName": "synthetic-node"},
        "status": {"phase": "Succeeded", "containerStatuses": [{
            "name": "benchmark", "restartCount": 0, "imageID": "synthetic@sha256:" + "a" * 64,
            "state": {"terminated": {"exitCode": 0}},
        }]},
    }
    write(root / "job-0.json", manifest)
    status = write(root / f"{job_name}-status.json", {"job": job, "pods": {"items": [pod]}})
    run = write(root / "run.json", {
        "stage": "release-regression", "state": "SUBMITTED", "source_sha": "a" * 40,
        "source_identity": {"clean": True, "commit": "a" * 40, "sha256": sources},
        "namespace": "synthetic", "source_hashes": sources,
        "jobs": [{"job": job_name, "output": str(output), "manifest": str(root / "job-0.json")}],
    })
    expected_sources = {
        name: digest for name, digest in sources.items()
        if name.startswith("src/openopt/") and "/" not in name[len("src/openopt/"):]
    }
    expected_sources.update({name: sources[name] for name in (audit.TEST, audit.execution.RUNNER)})
    report = {
        "verdict": "PASS", "returncode": 0, "targets": [f"{audit.TEST}::{audit.FUNCTION}"],
        "selection": "", "source_sha256": expected_sources,
        "tests": {"tests": 76, "failures": 0, "errors": 0, "skipped": 0},
        "hardware": {"node": "synthetic-node", "gpu": "NVIDIA RTX PRO 4500 Blackwell",
                     "torch": "synthetic", "cuda": "synthetic", "cpu_quota": "synthetic"},
        "telemetry_samples": 1,
    }
    result = write(result_dir / "result.json", report)
    write(output / "result.json", report)
    marker = f"OPENOPT_RELEASE_CUDA verdict=PASS tests=76 evidence={result_dir}"
    (result_dir / "execution.log").write_text(marker + "\n")
    (root / f"{pod_name}.log").write_text(marker + "\n")
    (result_dir / "pytest.log").write_text("76 passed in 1s\nreturncode=0\n")
    write(result_dir / "telemetry.json", [{"rc": 0, "gpu": "GPU-synthetic", "cpu": {"usage_usec": 1}}])
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite", {key: str(value) for key, value in report["tests"].items()})
    sidecars = []
    for index, (identity, (case, dual)) in enumerate(audit.expected_identities().items()):
        sidecar = write(result_dir / "pytest-tmp" / f"case-{index}" / "rotated-conic-extreme.json",
                        fabricated_sidecar(case, dual, sources))
        sidecars.append(sidecar)
        member = ET.SubElement(suite, "testcase", {
            "classname": identity.split("::")[0], "name": identity.split("::")[1], "time": "0.01",
        })
        properties = ET.SubElement(member, "properties")
        for name, value in (("rsoc_cuda_extremes_path", str(sidecar)),
                            ("rsoc_cuda_extremes_sha256", pin(sidecar))):
            ET.SubElement(properties, "property", {"name": name, "value": value})
    junit = result_dir / "pytest.xml"
    ET.ElementTree(suites).write(junit, encoding="utf-8", xml_declaration=True)
    return {"root": root, "current": current, "result": result, "run": run,
            "junit": junit, "status": status, "sidecars": sidecars, "sources": sources}


def invocation(archive):
    result = {"current_root": archive["current"]}
    for name in ("result", "run", "junit", "status"):
        result[name] = archive[name]
        result[name + "_sha256"] = pin(archive[name])
    return result


def test_synthetic_archive_is_complete_but_not_remote_attestation(archive):
    result = audit.accept(**invocation(archive))
    assert result["verdict"] == "RSOC_EXTREMES_REPLAY_PASS"
    assert len(result["members"]) == result["tests"]["tests"] == 76
    assert not result["qualified"] and result["remote_attestation"] == "NOT_PROVIDED"
    assert result["execution"]["node"] == "synthetic-node"
    assert archive["root"].is_dir()


@pytest.mark.parametrize("mutation", [
    "lost-head", "input", "orientation", "device", "source", "oracle",
    "fraction", "trace", "qualification", "missing", "nonfinite",
])
def test_rehashed_sidecar_corruption_is_rejected(archive, mutation):
    path = archive["sidecars"][0]
    original = json.loads(path.read_bytes())
    identity = next(iter(audit.expected_identities()))
    assert audit.replay(original, identity, archive["sources"], original["gpu_name"])["verdict"]
    changed = copy.deepcopy(original)
    if mutation == "lost-head":
        changed["positive_hex"][0] = 0.0.hex()
    elif mutation == "input":
        changed["input_hex"][-1] = 2.0.hex()
    elif mutation == "orientation":
        changed["dual"] = not changed["dual"]
    elif mutation == "device":
        changed["observed_device"] = "cpu"
    elif mutation == "source":
        changed["source_sha256"][audit.TEST] = "0" * 64
    elif mutation == "oracle":
        changed["expected_decimal"][0] = "0"
    elif mutation == "fraction":
        changed["exact_excess"]["numerator"] = "0"
    elif mutation == "trace":
        changed["trace"]["positive"]["operations"]["aten.linalg_vector_norm.default"] = 1
    elif mutation == "qualification":
        changed["qualification"] = "commercial-qualified"
    elif mutation == "missing":
        changed.pop("negative_hex")
    else:
        changed["positive_hex"][0] = "inf"
    write(path, changed)
    assert pin(path) != hashlib.sha256((json.dumps(original, indent=2) + "\n").encode()).hexdigest()
    with pytest.raises((ValueError, OverflowError)):
        audit.replay(json.loads(path.read_bytes()), identity, archive["sources"], original["gpu_name"])


@pytest.mark.parametrize("mutation", [
    "args", "env-value", "env-extra", "env-duplicate", "env-from", "restarted", "owner",
])
def test_coordinated_observed_status_mutations_rejected(archive, mutation):
    values = invocation(archive)
    assert audit.accept(**values)["verdict"] == "RSOC_EXTREMES_REPLAY_PASS"
    status = json.loads(archive["status"].read_bytes())
    specs = (status["job"]["spec"]["template"]["spec"], status["pods"]["items"][0]["spec"])
    for spec in specs:
        container = spec["containers"][0]
        if mutation == "args":
            container["args"] = ["--unsubmitted"]
        elif mutation == "env-value":
            container["env"][0]["value"] = "/synthetic-substitution"
        elif mutation == "env-extra":
            container["env"].append({"name": "PYTEST_ADDOPTS", "value": "-k empty"})
        elif mutation == "env-duplicate":
            container["env"].append(copy.deepcopy(container["env"][0]))
        elif mutation == "env-from":
            container["envFrom"] = [{"configMapRef": {"name": "unsubmitted"}}]
    if mutation == "restarted":
        status["pods"]["items"][0]["status"]["containerStatuses"][0]["restartCount"] = 1
    if mutation == "owner":
        status["pods"]["items"][0]["metadata"]["ownerReferences"][0]["uid"] = "wrong"
    write(archive["status"], status)
    values["status_sha256"] = pin(archive["status"])
    with pytest.raises(ValueError):
        audit.accept(**values)


def test_caller_hash_and_exact_denominator_required(archive):
    values = invocation(archive)
    values["run_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="caller pin"):
        audit.accept(**values)
    tree = ET.parse(archive["junit"])
    member = tree.getroot().find("testsuite/testcase")
    member.set("name", "test_unsubmitted")
    tree.write(archive["junit"], encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError):
        audit.accept(**invocation(archive))


def test_original_decimal_oracle_known_solution_and_domain():
    with localcontext() as context:
        context.prec = 1100
        projected = audit.projection([0., 0., 4.])
        assert projected[2] == 2
        assert abs(projected[0] - audit.Decimal(2).sqrt()) < audit.Decimal("1e-1090")
    assert len(audit.cases()) == 38 and len(audit.expected_identities()) == 76
    assert audit.cases()["f64-remote-unit-last"][0] == -2.0 ** 600
    assert audit.cases()["f32-remote-unit-last"][0] == -2.0 ** 100
