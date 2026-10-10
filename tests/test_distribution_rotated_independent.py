"""Independent frozen-wheel review; never build or install.

`run` executes a bounded CPU pytest gate and preserves its external receipts.
`numerical-worker` reads an existing installation with pinned source-only
loaders. That diagnostic is NOT acceptance of an install containing bytecode.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import importlib.abc
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "8ba0dafd74057527d5d3d63b4e40b641bf3cfac3"
REVIEW_ENV = "OPENOPT_DISTRIBUTION_INDEPENDENT_EVIDENCE"
COLLECTION_ENV = "OPENOPT_DISTRIBUTION_INDEPENDENT_COLLECTION"


def pytest_collection_finish(session):
    output = os.environ.get(COLLECTION_ENV)
    if output:
        write_json(output, {"nodes": [item.nodeid for item in session.items]})


def descriptor(payload):
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_gate():
    spec = importlib.util.spec_from_file_location(
        "_distribution_independent_producer", ROOT / "scripts/check_distribution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_bytes(expression="0BSD"):
    return ("Metadata-Version: 2.4\nName: openopt\nVersion: 0.1.0\n"
            f"License-Expression: {expression}\nLicense-File: LICENSE\n\n").encode()


def archive(entries):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as wheel:
        for name, payload in entries.items():
            wheel.writestr(name, payload)
    return zipfile.ZipFile(stream)


@pytest.fixture
def byte_fixture(tmp_path):
    modules = {"openopt/__init__.py": b"REVIEW = 7\n",
               "openpo/__init__.py": b"REVIEW = 11\n"}
    license_bytes = (ROOT / "LICENSE").read_bytes()
    target, source = tmp_path / "installed", tmp_path / "src"
    for base in (target, source):
        for name, payload in modules.items():
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    metadata = target / "openopt-0.1.0.dist-info/METADATA"
    metadata.parent.mkdir()
    metadata.write_bytes(metadata_bytes())
    license_path = metadata.parent / "licenses/LICENSE"
    license_path.parent.mkdir()
    license_path.write_bytes(license_bytes)
    manifest = {"schema_version": 1,
                "source": {"commit": FROZEN, "tree": "a" * 40, "clean": True},
                "modules": {name: descriptor(payload) for name, payload in modules.items()},
                "license_expression": "0BSD", "license": descriptor(license_bytes)}
    entries = {**modules, "openopt-0.1.0.dist-info/METADATA": metadata_bytes(),
               "openopt-0.1.0.dist-info/licenses/LICENSE": license_bytes}
    gate = load_gate()
    assert gate.validate_installed_bytes(target, manifest)["modules"] == manifest["modules"]
    with archive(entries) as wheel:
        assert gate.validate_wheel_sources(wheel, source) == 2
        gate.validate_wheel_license(wheel, license_bytes)
    return gate, target, source, manifest, entries


@pytest.mark.parametrize("mutation", ["path", "missing", "extra", "sha", "license", "0BSD"])
def test_wrong_wheel_path_module_sha_or_license_rejected(byte_fixture, mutation):
    gate, _, source, _, original = byte_fixture
    entries = dict(original)
    if mutation == "path":
        entries["openopt/../outside.py"] = entries.pop("openopt/__init__.py")
    elif mutation == "missing":
        del entries["openpo/__init__.py"]
    elif mutation == "extra":
        entries["openopt/unaccepted.py"] = b"REVIEW = 13\n"
    elif mutation == "sha":
        entries["openopt/__init__.py"] = b"REVIEW = 8\n"
    elif mutation == "license":
        entries["openopt-0.1.0.dist-info/licenses/LICENSE"] = b"not the accepted license\n"
    else:
        entries["openopt-0.1.0.dist-info/METADATA"] = metadata_bytes("MIT")
    with archive(entries) as wheel, pytest.raises(ValueError):
        gate.validate_wheel_sources(wheel, source)
        gate.validate_wheel_license(wheel, (ROOT / "LICENSE").read_bytes())


@pytest.mark.parametrize("mutation", ["path", "missing", "extra", "sha", "license", "0BSD",
                                    "cached_executable"])
def test_wrong_installed_path_module_sha_or_license_rejected(byte_fixture, mutation):
    gate, target, _, manifest, _ = byte_fixture
    module = target / "openopt/__init__.py"
    if mutation == "path":
        outside = target.parent / "outside.py"
        outside.write_bytes(module.read_bytes())
        module.unlink()
        module.symlink_to(outside)
    elif mutation == "missing":
        module.unlink()
    elif mutation == "extra":
        (module.parent / "unaccepted.py").write_bytes(b"REVIEW = 13\n")
    elif mutation == "sha":
        module.write_bytes(b"REVIEW = 8\n")
    elif mutation == "license":
        (target / "openopt-0.1.0.dist-info/licenses/LICENSE").write_bytes(b"modified\n")
    elif mutation == "0BSD":
        (target / "openopt-0.1.0.dist-info/METADATA").write_bytes(metadata_bytes("MIT"))
    else:
        (module.parent / "hidden.pyc").write_bytes(b"not a source-pinned executable")
    with pytest.raises(ValueError):
        gate.validate_installed_bytes(target, manifest)


def test_wrong_manifest_caller_sha_rejected(byte_fixture, tmp_path):
    gate, _, _, manifest, _ = byte_fixture
    path = tmp_path / "source-manifest.json"
    write_json(path, manifest)
    assert gate.load_source_manifest(path, descriptor(path.read_bytes())["sha256"])[0] == manifest
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        gate.load_source_manifest(path, "0" * 64)


@pytest.mark.parametrize("mutation", ["target_path", "module_sha", "manifest_sha", "0BSD"])
def test_actual_cli_rejects_before_import_with_receipt(byte_fixture, tmp_path, mutation):
    _, target, _, manifest, _ = byte_fixture
    manifest_path, output = tmp_path / "manifest.json", tmp_path / "refusal.json"
    write_json(manifest_path, manifest)
    pin = descriptor(manifest_path.read_bytes())["sha256"]
    if mutation == "target_path":
        target = tmp_path / "wrong-installation"
    elif mutation == "module_sha":
        (target / "openopt/__init__.py").write_bytes(b"REVIEW = 8\n")
    elif mutation == "manifest_sha":
        pin = "0" * 64
    else:
        (target / "openopt-0.1.0.dist-info/METADATA").write_bytes(metadata_bytes("MIT"))
    command = [sys.executable, "-I", "-B", str(ROOT / "scripts/check_distribution.py"),
               "--installed", str(target), "--source-manifest", str(manifest_path),
               "--source-manifest-sha256", pin, "--probe-report", str(output)]
    with (tmp_path / "refusal.log").open("x") as log:
        result = subprocess.run(command, cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT,
                                timeout=30, check=False)
    observed = json.loads(output.read_bytes())
    assert result.returncode == 1 and observed["verdict"] == "FAIL"
    first_causes = {
        "target_path": "installed package missing or symlink",
        "module_sha": "installed source membership or bytes differ from caller manifest",
        "manifest_sha": "caller manifest SHA256 mismatch",
        "0BSD": "installed metadata must declare openopt and exact 0BSD license",
    }
    assert observed["error"] == "ValueError: " + first_causes[mutation]
    assert "loaded_modules" not in observed and "rotated" not in observed


@pytest.mark.parametrize("kind", ["lp", "qp"])
def test_wrong_sign_dual_rejected_independently(kind):
    import numpy as np
    import openopt

    model, x, s, z, objective = original_model(openopt, kind)
    checked = direct_kkt(model, x, s, z, objective)
    assert checked["passed"]
    assert not direct_kkt(model, x, s, -z, objective)["passed"]
    evidence = openopt.check_conic_candidate(model, x, s, -z, atol=1e-7, rtol=0.)
    assert not evidence.passed and not evidence.checks["stationarity"]
    planted = SimpleNamespace(x=x, s=s, z=-z, objective=objective, device="cpu",
                              status="reference_solved", backend="clarabel_reference",
                              evidence=SimpleNamespace(passed=True))
    with pytest.raises(ValueError, match="original-coordinate"):
        load_gate().accept_rotated_candidate(openopt, model, planted,
                                            "rsoc_" + kind, "clarabel_reference")
    np.testing.assert_array_equal(model.A @ x + s, model.b)


def original_model(api, kind):
    import numpy as np

    A = np.array([[0., 0., 1.], [-1., 0., 0.],
                  [0., -1., 0.], [0., 0., -1.]])
    b = np.array([2., 0., 0., 0.])
    P = np.zeros((3, 3)) if kind == "lp" else np.diag([1., 4., 0.])
    q = np.array([1., 2., 0.]) if kind == "lp" else np.array([0., 0., 1.])
    z = np.array([-2., 1., 2., -2.] if kind == "lp" else [-5., 2., 4., -4.])
    return (api.ConicProblem(P, q, A, b, (api.ZeroCone(1), api.RotatedSecondOrderCone(3))),
            np.array([2., 1., 2.]), np.array([0., 2., 1., 2.]), z,
            4. if kind == "lp" else 6.)


def direct_kkt(model, x, s, z, expected_objective):
    """Direct original-coordinate algebra; no producer/checker PASS is consulted."""
    import numpy as np

    arrays = (x, s, z)
    if any(value.dtype != np.float64 or not np.isfinite(value).all() for value in arrays):
        raise ValueError("independent KKT requires finite real float64")
    tol = 1e-7
    def cone(v):
        return bool(v[0] >= -tol and v[1] >= -tol
                    and 2. * v[0] * v[1] - v[2] ** 2 >= -tol)
    objective = float(.5 * x @ model.P @ x + model.q @ x)
    # Stationarity makes this quadratic dual expression valid, including P=0.
    dual_objective = float(-.5 * x @ model.P @ x - model.b @ z)
    checks = {
        "primal_equation": bool(np.max(np.abs(model.A @ x + s - model.b)) <= tol),
        "zero_slack": bool(abs(s[0]) <= tol),
        "primal_rsoc": cone(s[1:]),
        "dual_rsoc": cone(z[1:]),
        "stationarity": bool(np.max(np.abs(model.P @ x + model.q + model.A.T @ z)) <= tol),
        "complementarity": bool(abs(s @ z) <= tol),
        "analytic_x": bool(np.allclose(x, [2., 1., 2.], atol=2e-6, rtol=0.)),
        "objective": bool(abs(objective - expected_objective) <= 2e-7),
        "signed_gap": bool(abs(objective - dual_objective) <= tol),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "objective": objective, "dual_objective": dual_objective,
            "signed_gap": objective - dual_objective,
            "primal_residual": float(np.max(np.abs(model.A @ x + s - model.b))),
            "stationarity_residual": float(np.max(np.abs(model.P @ x + model.q + model.A.T @ z)))}


class PinnedSourceFinder(importlib.abc.MetaPathFinder):
    """Diagnostic only: execute pinned installed .py bytes, never existing pyc."""

    def __init__(self, target, manifest):
        self.target, self.manifest, self.loaded = target, manifest, {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in {"openopt", "openpo"}:
            return None
        relative = fullname.replace(".", "/")
        package = relative + "/__init__.py"
        relative = package if package in self.manifest["modules"] else relative + ".py"
        if relative not in self.manifest["modules"]:
            raise ImportError("project module absent from pinned manifest: " + fullname)
        filename = self.target / relative
        owner = self
        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                payload = filename.read_bytes()
                if descriptor(payload) != owner.manifest["modules"][relative]:
                    raise ImportError("installed source drift: " + fullname)
                module.__file__ = str(filename)
                module.__cached__ = None
                exec(compile(payload, str(filename), "exec"), module.__dict__)
                owner.loaded[fullname] = {"path": str(filename), **descriptor(payload)}
        return importlib.util.spec_from_file_location(
            fullname, filename, loader=Loader(),
            submodule_search_locations=[str(filename.parent)] if relative == package else None)


def installed_tree(target):
    files = {}
    for package in ("openopt", "openpo"):
        root = target / package
        if not root.is_dir() or root.is_symlink():
            raise ValueError("installed project package missing or symlink")
        for path in root.rglob("*"):
            if path.is_symlink() or not path.resolve().is_relative_to(target):
                raise ValueError("installed path escapes target")
            if path.is_file():
                files[path.relative_to(target).as_posix()] = descriptor(path.read_bytes())
    for path in target.glob("*.dist-info/**/*"):
        if path.is_symlink() or not path.resolve().is_relative_to(target):
            raise ValueError("installed metadata escapes target")
        if path.is_file():
            files[path.relative_to(target).as_posix()] = descriptor(path.read_bytes())
    return files


def existing_inputs(evidence_dir):
    evidence_dir = evidence_dir.resolve()
    report = json.loads((evidence_dir / "distribution.json").read_bytes())
    manifest_path = evidence_dir / "source-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if report["source_manifest"]["sha256"] != descriptor(manifest_path.read_bytes())["sha256"]:
        raise ValueError("historical caller source manifest pin mismatch")
    if manifest["source"]["commit"] != FROZEN or manifest["source"]["clean"] is not True:
        raise ValueError("existing install is not the frozen clean producer")
    source = report["source"]
    if manifest["source"] != {key: source[key] for key in ("commit", "tree", "clean")}:
        raise ValueError("historical source manifest/receipt identity mismatch")
    for name, pin in source["sha256"].items():
        if descriptor((ROOT / name).read_bytes())["sha256"] != pin:
            raise ValueError("reviewed full producer source differs: " + name)
    if report.get("source_consistency", {}).get("verdict") != "PASS" or report["verdict"] != "PASS":
        raise ValueError("existing producer distribution has not passed source gate")
    gate = load_gate()
    gate.validate_source_manifest(manifest)
    expected = {path.relative_to(ROOT / "src").as_posix(): descriptor(path.read_bytes())
                for package in ("openopt", "openpo") for path in (ROOT / "src" / package).rglob("*.py")}
    if not expected or expected != manifest["modules"]:
        raise ValueError("existing source membership/bytes not equal reviewed checkout")
    if descriptor((ROOT / "LICENSE").read_bytes()) != manifest["license"]:
        raise ValueError("existing 0BSD license not equal reviewed source")
    if source["sha256"]["scripts/check_distribution.py"] != descriptor(
            (ROOT / "scripts/check_distribution.py").read_bytes())["sha256"]:
        raise ValueError("reviewed producer script differs from frozen evidence")
    wheels = list(evidence_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("existing evidence must contain exactly one wheel")
    wheel_path = wheels[0]
    if wheel_path.name != report["wheel"] or descriptor(wheel_path.read_bytes())["sha256"] != report["wheel_sha256"]:
        raise ValueError("existing wheel path/SHA differs from producer receipt")
    installed_receipt_path = evidence_dir / "installed-probe.json"
    if descriptor(installed_receipt_path.read_bytes()) != {
            key: report["installed_probe"][key] for key in ("sha256", "bytes")}:
        raise ValueError("historical installed receipt byte pin mismatch")
    installed_receipt = json.loads(installed_receipt_path.read_bytes())
    if installed_receipt["verdict"] != "PASS" or installed_receipt["source"] != manifest["source"]:
        raise ValueError("historical installed receipt source mismatch")
    with zipfile.ZipFile(wheel_path) as wheel:
        gate.validate_wheel_sources(wheel, ROOT / "src")
        gate.validate_wheel_license(wheel, (ROOT / "LICENSE").read_bytes())
        if {name: descriptor(wheel.read(name)) for name in manifest["modules"]} != expected:
            raise ValueError("wheel sources differ from reviewed manifest")
    target = (evidence_dir / "installed").resolve()
    before = installed_tree(target)
    if {name: value for name, value in before.items() if name.endswith(".py")} != expected:
        raise ValueError("existing installed Python set/bytes differ")
    metadata = list(target.glob("*.dist-info/METADATA"))
    if len(metadata) != 1:
        raise ValueError("existing install METADATA membership mismatch")
    parsed = BytesParser().parsebytes(metadata[0].read_bytes())
    if (parsed.get_all("Name") != ["openopt"] or parsed.get_all("License-Expression") != ["0BSD"]
            or parsed.get_all("License-File") != ["LICENSE"]
            or descriptor((metadata[0].parent / "licenses/LICENSE").read_bytes()) != manifest["license"]):
        raise ValueError("existing installed 0BSD mismatch")
    return target, manifest, {"source": manifest["source"], "wheel": {
        "path": str(wheel_path), **descriptor(wheel_path.read_bytes())},
        "source_manifest": {"path": str(manifest_path), **descriptor(manifest_path.read_bytes())},
        "historical_distribution_verdict": report["verdict"],
        "installed_files_before": before}


def numerical_worker(evidence_dir, output):
    import numpy as np
    import torch

    receipt = {"verdict": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
               "qualification": "source-only installed CPU diagnostic; not wheel acceptance",
               "cuda_execution": "UNVERIFIED", "commercial": "UNVERIFIED"}
    try:
        target, manifest, inputs = existing_inputs(evidence_dir)
        receipt.update(inputs)
        if any(name.split(".")[0] in {"openopt", "openpo"} for name in sys.modules):
            raise ValueError("diagnostic requires fresh process")
        finder = PinnedSourceFinder(target, manifest)
        sys.meta_path.insert(0, finder)
        import openopt
        import openpo

        assert all(getattr(openopt, name) is getattr(openpo, name) for name in openpo.__all__)
        rows = {}
        for kind in ("lp", "qp"):
            model, _, _, analytic_z, objective = original_model(openopt, kind)
            frozen = {name: getattr(model, name).copy() for name in ("P", "q", "A", "b")}
            results = [
                ("native", openopt.solve_native_conic(model, openopt.NativeConicConfig(
                    enabled=True, device="cpu", dtype=torch.float64, eps_abs=1e-9, eps_rel=0.))),
                ("clarabel_reference", openopt.solve_conic(model, openopt.ConicConfig(
                    backend="clarabel_reference", device="cpu", eps_abs=1e-10, eps_rel=1e-10))),
            ]
            for backend, solved in results:
                assert solved.status == ("solved" if backend == "native" else "reference_solved")
                assert solved.backend == ("native_torch_proximal_admm" if backend == "native" else backend)
                assert solved.device == "cpu"
                arrays = []
                for value in (solved.x, solved.s, solved.z):
                    if backend == "native":
                        assert isinstance(value, torch.Tensor)
                        assert value.device == torch.device("cpu") and value.dtype == torch.float64
                        value = value.detach().numpy()
                    assert isinstance(value, np.ndarray) and value.dtype == np.float64
                    arrays.append(value.copy())
                x, s, z = arrays
                algebra = direct_kkt(model, x, s, z, objective)
                assert algebra["passed"], algebra
                assert abs(solved.objective - algebra["objective"]) <= 1e-12
                original = openopt.check_conic_candidate(model, x, s, z, atol=1e-7, rtol=0.)
                assert original.passed
                wrong = openopt.check_conic_candidate(model, x, s, -z, atol=1e-7, rtol=0.)
                assert not wrong.passed and not direct_kkt(model, x, s, -z, objective)["passed"]
                assert all(np.array_equal(getattr(model, name), value) for name, value in frozen.items())
                rows[kind + ":" + backend] = {"x": x.tolist(), "s": s.tolist(), "z": z.tolist(),
                    "status": solved.status, "backend": solved.backend, "device": solved.device,
                    "direct_kkt": algebra,
                    "analytic_dual_max_error": float(np.max(np.abs(z - analytic_z))),
                    "installed_original_checker": original.to_dict(),
                    "wrong_sign_dual": wrong.to_dict()}
        receipt["cases"] = rows
        receipt["loaded_modules"] = finder.loaded
        receipt["actual_solves"] = {"native": 2, "clarabel_reference": 2, "commercial": 0}
        after = installed_tree(target)
        if after != inputs["installed_files_before"]:
            raise ValueError("installed files changed during diagnostic")
        receipt["installed_files_after"] = after
        receipt["loaded_bytes_verified"] = all(
            manifest["modules"][Path(row["path"]).relative_to(target).as_posix()]
            == {key: row[key] for key in ("sha256", "bytes")} for row in finder.loaded.values())
        assert receipt["loaded_bytes_verified"] and finder.loaded
        receipt["verdict"] = "PASS"
    except Exception as error:
        receipt.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output, receipt)


def test_actual_existing_wheel_bytes_and_installed_rsoc_lp_qp(tmp_path, request):
    evidence = os.environ.get(REVIEW_ENV)
    if not evidence:
        pytest.skip("explicit existing installation required; no build/install permitted")
    output = tmp_path / "installed-numerical.json"
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
               "numerical-worker", "--evidence-dir", evidence, "--output", str(output)]
    with (tmp_path / "numerical-worker.log").open("x") as log:
        result = subprocess.run(command, cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT,
                                timeout=90, check=False)
    receipt = json.loads(output.read_bytes())
    request.node.user_properties.extend([("installed_numerical_receipt", str(output)),
        ("installed_numerical_sha256", descriptor(output.read_bytes())["sha256"])])
    assert result.returncode == 0, receipt
    assert receipt["verdict"] == "PASS" and receipt["loaded_bytes_verified"]
    assert receipt["actual_solves"] == {"native": 2, "clarabel_reference": 2, "commercial": 0}
    assert len(receipt["cases"]) == 4 and receipt["loaded_modules"]
    assert receipt["installed_files_before"] == receipt["installed_files_after"]


def test_existing_install_standard_gate_actual_pass_and_loaded_bytes(tmp_path, request):
    evidence = os.environ.get(REVIEW_ENV)
    if not evidence:
        pytest.skip("explicit existing installation required; no build/install permitted")
    target, manifest, _ = existing_inputs(Path(evidence))
    output = tmp_path / "standard-producer.json"
    manifest_path = Path(evidence) / "source-manifest.json"
    command = [sys.executable, "-I", "-B", str(ROOT / "scripts/check_distribution.py"),
        "--installed", str(target), "--source-manifest", str(manifest_path),
        "--source-manifest-sha256", descriptor(manifest_path.read_bytes())["sha256"],
        "--probe-report", str(output)]
    before = installed_tree(target)
    with (tmp_path / "standard-producer.log").open("x") as log:
        result = subprocess.run(command, cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT,
                                timeout=60, check=False)
    receipt = json.loads(output.read_bytes())
    request.node.user_properties.extend([("standard_producer_receipt", str(output)),
        ("standard_producer_sha256", descriptor(output.read_bytes())["sha256"])])
    assert result.returncode == 0 and receipt["verdict"] == "PASS", receipt
    assert receipt["provenance_verdict"] == "PASS" and receipt["source"] == manifest["source"]
    assert receipt["rotated"]["actual_solves"] == {"native": 2, "clarabel_reference": 2, "commercial": 0}
    assert receipt["loaded_modules"] and receipt["byte_validation"]["modules"] == manifest["modules"]
    for loaded in receipt["loaded_modules"].values():
        path = Path(loaded["path"]).resolve()
        assert path.is_relative_to(target)
        assert descriptor(path.read_bytes()) == {key: loaded[key] for key in ("sha256", "bytes")}
        assert descriptor(path.read_bytes()) == manifest["modules"][path.relative_to(target).as_posix()]
    for row in receipt["rotated"]["cases"].values():
        assert len(row["rejected_controls"]) == 3
        assert all(not control["passed"] for control in row["rejected_controls"].values())
        for solved in row["solves"].values():
            assert len(solved["evidence"]["checks"]) == 9
            assert all(solved["evidence"]["checks"].values())
    assert not any(name.endswith((".pyc", ".pyo", ".so", ".pyd")) for name in before)
    assert receipt["source_manifest"]["sha256"] == descriptor(manifest_path.read_bytes())["sha256"]
    assert installed_tree(target) == before
    assert manifest["source"]["commit"] == FROZEN


def run_review(args):
    from xml.etree import ElementTree

    sys.path.insert(0, str(ROOT))
    from scripts import verify, verify_sharded

    base = args.artifact_dir.resolve()
    if base.is_relative_to(ROOT) or not base.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve()):
        raise ValueError("review artifacts must be external 3FS tmp")
    base.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="review-", dir=base))
    env = verify_sharded.cpu_env(ROOT, artifacts)
    env[REVIEW_ENV] = str(args.evidence_dir.resolve())
    receipt = {"verdict": "RUNNING", "producer_commit": FROZEN,
               "started_at": datetime.now(timezone.utc).isoformat(),
               "evidence_dir": env[REVIEW_ENV], "artifact_dir": str(artifacts),
               "qualification": "independent CPU review, no install/CUDA/commercial claim"}
    rc = 1
    try:
        before = verify.source_snapshot(ROOT)
        receipt["source"] = before
        collection_dir = artifacts / "collection"
        collection_dir.mkdir()
        collection_path = collection_dir / "nodes.json"
        collection_env = {**env, COLLECTION_ENV: str(collection_path)}
        verify_sharded.command_logged(
            [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q",
             "-p", "tests.test_distribution_rotated_independent",
             "-o", f"cache_dir={collection_dir / 'pytest-cache'}",
             "--basetemp", str(collection_dir / "pytest-tmp"),
             "tests/test_distribution_rotated_independent.py"],
            ROOT, collection_dir, 30, collection_env)
        nodes = json.loads(collection_path.read_bytes())["nodes"]
        if not nodes or len(nodes) != len(set(nodes)):
            raise ValueError("independent collection empty or duplicate")
        receipt["collection"] = {"path": str(collection_path), **descriptor(collection_path.read_bytes())}
        command = [sys.executable, "-B", str(ROOT / "scripts/verify.py"), "tests",
                   "--target", "tests/test_distribution_rotated_independent.py",
                   "--artifact-dir", str(artifacts / "verification")]
        verify_sharded.command_logged(command, ROOT, artifacts, 179, env)
        reports = list((artifacts / "verification").glob("tests-*/verification.json"))
        if len(reports) != 1:
            raise ValueError("independent verification receipt membership mismatch")
        report_path = reports[0]
        report = json.loads(report_path.read_bytes())
        receipt["verification"] = {"path": str(report_path), **descriptor(report_path.read_bytes())}
        if report["verdict"] != "PASS":
            raise ValueError("independent verification did not pass")
        cases = ElementTree.parse(report_path.parent / "pytest.xml").findall(".//testcase")
        if not cases or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
            raise ValueError("independent review requires nonempty, zero-skip complete tests")
        expected = Counter(verify_sharded.junit_key(node) for node in nodes)
        actual = Counter((case.attrib["classname"], case.attrib["name"]) for case in cases)
        if expected != actual or any(count != 1 for count in actual.values()):
            raise ValueError("independent exact collection/JUnit membership mismatch")
        receipt["junit"] = {"path": str(report_path.parent / "pytest.xml"),
            **descriptor((report_path.parent / "pytest.xml").read_bytes()),
            "missing": [], "extra": [], "duplicates": []}
        receipt["tests"] = len(cases)
        properties = {prop.attrib["name"]: prop.attrib["value"]
                      for prop in ElementTree.parse(report_path.parent / "pytest.xml").findall(".//property")}
        for label in ("installed_numerical", "standard_producer"):
            path = Path(properties[label + "_receipt"])
            if descriptor(path.read_bytes())["sha256"] != properties[label + "_sha256"]:
                raise ValueError("independent evidence receipt pin mismatch")
            receipt[label] = {"path": str(path), **descriptor(path.read_bytes())}
        numerical = json.loads(Path(receipt["installed_numerical"]["path"]).read_bytes())
        standard = json.loads(Path(receipt["standard_producer"]["path"]).read_bytes())
        receipt["source_consistency"] = verify.check_source_consistency(before, verify.source_snapshot(ROOT))
        assert receipt["source_consistency"]["verdict"] == "PASS"
        receipt["actual_solves"] = numerical["actual_solves"]
        receipt["standard_producer_actual_solves"] = standard["rotated"]["actual_solves"]
        assert numerical["verdict"] == standard["verdict"] == "PASS"
        receipt.update(verdict="PASS", accept_distribution=True,
                       distribution_verdict=standard["verdict"], first_cause=None)
        rc = 0
    except Exception as error:
        receipt.update(verdict="FAIL", accept_distribution=False, error=f"{type(error).__name__}: {error}")
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(artifacts / "review.json", receipt)
        print(f"DISTRIBUTION_INDEPENDENT verdict={receipt['verdict']} "
              f"accept_distribution={receipt.get('accept_distribution', False)} "
              f"report={artifacts / 'review.json'}", flush=True)
    return rc


def scan_review(args):
    base = args.artifact_dir.resolve()
    if base.is_relative_to(ROOT) or not base.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve()):
        raise ValueError("scan artifacts must be external 3FS tmp")
    base.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="path-scan-", dir=base))
    path = Path("/data/dev/infra/ops/h3c-gpu-cluster/repair-20261010/retention/secret_scan.py")
    spec = importlib.util.spec_from_file_location("_distribution_review_path_scan", path)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    controls = scanner.selftest(artifacts)
    names = ("tests/test_distribution_rotated_independent.py",
             "docs/research/distribution-rotated-independent.md")
    report = scanner.scan_snapshot(ROOT, names, [])
    report.update(controls=controls, scanner={"path": str(path), **descriptor(path.read_bytes())},
                  scope="exact two review paths; seven shapes; no credential reads",
                  known_credential_literal_scan="NOT_PERFORMED",
                  finished_at=datetime.now(timezone.utc).isoformat())
    write_json(artifacts / "scan.json", report)
    print(f"DISTRIBUTION_PATH_SCAN verdict={report['verdict']} "
          f"files={report['files_scanned']} hits={report['hits']} report={artifacts / 'scan.json'}")
    return 0 if report["verdict"] == "CLEAN" and report["hits"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--artifact-dir", type=Path, required=True)
    run.add_argument("--evidence-dir", type=Path, required=True)
    worker = commands.add_parser("numerical-worker")
    worker.add_argument("--evidence-dir", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    scan = commands.add_parser("scan")
    scan.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        return run_review(args)
    if args.command == "scan":
        return scan_review(args)
    numerical_worker(args.evidence_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
