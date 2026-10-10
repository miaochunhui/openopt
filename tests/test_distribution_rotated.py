"""Self-contained installed-byte controls and real original-model CPU solves."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import openopt
from scripts import check_distribution as gate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_bytes(tmp_path):
    target = tmp_path / "installed"
    modules = {"openopt/__init__.py": b"# isolated openopt\n", "openpo/__init__.py": b"# isolated openpo\n"}
    for name, payload in modules.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    metadata = target / "openopt-0.1.0.dist-info/METADATA"
    metadata.parent.mkdir()
    metadata.write_bytes(b"Metadata-Version: 2.4\nName: openopt\nVersion: 0.1.0\nLicense-Expression: 0BSD\nLicense-File: LICENSE\n\n")
    license_path = metadata.parent / "licenses/LICENSE"
    license_path.parent.mkdir()
    license_bytes = (ROOT / "LICENSE").read_bytes()
    license_path.write_bytes(license_bytes)
    manifest = {"schema_version": 1, "source": {"commit": "1" * 40, "tree": "2" * 40, "clean": True},
                "modules": {name: gate.byte_descriptor(payload) for name, payload in modules.items()},
                "license_expression": "0BSD", "license": gate.byte_descriptor(license_bytes)}
    return target, manifest, metadata, license_path


def test_installed_source_and_license_exact_bytes(isolated_bytes):
    target, manifest, _, _ = isolated_bytes
    report = gate.validate_installed_bytes(target, manifest)
    assert report["modules"] == manifest["modules"]
    assert report["license"] == manifest["license"]
    assert report["license_expression"] == "0BSD"


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "license", "license_expression", "duplicate_license", "duplicate_metadata", "symlink", "bytecode", "native_binary"])
def test_installed_bytes_tamper_rejected(isolated_bytes, mutation):
    target, manifest, metadata, license_path = isolated_bytes
    source = target / "openopt/__init__.py"
    if mutation == "changed":
        source.write_bytes(b"# changed installed bytes\n")
    elif mutation == "missing":
        source.unlink()
    elif mutation == "extra":
        (source.parent / "extra.py").write_bytes(b"# extra\n")
    elif mutation == "license":
        license_path.write_bytes(b"changed license\n")
    elif mutation == "license_expression":
        metadata.write_bytes(metadata.read_bytes().replace(b"0BSD", b"MIT"))
    elif mutation == "duplicate_license":
        metadata.write_bytes(metadata.read_bytes().replace(b"\n\n", b"\nLicense-Expression: MIT\n\n"))
    elif mutation == "duplicate_metadata":
        other = target / "other.dist-info/METADATA"
        other.parent.mkdir()
        other.write_bytes(metadata.read_bytes())
    elif mutation in {"bytecode", "native_binary"}:
        suffix = ".pyc" if mutation == "bytecode" else ".so"
        (source.parent / ("hidden" + suffix)).write_bytes(b"unpinned executable")
    else:
        outside = target.parent / "outside.py"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
    with pytest.raises(ValueError):
        gate.validate_installed_bytes(target, manifest)


@pytest.mark.parametrize("mutation", ["unknown", "bool_version", "empty", "traversal", "bool_bytes", "bad_hash", "bad_commit", "wrong_license"])
def test_manifest_schema_strict(isolated_bytes, mutation):
    _, original, _, _ = isolated_bytes
    manifest = deepcopy(original)
    if mutation == "unknown":
        manifest["unchecked"] = True
    elif mutation == "bool_version":
        manifest["schema_version"] = True
    elif mutation == "empty":
        manifest["modules"] = {}
    elif mutation == "traversal":
        manifest["modules"]["openopt/../outside.py"] = manifest["modules"]["openopt/__init__.py"]
    elif mutation == "bool_bytes":
        manifest["license"]["bytes"] = True
    elif mutation == "bad_hash":
        manifest["license"]["sha256"] = "not-a-sha"
    elif mutation == "bad_commit":
        manifest["source"]["commit"] = "unknown"
    else:
        manifest["license_expression"] = "MIT"
    with pytest.raises(ValueError):
        gate.validate_source_manifest(manifest)


def test_manifest_caller_pin_and_json_fail_loud(isolated_bytes, tmp_path):
    _, manifest, _, _ = isolated_bytes
    path = tmp_path / "manifest.json"
    payload = json.dumps(manifest).encode()
    path.write_bytes(payload)
    actual, descriptor = gate.load_source_manifest(path, gate.byte_descriptor(payload)["sha256"])
    assert actual == manifest and descriptor["bytes"] == len(payload)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        gate.load_source_manifest(path, "0" * 64)
    for invalid in (b"{}", b"{\"schema_version\":1,\"schema_version\":1}", b"{\"schema_version\":NaN}"):
        path.write_bytes(invalid)
        with pytest.raises(ValueError):
            gate.load_source_manifest(path, gate.byte_descriptor(invalid)["sha256"])


def test_frozen_source_manifest_recomputable_and_detects_change():
    from scripts.verify import source_snapshot

    snapshot = source_snapshot(ROOT)
    manifest = gate.make_source_manifest(ROOT, snapshot)
    assert manifest["modules"]["openopt/conic.py"] == gate.byte_descriptor((ROOT / "src/openopt/conic.py").read_bytes())
    assert manifest["license"] == gate.byte_descriptor((ROOT / "LICENSE").read_bytes())
    damaged = deepcopy(snapshot)
    damaged["sha256"]["src/openopt/conic.py"] = "0" * 64
    with pytest.raises(ValueError, match="freeze"):
        gate.make_source_manifest(ROOT, damaged)


def test_cached_checkout_import_cannot_pass_installed_probe(isolated_bytes):
    target, manifest, _, _ = isolated_bytes
    with pytest.raises(ValueError, match="fresh process"):
        gate.installed_probe(target, manifest)


def test_legacy_installed_cli_still_accepted_but_missing_packages_fail(tmp_path):
    result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/check_distribution.py"),
                             "--installed", str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "usage:" not in result.stderr
    assert "provenance=UNVERIFIED" in result.stdout


def test_manifest_path_without_pin_still_fails_loud(tmp_path):
    result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/check_distribution.py"),
                             "--installed", str(tmp_path), "--source-manifest", str(tmp_path / "manifest.json")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "caller SHA256" in result.stderr


@pytest.mark.parametrize("kind", ["rsoc_lp", "rsoc_qp"])
def test_original_model_exact_analytic_kkt(kind):
    model = gate.rotated_models(openopt)[kind]
    assert model.cones[1] == openopt.RotatedSecondOrderCone(3)
    x, s = np.array([2., 1., 2.]), np.array([0., 2., 1., 2.])
    z = np.array([-2., 1., 2., -2.] if kind == "rsoc_lp" else [-5., 2., 4., -4.])
    np.testing.assert_array_equal(model.A @ x + s, model.b)
    np.testing.assert_array_equal(model.P @ x + model.q + model.A.T @ z, np.zeros(3))
    assert 2 * s[1] * s[2] == s[3] ** 2
    assert 2 * z[1] * z[2] == z[3] ** 2
    assert s @ z == 0.
    evidence = openopt.check_conic_candidate(model, x, s, z, atol=1e-12, rtol=0.)
    assert evidence.passed and evidence.metrics["duality_gap"] == 0.
    assert evidence.metrics["primal_objective"] == {"rsoc_lp": 4., "rsoc_qp": 6.}[kind]


def test_real_four_cpu_solves_original_checker_and_receipt(tmp_path, request):
    result = gate.rotated_probe(openopt)
    assert result["actual_solves"] == {"native": 2, "clarabel_reference": 2, "commercial": 0}
    assert result["cuda_execution"] == "UNVERIFIED"
    for row in result["cases"].values():
        assert row["analytic_evidence"]["passed"]
        assert all(not control["passed"] for control in row["rejected_controls"].values())
        for solved in row["solves"].values():
            assert solved["evidence"]["passed"] and solved["actual_device"] == "cpu"
    path = tmp_path / "rotated-cpu-probe.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    request.node.user_properties.extend([("rotated_cpu_receipt", str(path)),
                                        ("rotated_cpu_receipt_sha256", gate.byte_descriptor(path.read_bytes())["sha256"])])


@pytest.mark.parametrize("mutation", ["wrong_x", "wrong_dual", "planted_slack", "status", "device", "backend", "objective", "nonfinite", "float32"])
def test_solver_embedded_pass_cannot_replace_original_checker(mutation):
    model = gate.rotated_models(openopt)["rsoc_lp"]
    x, s, z = torch.tensor([2., 1., 2.], dtype=torch.float64), torch.tensor([0., 2., 1., 2.], dtype=torch.float64), torch.tensor([-2., 1., 2., -2.], dtype=torch.float64)
    candidate = SimpleNamespace(x=x, s=s, z=z, status="solved", device="cpu",
                                backend="native_torch_proximal_admm", objective=4., evidence=SimpleNamespace(passed=True))
    if mutation == "wrong_x":
        candidate.x = x + 1.
    elif mutation == "wrong_dual":
        candidate.z = -z
    elif mutation == "planted_slack":
        candidate.x = torch.tensor([0., 0., 2.], dtype=torch.float64)
    elif mutation == "status":
        candidate.status = "max_iterations"
    elif mutation == "device":
        candidate.device = "cuda:0"
    elif mutation == "backend":
        candidate.backend = "external_fake"
    elif mutation == "objective":
        candidate.objective = 0.
    elif mutation == "nonfinite":
        candidate.z = z * float("nan")
    else:
        candidate.x = x.float()
    with pytest.raises(ValueError):
        gate.accept_rotated_candidate(openopt, model, candidate, "rsoc_lp", "native")


def test_gap_report_unavailable_or_tampered_rejected(monkeypatch):
    model = gate.rotated_models(openopt)["rsoc_lp"]
    candidate = SimpleNamespace(x=np.array([2., 1., 2.]), s=np.array([0., 2., 1., 2.]),
                                z=np.array([-2., 1., 2., -2.]), status="reference_solved", device="cpu",
                                backend="clarabel_reference", objective=4.)
    original = openopt.check_conic_candidate(model, candidate.x, candidate.s, candidate.z, atol=1e-7, rtol=0.)
    for gap in (None, 0.5):
        damaged = replace(original, metrics={**original.metrics, "duality_gap": gap})
        monkeypatch.setattr(openopt, "check_conic_candidate", lambda *args, **kwargs: damaged)
        with pytest.raises(ValueError, match="gap"):
            gate.accept_rotated_candidate(openopt, model, candidate, "rsoc_lp", "clarabel_reference")


def test_reference_cannot_hide_complex_or_wrong_dtype():
    model = gate.rotated_models(openopt)["rsoc_lp"]
    for dtype in (np.float32, np.complex128):
        candidate = SimpleNamespace(x=np.array([2., 1., 2.], dtype=dtype), s=np.array([0., 2., 1., 2.]),
                                    z=np.array([-2., 1., 2., -2.]), status="reference_solved", device="cpu",
                                    backend="clarabel_reference", objective=4.)
        with pytest.raises(ValueError, match="float64"):
            gate.accept_rotated_candidate(openopt, model, candidate, "rsoc_lp", "clarabel_reference")


def test_native_model_mutation_stops_before_reference(monkeypatch):
    called = []
    def mutate(model, config):
        model.q[0] += 1.
        return SimpleNamespace()
    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("must stop before reference")
    monkeypatch.setattr(openopt, "solve_native_conic", mutate)
    monkeypatch.setattr(openopt, "solve_conic", forbidden)
    with pytest.raises(ValueError, match="mutated original model"):
        gate.rotated_probe(openopt)
    assert not called
