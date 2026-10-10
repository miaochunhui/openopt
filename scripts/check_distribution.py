"""Build and validate a wheel without importing this checkout's source tree."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def byte_descriptor(payload: bytes) -> dict:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonfinite manifest JSON constant")


def validate_source_manifest(manifest: dict) -> None:
    fields = {"schema_version", "source", "modules", "license_expression", "license"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("invalid source manifest schema")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["license_expression"] != "0BSD":
        raise ValueError("invalid source manifest schema")
    source = manifest["source"]
    if (not isinstance(source, dict) or set(source) != {"commit", "tree", "clean"}
            or type(source["clean"]) is not bool
            or any(not isinstance(source[k], str) or not re.fullmatch(r"[0-9a-f]{40}", source[k]) for k in ("commit", "tree"))):
        raise ValueError("invalid source manifest identity")
    modules = manifest["modules"]
    if not isinstance(modules, dict) or not modules:
        raise ValueError("source manifest module membership empty")
    for name in modules:
        if (not isinstance(name, str) or not re.fullmatch(r"(?:openopt|openpo)/[A-Za-z0-9_/]+\.py", name)
                or Path(name).as_posix() != name):
            raise ValueError("unsafe source manifest module path")
    for d in [manifest["license"], *modules.values()]:
        if (not isinstance(d, dict) or set(d) != {"sha256", "bytes"}
                or not isinstance(d["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", d["sha256"])
                or type(d["bytes"]) is not int or d["bytes"] < 0):
            raise ValueError("invalid source manifest byte descriptor")
    if manifest["license"]["bytes"] == 0:
        raise ValueError("empty manifest license")


def make_source_manifest(root: Path, snapshot: dict) -> dict:
    descriptors = {}
    for name, digest in snapshot["sha256"].items():
        if name == "LICENSE" or (name.startswith(("src/openopt/", "src/openpo/")) and name.endswith(".py")):
            descriptor = byte_descriptor((root / name).read_bytes())
            if descriptor["sha256"] != digest:
                raise ValueError("source changed before manifest freeze")
            descriptors[name] = descriptor
    manifest = {
        "schema_version": 1,
        "source": {key: snapshot[key] for key in ("commit", "tree", "clean")},
        "modules": {name[4:]: d for name, d in descriptors.items() if name != "LICENSE"},
        "license_expression": "0BSD", "license": descriptors["LICENSE"],
    }
    validate_source_manifest(manifest)
    return manifest


def load_source_manifest(path: Path, pin: str) -> tuple[dict, dict]:
    if not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{64}", pin):
        raise ValueError("caller manifest SHA256 required")
    path = path.resolve()
    payload = path.read_bytes()
    descriptor = byte_descriptor(payload)
    if descriptor["sha256"] != pin:
        raise ValueError("caller manifest SHA256 mismatch")
    manifest = json.loads(payload, object_pairs_hook=_json_object, parse_constant=_invalid_constant)
    validate_source_manifest(manifest)
    return manifest, {"path": str(path), **descriptor}


def validate_installed_bytes(target: Path, manifest: dict) -> dict:
    validate_source_manifest(manifest)
    target = target.resolve()
    actual = {}
    for package in ("openopt", "openpo"):
        package_root = target / package
        if not package_root.is_dir() or package_root.is_symlink():
            raise ValueError("installed package missing or symlink")
        for path in package_root.rglob("*"):
            if path.is_symlink() or not path.resolve().is_relative_to(target):
                raise ValueError("installed package symlink/escape")
            if path.is_file() and path.suffix in {".pyc", ".pyo", ".so", ".pyd"}:
                raise ValueError("installed unpinned bytecode/native executable forbidden")
            if path.is_file() and path.suffix == ".py":
                actual[path.relative_to(target).as_posix()] = byte_descriptor(path.read_bytes())
    if actual != manifest["modules"]:
        raise ValueError("installed source membership or bytes differ from caller manifest")
    metadata_paths = list(target.glob("*.dist-info/METADATA"))
    if len(metadata_paths) != 1:
        raise ValueError("isolated install requires exactly one METADATA")
    metadata_path = metadata_paths[0]
    license_path = metadata_path.parent / "licenses/LICENSE"
    for path in (metadata_path, license_path):
        if path.is_symlink() or not path.resolve().is_relative_to(target) or not path.is_file():
            raise ValueError("installed metadata/license missing or escaping")
    metadata_payload = metadata_path.read_bytes()
    metadata = BytesParser().parsebytes(metadata_payload)
    if (metadata.get_all("Name") != ["openopt"]
            or metadata.get_all("License-Expression") != ["0BSD"]
            or metadata.get_all("License-File") != ["LICENSE"]):
        raise ValueError("installed metadata must declare openopt and exact 0BSD license")
    license_bytes = license_path.read_bytes()
    if byte_descriptor(license_bytes) != manifest["license"]:
        raise ValueError("installed license bytes differ from caller manifest")
    return {"modules": actual, "metadata": byte_descriptor(metadata_payload),
            "license": byte_descriptor(license_bytes), "license_expression": "0BSD"}


def rotated_models(api):
    """Original RSOC LP/QP, not transformed SOC inputs; analytic x=[2,1,2]."""
    import numpy as np

    A = np.vstack((np.array([[0., 0., 1.]]), -np.eye(3)))
    b = np.array([2., 0., 0., 0.])
    cones = (api.ZeroCone(1), api.RotatedSecondOrderCone(3))
    return {
        "rsoc_lp": api.ConicProblem(np.zeros((3, 3)), np.array([1., 2., 0.]), A.copy(), b.copy(), cones),
        "rsoc_qp": api.ConicProblem(np.diag([1., 4., 0.]), np.array([0., 0., 1.]), A.copy(), b.copy(), cones),
    }


def accept_rotated_candidate(api, model, result, kind: str, backend: str) -> dict:
    import numpy as np
    import torch

    if kind not in {"rsoc_lp", "rsoc_qp"} or backend not in {"native", "clarabel_reference"}:
        raise ValueError("unsupported installed RSOC case/backend")
    expected_status = "solved" if backend == "native" else "reference_solved"
    expected_backend = "native_torch_proximal_admm" if backend == "native" else "clarabel_reference"
    if result.status != expected_status or result.backend != expected_backend or result.device != "cpu":
        raise ValueError("installed RSOC solve status/backend/device failed")
    arrays = []
    for value in (result.x, result.s, result.z):
        if backend == "native":
            if not isinstance(value, torch.Tensor) or value.device != torch.device("cpu") or value.dtype != torch.float64:
                raise ValueError("installed native RSOC actual tensor device/dtype failed")
            value = value.detach().numpy()
        elif not isinstance(value, np.ndarray) or value.dtype != np.float64:
            raise ValueError("installed reference RSOC must return real float64 CPU arrays")
        value = value.copy()
        if not np.isfinite(value).all():
            raise ValueError("installed RSOC candidate nonfinite")
        arrays.append(value)
    x, s, z = arrays
    evidence = api.check_conic_candidate(model, x, s, z, atol=1e-7, rtol=0.)
    required = {"primal_equation", "primal_cone", "original_slack_cone", "dual_cone", "stationarity",
                "complementarity_block_0", "complementarity_block_1", "dual_objective_available", "duality_gap"}
    if not evidence.passed or set(evidence.checks) != required or not all(evidence.checks.values()):
        raise ValueError("installed RSOC independent original-coordinate checks failed")
    objective = float(.5 * x @ model.P @ x + model.q @ x)
    expected_objective = {"rsoc_lp": 4., "rsoc_qp": 6.}[kind]
    if (not np.allclose(x, [2., 1., 2.], atol=2e-6, rtol=0.)
            or abs(objective - expected_objective) > 2e-7
            or not isinstance(result.objective, (float, int)) or not np.isfinite(result.objective)
            or abs(result.objective - objective) > 1e-12):
        raise ValueError("installed RSOC analytic solution/objective failed")
    for name in ("primal_objective", "dual_objective", "duality_gap"):
        value = evidence.metrics[name]
        if value is None or not np.isfinite(value):
            raise ValueError("installed RSOC independent gap unavailable/nonfinite")
    if (abs(evidence.metrics["primal_objective"] - objective) > 1e-12
            or abs(evidence.metrics["primal_objective"] - evidence.metrics["dual_objective"]
                   - evidence.metrics["duality_gap"]) > 1e-12
            or abs(evidence.metrics["duality_gap"]) > 1e-7):
        raise ValueError("installed RSOC independent signed gap failed")
    return {"backend": result.backend, "status": result.status, "actual_device": result.device,
            "x": x.tolist(), "s": s.tolist(), "z": z.tolist(),
            "objective_recomputed": objective, "evidence": evidence.to_dict()}


def rotated_probe(api) -> dict:
    import numpy as np
    import torch

    rows = {}
    for kind, model in rotated_models(api).items():
        frozen = {name: getattr(model, name).copy() for name in ("P", "q", "A", "b")}
        solvers = [
            ("native", api.solve_native_conic, api.NativeConicConfig(
                enabled=True, device="cpu", dtype=torch.float64, eps_abs=1e-9, eps_rel=0.)),
            ("clarabel_reference", api.solve_conic, api.ConicConfig(
                backend="clarabel_reference", device="cpu", eps_abs=1e-10, eps_rel=1e-10)),
        ]
        results = []
        accepted = {}
        for backend, solve, config in solvers:
            result = solve(model, config)
            results.append((backend, result))
            if any(not np.array_equal(getattr(model, name), value) for name, value in frozen.items()):
                raise ValueError("installed RSOC solve mutated original model")
            accepted[backend] = accept_rotated_candidate(api, model, result, kind, backend)
            if any(not np.array_equal(getattr(model, name), value) for name, value in frozen.items()):
                raise ValueError("installed RSOC checker mutated original model")
        if not np.allclose(results[0][1].x.numpy(), results[1][1].x, atol=2e-6, rtol=0.):
            raise ValueError("installed same-model native/reference RSOC disagreement")
        analytic_z = [-2., 1., 2., -2.] if kind == "rsoc_lp" else [-5., 2., 4., -4.]
        analytic = api.check_conic_candidate(model, [2., 1., 2.], [0., 2., 1., 2.], analytic_z, atol=1e-12, rtol=0.)
        if not analytic.passed or analytic.metrics["duality_gap"] != 0.:
            raise ValueError("installed analytic original RSOC KKT/gap failed")
        controls = {
            "wrong_dual_sign": ([2., 1., 2.], [0., 2., 1., 2.], [-v for v in analytic_z]),
            "planted_slack": ([0., 0., 2.], [0., 2., 1., 2.], analytic_z),
            "wrong_tail": ([2., 1., 3.], [0., 2., 1., 2.], analytic_z),
        }
        rejected = {}
        for name, (x, s, z) in controls.items():
            observed = api.check_conic_candidate(model, x, s, z, atol=1e-7, rtol=0.)
            if observed.passed:
                raise ValueError("installed RSOC negative control falsely accepted")
            rejected[name] = observed.to_dict()
        rows[kind] = {"original_model": {name: value.tolist() for name, value in frozen.items()},
                      "cones": [{"type": type(cone).__name__, "dimension": cone.dimension} for cone in model.cones],
                      "analytic_x": [2., 1., 2.], "analytic_z": analytic_z,
                      "analytic_evidence": analytic.to_dict(), "solves": accepted, "rejected_controls": rejected}
    return {"verdict": "PASS", "scope": "installed CPU RSOC LP/QP numerical acceptance only",
            "actual_solves": {"native": 2, "clarabel_reference": 2, "commercial": 0}, "cases": rows,
            "cuda_execution": "UNVERIFIED", "performance": "UNVERIFIED"}


def validate_wheel_license(wheel: zipfile.ZipFile, expected: bytes) -> None:
    metadata_paths = [
        name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_paths) != 1:
        raise ValueError("wheel must contain exactly one distribution METADATA")
    path = metadata_paths[0]
    metadata = BytesParser().parsebytes(wheel.read(path))
    if metadata.get_all("License-Expression") != ["0BSD"]:
        raise ValueError("wheel must declare License-Expression: 0BSD")
    if metadata.get_all("License-File") != ["LICENSE"]:
        raise ValueError("wheel must declare exactly License-File: LICENSE")
    license_path = path.rsplit("/", 1)[0] + "/licenses/LICENSE"
    if license_path not in wheel.namelist() or wheel.read(license_path) != expected:
        raise ValueError("wheel license payload is absent or differs from LICENSE")
    if not expected.strip():
        raise ValueError("empty source license")


def validate_wheel_sources(wheel: zipfile.ZipFile, source: Path) -> int:
    expected = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*.py")
    }
    names = wheel.namelist()
    actual = {
        name for name in names
        if name.endswith(".py") and name.startswith(("openpo/", "openopt/"))
    }
    if not expected or actual != set(expected) or len(names) != len(set(names)):
        raise ValueError("wheel source membership is empty, duplicated or mismatched")
    changed = sorted(name for name, payload in expected.items() if wheel.read(name) != payload)
    if changed:
        raise ValueError(f"wheel source bytes differ: {changed}")
    return len(expected)


def installed_probe(target, manifest=None):
    target = Path(target).resolve()
    # -I ignores PYTHONDONTWRITEBYTECODE; do not create unpinned executables.
    sys.dont_write_bytecode = True
    verified_bytes = validate_installed_bytes(target, manifest) if manifest is not None else None
    if any(name == package or name.startswith(package + ".")
           for name in sys.modules for package in ("openopt", "openpo")):
        raise ValueError("installed probe requires fresh process; cached source imports forbidden")
    sys.path.insert(0, str(target))
    import openopt
    import openpo
    import torch

    for module in (openopt, openpo):
        if not Path(module.__file__).resolve().is_relative_to(target):
            raise ValueError(f"package loaded outside isolated wheel install: {module.__name__}")
    for name in openpo.__all__:
        if getattr(openopt, name) is not getattr(openpo, name):
            raise ValueError(f"reexport identity mismatch: {name}")
    problem = openopt.PortfolioProblem(
        alpha=torch.tensor([0.01, 0.02], dtype=torch.float64),
        risk=openopt.FactorRiskModel(
            exposures=torch.zeros((2, 1), dtype=torch.float64),
            factor_variance=torch.ones(1, dtype=torch.float64),
            specific_variance=torch.ones(2, dtype=torch.float64),
        ),
        lower=torch.zeros(2, dtype=torch.float64),
        upper=torch.ones(2, dtype=torch.float64),
        risk_aversion=2.0,
    )
    result = openopt.solve(problem, openopt.SolveConfig(iterations=300, tolerance=1e-9))
    if not result.converged or not torch.isfinite(result.weights).all():
        raise ValueError("isolated wheel solve did not converge")
    if float((result.weights.sum(dim=1) - 1.0).abs().max()) > 1e-6:
        raise ValueError("isolated wheel budget constraint failed")
    general = openopt.QPProblem(
        P=torch.eye(2, dtype=torch.float64),
        q=torch.tensor([-1.0, 2.0], dtype=torch.float64),
        A=torch.eye(2, dtype=torch.float64),
        l=torch.zeros(2, dtype=torch.float64),
        u=torch.ones(2, dtype=torch.float64),
    )
    solved = openopt.solve_qp(general)
    if solved.status != "solved" or not torch.allclose(
        solved.x, torch.tensor([1.0, 0.0], dtype=torch.float64), atol=1e-6,
    ):
        raise ValueError("isolated wheel general QP solve failed")
    batched = openopt.solve_batch_qp(openopt.BatchQPProblem.stack([general, general]))
    if batched.status != ("solved", "solved") or not torch.allclose(
        batched.x, solved.x.expand(2, -1), atol=1e-6,
    ):
        raise ValueError("isolated wheel batched QP solve failed")
    evidence = openopt.check_qp_candidate(general, solved.x, solved.y)
    if not evidence.passed:
        raise ValueError("isolated wheel independent evidence failed")
    sparse = openopt.OperatorQPProblem(
        general.P.to_sparse_coo(), general.q, general.A.to_sparse_coo(),
        general.l, general.u, assume_psd=True,
    )
    operator = openopt.solve_operator_qp(sparse)
    if operator.status != "solved" or not torch.allclose(operator.x, solved.x, atol=1e-6):
        raise ValueError("isolated wheel sparse QP solve failed")
    jacobi = openopt.solve_operator_qp(
        sparse, openopt.OperatorQPConfig(cg_preconditioner="jacobi"),
    )
    if jacobi.status != "solved" or not torch.allclose(jacobi.x, solved.x, atol=1e-6):
        raise ValueError("isolated wheel Jacobi QP solve failed")
    scaled = openopt.solve_scaled_qp(general)
    if scaled.status != "solved" or not scaled.evidence.passed:
        raise ValueError("isolated wheel restored-coordinate QP failed")
    gap = openopt.solve_qp(general, openopt.QPConfig(dual_gap_termination=True))
    if gap.status != "solved" or not gap.termination_evidence.passed:
        raise ValueError("isolated wheel independent-gap termination failed")
    conic = openopt.ConicProblem(
        general.P, general.q, -general.A, -general.l,
        (openopt.NonnegativeCone(2),),
    )
    conic_result = openopt.solve_conic(
        conic, openopt.ConicConfig(backend="clarabel_reference"),
    )
    if conic_result.status != "reference_solved" or not conic_result.evidence.passed:
        raise ValueError("isolated wheel conic CPU reference failed")
    native = openopt.solve_native_conic(
        conic, openopt.NativeConicConfig(enabled=True),
    )
    if native.status != "solved" or not native.evidence.passed or native.device != "cpu":
        raise ValueError("isolated wheel native conic numerical solve failed")
    polished = openopt.refine_qp_candidate(
        general, solved.x, solved.y, openopt.RefinementConfig(),
    )
    if not polished.accepted or not polished.evidence.passed:
        raise ValueError("isolated wheel original-coordinate QP refinement failed")
    fixed = openopt.QPProblem(
        general.P, general.q, general.A,
        torch.tensor([0.5, 0.0], dtype=torch.float64),
        torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    presolved = openopt.solve_presolved_qp(
        fixed, presolve=openopt.PresolveConfig(enabled=True),
    )
    if (
        presolved.status != "solved" or not presolved.evidence.passed
        or presolved.presolved.fixed_variables.tolist() != [0]
        or presolved.presolved.free_variables.tolist() != [1]
        or not torch.allclose(
            presolved.x, torch.tensor([0.5, 0.0], dtype=torch.float64), atol=1e-6,
        )
    ):
        raise ValueError("isolated wheel exact presolve/original postsolve failed")
    integer = openopt.MILPProblem(
        q=[-1.0], A=[[1.0]], l=[0.0], u=[1.5],
        lower=[0.0], upper=[2.0], integrality=[1],
    )
    integer_result = openopt.solve_milp(integer, openopt.MILPConfig(backend="highspy"))
    if integer_result.status != "optimal" or integer_result.incumbent != -1.0:
        raise ValueError("isolated wheel MILP CPU reference failed")
    for name in ("native_conic", "refinement", "presolve"):
        module = sys.modules[f"openopt.{name}"]
        if not Path(module.__file__).resolve().is_relative_to(target):
            raise ValueError(f"extension loaded outside isolated wheel: {name}")
    rotated = rotated_probe(openopt)
    loaded = {}
    for name, module in list(sys.modules.items()):
        if not (name in ("openopt", "openpo") or name.startswith(("openopt.", "openpo."))):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(target):
            raise ValueError("loaded extension outside isolated install")
        relative = path.relative_to(target).as_posix()
        descriptor = byte_descriptor(path.read_bytes())
        if manifest is not None and manifest["modules"].get(relative) != descriptor:
            raise ValueError("loaded extension source bytes not in caller manifest")
        loaded[name] = {"path": str(path), **descriptor}
    if manifest is not None and validate_installed_bytes(target, manifest) != verified_bytes:
        raise ValueError("installed bytes changed during numerical acceptance")
    print(f"WHEEL_IMPORT verdict=PASS api_objects={len(openpo.__all__)} version={openopt.__version__}")
    return {"verdict": "PASS", "source": manifest["source"] if manifest else None, "installed": str(target),
            "provenance_verdict": "PASS" if manifest else "UNVERIFIED",
            "byte_validation": verified_bytes, "loaded_modules": loaded, "rotated": rotated,
            "cuda_execution": "UNVERIFIED", "commercial_qualification": "UNVERIFIED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("/hf3fs-jd/data/tmp/openopt-distribution"))
    parser.add_argument("--installed", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--probe-report", type=Path)
    args = parser.parse_args()
    if args.installed:
        if bool(args.source_manifest) != bool(args.source_manifest_sha256):
            parser.error("source manifest path and caller SHA256 must be supplied together")
        output = args.probe_report.resolve() if args.probe_report else None
        if output is not None and output.is_relative_to(ROOT):
            raise ValueError("installed receipt must be outside repository")
        probe = {"verdict": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            manifest = None
            if args.source_manifest:
                manifest, descriptor = load_source_manifest(args.source_manifest, args.source_manifest_sha256)
                probe["source_manifest"] = descriptor
            probe.update(installed_probe(args.installed, manifest))
        except Exception as error:
            probe.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            probe["finished_at"] = datetime.now(timezone.utc).isoformat()
            if output is not None:
                with output.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(probe, indent=2, allow_nan=False) + "\n")
            verdict = probe["verdict"]
            provenance = probe.get("provenance_verdict", "UNVERIFIED")
            print(f"WHEEL_INSTALLED verdict={verdict} provenance={provenance} report={output}")
        return
    if args.source_manifest or args.source_manifest_sha256 or args.probe_report:
        parser.error("source manifest/report options are for installed acceptance only")
    base = args.artifact_dir.resolve()
    if base.is_relative_to(ROOT):
        raise ValueError("distribution artifacts must be outside the repository")
    base.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="wheel-", dir=base))
    target = artifacts / "installed"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(artifacts)
    env["PIP_NO_CACHE_DIR"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    report = {"verdict": "RUNNING", "at": datetime.now(timezone.utc).isoformat()}
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.verify import source_snapshot, check_source_consistency

        report["source"] = source_snapshot()
        manifest = make_source_manifest(ROOT, report["source"])
        manifest_path = artifacts / "source-manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
        manifest_pin = byte_descriptor(manifest_path.read_bytes())["sha256"]
        report["source_manifest"] = {"path": str(manifest_path), **byte_descriptor(manifest_path.read_bytes())}
        # Build from frozen, external input; setuptools must not mutate the worktree.
        stage = artifacts / "build-source"
        for name, expected in report["source"]["sha256"].items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
            if byte_descriptor(destination.read_bytes())["sha256"] != expected:
                raise ValueError("source changed during external build staging")
        probe_path = artifacts / "installed-probe.json"
        with (artifacts / "distribution.log").open("x") as log:
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                 "--no-index", "--wheel-dir", str(artifacts), str(stage)], env=env, stdout=log,
                stderr=subprocess.STDOUT, check=True,
            )
            wheels = list(artifacts.glob("*.whl"))
            if len(wheels) != 1:
                raise ValueError(f"expected exactly one wheel: {wheels}")
            with zipfile.ZipFile(wheels[0]) as wheel:
                validate_wheel_license(wheel, (ROOT / "LICENSE").read_bytes())
                modules = validate_wheel_sources(wheel, stage / "src")
                wheel_modules = {name: byte_descriptor(wheel.read(name)) for name in manifest["modules"]}
                if wheel_modules != manifest["modules"]:
                    raise ValueError("wheel bytes differ from caller source manifest")
                report["wheel_modules"] = wheel_modules
                report["license"] = manifest["license"]
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
                 "--target", str(target), str(wheels[0])], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
            subprocess.run(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--installed", str(target),
                 "--source-manifest", str(manifest_path), "--source-manifest-sha256", manifest_pin,
                 "--probe-report", str(probe_path)],
                cwd=artifacts, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        installed = json.loads(probe_path.read_bytes(), object_pairs_hook=_json_object,
                               parse_constant=_invalid_constant)
        if (installed["verdict"] != "PASS" or installed["source"] != manifest["source"]
                or installed["source_manifest"]["sha256"] != manifest_pin
                or installed["rotated"]["verdict"] != "PASS"):
            raise ValueError("installed numerical/source receipt did not pass")
        report["installed_probe"] = {"path": str(probe_path), **byte_descriptor(probe_path.read_bytes())}
        report["rotated"] = installed["rotated"]
        report["source_consistency"] = check_source_consistency(
            report["source"], source_snapshot(),
        )
        if report["source_consistency"]["verdict"] != "PASS":
            raise RuntimeError("source changed during wheel acceptance")
        report.update(
            verdict="PASS", wheel=wheels[0].name, modules=modules,
            wheel_sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
            license_expression="0BSD", license_payload_verified=True,
        )
    except Exception as error:
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (artifacts / "distribution.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"OPENOPT_DISTRIBUTION verdict={report['verdict']} report={artifacts / 'distribution.json'}")


if __name__ == "__main__":
    main()
