"""Single-attempt CPU qualification of caller-pinned, externally frozen QPs.

No acquisition, installation, retries, planted witnesses, GPU fallback, license
claim or speed ranking. Replay checks saved host vectors without solving again.
Exit 0=PASS, 1=UNQUALIFIED, 2=FAIL; a synthetic run can never be public PASS.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import MISSING, asdict, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import subprocess
import time
import traceback
from types import FunctionType, MethodType, ModuleType, SimpleNamespace
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import public_qp_corpus as corpus

VERSION = "public-qp-qualification-v1"
V3 = "public-qp-worker-v3"
V3_BASE = "acc199306936922b6d75330884310a58cde84a9b"
V3_ARITHMETIC = "reproducible_binary64_v2"
V3_MODES = ("cold", "hot")
V3_GATES = {"eps_abs": 1e-7, "eps_rel": 1e-7,
            "direction_atol": 1e-8, "direction_rtol": 1e-8}
V3_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
          "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
V3_SOURCES = {
    "runner.py": "scripts/qualify_public_qp.py",
    "importer.py": "scripts/public_qp_corpus.py",
    "contracts.py": "scripts/experiment_public_robustness.py",
    "qp.py": "src/openopt/qp.py", "certificates.py": "src/openopt/certificates.py",
    "scaling.py": "src/openopt/scaling.py", "refinement.py": "src/openopt/refinement.py",
}
SHARED_TMP = Path("/hf3fs-jd/data/tmp").resolve()
SOLVERS = ("openopt", "osqp", "clarabel")
VECTOR_NAMES = ("x", "y", "primal_direction", "dual_direction", "feasible_base",
                "reported_objective")
STATUS = {
    "openopt": {"solved": "optimal", "infeasible": "infeasible", "unbounded": "unbounded"},
    "osqp": {"solved": "optimal", "primal infeasible": "infeasible",
             "dual infeasible": "unbounded"},
    "clarabel": {"Solved": "optimal", "PrimalInfeasible": "infeasible",
                 "DualInfeasible": "unbounded"},
}
SCOPE = "CPU numerical public-model qualification; not industrial/commercial/GPU qualification"


class UnsupportedAPI(RuntimeError):
    """Required opt-in API is absent; never silently disable it."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_relative_to(SHARED_TMP) or path.is_relative_to(ROOT):
        raise ValueError(f"output must be outside worktree under {SHARED_TMP}: {path}")
    return path


def copy_file(source: Path, target: Path) -> None:
    raw = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
    if target.read_bytes() != raw:
        raise OSError(f"copy roundtrip mismatch: {target}")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_core(directory: Path) -> SimpleNamespace:
    # A private namespace avoids importing a different installed openopt package.
    name = "_public_qp_core_" + digest(directory / "qp.py")[:16]
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    checker = load_module(directory / "certificates.py", name + ".certificates")
    qp = load_module(directory / "qp.py", name + ".qp")
    return SimpleNamespace(qp=qp, checker=checker)


def config_values(args: argparse.Namespace) -> dict:
    if args.device != "cpu":
        raise ValueError("CPU runner rejects explicit GPU/non-CPU requests; no fallback")
    for name in ("eps_abs", "eps_rel", "direction_atol", "direction_rtol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0 or (name.endswith(("rel", "rtol")) and value >= 1):
            raise ValueError(f"{name} must be finite, positive; relative tolerance < 1")
    for name in ("max_iterations", "check_interval", "adaptive_rho_interval", "max_dense_bytes"):
        value = getattr(args, name)
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return {
        "device": "cpu", "eps_abs": args.eps_abs, "eps_rel": args.eps_rel,
        "max_iterations": args.max_iterations, "check_interval": args.check_interval,
        "adaptive_rho": True, "adaptive_rho_interval": args.adaptive_rho_interval,
        "dual_gap_termination": True, "direction_termination": True,
        "direction_atol": args.direction_atol, "direction_rtol": args.direction_rtol,
    }


def source_eligibility(frozen: dict, directory: Path, acceptance: dict | None,
                       frozen_pin: str) -> list[dict]:
    """A public label is not authenticity: require separately pinned caller review.

    The review is an external trust anchor, not an inferred license or a network
    authenticity check. Reserved/test URLs and synthetic declarations still fail.
    """
    accepted = {}
    if acceptance is not None:
        required = {"schema_version", "verdict", "corpus_sha256", "cases"}
        rights = {"license_verified", "public_redistribution", "official_repository",
                  "resolved_commit", "review_scope"}
        if (set(acceptance) not in (required, required | rights)
                or acceptance["schema_version"] != 1
                or acceptance["verdict"] != "PUBLIC_SOURCE_ACCEPTED"
                or acceptance["corpus_sha256"] != frozen_pin):
            raise ValueError("public acceptance schema/verdict/corpus pin mismatch")
        if set(acceptance) == required | rights and (
            acceptance["license_verified"] is not False
            or acceptance["public_redistribution"] is not False
        ):
            raise ValueError("source review cannot grant legal license/redistribution permission")
        public_ids = [row["case_id"] for row in frozen["cases"] if row["origin"] == "public"]
        if not isinstance(acceptance["cases"], list) or not public_ids:
            raise ValueError("empty public acceptance rejected")
        if [row.get("case_id") for row in acceptance["cases"]] != public_ids:
            raise ValueError("public acceptance membership/order mismatch")
        for row in acceptance["cases"]:
            if set(row) != {"case_id", "source_url", "input_sha256", "model_sha256",
                            "source_review", "evidence_sha256"}:
                raise ValueError("incomplete public source acceptance")
            corpus._sha(row["evidence_sha256"], "public evidence_sha256")
            if not isinstance(row["source_review"], str) or not row["source_review"].strip():
                raise ValueError("missing independent public source review")
            accepted[row["case_id"]] = row
    result = []
    for row in frozen["cases"]:
        metadata = corpus.read_json(directory / row["case_id"] / "metadata.json")
        entry = metadata["provenance"]
        host = urlsplit(entry["source_url"]).hostname.lower()
        reserved = (host in {"localhost", "example.com", "example.org", "example.net"}
                    or host.endswith((".invalid", ".test", ".localhost", ".example"))
                    or host.startswith(("127.", "0.")) or host == "::1")
        synthetic = bool(re.search(r"\b(synthetic|fixture|planted|hand-authored)\b",
                                   entry["source_url"] + " " + entry["license"], re.I))
        reason = None
        if entry["origin"] != "public":
            reason = "synthetic format/function evidence, not public or industrial coverage"
        elif reserved or synthetic:
            reason = "public declaration rejected: reserved/test source or synthetic marker"
        elif row["case_id"] not in accepted:
            reason = "public origin is only a declaration; independent pinned source review absent"
        else:
            review = accepted[row["case_id"]]
            expected = {"source_url": entry["source_url"], "input_sha256": entry["input_sha256"],
                        "model_sha256": metadata["model_sha256"]}
            if any(review[key] != value for key, value in expected.items()):
                raise ValueError(f"public acceptance source/model mismatch: {row['case_id']}")
        result.append({"case_id": row["case_id"], "origin": entry["origin"],
                       "eligible": reason is None, "reason": reason})
    return result


def load_model(directory: Path, case_id: str) -> dict:
    metadata = corpus.read_json(directory / case_id / "metadata.json")
    with np.load(directory / case_id / "arrays.npz", allow_pickle=False) as archive:
        return {**metadata, "arrays": {name: archive[name].copy() for name in corpus.ARRAY_NAMES}}


def problem(view: dict) -> SimpleNamespace:
    return SimpleNamespace(**{name: view[name].copy() for name in ("P", "q", "A", "l", "u")})


def clarabel_form(view: dict, clarabel) -> tuple:
    rows, bounds, mapping = [], [], []
    equal = np.flatnonzero(view["l"] == view["u"])
    for i in equal:
        rows.append(view["A"][i])
        bounds.append(view["u"][i])
        mapping.append((int(i), 1.0))
    for i in np.flatnonzero(view["l"] != view["u"]):
        for key, sign in (("u", 1.0), ("l", -1.0)):
            if np.isfinite(view[key][i]):
                rows.append(sign * view["A"][i])
                bounds.append(sign * view[key][i])
                mapping.append((int(i), sign))
    cones = [clarabel.ZeroConeT(len(equal))] if len(equal) else []
    if len(rows) > len(equal):
        cones.append(clarabel.NonnegativeConeT(len(rows) - len(equal)))
    return (np.asarray(rows, dtype=np.float64).reshape(len(rows), len(view["q"])),
            np.asarray(bounds), cones, mapping)


def run_solver(model: dict, solver: str, values: dict, core: SimpleNamespace) -> dict:
    """One fresh solve, zero warm start; no expected solution enters adapters."""
    view = corpus.solver_arrays(model)
    if solver == "openopt":
        import torch

        required = {"adaptive_rho", "dual_gap_termination", "direction_termination"}
        fields = getattr(core.qp.QPConfig, "__dataclass_fields__", {})
        if not required <= set(fields):
            raise UnsupportedAPI("source-root QPConfig lacks required adaptive/gap/direction API")
        config = core.qp.QPConfig(**values, dtype=torch.float64)
        config.validate()
        p = core.qp.QPProblem(*[
            torch.tensor(view[name], dtype=torch.float64, device="cpu")
            for name in ("P", "q", "A", "l", "u")
        ])
        solved = core.qp.solve_qp(p, config)
        if solved.x.device.type != "cpu" or solved.y.device.type != "cpu":
            raise RuntimeError("CPU placement mismatch; no device fallback permitted")
        vectors = {"x": solved.x.detach().numpy(), "y": solved.y.detach().numpy(),
                   "reported_objective": np.asarray(solved.objective)}
        for source, target in (("infeasibility_direction", "primal_direction"),
                               ("unbounded_direction", "dual_direction"),
                               ("feasible_base", "feasible_base")):
            value = getattr(solved, source, None)
            if value is not None:
                if value.device.type != "cpu":
                    raise RuntimeError(f"non-CPU host evidence: {source}")
                vectors[target] = value.detach().numpy()
        return {"status": solved.status, "iterations": int(solved.iterations),
                "observed_device": "cpu", "vectors": vectors,
                "settings": {**asdict(config), "dtype": str(config.dtype)},
                "rho_updates": solved.rho_updates, "final_rho": solved.final_rho}
    from scipy import sparse

    if solver == "osqp":
        import osqp

        settings = {
            "eps_abs": values["eps_abs"], "eps_rel": values["eps_rel"],
            "eps_prim_inf": values["direction_atol"], "eps_dual_inf": values["direction_atol"],
            "max_iter": values["max_iterations"], "check_termination": values["check_interval"],
            "adaptive_rho": True, "adaptive_rho_interval": values["adaptive_rho_interval"],
            "scaling": 0, "scaled_termination": False, "polishing": False,
            "warm_starting": False, "verbose": False,
        }
        workspace = osqp.OSQP()
        workspace.setup(P=sparse.csc_matrix(np.triu(view["P"])), q=view["q"].copy(),
                        A=sparse.csc_matrix(view["A"]), l=view["l"].copy(), u=view["u"].copy(),
                        **settings)
        solved = workspace.solve(raise_error=False)
        vectors = {key: value for key, value in {
            "x": solved.x, "y": solved.y, "primal_direction": solved.prim_inf_cert,
            "dual_direction": solved.dual_inf_cert, "feasible_base": solved.x,
            "reported_objective": np.asarray(solved.info.obj_val),
        }.items() if value is not None}
        effective = {key: getattr(workspace.settings, key) for key in settings}
        for key in ("rho", "sigma", "alpha", "adaptive_rho_tolerance", "adaptive_rho_fraction"):
            effective[key] = getattr(workspace.settings, key)
        return {"status": solved.info.status, "iterations": int(solved.info.iter),
                "observed_device": "cpu", "settings": effective, "vectors": vectors}
    if solver == "clarabel":
        import clarabel

        A, b, cones, mapping = clarabel_form(view, clarabel)
        settings = clarabel.DefaultSettings()
        requested = {
            "verbose": False, "max_threads": 1, "max_iter": values["max_iterations"],
            "tol_gap_abs": values["eps_abs"], "tol_gap_rel": values["eps_rel"],
            "tol_feas": values["eps_abs"], "tol_infeas_abs": values["direction_atol"],
            "tol_infeas_rel": values["direction_rtol"],
            "equilibrate_enable": False, "presolve_enable": False,
        }
        for key, value in requested.items():
            setattr(settings, key, value)
        workspace = clarabel.DefaultSolver(sparse.csc_matrix(np.triu(view["P"])),
                                          view["q"].copy(), sparse.csc_matrix(A),
                                          b, cones, settings)
        solved = workspace.solve()
        z = np.asarray(solved.z, dtype=np.float64)
        if z.shape != (len(mapping),):
            raise ValueError("Clarabel dual/finite-bound mapping dimension mismatch")
        y = np.zeros(len(view["l"]))
        for value, (index, sign) in zip(z, mapping):
            y[index] += sign * value
        x = np.asarray(solved.x, dtype=np.float64)
        return {
            "status": str(solved.status), "iterations": int(solved.iterations),
            "observed_device": "cpu", "settings": {
                key: getattr(settings, key) for key in requested
            }, "dual_mapping": [list(item) for item in mapping],
            "vectors": {"x": x, "y": y, "primal_direction": y, "dual_direction": x,
                        "feasible_base": x, "reported_objective": np.asarray(solved.obj_val)},
        }
    raise ValueError(f"unknown backend: {solver}")


def assess(model: dict, solver: str, raw: dict, values: dict, checker) -> dict:
    view = corpus.solver_arrays(model)
    p, vectors = problem(view), raw["vectors"]
    prop = STATUS[solver].get(raw["status"], "unknown")
    candidate = direction = None
    objectives = None
    reported_ok = False
    if "x" in vectors and "y" in vectors:
        candidate = checker.check_qp_candidate(
            p, vectors["x"], vectors["y"], atol=values["eps_abs"], rtol=values["eps_rel"],
        ).to_dict()
        metrics = candidate["metrics"]
        primal = metrics["primal_objective"]
        dual = metrics["dual_objective"]
        objectives = {
            "sense": model["objective_sense"], "offset": float(model["arrays"]["offset"]),
            "primal_original": corpus.original_objective(model, vectors["x"]),
            "dual_original": None if dual is None else view["objective_sign"] * (dual + view["offset"]),
            "signed_gap_original": None if metrics["duality_gap"] is None
            else view["objective_sign"] * metrics["duality_gap"],
            "normalized_primal_without_offset": primal,
            "normalized_dual_without_offset": dual,
        }
        if "reported_objective" in vectors:
            reported = float(vectors["reported_objective"])
            reported_ok = math.isfinite(reported) and abs(reported - primal) <= (
                values["eps_abs"] + values["eps_rel"] * max(abs(reported), abs(primal))
            )
            objectives["reported_original"] = (
                view["objective_sign"] * (reported + view["offset"])
                if math.isfinite(reported) else None
            )
    if prop == "infeasible" and "primal_direction" in vectors:
        direction = checker.check_primal_infeasibility(
            p, vectors["primal_direction"], atol=values["direction_atol"],
            rtol=values["direction_rtol"],
        ).to_dict()
    elif prop == "unbounded" and "dual_direction" in vectors:
        # Only this solve's returned host x/base is eligible, never an oracle,
        # constructed point, reference's solution or an auxiliary feasibility solve.
        direction = checker.check_unbounded_direction(
            p, vectors["dual_direction"], feasible_x=vectors.get("feasible_base"),
            atol=values["direction_atol"], rtol=values["direction_rtol"],
        ).to_dict()
    passed = (
        prop == "optimal" and candidate is not None and candidate["passed"] and reported_ok
    ) or (prop in ("infeasible", "unbounded") and direction is not None and direction["passed"])
    return {
        "verdict": "QUALIFIED" if passed else "UNQUALIFIED", "property": prop,
        "strict_status": prop != "unknown", "candidate": candidate, "direction": direction,
        "objective": objectives, "reported_objective_matches": reported_ok,
        "quality_scope": "same independent CPU checker; original x, original bound rows, no regularization",
        "inaccurate_statuses_accepted": False,
    }


def save_vectors(path: Path, vectors: dict) -> None:
    if set(vectors) - set(VECTOR_NAMES):
        raise ValueError("unknown host vector sidecar member")
    arrays = {key: np.asarray(value, dtype="<f8", order="C") for key, value in vectors.items()}
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(
            saved[key].shape != value.shape or saved[key].dtype != value.dtype
            or saved[key].tobytes() != value.tobytes() for key, value in arrays.items()
        ):
            raise OSError(f"host sidecar roundtrip mismatch: {path}")


def read_vectors(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != len(set(saved.files)) or set(saved.files) - set(VECTOR_NAMES):
            raise ValueError("host sidecar membership mismatch")
        result = {key: saved[key].copy() for key in saved.files}
    if any(value.dtype != np.dtype("<f8") for value in result.values()):
        raise ValueError("host sidecar must use CPU float64 numeric arrays")
    return result


def comparisons(rows: list[dict], values: dict) -> list[dict]:
    result = []
    for case_id in dict.fromkeys(row["case_id"] for row in rows):
        selected = {row["solver"]: row for row in rows if row["case_id"] == case_id}
        for reference in SOLVERS[1:]:
            a, b = selected["openopt"], selected[reference]
            qa, qb = a.get("assessment"), b.get("assessment")
            same = qa is not None and qb is not None and qa["property"] == qb["property"]
            qualified = a["verdict"] == b["verdict"] == "QUALIFIED"
            error = limit = None
            agreed = bool(same and qualified)
            if agreed and qa["property"] == "optimal":
                x = qa["objective"]["normalized_primal_without_offset"]
                y = qb["objective"]["normalized_primal_without_offset"]
                error = abs(x - y)
                limit = values["eps_abs"] + values["eps_rel"] * max(abs(x), abs(y))
                agreed = error <= limit
            result.append({"case_id": case_id, "reference": reference,
                           "verdict": "AGREED" if agreed else "UNQUALIFIED",
                           "same_property": bool(same), "both_independently_qualified": qualified,
                           "objective_difference": error, "objective_limit": limit,
                           "denominator": 1})
    return result


def summarize(rows: list[dict], eligibility: list[dict], values: dict) -> dict:
    pairs = comparisons(rows, values)
    counts = {
        solver: {"denominator": sum(row["solver"] == solver for row in rows),
                 "attempted": sum(row["solver"] == solver and row["attempted"] for row in rows),
                 **{state: sum(row["solver"] == solver and row["verdict"] == state for row in rows)
                    for state in ("QUALIFIED", "UNQUALIFIED", "UNSUPPORTED", "FAIL", "NOT_ATTEMPTED")}}
        for solver in SOLVERS
    }
    failed = any(row["verdict"] == "FAIL" for row in rows)
    numeric = bool(rows) and all(row["verdict"] == "QUALIFIED" for row in rows)
    numeric = numeric and all(pair["verdict"] == "AGREED" for pair in pairs)
    public = bool(eligibility) and all(row["eligible"] for row in eligibility)
    verdict = "FAIL" if failed else "PASS" if numeric and public else "UNQUALIFIED"
    return {"verdict": verdict, "exit_code": {"PASS": 0, "UNQUALIFIED": 1, "FAIL": 2}[verdict],
            "numeric_verdict": "PASS" if numeric else "UNQUALIFIED",
            "backend_counts": counts, "comparisons": pairs, "planned_denominator": len(rows),
            "attempted_count": sum(row["attempted"] for row in rows),
            "public_qualified_count": sum(
                entry["eligible"] and all(
                    row["verdict"] == "QUALIFIED" for row in rows if row["case_id"] == entry["case_id"]
                ) and all(pair["verdict"] == "AGREED" for pair in pairs
                          if pair["case_id"] == entry["case_id"]) for entry in eligibility),
            "industrial_qualified_count": 0}


def inventory(output: Path) -> dict:
    return {str(path.relative_to(output)): {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(output.rglob("*")) if path.is_file() and path != output / "result.json"}


def qualify(args: argparse.Namespace) -> tuple[Path, int]:
    output = external(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "runner_version": VERSION, "scope": SCOPE,
              "verdict": "RUNNING", "exit_code": 2, "rows": [], "source_eligibility": [],
              "started_at": datetime.now(timezone.utc).isoformat(),
              "device": "cpu", "solvers": list(SOLVERS), "automatic_retry": False,
              "sameoracle": "independent check_qp_candidate / direction checks for every backend",
              "license_claim": False, "speedup_claim": False, "industrial_qualified_count": 0,
              "license_verified": False, "public_redistribution": False,
              "source_root": str(args.source_root.expanduser().resolve()),
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]}
    current = None
    try:
        values = config_values(args)
        report["config"] = values
        report["max_dense_bytes"] = args.max_dense_bytes
        frozen_path = args.corpus.expanduser().resolve(strict=True)
        source = args.source_root.expanduser().resolve(strict=True)
        importer_path = source / "scripts" / "public_qp_corpus.py"
        importer = load_module(importer_path, "_public_qp_run_importer_" + digest(importer_path)[:16])
        importer.verify_frozen(frozen_path, args.manifest_sha256, args.max_dense_bytes)
        frozen = importer.read_json(frozen_path / "corpus.json")
        report["corpus_sha256"] = args.manifest_sha256
        report["planned_cases"] = [row["case_id"] for row in frozen["cases"]]
        report["rows"] = [
            {"case_id": row["case_id"], "model_sha256": row["model_sha256"],
             "solver": solver, "attempt": 1, "denominator": 1, "attempted": False,
             "verdict": "NOT_ATTEMPTED"} for row in frozen["cases"] for solver in SOLVERS
        ]
        for filename in ("corpus.json", "input-manifest.json"):
            copy_file(frozen_path / filename, output / "frozen" / filename)
        for row in frozen["cases"]:
            for filename in ("raw.mps", "arrays.npz", "metadata.json", "reader.log"):
                copy_file(frozen_path / row["case_id"] / filename,
                          output / "frozen" / row["case_id"] / filename)
        importer.verify_frozen(output / "frozen", args.manifest_sha256, args.max_dense_bytes)
        acceptance = None
        if (args.public_acceptance is None) != (args.public_acceptance_sha256 is None):
            raise ValueError("public acceptance requires both file and externally pinned SHA256")
        if args.public_acceptance is not None:
            corpus._sha(args.public_acceptance_sha256, "public_acceptance_sha256")
            copy_file(args.public_acceptance, output / "public-acceptance.json")
            if digest(output / "public-acceptance.json") != args.public_acceptance_sha256:
                raise ValueError("public acceptance SHA256 mismatch")
            acceptance = corpus.read_json(output / "public-acceptance.json")
            report["public_acceptance_sha256"] = args.public_acceptance_sha256
        report["source_eligibility"] = source_eligibility(
            frozen, output / "frozen", acceptance, args.manifest_sha256,
        )
        copy_file(Path(__file__), output / "code" / "runner.py")
        copy_file(importer_path, output / "code" / "importer.py")
        if digest(output / "code" / "importer.py") != frozen["source_hashes"]["scripts/public_qp_corpus.py"]:
            raise ValueError("frozen importer source differs from current verified parser")
        for filename in ("qp.py", "certificates.py"):
            copy_file(source / "src" / "openopt" / filename, output / "code" / filename)
        core = load_core(output / "code")
        import torch

        torch.set_num_threads(1)
        report["cpu_threads"] = torch.get_num_threads()
        report["software"] = {"python": sys.version, "python_executable": sys.executable}
        for name in ("numpy", "torch", "highspy", "scipy", "osqp", "clarabel"):
            try:
                report["software"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report["software"][name] = "MISSING; backend attempt must fail, not skip"
        for row in report["rows"]:
            current = row
            folder = output / "attempts" / row["case_id"]
            folder.mkdir(parents=True, exist_ok=True)
            row["attempted"] = True
            print(f"PUBLIC_QP_ATTEMPT case={row['case_id']} solver={row['solver']} attempt=1", flush=True)
            try:
                model = load_model(output / "frozen", row["case_id"])
                before = corpus.model_hash(model)
                raw = run_solver(model, row["solver"], values, core)
                vectors_path = folder / f"{row['solver']}.npz"
                save_vectors(vectors_path, raw["vectors"])
                row["vectors_path"] = str(vectors_path.relative_to(output))
                row["raw"] = {key: value for key, value in raw.items() if key != "vectors"}
                if corpus.model_hash(model) != before or before != row["model_sha256"]:
                    raise RuntimeError("backend mutated frozen original-coordinate inputs")
                row["assessment"] = assess(model, row["solver"], raw, values, core.checker)
                row["verdict"] = row["assessment"]["verdict"]
            except Exception as error:
                row["verdict"] = "UNSUPPORTED" if isinstance(error, UnsupportedAPI) else "FAIL"
                row["error"] = {"type": type(error).__name__, "message": str(error)}
                log = folder / f"{row['solver']}-failure.log"
                with log.open("x", encoding="utf-8") as stream:
                    stream.write(traceback.format_exc())
                row["failure_log"] = str(log.relative_to(output))
                traceback.print_exc()
            corpus.write_json(folder / f"{row['solver']}.json", row)
        report.update(summarize(report["rows"], report["source_eligibility"], values))
    except Exception as error:
        report["verdict"], report["exit_code"] = "FAIL", 2
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "case_id": None if current is None else current["case_id"],
                           "solver": None if current is None else current["solver"]}
        with (output / "failure.log").open("x", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        traceback.print_exc()
        if report["rows"] and "config" in report:
            report.update(summarize(report["rows"], report["source_eligibility"], report["config"]))
            report["verdict"], report["exit_code"] = "FAIL", 2
        else:
            report["planned_denominator"] = len(report["rows"])
            report["attempted_count"] = sum(row["attempted"] for row in report["rows"])
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["files"] = inventory(output)
        corpus.write_json(output / "result.json", report)
    return output / "result.json", report["exit_code"]


def replay(result_path: Path, result_pin: str) -> dict:
    """Read-only numerical replay, anchored by the caller's result.json SHA."""
    corpus._sha(result_pin, "result_sha256")
    result_path = result_path.expanduser().resolve(strict=True)
    if digest(result_path) != result_pin:
        raise ValueError("result SHA256 mismatch")
    output = result_path.parent
    report = corpus.read_json(result_path)
    if report.get("schema_version") != 1 or report.get("runner_version") != VERSION:
        raise ValueError("unsupported result schema/version")
    files = report.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("empty replay inventory")
    for relative, pin in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe replay artifact path")
        path = (output / path).resolve(strict=True)
        if not path.is_relative_to(output) or digest(path) != pin["sha256"] or path.stat().st_size != pin["bytes"]:
            raise ValueError(f"replay artifact SHA256/size mismatch: {relative}")
    if report["verdict"] not in ("PASS", "UNQUALIFIED", "FAIL"):
        raise ValueError("RUNNING/unknown is not a replayable terminal result")
    if "error" in report:
        if "failure.log" not in files:
            raise ValueError("FAIL result lacks first-cause log")
        return {"verdict": "REPLAYED", "qualification_verdict": "FAIL",
                "qualification_exit_code": 2, "result_sha256": result_pin,
                "planned_denominator": report["planned_denominator"],
                "attempted_count": report["attempted_count"], "numerically_replayed": 0}
    frozen_dir = output / "frozen"
    importer = load_module(output / "code" / "importer.py", "_public_qp_replay_importer")
    importer.verify_frozen(frozen_dir, report["corpus_sha256"], report["max_dense_bytes"])
    saved_runner = load_module(output / "code" / "runner.py", "_public_qp_replay_runner")
    saved_runner.corpus = importer
    frozen = corpus.read_json(frozen_dir / "corpus.json")
    expected = [(row["case_id"], solver, row["model_sha256"])
                for row in frozen["cases"] for solver in SOLVERS]
    actual = [(row["case_id"], row["solver"], row["model_sha256"]) for row in report["rows"]]
    if actual != expected or report["solvers"] != list(SOLVERS):
        raise ValueError("replay case/backend denominator mismatch")
    acceptance = None
    if "public_acceptance_sha256" in report:
        if digest(output / "public-acceptance.json") != report["public_acceptance_sha256"]:
            raise ValueError("replay public acceptance pin mismatch")
        acceptance = corpus.read_json(output / "public-acceptance.json")
    eligibility = saved_runner.source_eligibility(
        frozen, frozen_dir, acceptance, report["corpus_sha256"],
    )
    if eligibility != report["source_eligibility"]:
        raise ValueError("replay public source eligibility mismatch")
    core = load_core(output / "code")
    checked = 0
    for row in report["rows"]:
        if row["attempt"] != 1 or row["denominator"] != 1:
            raise ValueError("replay automatic retry/denominator mismatch")
        if row["verdict"] in ("FAIL", "UNSUPPORTED"):
            if row.get("failure_log") not in files or not row.get("error") or not row["attempted"]:
                raise ValueError("failed/unsupported attempt lacks first-cause evidence")
        elif row["verdict"] in ("QUALIFIED", "UNQUALIFIED"):
            if row.get("vectors_path") not in files or not row["attempted"]:
                raise ValueError("replay missing host sidecar")
            raw = {**row["raw"], "vectors": read_vectors(output / row["vectors_path"])}
            measured = saved_runner.assess(
                saved_runner.load_model(frozen_dir, row["case_id"]), row["solver"],
                raw, report["config"], core.checker,
            )
            if measured != row["assessment"] or measured["verdict"] != row["verdict"]:
                raise ValueError("saved host evidence numerical/status replay mismatch")
            checked += 1
        else:
            raise ValueError("successful execution left a missing/unknown attempt")
        row_path = f"attempts/{row['case_id']}/{row['solver']}.json"
        if row_path not in files or corpus.read_json(output / row_path) != row:
            raise ValueError("attempt/result row mismatch")
    measured_summary = saved_runner.summarize(report["rows"], eligibility, report["config"])
    if any(report.get(key) != value for key, value in measured_summary.items()):
        raise ValueError("replay aggregate/denominator/qualification mismatch")
    return {"verdict": "REPLAYED", "qualification_verdict": report["verdict"],
            "qualification_exit_code": report["exit_code"], "result_sha256": result_pin,
            "planned_denominator": report["planned_denominator"],
            "attempted_count": report["attempted_count"], "numerically_replayed": checked}


def v3_exact(actual, expected, context: str) -> None:
    """Types, complete field sets and exact values are part of the contract."""
    if type(actual) is not type(expected):
        raise ValueError(f"{context}: type mismatch")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{context}: field membership mismatch")
        for key in expected:
            v3_exact(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{context}: length mismatch")
        for index, value in enumerate(expected):
            v3_exact(actual[index], value, f"{context}[{index}]")
    elif actual != expected:
        raise ValueError(f"{context}: value mismatch")


def v3_sources() -> dict:
    """The trusted local producer and immutable base, not rehashed sidecar code."""
    pins = {}
    for name, relative in V3_SOURCES.items():
        path = ROOT / relative
        raw = path.read_bytes()
        if name != "runner.py":
            found = subprocess.run(["git", "show", f"{V3_BASE}:{relative}"], cwd=ROOT,
                                   capture_output=True, check=False)
            if found.returncode or raw != found.stdout:
                raise ValueError(f"v3 dependency differs from reviewed base: {relative}")
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def v3_core(directory: Path):
    # Capture once, then compile those buffers, never a second pathname read.
    # This low-level handle is not trusted until v3_loaded_sources validates it.
    buffers = {name: (directory / name).read_bytes() for name in
               ("certificates.py", "qp.py", "scaling.py", "refinement.py")}
    return _v3_buffer_core(directory, buffers)


def _v3_buffer_core(directory: Path, buffers: dict) -> SimpleNamespace:
    package_name = "_public_qp_v3_" + hashlib.sha256(
        str(directory.resolve()).encode() + b"".join(buffers.values())).hexdigest()[:16]
    # Each load has a separate cache identity, including the trusted comparison.
    package_name += "_" + str(time.monotonic_ns())
    package = ModuleType(package_name)
    package.__path__ = [str(directory.resolve())]
    sys.modules[package_name] = package
    modules = {}
    try:
        for filename, raw in buffers.items():
            name = filename[:-3]
            path = (directory / filename).resolve(strict=True)
            spec = importlib.util.spec_from_file_location(package_name + "." + name, path)
            if spec is None:
                raise ValueError(f"v3 loaded source spec missing: {filename}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module.__name__] = module
            setattr(package, name, module)
            exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
            modules[name] = module
    except BaseException:
        for name in (package_name, *(package_name + "." + key for key in modules)):
            sys.modules.pop(name, None)
        raise
    return SimpleNamespace(qp=modules["qp"], checker=modules["certificates"],
                           scaling=modules["scaling"], refinement=modules["refinement"])


def _v3_live_binding(actual_modules: dict, trusted_modules: dict) -> None:
    """Compare live objects to independently executed, hash-checked source buffers.

    No candidate, solver or actual constructor is executed by this comparison.
    Generated dataclass and known decorator code/closures are compared too.
    """
    paired = {id(value): actual_modules[name] for name, value in trusted_modules.items()}
    for name, module in trusted_modules.items():
        for key, value in vars(module).items():
            if isinstance(value, type) and value.__module__ == module.__name__:
                paired[id(value)] = vars(actual_modules[name]).get(key)
    visited = set()

    def fail(label):
        raise ValueError(f"v3 actual loaded source binding differs: {label}")

    def compare(actual, wanted, label):
        pair = (id(actual), id(wanted))
        if pair in visited:
            return
        visited.add(pair)
        if id(wanted) in paired:
            if actual is not paired[id(wanted)]:
                fail(label + " alias")
            return
        if actual is wanted:
            return
        if is_dataclass(wanted) and not isinstance(wanted, type):
            expected_class = paired.get(id(type(wanted)))
            if type(actual) is not expected_class:
                fail(label + " default class")
            compare(vars(actual), vars(wanted), label + " default fields")
            return
        if type(actual) is not type(wanted):
            fail(label + " type")
        if isinstance(wanted, FunctionType):
            if actual.__code__ != wanted.__code__:
                fail(label + " code")
            expected_globals = next(
                (vars(actual_modules[name]) for name, module in trusted_modules.items()
                 if wanted.__globals__ is vars(module)), wanted.__globals__)
            if actual.__globals__ is not expected_globals:
                fail(label + " globals")
            compare(actual.__defaults__, wanted.__defaults__, label + " defaults")
            compare(actual.__kwdefaults__, wanted.__kwdefaults__, label + " kwdefaults")
            compare(vars(actual), vars(wanted), label + " wrapper attributes")
            if len(actual.__closure__ or ()) != len(wanted.__closure__ or ()):
                fail(label + " closure")
            for index, (left, right) in enumerate(zip(
                    actual.__closure__ or (), wanted.__closure__ or ())):
                compare(left.cell_contents, right.cell_contents, f"{label} closure[{index}]")
        elif isinstance(wanted, MethodType):
            compare(actual.__func__, wanted.__func__, label + " method")
            compare(actual.__self__, wanted.__self__, label + " receiver")
        elif isinstance(wanted, property):
            for key in ("fget", "fset", "fdel"):
                compare(getattr(actual, key), getattr(wanted, key), label + "." + key)
        elif isinstance(wanted, (staticmethod, classmethod)):
            compare(actual.__func__, wanted.__func__, label + " descriptor")
        elif type(wanted) is dict:
            if actual.keys() != wanted.keys():
                fail(label + " membership")
            for key in wanted:
                compare(actual[key], wanted[key], label + "." + str(key))
        elif type(wanted) in (tuple, list):
            if len(actual) != len(wanted):
                fail(label + " length")
            for index, value in enumerate(wanted):
                compare(actual[index], value, f"{label}[{index}]")
        elif type(wanted) in (str, int, float, bool, bytes, type(None), set, frozenset):
            if actual != wanted:
                fail(label + " value")
        elif type(wanted).__module__.startswith("torch.") and hasattr(wanted, "__dict__"):
            # no_grad's context-factory receiver: not an arbitrary source object.
            compare(vars(actual), vars(wanted), label + " torch decorator state")
        else:
            fail(label + " unsupported/nonidentical object")

    # Classes are registered before resolving captured config-instance defaults.
    for name, wanted_module in trusted_modules.items():
        actual_module = actual_modules[name]
        wanted_vars = {key: value for key, value in vars(wanted_module).items()
                       if not key.startswith("__")}
        actual_vars = {key: value for key, value in vars(actual_module).items()
                       if not key.startswith("__")}
        if actual_vars.keys() != wanted_vars.keys():
            fail(name + " module membership")
        for key, wanted in wanted_vars.items():
            actual = actual_vars[key]
            label = name + "." + key
            if isinstance(wanted, type) and wanted.__module__ == wanted_module.__name__:
                if (type(actual) is not type or actual.__module__ != actual_module.__name__
                        or actual.__bases__ != (object,) or actual.__new__ is not object.__new__):
                    fail(label + " class identity")
                left, right = vars(actual), vars(wanted)
                if left.keys() != right.keys():
                    fail(label + " class membership")
                for attr, value in right.items():
                    if attr in ("__module__", "__dict__", "__weakref__", "__dataclass_fields__",
                                "__dataclass_params__"):
                        continue
                    compare(left[attr], value, label + "." + attr)
                if is_dataclass(wanted):
                    if not is_dataclass(actual):
                        fail(label + " dataclass")
                    for attr in ("init", "repr", "eq", "order", "unsafe_hash", "frozen"):
                        compare(getattr(actual.__dataclass_params__, attr),
                                getattr(wanted.__dataclass_params__, attr),
                                label + " dataclass parameter " + attr)
                    if [f.name for f in fields(actual)] != [f.name for f in fields(wanted)]:
                        fail(label + " dataclass fields")
                    for a, w in zip(fields(actual), fields(wanted)):
                        for attr in ("default", "default_factory", "init", "repr", "hash",
                                     "compare", "kw_only", "type", "metadata"):
                            value = getattr(w, attr)
                            if attr == "metadata":
                                compare(dict(getattr(a, attr)), dict(value), label + "." + a.name)
                            elif value is MISSING:
                                if getattr(a, attr) is not MISSING:
                                    fail(label + "." + a.name + " missing default")
                            else:
                                compare(getattr(a, attr), value, label + "." + a.name + "." + attr)
            else:
                compare(actual, wanted, label)


def v3_contracts_module():
    from scripts import experiment_public_robustness

    return experiment_public_robustness


def v3_settings() -> dict:
    """No defaults derived from saved metadata, or from an unreviewed new API."""
    contracts = v3_contracts_module()
    return {
        "openopt": {
            "solver": {
                "device": "cpu", "dtype": "torch.float64", "rho": 1.0, "sigma": 1e-6,
                "relaxation": 1.6, "max_iterations": 2000, "check_interval": 25,
                "eps_abs": 1e-10, "eps_rel": 1e-10, "linear_solver": "cholesky",
                "adaptive_rho": True, "adaptive_rho_interval": 100,
                "adaptive_rho_tolerance": 5.0, "rho_min": 1e-6, "rho_max": 1e6,
                "dual_gap_termination": True, "direction_termination": True,
                "direction_atol": 1e-8, "direction_rtol": 1e-8,
            },
            "scaling": {"method": "ruiz", "iterations": 10, "scale_min": 1e-6,
                        "scale_max": 1e6, "objective_scaling": True,
                        "check_atol": 1e-7, "check_rtol": 1e-7},
            "refinement": {"device": "cpu", "dtype": "torch.float64",
                           "factor_dtype": "torch.float64", "max_refinement_steps": 3,
                           "active_atol": 1e-5, "active_rtol": 1e-7, "check_atol": 1e-7,
                           "check_rtol": 1e-7, "check_arithmetic": V3_ARITHMETIC},
        },
        "osqp": {"requested": {**contracts.OSQP_REQUESTED, "max_iter": 2000},
                 "effective": {**contracts.OSQP_EFFECTIVE, "max_iter": 2000}},
        "clarabel": {"requested": dict(contracts.CLARABEL_EFFECTIVE),
                     "effective": dict(contracts.CLARABEL_EFFECTIVE)},
    }


def v3_configs(core):
    import torch

    configs = []
    for name, cls in (("solver", core.qp.QPConfig), ("scaling", core.scaling.ScalingConfig),
                      ("refinement", core.refinement.RefinementConfig)):
        values = dict(v3_settings()["openopt"][name])
        for key in ("dtype", "factor_dtype"):
            if key in values:
                values[key] = torch.float64
        config = cls(**values)
        observed = {key: str(value) if key in ("dtype", "factor_dtype") else value
                    for key, value in asdict(config).items()}
        v3_exact(observed, v3_settings()["openopt"][name], f"v3 {name} API/settings")
        configs.append(config)
    return tuple(configs)


def v3_execution_contract() -> dict:
    return {
        "schema_version": 3, "version": V3, "base_commit": V3_BASE,
        "scope": "bounded synthetic CPU worker execution only; not public qualification",
        "gates": dict(V3_GATES), "settings": v3_settings(), "modes": list(V3_MODES),
        "checker_arithmetic": V3_ARITHMETIC,
        "cold_hot": {
            "attempts_per_mode": 1, "solves_per_worker": 2, "warm_start": False,
            "cold": "fresh setup + solve + original-coordinate refinement/host output",
            "hot": "same workspace; new zero-initialized solve; no previous x/y input",
            "openopt_initialization": "x=y=0; z=projection(0); reset rho and factor",
            "osqp_initialization": "warm_starting=0; no warm_start call",
            "clarabel_initialization": "DefaultSolver.solve default initialization; no warm-start API",
            "hot_excludes": "module/runtime/input setup, equilibration and initial factorization",
            "timing_claim": False,
        },
        "environment": dict(V3_ENV), "torch_intraop": 1, "torch_interop": 1,
        "dimension_limit": {"max_n": 8, "max_m": 64, "max_cases": 4},
        "public_execution_authorized": False, "license_verified": False,
        "public_redistribution": False, "automatic_retry": False, "tiny_dual_clipping": False,
    }


def v3_runtime() -> dict:
    """Observe selected loaded entry/native bytes and query actual native threads."""
    import importlib
    import torch

    contracts = v3_contracts_module()
    modules = {}
    for name in ("numpy", "numpy._core._multiarray_umath", "scipy", "scipy.linalg._flapack",
                 "scipy.sparse", "torch", "torch._C", "osqp", "osqp.interface",
                 "osqp.ext_builtin", "clarabel", "clarabel.clarabel", "highspy",
                 "highspy._core"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        modules[name] = {"file": path.name, "sha256": digest(path)}
    versions = {name: importlib.metadata.version(name)
                for name in (*contracts.PACKAGE_VERSIONS, "highspy")}
    v3_exact({name: versions[name] for name in contracts.PACKAGE_VERSIONS},
             contracts.PACKAGE_VERSIONS, "v3 runtime packages")
    v3_exact(sys.version.split()[0], "3.10.20", "v3 runtime Python")
    v3_exact(versions["highspy"], "1.15.1", "v3 runtime highspy")
    for name in contracts.PACKAGE_VERSIONS:
        v3_exact(str(importlib.import_module(name).__version__), versions[name],
                 f"v3 loaded version {name}")
    environment = {key: os.environ.get(key) for key in V3_ENV}
    v3_exact(environment, V3_ENV, "v3 thread environment")
    v3_exact(torch.get_num_threads(), 1, "v3 torch intraop threads")
    v3_exact(torch.get_num_interop_threads(), 1, "v3 torch interop threads")
    libraries = {}
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith("/"):
            path = Path(fields[-1])
            if "site-packages" in str(path) and (
                "openblas" in path.name.lower() or "libgomp" in path.name.lower()
                or "libmkl" in path.name.lower()
            ):
                paths.add(path.resolve(strict=True))
    for path in sorted(paths):
        library = ctypes.CDLL(str(path))
        queries = {}
        for symbol in ("openblas_get_num_threads64_", "openblas_get_num_threads",
                       "scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
                       "MKL_Get_Max_Threads", "omp_get_max_threads"):
            query = getattr(library, symbol, None)
            if query is not None:
                query.argtypes, query.restype = (), ctypes.c_int
                count = int(query())
                v3_exact(count, 1, f"v3 native threads {path.name}.{symbol}")
                queries[symbol] = count
        if not queries:
            raise ValueError(f"v3 loaded threading library unobservable: {path.name}")
        libraries[path.name] = {"sha256": digest(path), "thread_queries": queries}
    if not libraries:
        raise ValueError("v3 native thread observation missing")
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "packages": versions, "modules": modules, "thread_libraries": libraries,
            "environment": environment, "torch_intraop": 1, "torch_interop": 1,
            "closure": "selected loaded modules/thread libraries; not full dependency/hardware closure"}


def v3_initialize_threads() -> None:
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def v3_member(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or relative != Path(relative).as_posix()
            or Path(relative).is_absolute() or ".." in Path(relative).parts):
        raise ValueError("v3 unsafe inventory path")
    current = root
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("v3 symlink member rejected")
    if not current.resolve(strict=True).is_relative_to(root) or not current.is_file():
        raise ValueError("v3 member escapes root")
    return current


def v3_validate_context(output: Path, plan: dict) -> SimpleNamespace:
    v3_exact(plan["contract"], v3_execution_contract(), "v3 predeclared contract")
    pins = v3_sources()
    v3_exact(plan["sources"], pins, "v3 source pins")
    for name, pin in pins.items():
        v3_exact(digest(v3_member(output, "code/" + name)), pin, f"v3 saved source {name}")
    v3_exact(plan["runtime"], v3_runtime(), "v3 observed runtime")
    corpus.verify_frozen(output / "frozen", plan["corpus_sha256"], corpus.DEFAULT_MAX_DENSE_BYTES)
    frozen = corpus.read_json(output / "frozen/corpus.json")
    if not 1 <= len(frozen["cases"]) <= 4:
        raise ValueError("v3 bounded case count rejected")
    expected = []
    for case in frozen["cases"]:
        if case["origin"] != "synthetic":
            raise ValueError("v3 public execution not authorized in this scope")
        model = load_model(output / "frozen", case["case_id"])
        view = corpus.solver_arrays(model)
        if not 1 <= len(view["q"]) <= 8 or len(view["l"]) > 64:
            raise ValueError("v3 bounded reproducible arithmetic dimensions rejected")
        expected.extend({"case_id": case["case_id"], "model_sha256": case["model_sha256"],
                         "solver": solver, "mode": mode, "attempt": 1, "denominator": 1}
                        for solver in SOLVERS for mode in V3_MODES)
    v3_exact(plan["rows"], expected, "v3 complete case/backend/mode plan")
    core = v3_core(output / "code")
    v3_loaded_sources(output, core)
    v3_configs(core)
    v3_exact(core.checker.REPRODUCIBLE_ARITHMETIC, V3_ARITHMETIC, "v3 checker API")
    return core


def v3_loaded_sources(output: Path, core) -> dict:
    pins = v3_sources()
    modules = {"qp.py": core.qp, "certificates.py": core.checker,
               "scaling.py": core.scaling, "refinement.py": core.refinement}
    loaded = {}
    buffers = {}
    package_name = core.qp.__package__
    package = sys.modules.get(package_name)
    if not isinstance(package, ModuleType) or package.__path__ != [str((output / "code").resolve())]:
        raise ValueError("v3 actual loaded source package identity differs")
    for name, module in modules.items():
        expected = v3_member(output, "code/" + name).resolve(strict=True)
        origin = Path(module.__spec__.origin).resolve(strict=True)
        if origin != expected or Path(module.__file__).resolve(strict=True) != expected:
            raise ValueError(f"v3 loaded core module escaped frozen source: {name}")
        if (module.__name__ != package_name + "." + name[:-3]
                or module.__package__ != package_name
                or sys.modules.get(module.__name__) is not module
                or getattr(package, name[:-3], None) is not module):
            raise ValueError(f"v3 actual loaded source module/cache identity differs: {name}")
        buffers[name] = origin.read_bytes()
        v3_exact(hashlib.sha256(buffers[name]).hexdigest(), pins[name],
                 f"v3 loaded source bytes {name}")
        loaded[name] = {"file": name, "sha256": pins[name]}
    # Dependency order matters; the trusted source never imports an artifact
    # through an ordinary loader, and its temporary cache is always removed.
    ordered = {name: buffers[name] for name in
               ("certificates.py", "qp.py", "scaling.py", "refinement.py")}
    trusted = _v3_buffer_core(output / "code", ordered)
    trusted_modules = {"qp.py": trusted.qp, "certificates.py": trusted.checker,
                       "scaling.py": trusted.scaling, "refinement.py": trusted.refinement}
    try:
        _v3_live_binding(modules, trusted_modules)
        for name in modules:
            v3_exact(digest(v3_member(output, "code/" + name)), pins[name],
                     f"v3 loaded source post-check bytes {name}")
    finally:
        trusted_package = trusted.qp.__package__
        for module in trusted_modules.values():
            sys.modules.pop(module.__name__, None)
        sys.modules.pop(trusted_package, None)
    return loaded


class _V3Checker:
    def __init__(self, checker):
        self.checker = checker

    def check_qp_candidate(self, *args, **kwargs):
        return self.checker.check_qp_candidate(*args, **kwargs, arithmetic=V3_ARITHMETIC)


def v3_assess(model, solver, raw, core):
    if STATUS[solver].get(raw["status"]) not in ("optimal", None):
        raise ValueError("v3 bounded scope does not accept direction qualification")
    return assess(model, solver, raw, V3_GATES, _V3Checker(core.checker))


def v3_tensor(value):
    if value.device.type != "cpu":
        raise ValueError("v3 non-CPU result; no fallback")
    return value.detach().numpy().copy()


def v3_setup(model: dict, solver: str, core) -> SimpleNamespace:
    import torch
    from scipy import sparse

    view = corpus.solver_arrays(model)
    settings = v3_settings()
    if solver == "openopt":
        config, scaling, refinement = v3_configs(core)
        p = core.qp.QPProblem(*[
            torch.tensor(view[key], dtype=torch.float64, device="cpu")
            for key in ("P", "q", "A", "l", "u")])
        prepared = core.scaling.equilibrate_qp(p, config, scaling)
        workspace = core.qp.prepare_qp(prepared.scaled_problem, config)
        return SimpleNamespace(workspace=workspace, p=p, prepared=prepared,
                               scaling=scaling, refinement=refinement, requested=settings["openopt"])
    if solver == "osqp":
        import osqp

        requested = settings["osqp"]["requested"]
        workspace = osqp.OSQP(algebra="builtin")
        workspace.setup(P=sparse.csc_matrix(np.triu(view["P"])), q=view["q"].copy(),
                        A=sparse.csc_matrix(view["A"]), l=view["l"].copy(), u=view["u"].copy(),
                        **requested)
        return SimpleNamespace(workspace=workspace, requested=requested)
    import clarabel

    A, b, cones, mapping = clarabel_form(view, clarabel)
    requested = settings["clarabel"]["requested"]
    effective = clarabel.DefaultSettings()
    for key, value in requested.items():
        setattr(effective, key, float("inf") if value == "Infinity" else value)
    workspace = clarabel.DefaultSolver(sparse.csc_matrix(np.triu(view["P"])),
                                      view["q"].copy(), sparse.csc_matrix(A), b, cones, effective)
    return SimpleNamespace(workspace=workspace, requested=requested, mapping=mapping)


def v3_observed_settings(state, solver: str) -> dict:
    if solver == "openopt":
        return {
            "solver": {key: str(value) if key == "dtype" else value
                       for key, value in asdict(state.workspace.config).items()},
            "scaling": asdict(state.scaling),
            "refinement": {key: str(value) if key in ("dtype", "factor_dtype") else value
                           for key, value in asdict(state.refinement).items()},
        }
    settings = state.workspace.settings if solver == "osqp" else state.workspace.get_settings()
    return v3_contracts_module().settings_snapshot(settings)


def v3_check_settings(raw: dict, solver: str) -> None:
    contract = v3_settings()[solver]
    requested = contract if solver == "openopt" else contract["requested"]
    effective = contract if solver == "openopt" else contract["effective"]
    v3_exact(raw["requested"], requested, f"v3 {solver} requested")
    for stage in ("settings", "settings_before_solve", "settings_after_solve"):
        v3_exact(raw[stage], effective, f"v3 {solver} {stage}")


def v3_observe_call(state, model, solver, core) -> tuple[dict, dict]:
    """Count the real bound workspace entry, not an increment beside a call."""
    counts = {"workspace_solve": 0, "backend_native_solve": 0, "other_solver": 0, "warm_start": 0,
              "refinement_proposal": 0}
    previous = sys.getprofile()
    method = state.workspace.solve
    code = getattr(inspect.unwrap(getattr(method, "__func__", method)), "__code__", None)
    native = state.workspace._solver if solver == "osqp" else (
        state.workspace if solver == "clarabel" else None)
    native_class = type(native) if native is not None else None
    original_native = native_class.solve if native_class is not None else None

    def native_call(self, *args, **kwargs):
        if self is not native:
            counts["other_solver"] += 1
        elif solver == "clarabel":
            counts["workspace_solve"] += 1
        else:
            counts["backend_native_solve"] += 1
        return original_native(self, *args, **kwargs)

    def profile(frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            if code is not None and frame.f_code is code and frame.f_locals.get("self") is state.workspace:
                counts["workspace_solve"] += 1
            elif name in ("solve", "solve_qp"):
                counts["other_solver"] += 1
            elif name == "warm_start":
                counts["warm_start"] += 1
            elif name == "refine_qp_candidate":
                counts["refinement_proposal"] += 1
        if previous is not None:
            previous(frame, event, arg)

    if native_class is not None:
        native_class.solve = native_call
    sys.setprofile(profile)
    try:
        raw, extra = v3_solve(state, model, solver, core)
    finally:
        sys.setprofile(previous)
        if native_class is not None:
            native_class.solve = original_native
    v3_exact(counts, {"workspace_solve": 1, "backend_native_solve": 1 if solver == "osqp" else 0,
                     "other_solver": 0, "warm_start": 0,
                     "refinement_proposal": 1 if solver == "openopt" else 0},
             f"v3 observed actual solver calls {counts}")
    raw["actual_calls"] = counts
    return raw, extra


def v3_solve(state, model, solver: str, core) -> tuple[dict, dict]:
    before = v3_observed_settings(state, solver)
    effective = v3_settings()[solver]
    v3_exact(before, effective if solver == "openopt" else effective["effective"],
             f"v3 {solver} before actual solve")
    started = time.perf_counter()
    solved = state.workspace.solve(raise_error=False) if solver == "osqp" else state.workspace.solve()
    solve_seconds = time.perf_counter() - started
    extra, detail = {}, {}
    if solver == "openopt":
        x, y = state.prepared.scaling.restore(solved.x, solved.y)
        refined = core.refinement.refine_qp_candidate(state.p, x, y, state.refinement)
        extra = {"scaled_x": v3_tensor(solved.x), "scaled_y": v3_tensor(solved.y),
                 "restored_x": v3_tensor(x), "restored_y": v3_tensor(y),
                 "variable_scale": v3_tensor(state.prepared.scaling.variable_scale),
                 "constraint_scale": v3_tensor(state.prepared.scaling.constraint_scale),
                 "objective_scale": np.asarray(state.prepared.scaling.objective_scale)}
        if refined.attempted_x is not None:
            extra.update(attempted_x=v3_tensor(refined.attempted_x),
                         attempted_y=v3_tensor(refined.attempted_y))
        vectors = {"x": v3_tensor(refined.x), "y": v3_tensor(refined.y)}
        detail = {
            "solver_residuals": {"coordinate": "scaled", "primal": solved.primal_residual,
                                 "dual": solved.dual_residual, "normal": solved.normal_cone_residual},
            "rho_updates": solved.rho_updates, "final_rho": solved.final_rho,
            "refinement": {"status": refined.status, "reason": refined.reason,
                           "diagnostics": refined.diagnostics,
                           "baseline": refined.baseline_evidence.to_dict(),
                           "returned": refined.evidence.to_dict(),
                           "attempted": None if refined.attempted_evidence is None
                           else refined.attempted_evidence.to_dict()},
        }
        objective = float(.5 * refined.x @ state.p.P @ refined.x + state.p.q @ refined.x)
        status, iterations = solved.status, solved.iterations
    elif solver == "osqp":
        vectors = {"x": solved.x.copy(), "y": solved.y.copy()}
        objective, status, iterations = solved.info.obj_val, solved.info.status, int(solved.info.iter)
        detail["solver_residuals"] = {"coordinate": "OSQP reporting",
                                     "primal": solved.info.prim_res, "dual": solved.info.dual_res}
    else:
        z = np.asarray(solved.z, dtype=np.float64)
        if z.shape != (len(state.mapping),):
            raise ValueError("v3 Clarabel original dual mapping shape")
        y = np.zeros(len(corpus.solver_arrays(model)["l"]))
        for value, (index, sign) in zip(z, state.mapping):
            y[index] += sign * value
        extra["cone_z"] = z
        vectors = {"x": np.asarray(solved.x), "y": y}
        objective, status, iterations = solved.obj_val, str(solved.status), int(solved.iterations)
        detail["dual_mapping"] = [list(item) for item in state.mapping]
        detail["solver_residuals"] = {"coordinate": "Clarabel reporting",
                                     "primal": solved.r_prim, "dual": solved.r_dual}
    vectors["reported_objective"] = np.asarray(objective)
    after = v3_observed_settings(state, solver)
    raw = {"status": status, "iterations": iterations, "observed_device": "cpu",
           "requested": state.requested, "settings": after, "settings_before_solve": before,
           "settings_after_solve": after, "solve_seconds": solve_seconds, **detail,
           "vectors": vectors}
    v3_check_settings(raw, solver)
    return raw, extra


def v3_worker(output: Path, plan_pin: str, index: int) -> int:
    output = external(output)
    plan_path = v3_member(output, "plan.json")
    corpus._sha(plan_pin, "v3 caller worker plan SHA256")
    if digest(plan_path) != plan_pin:
        raise ValueError("v3 caller worker plan pin mismatch")
    plan = corpus.read_json(plan_path)
    folder = output / "workers" / f"{index:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 3, "version": V3, "plan_sha256": plan_pin,
               "worker_index": index, "rows": [], "verdict": "RUNNING",
               "started_at": datetime.now(timezone.utc).isoformat(), "solve_calls": 0}
    try:
        v3_initialize_threads()
        core = v3_validate_context(output, plan)
        loaded_sources = v3_loaded_sources(output, core)
        if type(index) is not int or not 0 <= index < len(plan["rows"]) // 2:
            raise ValueError("v3 worker index outside complete plan")
        initials = plan["rows"][2 * index:2 * index + 2]
        model = load_model(output / "frozen", initials[0]["case_id"])
        before = corpus.model_hash(model)
        solver = initials[0]["solver"]
        state = None
        for initial in initials:
            row = {**initial, "attempted": False, "verdict": "NOT_ATTEMPTED"}
            receipt["rows"].append(row)
            begin = time.perf_counter()
            if initial["mode"] == "cold":
                state = v3_setup(model, solver, core)
            row["attempted"] = True
            raw, extra = v3_observe_call(state, model, solver, core)
            receipt["solve_calls"] += raw["actual_calls"]["workspace_solve"]
            row["elapsed_seconds"] = time.perf_counter() - begin
            v3_exact(v3_runtime(), plan["runtime"], "v3 worker post-solve runtime")
            v3_exact(v3_loaded_sources(output, core), loaded_sources, "v3 worker loaded sources")
            v3_exact(corpus.model_hash(model), before, "v3 immutable original arrays")
            vectors = folder / f"{initial['mode']}.npz"
            save_vectors(vectors, raw["vectors"])
            transforms = folder / f"{initial['mode']}-transform.npz"
            v3_contracts_module().save_extra(transforms, extra)
            row.update(raw={key: value for key, value in raw.items() if key != "vectors"},
                       vectors_path=str(vectors.relative_to(output)),
                       transform_path=str(transforms.relative_to(output)),
                       input_model_before=before, input_model_after=corpus.model_hash(model),
                       runtime=plan["runtime"], plan_sha256=plan_pin,
                       loaded_sources=loaded_sources,
                       assessment=v3_assess(model, solver, raw, core))
            row["verdict"] = row["assessment"]["verdict"]
        receipt["verdict"] = "WORKER_COMPLETE_NOT_QUALIFICATION"
    except Exception as error:
        receipt.update(verdict="FAIL", error={"type": type(error).__name__, "message": str(error)})
        if receipt["rows"]:
            receipt["rows"][-1].update(verdict="FAIL", error=receipt["error"])
        with (folder / "failure.log").open("x", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        traceback.print_exc()
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    corpus.write_json(folder / "result.json", receipt)
    return 2 if receipt["verdict"] == "FAIL" else 0


def v3_run(args) -> tuple[Path, int]:
    output = external(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 3, "runner_version": V3, "verdict": "RUNNING",
              "rows": [], "started_at": datetime.now(timezone.utc).isoformat(),
              "public_qualified_count": 0, "industrial_qualified_count": 0,
              "worker_execution_verified": False, "exit_code": 2}
    try:
        v3_exact(args.version, V3, "v3 explicit executable version")
        if (args.device != "cpu" or args.source_root.resolve() != ROOT
                or args.public_acceptance is not None or args.public_acceptance_sha256 is not None
                or args.max_dense_bytes != corpus.DEFAULT_MAX_DENSE_BYTES):
            raise ValueError("v3 no source/device/acceptance override or public execution")
        default = parser().parse_args(["run", "--corpus", str(args.corpus),
                                      "--manifest-sha256", args.manifest_sha256,
                                      "--artifact-dir", str(args.artifact_dir)])
        v3_exact(config_values(args), config_values(default), "v3 rejects v1 setting overrides")
        v3_initialize_threads()
        pins = v3_sources()
        frozen_path = args.corpus.expanduser().resolve(strict=True)
        corpus.verify_frozen(frozen_path, args.manifest_sha256, corpus.DEFAULT_MAX_DENSE_BYTES)
        frozen = corpus.read_json(frozen_path / "corpus.json")
        if any(row["origin"] != "synthetic" for row in frozen["cases"]):
            raise ValueError("v3 public execution not authorized in this scope")
        for filename in ("corpus.json", "input-manifest.json"):
            copy_file(frozen_path / filename, output / "frozen" / filename)
        for row in frozen["cases"]:
            for filename in ("raw.mps", "arrays.npz", "metadata.json", "reader.log"):
                copy_file(frozen_path / row["case_id"] / filename,
                          output / "frozen" / row["case_id"] / filename)
        for name, relative in V3_SOURCES.items():
            copy_file(ROOT / relative, output / "code" / name)
        plan = {"contract": v3_execution_contract(), "sources": pins,
                "runtime": v3_runtime(), "corpus_sha256": args.manifest_sha256,
                "rows": [{"case_id": case["case_id"], "model_sha256": case["model_sha256"],
                          "solver": solver, "mode": mode, "attempt": 1, "denominator": 1}
                         for case in frozen["cases"] for solver in SOLVERS for mode in V3_MODES]}
        v3_validate_context(output, plan)
        corpus.write_json(output / "plan.json", plan)
        plan_pin = digest(output / "plan.json")
        report.update(plan_sha256=plan_pin, contract=plan["contract"], sources=pins,
                      runtime=plan["runtime"], corpus_sha256=args.manifest_sha256)
        report["rows"] = [{**row, "attempted": False, "verdict": "NOT_ATTEMPTED"}
                          for row in plan["rows"]]
        for index in range(len(plan["rows"]) // 2):
            log = output / f"worker-{index:03d}.log"
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker",
                       "--version", V3, "--artifact-dir", str(output),
                       "--plan-sha256", plan_pin, "--worker-index", str(index)]
            with log.open("xb") as stream:
                process = subprocess.run(command, cwd=ROOT, env={**os.environ, **V3_ENV},
                                         stdout=stream, stderr=subprocess.STDOUT, check=False)
            worker = corpus.read_json(output / "workers" / f"{index:03d}" / "result.json")
            for offset, worker_row in enumerate(worker["rows"]):
                report["rows"][index * 2 + offset] = worker_row
            if process.returncode != 0 or worker["verdict"] != "WORKER_COMPLETE_NOT_QUALIFICATION":
                raise RuntimeError(f"v3 worker failed index={index} rc={process.returncode}; {worker.get('error')}")
            v3_exact(worker["solve_calls"], 2, "v3 worker solve count")
            report["rows"][index * 2:index * 2 + 2] = worker["rows"]
        report.update(verdict="EXECUTION_COMPLETE_NOT_QUALIFICATION", exit_code=0,
                      worker_execution_verified=True,
                      numeric_verdict="PASS" if all(row["verdict"] == "QUALIFIED"
                                                   for row in report["rows"]) else "UNQUALIFIED")
    except Exception as error:
        report.update(verdict="FAIL", error={"type": type(error).__name__, "message": str(error)})
        with (output / "failure.log").open("x", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        traceback.print_exc()
    report["planned_denominator"] = len(report["rows"])
    report["attempted_count"] = sum(row["attempted"] for row in report["rows"])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["files"] = inventory(output)
    corpus.write_json(output / "result.json", report)
    return output / "result.json", report["exit_code"]


def v3_check_transform(model, row, vectors, extra, core):
    import torch

    solver = row["solver"]
    if solver == "osqp":
        v3_exact(extra, {}, "v3 OSQP transformation")
        return
    if solver == "clarabel":
        import clarabel

        v3_exact(sorted(extra), ["cone_z"], "v3 Clarabel transform fields")
        _, _, _, mapping = clarabel_form(corpus.solver_arrays(model), clarabel)
        v3_exact(row["raw"]["dual_mapping"], [list(item) for item in mapping], "v3 cone mapping")
        y = np.zeros_like(vectors["y"])
        if extra["cone_z"].shape != (len(mapping),):
            raise ValueError("v3 cone dual dimension mismatch")
        for z, (index, sign) in zip(extra["cone_z"], mapping):
            y[index] += sign * z
        if not np.array_equal(y, vectors["y"]):
            raise ValueError("v3 original signed dual reconstruction mismatch")
        return
    required = {"scaled_x", "scaled_y", "restored_x", "restored_y", "variable_scale",
                "constraint_scale", "objective_scale"}
    if set(extra) not in (required, required | {"attempted_x", "attempted_y"}):
        raise ValueError("v3 refinement transform membership")
    config, scaling, _ = v3_configs(core)
    view = corpus.solver_arrays(model)
    p = core.qp.QPProblem(*[torch.tensor(view[key], dtype=torch.float64)
                           for key in ("P", "q", "A", "l", "u")])
    # Transform reconstruction is not a solve/refinement proposal.
    transformed = core.scaling.equilibrate_qp(p, config, scaling)
    for key, value in (("variable_scale", transformed.scaling.variable_scale),
                       ("constraint_scale", transformed.scaling.constraint_scale)):
        if not np.array_equal(v3_tensor(value), extra[key]):
            raise ValueError("v3 deterministic scaling mismatch")
    if float(extra["objective_scale"]) != transformed.scaling.objective_scale:
        raise ValueError("v3 objective scaling mismatch")
    x, y = transformed.scaling.restore(torch.tensor(extra["scaled_x"]),
                                      torch.tensor(extra["scaled_y"]))
    if not np.array_equal(v3_tensor(x), extra["restored_x"]) or not np.array_equal(
        v3_tensor(y), extra["restored_y"]
    ):
        raise ValueError("v3 original-coordinate restoration mismatch")
    detail = row["raw"]["refinement"]
    for stage, x, y in (("baseline", extra["restored_x"], extra["restored_y"]),
                        ("returned", vectors["x"], vectors["y"])):
        measured = core.checker.check_qp_candidate(problem(view), x, y, atol=1e-7, rtol=1e-7,
                                                  arithmetic=V3_ARITHMETIC).to_dict()
        v3_exact(detail[stage], measured, f"v3 refinement {stage} evidence")
    if "attempted_x" in extra:
        measured = core.checker.check_qp_candidate(
            problem(view), extra["attempted_x"], extra["attempted_y"],
            atol=1e-7, rtol=1e-7, arithmetic=V3_ARITHMETIC).to_dict()
        v3_exact(detail["attempted"], measured, "v3 attempted evidence")
    elif detail["attempted"] is not None:
        raise ValueError("v3 missing attempted refinement vectors")
    if detail["status"] == "accepted":
        if not detail["returned"]["passed"] or core.refinement._nonworsening(
            SimpleNamespace(metrics=detail["baseline"]["metrics"]),
            SimpleNamespace(metrics=detail["returned"]["metrics"])
        ):
            raise ValueError("v3 accepted refinement failed original nonworsening gates")
    elif detail["status"] not in ("unknown", "rejected") or not (
        np.array_equal(vectors["x"], extra["restored_x"])
        and np.array_equal(vectors["y"], extra["restored_y"])
    ):
        raise ValueError("v3 rejected refinement changed original returned values")


def v3_replay(path: Path, pin: str) -> dict:
    v3_initialize_threads()
    corpus._sha(pin, "v3 caller result pin")
    if path.is_symlink():
        raise ValueError("v3 symlink result rejected")
    path = path.resolve(strict=True)
    if digest(path) != pin:
        raise ValueError("v3 caller result SHA256 mismatch")
    report, output = corpus.read_json(path), path.parent
    v3_exact(report["schema_version"], 3, "v3 result schema")
    v3_exact(report["runner_version"], V3, "v3 result version; v1/v2 never promoted")
    if report["verdict"] != "EXECUTION_COMPLETE_NOT_QUALIFICATION" or report.get("error"):
        raise ValueError("v3 incomplete/failed run remains failed; not replay-qualified")
    files = report["files"]
    actual = {str(item.relative_to(output)) for item in output.rglob("*") if item.is_file()}
    if not files or actual != set(files) | {"result.json"}:
        raise ValueError("v3 exact inventory membership mismatch")
    for relative, expected in files.items():
        item = v3_member(output, relative)
        v3_exact(expected, {"sha256": digest(item), "bytes": item.stat().st_size},
                 "v3 inventory byte hash")
    plan_path = v3_member(output, "plan.json")
    v3_exact(digest(plan_path), report["plan_sha256"], "v3 result/plan pin")
    plan = corpus.read_json(plan_path)
    core = v3_validate_context(output, plan)
    loaded_sources = v3_loaded_sources(output, core)
    for key in ("contract", "sources", "runtime", "corpus_sha256"):
        v3_exact(report[key], plan[key], f"v3 result/plan {key}")
    if len(report["rows"]) != len(plan["rows"]):
        raise ValueError("v3 complete denominator mismatch")
    for index in range(len(plan["rows"]) // 2):
        worker = corpus.read_json(v3_member(output, f"workers/{index:03d}/result.json"))
        v3_exact(worker["schema_version"], 3, "v3 worker schema")
        v3_exact(worker["version"], V3, "v3 worker version")
        v3_exact(worker["worker_index"], index, "v3 worker index")
        v3_exact(worker["plan_sha256"], report["plan_sha256"], "v3 worker plan pin")
        v3_exact(worker["solve_calls"], 2, "v3 cold/hot actual solve counts")
        v3_exact(worker["verdict"], "WORKER_COMPLETE_NOT_QUALIFICATION", "v3 worker terminal state")
        v3_exact(worker["rows"], report["rows"][2 * index:2 * index + 2],
                 "v3 worker/parent rows")
        v3_member(output, f"worker-{index:03d}.log")
    for initial, row in zip(plan["rows"], report["rows"]):
        v3_exact({key: row[key] for key in initial}, initial, "v3 case/backend/mode membership")
        v3_exact(row["attempted"], True, "v3 actual attempt")
        v3_exact(row["plan_sha256"], report["plan_sha256"], "v3 row plan pin")
        v3_exact(row["runtime"], plan["runtime"], "v3 row runtime")
        v3_exact(row["loaded_sources"], loaded_sources, "v3 row loaded core identity")
        worker_index = plan["rows"].index(initial) // 2
        for key, suffix in (("vectors_path", ".npz"), ("transform_path", "-transform.npz")):
            v3_exact(row[key], f"workers/{worker_index:03d}/{initial['mode']}{suffix}",
                     "v3 mode-bound sidecar path")
        raw = row["raw"]
        v3_check_settings(raw, row["solver"])
        v3_exact(raw["actual_calls"], {"workspace_solve": 1,
                 "backend_native_solve": 1 if row["solver"] == "osqp" else 0,
                 "other_solver": 0, "warm_start": 0,
                 "refinement_proposal": 1 if row["solver"] == "openopt" else 0},
                 "v3 observed callcount contract")
        v3_exact(raw["observed_device"], "cpu", "v3 actual placement")
        if type(raw["iterations"]) is not int or not 1 <= raw["iterations"] <= (
            200 if row["solver"] == "clarabel" else 2000
        ):
            raise ValueError("v3 iteration count outside predeclared bound")
        for value in (row["elapsed_seconds"], raw["solve_seconds"]):
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError("v3 invalid measured timing")
        model = load_model(output / "frozen", row["case_id"])
        for key in ("model_sha256", "input_model_before", "input_model_after"):
            v3_exact(row[key], corpus.model_hash(model), "v3 original model identity")
        vectors = read_vectors(v3_member(output, row["vectors_path"]))
        measured = v3_assess(model, row["solver"], {**raw, "vectors": vectors}, core)
        v3_exact(row["assessment"], measured, "v3 independent original six-gate replay")
        v3_exact(row["verdict"], measured["verdict"], "v3 original failed verdict retained")
        extra = v3_contracts_module().read_extra(v3_member(output, row["transform_path"]))
        v3_check_transform(model, row, vectors, extra, core)
    v3_exact(report["planned_denominator"], len(plan["rows"]), "v3 full denominator")
    v3_exact(report["attempted_count"], len(plan["rows"]), "v3 actual attempted denominator")
    v3_exact(report["numeric_verdict"], "PASS" if all(row["verdict"] == "QUALIFIED"
             for row in report["rows"]) else "UNQUALIFIED", "v3 numerical summary")
    for key in ("public_qualified_count", "industrial_qualified_count"):
        v3_exact(report[key], 0, "v3 no qualification upgrade")
    v3_exact(report["worker_execution_verified"], True, "v3 execution binding")
    v3_exact(report["exit_code"], 0, "v3 completion exit code")
    return {"verdict": "EXECUTION_REPLAYED_NOT_QUALIFICATION", "schema_version": 3,
            "runner_version": V3, "result_sha256": pin, "plan_sha256": report["plan_sha256"],
            "numerically_replayed": len(report["rows"]), "solve_calls": 0,
            "numeric_verdict": report["numeric_verdict"], "public_qualified_count": 0,
            "checked_at": datetime.now(timezone.utc).isoformat()}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--version", choices=(VERSION, V3), default=VERSION)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--manifest-sha256", required=True, help="externally pinned corpus.json SHA256")
    run.add_argument("--artifact-dir", type=Path, required=True)
    run.add_argument("--source-root", type=Path, default=ROOT)
    run.add_argument("--device", default="cpu", help="only cpu accepted; GPU explicitly refused")
    run.add_argument("--public-acceptance", type=Path)
    run.add_argument("--public-acceptance-sha256")
    run.add_argument("--max-iterations", type=int, default=5000)
    run.add_argument("--check-interval", type=int, default=25)
    run.add_argument("--adaptive-rho-interval", type=int, default=100)
    run.add_argument("--eps-abs", type=float, default=1e-7)
    run.add_argument("--eps-rel", type=float, default=1e-7)
    run.add_argument("--direction-atol", type=float, default=1e-8)
    run.add_argument("--direction-rtol", type=float, default=1e-8)
    run.add_argument("--max-dense-bytes", type=int, default=corpus.DEFAULT_MAX_DENSE_BYTES)
    check = commands.add_parser("replay")
    check.add_argument("--result", type=Path, required=True)
    check.add_argument("--result-sha256", required=True)
    check.add_argument("--version", choices=(VERSION, V3), default=VERSION)
    worker = commands.add_parser("worker", help="explicit v3 bounded synthetic CPU worker")
    worker.add_argument("--version", choices=(V3,), required=True)
    worker.add_argument("--artifact-dir", type=Path, required=True)
    worker.add_argument("--plan-sha256", required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            if args.version == V3:
                result, code = v3_run(args)
            else:
                result, code = qualify(args)
            print("PUBLIC_QP_RESULT " + json.dumps({
                "result": str(result), "result_sha256": digest(result),
                "verdict": corpus.read_json(result)["verdict"], "exit_code": code,
            }, sort_keys=True), flush=True)
            return code
        if args.command == "worker":
            return v3_worker(args.artifact_dir, args.plan_sha256, args.worker_index)
        if args.version == V3:
            result = v3_replay(args.result, args.result_sha256)
            print("PUBLIC_QP_REPLAY " + json.dumps(result, sort_keys=True), flush=True)
            return 0
        result = replay(args.result, args.result_sha256)
        print("PUBLIC_QP_REPLAY " + json.dumps(result, sort_keys=True), flush=True)
        # Integrity success does not turn UNQUALIFIED/FAIL into CLI success.
        return result["qualification_exit_code"]
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
