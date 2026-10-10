"""Single-attempt CPU qualification of caller-pinned, externally frozen QPs.

No acquisition, installation, retries, planted witnesses, GPU fallback, license
claim or speed ranking. Replay checks saved host vectors without solving again.
Exit 0=PASS, 1=UNQUALIFIED, 2=FAIL; a synthetic run can never be public PASS.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import traceback
from types import ModuleType, SimpleNamespace
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import public_qp_corpus as corpus

VERSION = "public-qp-qualification-v1"
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


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
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
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            result, code = qualify(args)
            print("PUBLIC_QP_RESULT " + json.dumps({
                "result": str(result), "result_sha256": digest(result),
                "verdict": corpus.read_json(result)["verdict"], "exit_code": code,
            }, sort_keys=True), flush=True)
            return code
        result = replay(args.result, args.result_sha256)
        print("PUBLIC_QP_REPLAY " + json.dumps(result, sort_keys=True), flush=True)
        # Integrity success does not turn UNQUALIFIED/FAIL into CLI success.
        return result["qualification_exit_code"]
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
