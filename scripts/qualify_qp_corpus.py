"""Qualify identical frozen synthetic QPs against independent constructed oracles.

One cold solve per backend; this is correctness/status evidence, not a speed
ranking. Exit 0=all selected rows qualified, 1=unqualified status/quality,
2=execution/input error. Exceptions stop execution and retain their traceback.
No GPU scheduling, solver tuning, retries, silent skips or fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from openopt.corpus import (
    FAMILIES, GENERATOR_VERSION, SCOPE, CorpusCase, build_corpus,
    check_infeasibility, check_unboundedness, generate_case, quality, validate_oracle,
)
from openopt.qp import QPConfig, solve_qp

SHARED_TMP = Path("/hf3fs-jd/data/tmp").resolve()
SOLVERS = ("openopt", "osqp", "clarabel")
OSQP_SETTING_NAMES = (
    "eps_abs", "eps_rel", "eps_prim_inf", "eps_dual_inf", "scaling",
    "scaled_termination", "max_iter", "adaptive_rho", "check_termination",
)


def write_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    if path.stat().st_size != len(text.encode("utf-8")):
        raise OSError(f"artifact byte count mismatch: {path}")


def clarabel_form(case: CorpusCase) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Convert only finite bounds and retain the signed original-row mapping."""
    import clarabel

    rows, bounds, mapping = [], [], []
    equality = np.flatnonzero(case.l == case.u)
    for index in equality:
        rows.append(case.A[index])
        bounds.append(case.u[index])
        mapping.append((int(index), 1.0))
    for index in np.flatnonzero(case.l != case.u):
        if np.isfinite(case.u[index]):
            rows.append(case.A[index])
            bounds.append(case.u[index])
            mapping.append((int(index), 1.0))
        if np.isfinite(case.l[index]):
            rows.append(-case.A[index])
            bounds.append(-case.l[index])
            mapping.append((int(index), -1.0))
    cones = []
    if len(equality):
        cones.append(clarabel.ZeroConeT(len(equality)))
    if len(rows) > len(equality):
        cones.append(clarabel.NonnegativeConeT(len(rows) - len(equality)))
    matrix = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(case.q))
    return matrix, np.asarray(bounds, dtype=np.float64), cones, mapping


def run_solver(case: CorpusCase, solver: str, config: QPConfig) -> dict:
    """Backend adapters return raw vectors/status, not a truth assertion."""
    start = time.perf_counter()
    if solver == "openopt":
        if torch.device(config.device).type == "cuda":
            torch.cuda.synchronize(torch.device(config.device))
        result = solve_qp(case.problem(), config)
        x, y = result.x.detach().cpu().numpy(), result.y.detach().cpu().numpy()
        if torch.device(config.device).type == "cuda":
            torch.cuda.synchronize(torch.device(config.device))
        return {
            "status": result.status, "iterations": result.iterations, "x": x, "y": y,
            "reported_objective": result.objective,
            "primal_certificate": None if result.infeasibility_direction is None
            else result.infeasibility_direction.cpu().numpy(),
            "dual_certificate": None if result.unbounded_direction is None
            else result.unbounded_direction.cpu().numpy(),
            "feasible_base": None if result.feasible_base is None else result.feasible_base.cpu().numpy(),
            "observed_device": str(result.x.device), "elapsed_s": time.perf_counter() - start,
        }
    if solver == "osqp":
        import osqp
        from scipy import sparse

        workspace = osqp.OSQP()
        workspace.setup(
            P=sparse.csc_matrix(np.triu(case.P)), q=case.q.copy(),
            A=sparse.csc_matrix(case.A), l=case.l.copy(), u=case.u.copy(),
            eps_abs=config.eps_abs, eps_rel=config.eps_rel,
            max_iter=config.max_iterations, verbose=False, polishing=False,
            warm_starting=False,
        )
        # Non-optimal statuses are data, not OSQP API exceptions. They still
        # fail qualification below; no max_iter/infeasible exception is hidden.
        result = workspace.solve(raise_error=False)
        return {
            "status": result.info.status, "iterations": result.info.iter,
            "x": result.x, "y": result.y, "primal_certificate": result.prim_inf_cert,
            "dual_certificate": result.dual_inf_cert, "observed_device": "cpu",
            "settings": {name: getattr(workspace.settings, name) for name in OSQP_SETTING_NAMES},
            "elapsed_s": time.perf_counter() - start,
        }
    if solver == "clarabel":
        import clarabel
        from scipy import sparse

        A, b, cones, mapping = clarabel_form(case)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = 1
        settings.max_iter = config.max_iterations
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = min(
            config.eps_abs, config.eps_rel,
        )
        workspace = clarabel.DefaultSolver(
            sparse.csc_matrix(np.triu(case.P)), case.q.copy(),
            sparse.csc_matrix(A), b, cones, settings,
        )
        result = workspace.solve()
        z = np.asarray(result.z, dtype=np.float64)
        if z.shape != (len(mapping),):
            raise ValueError("Clarabel dual dimension does not match finite-bound mapping")
        y = np.zeros(len(case.l))
        for value, (index, sign) in zip(z, mapping):
            y[index] += sign * value
        x = np.asarray(result.x, dtype=np.float64)
        return {
            "status": str(result.status), "iterations": result.iterations,
            "x": x, "y": y, "primal_certificate": y, "dual_certificate": x,
            "observed_device": "cpu", "elapsed_s": time.perf_counter() - start,
        }
    raise ValueError(f"unknown solver: {solver}")


def assess(case: CorpusCase, solver: str, raw: dict, residual_tol: float, gap_tol: float) -> dict:
    expected = case.oracle.expected
    accepted = {
        "openopt": {"optimal": "solved", "infeasible": "infeasible", "unbounded": "unbounded"},
        "osqp": {"optimal": "solved", "infeasible": "primal infeasible", "unbounded": "dual infeasible"},
        "clarabel": {"optimal": "Solved", "infeasible": "PrimalInfeasible", "unbounded": "DualInfeasible"},
    }
    target = accepted[solver][expected]
    status_ok = target is not None and raw["status"] == target
    candidate = None
    certificate = None
    if expected == "optimal" or solver == "openopt":
        candidate = quality(case, raw["x"], raw["y"], residual_tol, gap_tol)
    if status_ok and expected == "infeasible":
        if raw.get("primal_certificate") is not None:
            certificate = check_infeasibility(case, raw["primal_certificate"], residual_tol)
    if status_ok and expected == "unbounded":
        base = raw.get("feasible_base") if solver == "openopt" else case.oracle.x
        if raw.get("dual_certificate") is not None and base is not None:
            certificate = check_unboundedness(
                case, raw["dual_certificate"], base, residual_tol,
            )
    measured = candidate if expected == "optimal" else certificate
    passed = status_ok and measured is not None and measured["verdict"] == "PASS"
    reason = None
    if not status_ok:
        reason = "solver status did not establish the independently expected property"
        if solver == "openopt" and expected != "optimal" and raw["status"] == "max_iterations":
            reason = "prototype lacks infeasible/unbounded certificates; max_iterations is not a solver bug"
    elif not passed:
        reason = "independent residual/objective/certificate check failed"
    return {
        "verdict": "QUALIFIED" if passed else "UNQUALIFIED",
        "solver_status": raw["status"], "status_matches_expected": status_ok,
        "expected_property": expected, "candidate_quality": candidate,
        "certificate_quality": certificate, "reason": reason,
        "inaccurate_statuses_are_accepted": False,
        "quality_scope": "independent CPU NumPy using frozen input, never solver residuals",
    }


def select_cases(args: argparse.Namespace) -> list[CorpusCase]:
    if args.case:
        if args.size or args.seed:
            raise ValueError("--case embeds n/seed; do not combine it with --size/--seed")
        if len(args.case) != len(set(args.case)):
            raise ValueError("duplicate case selection")
        cases = []
        for case_id in args.case:
            match = re.fullmatch(r"([a-z_]+)-n([0-9]+)-seed([0-9]+)", case_id)
            if match is None:
                raise ValueError(f"invalid case ID: {case_id}")
            case = generate_case(match[1], int(match[2]), int(match[3]))
            if case.case_id != case_id:
                raise ValueError(f"noncanonical case ID: {case_id}")
            cases.append(case)
        return cases
    return build_corpus(
        args.size if args.size is not None else (8, 24, 64),
        args.seed if args.seed is not None else (20261010,),
        args.family if args.family is not None else FAMILIES,
    )


def diagnose_osqp(source: Path, output: Path, certificate_tolerance: float) -> dict:
    """Read retained failures, then vary only eps_prim_inf; never replace them."""
    import osqp
    from scipy import sparse

    source = source.resolve()
    original = json.loads((source / "result.json").read_text())
    failures = [
        row for row in original["rows"]
        if row["solver"] == "osqp" and row["expected_property"] == "infeasible"
        and row["verdict"] == "UNQUALIFIED"
    ]
    if not failures:
        raise ValueError("diagnostic source has no failed OSQP infeasibility rows")
    if not np.isfinite(certificate_tolerance) or certificate_tolerance <= 0:
        raise ValueError("diagnostic eps_prim_inf must be finite and positive")
    config = original["config"]
    gate = original["quality_tolerances"]["residual"]
    results = []
    for row in failures:
        case_id = row["case_id"]
        match = re.fullmatch(r"([a-z_]+)-n([0-9]+)-seed([0-9]+)", case_id)
        if match is None:
            raise ValueError(f"invalid retained case ID: {case_id}")
        case = generate_case(match[1], int(match[2]), int(match[3]))
        if case.problem_hash != row["problem_hash"]:
            raise ValueError(f"retained problem hash mismatch: {case_id}")
        with np.load(source / case_id / "problem.npz", allow_pickle=False) as saved:
            for name in ("P", "q", "A", "l", "u"):
                if not np.array_equal(saved[name], getattr(case, name)):
                    raise ValueError(f"retained problem bytes mismatch: {case_id}/{name}")
        with np.load(source / case_id / "osqp-vectors.npz", allow_pickle=False) as saved:
            certificate = saved["primal_certificate"].copy()
        normalized = certificate / np.linalg.norm(certificate, np.inf)
        # Separate scalar column-dot calculation avoids the checking function's
        # matvec path, and uses the saved vector with no row/sign remapping.
        scalar_residual = max(
            abs(sum(float(a) * float(y) for a, y in zip(column, normalized)))
            for column in case.A.T
        )
        measured = check_infeasibility(case, certificate, gate)
        if measured != row["certificate_quality"] or not np.isclose(
            scalar_residual, measured["annihilation_residual"], rtol=1e-12, atol=1e-15,
        ):
            raise ValueError(f"saved-vector independent recomputation mismatch: {case_id}")
        workspace = osqp.OSQP()
        workspace.setup(
            P=sparse.csc_matrix(np.triu(case.P)), q=case.q.copy(),
            A=sparse.csc_matrix(case.A), l=case.l.copy(), u=case.u.copy(),
            eps_abs=config["eps_abs"], eps_rel=config["eps_rel"],
            max_iter=config["max_iterations"], verbose=False, polishing=False,
            warm_starting=False,
        )
        baseline_settings = {
            name: getattr(workspace.settings, name) for name in OSQP_SETTING_NAMES
        }
        # The original failed solve is NOT repeated. Probe just the changed
        # infeasibility stopping tolerance, retaining the old failure as truth.
        workspace.update_settings(eps_prim_inf=certificate_tolerance)
        changed_settings = {
            name: getattr(workspace.settings, name) for name in OSQP_SETTING_NAMES
        }
        changed_fields = [
            name for name in OSQP_SETTING_NAMES
            if baseline_settings[name] != changed_settings[name]
        ]
        if changed_fields != ["eps_prim_inf"]:
            raise ValueError(f"unexpected diagnostic settings change: {changed_fields}")
        result = workspace.solve(raise_error=False)
        strict = check_infeasibility(case, result.prim_inf_cert, gate) \
            if result.info.status == "primal infeasible" else None
        strict_check = check_infeasibility(case, result.prim_inf_cert, 1e-8) \
            if strict is not None else None
        folder = output / case_id
        folder.mkdir()
        np.savez(
            folder / "certificate-comparison.npz", original=certificate,
            changed_eps_prim_inf=result.prim_inf_cert, exact=case.oracle.certificate,
        )
        comparison = {
            "case_id": case_id, "problem_hash": case.problem_hash,
            "source": str(source / case_id / "osqp.json"),
            "original_status": row["solver_status"], "original_verdict": row["verdict"],
            "original_independent_quality": measured,
            "original_scalar_column_dot_residual": scalar_residual,
            "original_distance_to_constructed_direction": float(
                np.max(np.abs(normalized - case.oracle.certificate))
            ),
            "baseline_effective_settings": baseline_settings,
            "changed_effective_settings": changed_settings, "changed_fields": changed_fields,
            "changed_status": result.info.status, "changed_iterations": int(result.info.iter),
            "changed_certificate_quality_same_gate": strict,
            "changed_certificate_quality_at_1e_8": strict_check,
            "first_cause": (
                "baseline adapter left eps_prim_inf at 1e-4 while independent gate is 5e-6; "
                "returned direction is approximate at that stopping threshold, not remapped"
            ),
            "original_failure_is_not_reclassified": True,
            "diagnostic_is_not_a_default_setting_change": True,
        }
        write_json(folder / "diagnosis.json", comparison)
        results.append(comparison)
        print(
            f"OSQP_DIAG case={case_id} original_residual={scalar_residual:.9g} "
            f"default_eps_prim_inf={baseline_settings['eps_prim_inf']} "
            f"changed_status={result.info.status} changed_quality={strict}", flush=True,
        )
    return {
        "mode": "retained-failure plus one changed-tolerance diagnostic per case",
        "source": str(source), "cases": results, "same_independent_gate": gate,
        "original_solver_eps_abs_rel": [config["eps_abs"], config["eps_rel"]],
        "strict_1e_8_is_diagnostic_only_not_original_qualification_gate": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--case", action="append", help="canonical family-nN-seedS; repeatable")
    selection.add_argument("--family", action="append", choices=FAMILIES)
    result.add_argument("--size", action="append", type=int)
    result.add_argument("--seed", action="append", type=int)
    result.add_argument("--device", default="cpu", help="strict openopt CPU/CUDA device; references use CPU")
    result.add_argument("--solver", action="append", choices=SOLVERS)
    result.add_argument("--max-iterations", type=int, default=5000)
    result.add_argument("--eps-abs", type=float, default=1e-7)
    result.add_argument("--eps-rel", type=float, default=1e-7)
    result.add_argument("--residual-tol", type=float, default=5e-6)
    result.add_argument("--gap-tol", type=float, default=1e-6)
    result.add_argument("--adaptive-rho", action="store_true")
    result.add_argument("--dual-gap-termination", action="store_true")
    result.add_argument("--direction-termination", action="store_true")
    result.add_argument(
        "--diagnose-osqp", type=Path,
        help="read failed OSQP infeasible rows from a retained run; separate one-variable probe",
    )
    result.add_argument("--diagnostic-eps-prim-inf", type=float, default=1e-7)
    result.add_argument(
        "--artifact-dir", type=Path, required=True,
        help="new exclusive directory under /hf3fs-jd/data/tmp; never overwrites evidence",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.artifact_dir.expanduser().resolve()
    if output.is_relative_to(ROOT) or not output.is_relative_to(SHARED_TMP):
        raise ValueError("artifacts must be outside the worktree and under 3FS tmp")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE, "generator_version": GENERATOR_VERSION,
        "verdict": "RUNNING", "exit_code": None, "rows": [], "planned_cases": [],
        "commercial": "UNVERIFIED; no licensed matched runs",
        "timing_scope": "single cold diagnostic solve including setup/transfers; not a performance ranking",
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
    }
    start = time.perf_counter()
    code = 2
    with (output / "execution.log").open("x", encoding="utf-8") as log:
        def event(text: str) -> None:
            log.write(text + "\n")
            log.flush()
            print(text, flush=True)

        current_case, current_solver = None, None
        try:
            torch.set_num_threads(1)
            config = QPConfig(
                device=args.device, max_iterations=args.max_iterations,
                eps_abs=args.eps_abs, eps_rel=args.eps_rel,
                adaptive_rho=args.adaptive_rho, dual_gap_termination=args.dual_gap_termination,
                direction_termination=args.direction_termination,
            )
            config.validate()
            if not all(np.isfinite(v) and v > 0 for v in (args.residual_tol, args.gap_tol)):
                raise ValueError("quality tolerances must be finite and positive")
            solvers = args.solver if args.solver is not None else list(SOLVERS)
            if not solvers or len(solvers) != len(set(solvers)):
                raise ValueError("solver selection must be nonempty and unique")
            cases = select_cases(args)
            if not cases:
                raise ValueError("empty corpus selection")
            report["planned_cases"] = [case.case_id for case in cases]
            report["solvers"] = solvers
            report["software"] = {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", *(s for s in solvers if s != "openopt"),
                             *(("scipy",) if any(s != "openopt" for s in solvers) else ()))
            }
            report["source_hashes"] = {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("src/openopt/corpus.py", "src/openopt/qp.py",
                             "scripts/qualify_qp_corpus.py")
            }
            if args.diagnose_osqp is not None:
                if args.case or args.family or args.size or args.seed or args.solver:
                    raise ValueError("diagnosis selects retained failures; do not mix case/solver filters")
                report["mode"] = "OSQP_CERTIFICATE_DIAGNOSIS"
                report["planned_cases"] = []
                report["solvers"] = []
                report["diagnosis"] = diagnose_osqp(
                    args.diagnose_osqp, output, args.diagnostic_eps_prim_inf,
                )
                report["verdict"] = "DIAGNOSED"
                code = 0
                return code
            report["config"] = {**asdict(config), "dtype": str(config.dtype)}
            report["quality_tolerances"] = {"residual": args.residual_tol, "relative_gap": args.gap_tol}
            report["python_executable"] = sys.executable
            report["cpu_threads"] = torch.get_num_threads()
            for case in cases:
                current_case = case.case_id
                folder = output / case.case_id
                folder.mkdir()
                metadata = case.metadata()
                metadata["oracle_validation"] = validate_oracle(case)
                write_json(folder / "metadata.json", metadata)
                saved = {name: getattr(case, name) for name in ("P", "q", "A", "l", "u")}
                saved.update({
                    name: value for name in ("x", "y", "certificate", "ray")
                    if (value := getattr(case.oracle, name)) is not None
                })
                np.savez(folder / "problem.npz", **saved)
                with np.load(folder / "problem.npz", allow_pickle=False) as frozen:
                    if set(frozen.files) != set(saved) or any(
                        not np.array_equal(frozen[name], value) for name, value in saved.items()
                    ):
                        raise OSError(f"frozen problem mismatch case={current_case}")
                frozen_hash = case.problem_hash
                for solver in solvers:
                    current_solver = solver
                    event(f"CORPUS_START case={current_case} solver={solver} hash={frozen_hash}")
                    raw = run_solver(case, solver, config)
                    if case.problem_hash != frozen_hash:
                        raise RuntimeError(f"backend mutated matched inputs solver={solver}")
                    row = {
                        "case_id": current_case, "class": case.family, "seed": case.seed,
                        "problem_hash": frozen_hash, "solver": solver,
                        "iterations": int(raw["iterations"]), "elapsed_s": raw["elapsed_s"],
                        "observed_device": raw["observed_device"],
                        **assess(case, solver, raw, args.residual_tol, args.gap_tol),
                    }
                    if "settings" in raw:
                        row["effective_settings"] = raw["settings"]
                    if solver == "openopt":
                        requested = torch.device(config.device)
                        actual = torch.device(raw["observed_device"])
                        if requested.type != actual.type or (
                            requested.index is not None and requested.index != actual.index
                        ):
                            raise RuntimeError("explicit openopt device placement mismatch")
                    np.savez(folder / f"{solver}-vectors.npz", **{
                        name: np.asarray(raw[name]) for name in
                        ("x", "y", "primal_certificate", "dual_certificate", "feasible_base")
                        if name in raw and raw[name] is not None
                    })
                    write_json(folder / f"{solver}.json", row)
                    report["rows"].append(row)
                    event(
                        f"CORPUS_ROW verdict={row['verdict']} case={current_case} "
                        f"solver={solver} status={row['solver_status']} reason={row['reason']}"
                    )
            expected_count = len(cases) * len(solvers)
            if not report["rows"] or len(report["rows"]) != expected_count:
                raise RuntimeError("missing or empty qualification rows")
            failures = [row for row in report["rows"] if row["verdict"] != "QUALIFIED"]
            report["unresolved_rows"] = failures
            report["qualified_count"] = expected_count - len(failures)
            report["unqualified_count"] = len(failures)
            report["verdict"] = "PASS" if not failures else "UNQUALIFIED"
            code = 0 if not failures else 1
        except Exception as error:
            report["verdict"] = "ERROR"
            report["error"] = {
                "type": type(error).__name__, "message": str(error),
                "case_id": current_case, "solver": current_solver,
            }
            detail = traceback.format_exc()
            with (output / "failure.log").open("x", encoding="utf-8") as failure:
                failure.write(detail)
            log.write(detail)
            log.flush()
            event(
                f"CORPUS_ERROR case={current_case} solver={current_solver} "
                f"error={type(error).__name__}: {error}"
            )
        finally:
            report["exit_code"] = code
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            report["wall_s"] = time.perf_counter() - start
            completed = {row["case_id"] for row in report["rows"]}
            report["cases_with_no_completed_rows"] = [
                name for name in report["planned_cases"] if name not in completed
            ]
            completed_pairs = {(row["case_id"], row["solver"]) for row in report["rows"]}
            report["missing_rows"] = [
                {"case_id": name, "solver": solver}
                for name in report["planned_cases"] for solver in report.get("solvers", [])
                if (name, solver) not in completed_pairs
            ]
            report["unresolved_rows"] = [
                row for row in report["rows"] if row["verdict"] != "QUALIFIED"
            ]
            write_json(output / "result.json", report)
            event(
                f"OPENOPT_CORPUS verdict={report['verdict']} rc={code} "
                f"rows={len(report['rows'])} result={output / 'result.json'}"
            )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
