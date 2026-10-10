"""Explicit CPU robustness arms on an unchanged caller-pinned public corpus.

Baseline attempts are copied and replayed, never rerun or requalified.
Every new arm has one attempt and unchanged original-coordinate 1e-7 gates.
No retries, multiplier clipping, installation, network or GPU submissions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import public_qp_corpus as corpus
from scripts import qualify_public_qp as runner

VERSION = "public-robustness-v1"
BASE_COMMIT = "e566b0b8163f099a29a0d554fa3b87ad15e4fb67"
GATES = {"eps_abs": 1e-7, "eps_rel": 1e-7,
         "direction_atol": 1e-8, "direction_rtol": 1e-8}
ARMS = (
    {"arm": "openopt-ruiz-refinement", "solver": "openopt", "solve_eps": 1e-10,
     "max_iterations": 10000, "ruiz_iterations": 10, "scale_min": 1e-6,
     "scale_max": 1e6, "objective_scaling": True, "refinement_steps": 3,
     "active_atol": 1e-5, "active_rtol": 1e-7},
    {"arm": "osqp-default-scaling-strict", "solver": "osqp", "solve_eps": 1e-10,
     "max_iterations": 10000},
    {"arm": "clarabel-default-equilibration-strict", "solver": "clarabel",
     "solve_eps": 1e-10, "max_iterations": 200},
)
DEPENDENCIES = {
    "importer.py": "scripts/public_qp_corpus.py",
    "runner.py": "scripts/qualify_public_qp.py",
    "qp.py": "src/openopt/qp.py",
    "certificates.py": "src/openopt/certificates.py",
    "scaling.py": "src/openopt/scaling.py",
    "refinement.py": "src/openopt/refinement.py",
}
STATES = ("QUALIFIED", "UNQUALIFIED", "FAIL", "BUDGET_EXHAUSTED")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON member")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON token: {value}")

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), "JSON object required")
    return value


def member(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    require(parts and not relative.startswith("/") and ".." not in parts
            and "\\" not in relative and relative == "/".join(parts), "unsafe artifact member")
    path = root
    for part in parts:
        path /= part
        require(not path.is_symlink(), "symlink artifact refused")
    require(path.resolve(strict=True).is_relative_to(root) and path.is_file(),
            "artifact escapes evidence root")
    return path


def check_inventory(root: Path, files: dict, result_name: str) -> None:
    require(isinstance(files, dict) and files, "empty inventory")
    for relative, pin in files.items():
        path = member(root, relative)
        require(pin == {"bytes": path.stat().st_size, "sha256": digest(path)},
                f"sidecar byte/hash mismatch: {relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    require(actual == set(files) | {result_name}, "artifact inventory membership mismatch")


def inventory(root: Path) -> dict:
    return {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(root.rglob("*")) if path.is_file() and path != root / "result.json"}


def load_dependencies(directory: Path):
    saved_runner = runner.load_module(directory / "runner.py", "_robustness_runner_" + digest(
        directory / "runner.py")[:16])
    importer = runner.load_module(directory / "importer.py", "_robustness_importer_" + digest(
        directory / "importer.py")[:16])
    saved_runner.corpus = importer
    core = saved_runner.load_core(directory)
    package = core.qp.__package__
    core.scaling = runner.load_module(directory / "scaling.py", package + ".scaling")
    core.refinement = runner.load_module(directory / "refinement.py", package + ".refinement")
    return saved_runner, importer, core


def baseline_check(path: Path, pin: str) -> dict:
    corpus._sha(pin, "caller baseline pin")
    require(not path.is_symlink(), "symlink baseline refused")
    path = path.resolve(strict=True)
    require(not path.is_relative_to(ROOT), "baseline must be external")
    require(digest(path) == pin, "baseline caller SHA256 mismatch")
    report = read_json(path)
    check_inventory(path.parent, report["files"], path.name)
    require(report.get("verdict") == "UNQUALIFIED" and "error" not in report,
            "expected complete negative baseline, not a failed run")
    require(all(report["config"].get(key) == value for key, value in GATES.items()),
            "baseline original gate differs from fixed 1e-7 contract")
    replay = runner.replay(path, pin)
    require(replay["numerically_replayed"] == len(report["rows"]) > 0,
            "baseline candidate replay incomplete")
    return report


def tensor_problem(view: dict, qp):
    import torch

    return qp.QPProblem(*[torch.tensor(view[key], dtype=torch.float64, device="cpu")
                          for key in ("P", "q", "A", "l", "u")])


def numpy_tensor(value):
    require(value.device.type == "cpu", "explicit CPU placement mismatch")
    return value.detach().numpy().copy()


def config_json(config) -> dict:
    return {key: str(value) if key in ("dtype", "factor_dtype") else value
            for key, value in asdict(config).items()}


def openopt_configs(core):
    import torch

    arm = ARMS[0]
    config = core.qp.QPConfig(
        device="cpu", dtype=torch.float64, max_iterations=arm["max_iterations"],
        eps_abs=arm["solve_eps"], eps_rel=arm["solve_eps"], check_interval=25,
        adaptive_rho=True, adaptive_rho_interval=100, dual_gap_termination=True,
        direction_termination=True, direction_atol=1e-8, direction_rtol=1e-8,
    )
    scaling = core.scaling.ScalingConfig(
        iterations=arm["ruiz_iterations"], scale_min=arm["scale_min"],
        scale_max=arm["scale_max"], objective_scaling=arm["objective_scaling"],
        check_atol=GATES["eps_abs"], check_rtol=GATES["eps_rel"],
    )
    refinement = core.refinement.RefinementConfig(
        device="cpu", dtype=torch.float64, factor_dtype=torch.float64,
        max_refinement_steps=arm["refinement_steps"], active_atol=arm["active_atol"],
        active_rtol=arm["active_rtol"], check_atol=1e-7, check_rtol=1e-7,
    )
    return config, scaling, refinement


def run_arm(model: dict, arm: dict, core, saved_runner) -> tuple[dict, dict]:
    """One solve per arm; refinement is a separately retained candidate proposal."""
    view = corpus.solver_arrays(model)
    transfers = {"device": "cpu", "dtype": "float64", "gpu_transfer_count": 0,
                 "host_input_bytes": sum(view[key].nbytes for key in ("P", "q", "A", "l", "u"))}
    extra = {}
    begin = time.perf_counter()
    if arm["solver"] == "openopt":
        p = tensor_problem(view, core.qp)
        transfers["host_to_cpu_tensor_seconds"] = time.perf_counter() - begin
        config, scaling, refinement = openopt_configs(core)
        begin = time.perf_counter()
        prepared = core.scaling.equilibrate_qp(p, config, scaling)
        transfers["equilibration_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        solved = core.qp.solve_qp(prepared.scaled_problem, config)
        transfers["solve_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        x, y = prepared.scaling.restore(solved.x, solved.y)
        transfers["restore_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        refined = core.refinement.refine_qp_candidate(p, x, y, refinement)
        transfers["refinement_seconds"] = time.perf_counter() - begin
        extra = {
            "scaled_x": numpy_tensor(solved.x), "scaled_y": numpy_tensor(solved.y),
            "restored_x": numpy_tensor(x), "restored_y": numpy_tensor(y),
            "variable_scale": numpy_tensor(prepared.scaling.variable_scale),
            "constraint_scale": numpy_tensor(prepared.scaling.constraint_scale),
            "objective_scale": np.asarray(prepared.scaling.objective_scale),
        }
        if refined.attempted_x is not None:
            extra["attempted_x"] = numpy_tensor(refined.attempted_x)
            extra["attempted_y"] = numpy_tensor(refined.attempted_y)
        vectors = {"x": numpy_tensor(refined.x), "y": numpy_tensor(refined.y)}
        vectors["reported_objective"] = np.asarray(float(
            0.5 * refined.x @ p.P @ refined.x + p.q @ refined.x))
        raw = {
            # An accepted transformation does not upgrade an exhausted solver status.
            "status": solved.status, "iterations": solved.iterations, "observed_device": "cpu",
            "settings": {"solver": config_json(config), "scaling": asdict(scaling),
                         "refinement": config_json(refinement)},
            "solver_residuals": {"coordinate": "scaled",
                                "primal": solved.primal_residual, "dual": solved.dual_residual,
                                "normal_cone": solved.normal_cone_residual},
            "rho_updates": solved.rho_updates, "final_rho": solved.final_rho,
            "refinement": {"status": refined.status, "reason": refined.reason,
                           "diagnostics": refined.diagnostics,
                           "baseline_evidence": refined.baseline_evidence.to_dict(),
                           "returned_evidence": refined.evidence.to_dict(),
                           "attempted_evidence": None if refined.attempted_evidence is None
                           else refined.attempted_evidence.to_dict()},
            "vectors": vectors,
        }
        restored_raw = {**raw, "vectors": {"x": extra["restored_x"], "y": extra["restored_y"],
                                          "reported_objective": np.asarray(float(qp_objective(
                                              view, extra["restored_x"])))}}
        raw["before_refinement_assessment"] = saved_runner.assess(
            model, "openopt", restored_raw, GATES, core.checker)
        if "attempted_x" in extra:
            raw["attempted_assessment"] = saved_runner.assess(
                model, "openopt", {**raw, "vectors": {
                    "x": extra["attempted_x"], "y": extra["attempted_y"],
                    "reported_objective": np.asarray(qp_objective(view, extra["attempted_x"]))}},
                GATES, core.checker)
    else:
        from scipy import sparse

        begin = time.perf_counter()
        if arm["solver"] == "osqp":
            import osqp

            workspace = osqp.OSQP()
            requested = {"eps_abs": arm["solve_eps"], "eps_rel": arm["solve_eps"],
                         "eps_prim_inf": 1e-8, "eps_dual_inf": 1e-8,
                         "max_iter": arm["max_iterations"], "verbose": False,
                         "warm_starting": False}
            workspace.setup(P=sparse.csc_matrix(np.triu(view["P"])), q=view["q"].copy(),
                            A=sparse.csc_matrix(view["A"]), l=view["l"].copy(),
                            u=view["u"].copy(), **requested)
            transfers["adapter_setup_seconds"] = time.perf_counter() - begin
            begin = time.perf_counter()
            solved = workspace.solve(raise_error=False)
            transfers["solve_seconds"] = time.perf_counter() - begin
            setting_names = (
                "scaling", "scaled_termination", "polishing", "warm_starting", "rho", "sigma",
                "alpha", "adaptive_rho", "adaptive_rho_interval", "adaptive_rho_tolerance",
                "adaptive_rho_fraction", "check_termination",
            )
            effective = {key: getattr(workspace.settings, key)
                         for key in (*requested, *setting_names)}
            raw = {"status": solved.info.status, "iterations": int(solved.info.iter),
                   "observed_device": "cpu", "settings": effective, "requested": requested,
                   "solver_residuals": {"coordinate": "OSQP original reporting",
                                       "primal": solved.info.prim_res, "dual": solved.info.dual_res},
                   "vectors": {"x": solved.x, "y": solved.y,
                               "reported_objective": np.asarray(solved.info.obj_val)}}
        else:
            import clarabel

            A, b, cones, mapping = saved_runner.clarabel_form(view, clarabel)
            settings = clarabel.DefaultSettings()
            requested = {"verbose": False, "max_threads": 1, "max_iter": arm["max_iterations"],
                         "tol_gap_abs": arm["solve_eps"], "tol_gap_rel": arm["solve_eps"],
                         "tol_feas": arm["solve_eps"], "tol_infeas_abs": 1e-8,
                         "tol_infeas_rel": 1e-8, "presolve_enable": False}
            for key, value in requested.items():
                setattr(settings, key, value)
            workspace = clarabel.DefaultSolver(
                sparse.csc_matrix(np.triu(view["P"])), view["q"].copy(),
                sparse.csc_matrix(A), b, cones, settings)
            transfers["adapter_setup_seconds"] = time.perf_counter() - begin
            begin = time.perf_counter()
            solved = workspace.solve()
            transfers["solve_seconds"] = time.perf_counter() - begin
            z = np.asarray(solved.z, dtype=np.float64)
            require(z.shape == (len(mapping),), "Clarabel mapping shape mismatch")
            y = np.zeros(len(view["l"]))
            for value, (index, sign) in zip(z, mapping):
                y[index] += sign * value
            extra["cone_z"] = z
            raw = {"status": str(solved.status), "iterations": int(solved.iterations),
                   "observed_device": "cpu", "requested": requested,
                   "settings": {key: getattr(settings, key) for key in (
                       *requested, "equilibrate_enable", "equilibrate_max_iter",
                       "equilibrate_min_scaling", "equilibrate_max_scaling",
                       "iterative_refinement_enable", "iterative_refinement_reltol",
                       "iterative_refinement_abstol", "iterative_refinement_max_iter")},
                   "dual_mapping": [list(item) for item in mapping],
                   "solver_residuals": {"coordinate": "Clarabel original reporting",
                                       "primal": solved.r_prim, "dual": solved.r_dual},
                   "vectors": {"x": np.asarray(solved.x), "y": y,
                               "reported_objective": np.asarray(solved.obj_val)}}
    raw["transfers"] = transfers
    return raw, extra


def qp_objective(view: dict, x) -> float:
    return float(0.5 * x @ view["P"] @ x + view["q"] @ x)


def save_extra(path: Path, values: dict) -> None:
    values = {key: np.asarray(value, dtype="<f8") for key, value in values.items()}
    with path.open("xb") as stream:
        np.savez(stream, **values)
    saved = read_extra(path)
    require(saved.keys() == values.keys() and all(
        saved[key].shape == value.shape and saved[key].tobytes() == value.tobytes()
        for key, value in values.items()), "transformation sidecar roundtrip mismatch")


def read_extra(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), "duplicate transformation member")
        values = {key: saved[key].copy() for key in saved.files}
    require(all(value.dtype == np.dtype("<f8") and np.isfinite(value).all()
                for value in values.values()), "nonfinite/nonfloat transformation evidence")
    return values


def worker(output: Path, index: int) -> int:
    plan = read_json(output / "plan.json")
    row = dict(plan["rows"][index])
    folder = output / "attempts" / row["case_id"]
    folder.mkdir(parents=True, exist_ok=True)
    row.update({"attempted": True, "started_at": now()})
    begin = time.perf_counter()
    try:
        saved_runner, importer, core = load_dependencies(output / "code")
        import torch

        torch.set_num_threads(1)
        model = saved_runner.load_model(output / "baseline/frozen", row["case_id"])
        before = importer.model_hash(model)
        require(before == row["model_sha256"], "worker original model hash mismatch")
        raw, extra = run_arm(model, ARMS[index % len(ARMS)], core, saved_runner)
        require(importer.model_hash(model) == before, "arm mutated original arrays")
        vectors = folder / (row["arm"] + ".npz")
        saved_runner.save_vectors(vectors, raw["vectors"])
        extras = folder / (row["arm"] + "-transform.npz")
        save_extra(extras, extra)
        row.update({"vectors_path": str(vectors.relative_to(output)),
                    "transform_path": str(extras.relative_to(output)),
                    "raw": {key: value for key, value in raw.items() if key != "vectors"},
                    "input_model_before": before, "input_model_after": importer.model_hash(model),
                    "assessment": saved_runner.assess(model, row["solver"], raw, GATES, core.checker)})
        row["verdict"] = row["assessment"]["verdict"]
    except Exception as error:
        row.update({"verdict": "FAIL", "error": {"type": type(error).__name__, "message": str(error)}})
        log = folder / (row["arm"] + "-failure.log")
        log.write_text(traceback.format_exc(), encoding="utf-8")
        row["failure_log"] = str(log.relative_to(output))
        traceback.print_exc()
    row.update({"elapsed_seconds": time.perf_counter() - begin, "finished_at": now()})
    corpus.write_json(folder / (row["arm"] + "-worker-result.json"), row)
    return 2 if row["verdict"] == "FAIL" else 0


def summary(rows: list[dict]) -> dict:
    counts = {arm["arm"]: {"denominator": sum(row["arm"] == arm["arm"] for row in rows),
                          **{state: sum(row["arm"] == arm["arm"] and row["verdict"] == state
                                       for row in rows) for state in STATES}}
              for arm in ARMS}
    return {"arm_counts": counts, "planned_denominator": len(rows),
            "attempted_count": sum(row["attempted"] for row in rows),
            "numeric_verdict": "PASS" if rows and all(
                row["verdict"] == "QUALIFIED" for row in rows) else "UNQUALIFIED"}


def run(baseline: Path, pin: str, output: Path, budget: float = 180) -> tuple[Path, int]:
    require(0 < budget <= 180, "CPU experiment budget must be in (0,180] seconds")
    output = runner.external(output)
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    report = {"schema_version": 1, "version": VERSION, "verdict": "RUNNING", "rows": [],
              "started_at": now(), "cpu_budget_seconds": budget, "gates": GATES, "arms": list(ARMS),
              "source_base_commit": BASE_COMMIT,
              "producer_sha256": digest(Path(__file__)), "baseline_sha256": pin,
              "baseline_attempts_modified": False, "automatic_retry": False,
              "tiny_dual_clipping": False, "license_verified": False,
              "public_redistribution": False, "commercial_claim": False, "speedup_claim": False}
    code = 2
    try:
        old = baseline_check(baseline, pin)
        require(digest(ROOT / DEPENDENCIES["importer.py"]) == old["files"]["code/importer.py"]["sha256"],
                "current importer differs from baseline")
        require(digest(ROOT / DEPENDENCIES["certificates.py"]) == old["files"]["code/certificates.py"]["sha256"],
                "current checker differs from baseline")
        for relative in (*old["files"], baseline.name):
            runner.copy_file(member(baseline.parent.resolve(), relative), output / "baseline" / relative)
        for name, relative in DEPENDENCIES.items():
            runner.copy_file(ROOT / relative, output / "code" / name)
        report["source_hashes"] = {relative: digest(output / "code" / name)
                                   for name, relative in DEPENDENCIES.items()}
        runner.copy_file(Path(__file__), output / "code/producer.py")
        frozen = read_json(output / "baseline/frozen/corpus.json")
        report["corpus_sha256"] = old["corpus_sha256"]
        report["public_acceptance_sha256"] = old.get("public_acceptance_sha256")
        report["original_verdict"] = old["verdict"]
        report["baseline_rows"] = old["rows"]
        report["source_eligibility"] = old["source_eligibility"]
        report["rows"] = [
            {"case_id": case["case_id"], "model_sha256": case["model_sha256"],
             "solver": arm["solver"], "arm": arm["arm"], "attempt": 1, "denominator": 1,
             "attempted": False, "verdict": "BUDGET_EXHAUSTED"}
            for case in frozen["cases"] for arm in ARMS]
        corpus.write_json(output / "plan.json", {"rows": report["rows"], "arms": list(ARMS), "gates": GATES})
        for index, initial in enumerate(report["rows"]):
            remaining = budget - (time.perf_counter() - begin)
            folder = output / "attempts" / initial["case_id"]
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-B", str(Path(__file__)), "worker",
                       "--artifact-dir", str(output), "--row-index", str(index)]
            log_path = folder / (initial["arm"] + "-worker.log")
            timed_out = False
            if remaining > 0:
                with log_path.open("xb") as stream:
                    try:
                        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                 timeout=remaining, check=False)
                        worker_code = process.returncode
                    except subprocess.TimeoutExpired:
                        timed_out, worker_code = True, 124
            else:
                log_path.write_text("BUDGET_EXHAUSTED: worker not started\n", encoding="utf-8")
                worker_code = 124
            worker_path = folder / (initial["arm"] + "-worker-result.json")
            if worker_path.is_file() and not timed_out:
                row = read_json(worker_path)
                require(worker_code == (2 if row["verdict"] == "FAIL" else 0),
                        "worker exit/status mismatch")
            else:
                row = {**initial, "attempted": remaining > 0,
                       "verdict": "BUDGET_EXHAUSTED" if worker_code == 124 else "FAIL",
                       "worker_exit_code": worker_code, "finished_at": now(),
                       "error": {"type": "CPU_BUDGET" if worker_code == 124 else "WorkerFailure",
                                 "message": "Explicit budget exhaustion, not solver qualification"
                                 if worker_code == 124 else "worker failed without result"}}
            row["worker_exit_code"] = worker_code
            row["worker_log"] = str(log_path.relative_to(output))
            corpus.write_json(folder / (initial["arm"] + ".json"), row)
            report["rows"][index] = row
            corpus.write_json(output / f"progress-{index}.json",
                              {"checked_at": now(), "rows": report["rows"]})
        report.update(summary(report["rows"]))
        report["verdict"] = "EXPERIMENT_COMPLETE_NOT_QUALIFICATION"
        code = 2 if any(row["verdict"] in ("FAIL", "BUDGET_EXHAUSTED")
                        for row in report["rows"]) else 0
    except Exception as error:
        report.update({"verdict": "ERROR", "error": {"type": type(error).__name__, "message": str(error)}})
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    report.update({"exit_code": code, "finished_at": now(),
                   "elapsed_seconds": time.perf_counter() - begin})
    (output / "experiment.log").write_text(json.dumps({
        key: report[key] for key in ("verdict", "exit_code", "started_at", "finished_at",
                                   "elapsed_seconds")}, sort_keys=True) + "\n", encoding="utf-8")
    report["files"] = inventory(output)
    corpus.write_json(output / "result.json", report)
    return output / "result.json", code


def check_transform(model: dict, row: dict, vectors: dict, extra: dict, core, saved_runner) -> None:
    view = corpus.solver_arrays(model)
    if row["solver"] == "openopt":
        import torch

        required = {"scaled_x", "scaled_y", "restored_x", "restored_y", "variable_scale",
                    "constraint_scale", "objective_scale"}
        require(set(extra) in (required, required | {"attempted_x", "attempted_y"}),
                "scaling/refinement evidence membership mismatch")
        s = core.scaling.DiagonalScaling(torch.tensor(extra["variable_scale"]),
                                        torch.tensor(extra["constraint_scale"]),
                                        float(extra["objective_scale"]))
        require(all(np.all(value >= ARMS[0]["scale_min"]) and np.all(value <= ARMS[0]["scale_max"])
                    for value in (extra["variable_scale"], extra["constraint_scale"],
                                  extra["objective_scale"])), "Ruiz factor exceeds explicit bounds")
        x, y = s.restore(torch.tensor(extra["scaled_x"]), torch.tensor(extra["scaled_y"]))
        require(np.array_equal(numpy_tensor(x), extra["restored_x"])
                and np.array_equal(numpy_tensor(y), extra["restored_y"]),
                "restoration equations mismatch")
        config, scaling, refinement = openopt_configs(core)
        require(row["raw"]["settings"] == {
            "solver": config_json(config), "scaling": asdict(scaling),
            "refinement": config_json(refinement)}, "predeclared Openopt settings mismatch")
        equilibrated = core.scaling.equilibrate_qp(tensor_problem(view, core.qp), config, scaling)
        require(np.array_equal(numpy_tensor(equilibrated.scaling.variable_scale),
                               extra["variable_scale"])
                and np.array_equal(numpy_tensor(equilibrated.scaling.constraint_scale),
                                   extra["constraint_scale"])
                and equilibrated.scaling.objective_scale == float(extra["objective_scale"]),
                "Ruiz factors differ from deterministic original model transformation")
        detail = row["raw"]["refinement"]
        p = saved_runner.problem(view)
        for stage, x, y in (
            ("baseline_evidence", extra["restored_x"], extra["restored_y"]),
            ("returned_evidence", vectors["x"], vectors["y"]),
        ):
            measured = core.checker.check_qp_candidate(p, x, y, atol=1e-7, rtol=1e-7).to_dict()
            require(measured == detail[stage], "refinement saved evidence mismatch")
        before = saved_runner.assess(model, "openopt", {**row["raw"], "vectors": {
            "x": extra["restored_x"], "y": extra["restored_y"],
            "reported_objective": np.asarray(qp_objective(view, extra["restored_x"]))}}, GATES, core.checker)
        require(before == row["raw"]["before_refinement_assessment"], "pre-refinement assessment mismatch")
        if "attempted_x" in extra:
            measured = core.checker.check_qp_candidate(
                p, extra["attempted_x"], extra["attempted_y"], atol=1e-7, rtol=1e-7).to_dict()
            require(measured == detail["attempted_evidence"], "attempted refinement mismatch")
            attempted = saved_runner.assess(model, "openopt", {**row["raw"], "vectors": {
                "x": extra["attempted_x"], "y": extra["attempted_y"],
                "reported_objective": np.asarray(qp_objective(view, extra["attempted_x"]))}},
                GATES, core.checker)
            require(attempted == row["raw"]["attempted_assessment"], "attempted assessment mismatch")
        else:
            require(detail["attempted_evidence"] is None, "missing proposed candidate")
        if detail["status"] == "accepted":
            old, new = detail["baseline_evidence"], detail["returned_evidence"]
            require(new["passed"] and all(new["metrics"][key] is not None and (
                old["metrics"][key] is None or abs(new["metrics"][key]) <= abs(old["metrics"][key]))
                for key in ("primal_feasibility", "stationarity", "bound_sign",
                            "complementarity", "duality_gap")), "nonworsening refinement gate mismatch")
        else:
            require(detail["status"] in ("unknown", "rejected")
                    and np.array_equal(vectors["x"], extra["restored_x"])
                    and np.array_equal(vectors["y"], extra["restored_y"]),
                    "rejected refinement changed baseline values")
    elif row["solver"] == "clarabel":
        import clarabel

        require(set(extra) == {"cone_z"}, "Clarabel cone sidecar membership mismatch")
        _, _, _, mapping = saved_runner.clarabel_form(view, clarabel)
        require(row["raw"]["dual_mapping"] == [list(item) for item in mapping],
                "Clarabel mapping differs from original bounds")
        y = np.zeros_like(vectors["y"])
        require(extra["cone_z"].shape == (len(mapping),), "Clarabel cone dual shape mismatch")
        for z, (index, sign) in zip(extra["cone_z"], mapping):
            y[index] += sign * z
        require(np.array_equal(y, vectors["y"]), "Clarabel signed dual mapping mismatch")
    else:
        require(not extra, "OSQP has unexpected transformation members")


def replay(path: Path, pin: str) -> dict:
    corpus._sha(pin, "caller experiment pin")
    require(not path.is_symlink(), "symlink experiment refused")
    path = path.resolve(strict=True)
    require(digest(path) == pin, "experiment caller SHA256 mismatch")
    report, output = read_json(path), path.parent
    check_inventory(output, report["files"], path.name)
    require(report.get("schema_version") == 1 and report.get("version") == VERSION
            and report.get("verdict") == "EXPERIMENT_COMPLETE_NOT_QUALIFICATION",
            "incomplete/unsupported experiment")
    require(report["gates"] == GATES and report["arms"] == list(ARMS),
            "fixed arms/original tolerance mismatch")
    require(report["source_base_commit"] == BASE_COMMIT and report["source_hashes"] == {
        relative: digest(member(output, "code/" + name)) for name, relative in DEPENDENCIES.items()},
        "frozen source inventory mismatch")
    for flag in ("baseline_attempts_modified", "automatic_retry", "tiny_dual_clipping",
                 "license_verified", "public_redistribution", "commercial_claim", "speedup_claim"):
        require(report[flag] is False, "forbidden upgrade/claim")
    require(report["producer_sha256"] == digest(output / "code/producer.py")
            == digest(Path(__file__)), "diagnostic producer source mismatch")
    old = baseline_check(output / "baseline/result.json", report["baseline_sha256"])
    require(old["rows"] == report["baseline_rows"] and old["verdict"] == report["original_verdict"]
            and old["corpus_sha256"] == report["corpus_sha256"]
            and old.get("public_acceptance_sha256") == report["public_acceptance_sha256"]
            and old["source_eligibility"] == report["source_eligibility"], "baseline/source altered")
    saved_runner, importer, core = load_dependencies(output / "code")
    for name in ("importer.py", "certificates.py"):
        require(digest(output / "code" / name) == old["files"]["code/" + name]["sha256"],
                "original parser/checker rule changed")
    frozen = read_json(output / "baseline/frozen/corpus.json")
    expected = [(case["case_id"], arm["arm"], arm["solver"], case["model_sha256"])
                for case in frozen["cases"] for arm in ARMS]
    require(expected and [(row["case_id"], row["arm"], row["solver"], row["model_sha256"])
                          for row in report["rows"]] == expected, "arm denominator/membership mismatch")
    plan = read_json(output / "plan.json")
    require(plan["arms"] == list(ARMS) and plan["gates"] == GATES
            and [(row["case_id"], row["arm"], row["solver"], row["model_sha256"])
                 for row in plan["rows"]] == expected, "plan mismatch")
    count = 0
    for row in report["rows"]:
        require(row["attempt"] == 1 and row["denominator"] == 1 and row["verdict"] in STATES,
                "retry or invalid denominator/state")
        require(read_json(member(output, f"attempts/{row['case_id']}/{row['arm']}.json")) == row,
                "attempt/result mismatch")
        member(output, row["worker_log"])
        worker_result = f"attempts/{row['case_id']}/{row['arm']}-worker-result.json"
        if row["verdict"] in ("QUALIFIED", "UNQUALIFIED") or row.get("failure_log"):
            require(read_json(member(output, worker_result)) == {
                key: value for key, value in row.items()
                if key not in ("worker_exit_code", "worker_log")}, "worker/final row mismatch")
        if row["verdict"] in ("QUALIFIED", "UNQUALIFIED"):
            require(row["attempted"] and row["worker_exit_code"] == 0, "successful worker mismatch")
            model = saved_runner.load_model(output / "baseline/frozen", row["case_id"])
            require(importer.model_hash(model) == row["model_sha256"]
                    == row["input_model_before"] == row["input_model_after"],
                    "arm original arrays changed")
            vectors = saved_runner.read_vectors(member(output, row["vectors_path"]))
            raw = {**row["raw"], "vectors": vectors}
            assessed = saved_runner.assess(model, row["solver"], raw, GATES, core.checker)
            require(assessed == row["assessment"] and assessed["verdict"] == row["verdict"],
                    "original full gate replay mismatch")
            check_transform(model, row, vectors, read_extra(member(output, row["transform_path"])),
                            core, saved_runner)
            if row["solver"] == "osqp":
                require(row["raw"]["settings"]["scaling"] == 10
                        and row["raw"]["settings"]["eps_abs"] == ARMS[1]["solve_eps"]
                        and row["raw"]["settings"]["eps_rel"] == ARMS[1]["solve_eps"],
                        "OSQP requested/default scaling settings mismatch")
            elif row["solver"] == "clarabel":
                require(row["raw"]["settings"]["equilibrate_enable"] is True
                        and row["raw"]["settings"]["tol_feas"] == ARMS[2]["solve_eps"],
                        "Clarabel requested/default equilibration mismatch")
            else:
                require(row["raw"]["settings"]["solver"]["eps_abs"] == ARMS[0]["solve_eps"]
                        and row["raw"]["settings"]["solver"]["eps_rel"] == ARMS[0]["solve_eps"]
                        and row["raw"]["settings"]["scaling"]["check_atol"] == 1e-7
                        and row["raw"]["settings"]["refinement"]["check_atol"] == 1e-7,
                        "Openopt solve/check settings mismatch")
            count += 1
        else:
            require(row.get("error") and row["worker_exit_code"] != 0,
                    "failed/budget row missing first cause")
            if row["verdict"] == "FAIL" and row.get("failure_log"):
                member(output, row["failure_log"])
    require(all(report[key] == value for key, value in summary(report["rows"]).items()),
            "full failure denominator summary mismatch")
    return {"verdict": "EXPERIMENT_REPLAYED_NOT_QUALIFICATION", "checked_at": now(),
            "result_sha256": pin, "original_verdict": old["verdict"],
            "baseline_replayed": len(old["rows"]), "new_numerically_replayed": count,
            **summary(report["rows"])}


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    experiment = commands.add_parser("run")
    experiment.add_argument("--baseline", type=Path, required=True)
    experiment.add_argument("--baseline-sha256", required=True)
    experiment.add_argument("--artifact-dir", type=Path, required=True)
    experiment.add_argument("--cpu-budget-seconds", type=float, default=180)
    check = commands.add_parser("replay")
    check.add_argument("--result", type=Path, required=True)
    check.add_argument("--result-sha256", required=True)
    child = commands.add_parser("worker", help=argparse.SUPPRESS)
    child.add_argument("--artifact-dir", type=Path, required=True)
    child.add_argument("--row-index", type=int, required=True)
    args = cli.parse_args(argv)
    try:
        if args.action == "worker":
            return worker(args.artifact_dir, args.row_index)
        if args.action == "run":
            path, code = run(args.baseline, args.baseline_sha256, args.artifact_dir,
                             args.cpu_budget_seconds)
            print("PUBLIC_ROBUSTNESS " + json.dumps({
                "result": str(path), "result_sha256": digest(path),
                "verdict": read_json(path)["verdict"], "exit_code": code}, sort_keys=True), flush=True)
            return code
        result = replay(args.result, args.result_sha256)
        print("PUBLIC_ROBUSTNESS " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
