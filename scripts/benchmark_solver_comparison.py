"""Same-host frozen dense QP comparison, not a commercial qualification.

CLI and main(argv=None) return 0 for measured numerical PASS (missing optional
SDKs remain UNVERIFIED), 1 for execution/quality failure, 2 for invalid arguments.
run_comparison(args, report) fills a JSON-compatible report. No GPU submission,
dependency installation, credential access, or fallback is performed here.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# CLI limits are established before NumPy/Torch initialize native thread pools.
if __name__ == "__main__":
    boot = argparse.ArgumentParser(add_help=False)
    boot.add_argument("--cpu-threads", type=int, default=1)
    boot_args, _ = boot.parse_known_args()
    if boot_args.cpu_threads > 0:
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[variable] = str(boot_args.cpu_threads)

import numpy as np
import torch

from openopt import QPConfig, QPProblem, prepare_qp
from scripts.benchmark_general_qp import generate


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def arrays(problem: QPProblem) -> tuple[np.ndarray, ...]:
    return tuple(t.detach().cpu().numpy() for t in (
        problem.P, problem.q, problem.A, problem.l, problem.u,
    ))


def problem_hash(problem: QPProblem) -> str:
    digest = hashlib.sha256()
    for key, value in zip(("P", "q", "A", "l", "u"), arrays(problem), strict=True):
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def norm(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def validate_problem(problem: QPProblem) -> dict:
    """Reference setup validates the full PSD matrix, not just its diagonal."""
    P, q, A, l, u = arrays(problem)
    if q.ndim != 1 or not q.size or P.shape != (q.size, q.size):
        raise ValueError("nonempty P/q dimensions required")
    if A.ndim != 2 or A.shape[1] != q.size or l.shape != (A.shape[0],) or u.shape != l.shape:
        raise ValueError("A/l/u dimensions mismatch")
    if not all(np.isfinite(t).all() for t in (P, q, A)):
        raise ValueError("P/q/A must be finite")
    if np.isnan(l).any() or np.isnan(u).any() or np.isposinf(l).any() or np.isneginf(u).any():
        raise ValueError("invalid bounds")
    if (l > u).any():
        raise ValueError("lower bound exceeds upper bound")
    tolerance = 20 * np.finfo(P.dtype).eps * max(1.0, norm(P))
    if norm(P - P.T) > tolerance:
        raise ValueError("P must be symmetric")
    eigen_min = float(np.linalg.eigvalsh((P + P.T) * 0.5).min())
    if eigen_min < -tolerance:
        raise ValueError("P must be positive semidefinite (full PSD check)")
    return {"method": "full_eigvalsh", "minimum_eigenvalue": eigen_min}


def check_quality(problem: QPProblem, x, y, expected, eps_abs: float, eps_rel: float) -> dict:
    """Recompute original-coordinate KKT/objective without trusting solver metrics."""
    P, q, A, l, u = arrays(problem)
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != q.shape or y.shape != l.shape:
        return {"verdict": "FAIL", "reason": "solution dimensions mismatch"}
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"verdict": "FAIL", "reason": "solution contains nonfinite values"}
    Ax, Px, ATy = A @ x, P @ x, A.T @ y
    projected = np.clip(Ax, l, u)
    primal, dual = norm(Ax - projected), norm(Px + q + ATy)
    normal = norm(projected - np.clip(projected + y, l, u))
    primal_tol = eps_abs + eps_rel * max(norm(Ax), norm(projected))
    dual_tol = eps_abs + eps_rel * max(norm(Px), norm(q), norm(ATy))
    objective = float(0.5 * x @ P @ x + q @ x)
    optimum = float(0.5 * expected @ P @ expected + q @ expected)
    gap = abs(objective - optimum) / max(1.0, abs(optimum))
    passed = primal <= primal_tol and dual <= dual_tol and normal <= primal_tol
    passed = passed and gap <= eps_abs + eps_rel
    return {
        "verdict": "PASS" if passed else "FAIL",
        "primal_residual": primal, "dual_residual": dual,
        "normal_cone_residual": normal, "primal_tolerance": primal_tol,
        "dual_tolerance": dual_tol, "objective": objective,
        "witness_objective": optimum, "relative_objective_gap": gap,
        "objective_gap_tolerance": eps_abs + eps_rel,
        "solution_error": norm(x - expected),
    }


def openopt_config(device: str, variant: str, eps_abs: float, eps_rel: float,
                   max_iterations: int, config_type=QPConfig):
    names = {field.name for field in fields(config_type)}
    kwargs = dict(device=device, eps_abs=eps_abs, eps_rel=eps_rel, max_iterations=max_iterations)
    if variant == "adaptive" and "adaptive_rho" not in names:
        raise ValueError("adaptive requested but QPConfig.adaptive_rho is unavailable")
    if variant not in ("baseline", "adaptive"):
        raise ValueError("unknown variant")
    if "adaptive_rho" in names:
        kwargs["adaptive_rho"] = variant == "adaptive"
    config = config_type(**kwargs)
    config.validate()
    return config


def clarabel_form(problem: QPProblem):
    """Return conic rows and original-row/sign map (including one-sided bounds)."""
    _, _, A, l, u = arrays(problem)
    equal = np.flatnonzero(l == u)
    upper = np.flatnonzero((l != u) & np.isfinite(u))
    lower = np.flatnonzero((l != u) & np.isfinite(l))
    matrix = np.concatenate((A[equal], A[upper], -A[lower]), axis=0)
    bounds = np.concatenate((u[equal], u[upper], -l[lower]))
    indices = np.concatenate((equal, upper, lower))
    signs = np.concatenate((np.ones(equal.size + upper.size), -np.ones(lower.size)))
    return matrix, bounds, indices, signs, equal.size


class OpenoptRunner:
    def __init__(self, problem, config):
        self.problem, self.config = problem, config

    def prepare(self):
        return prepare_qp(self.problem, self.config)

    def solve(self, workspace):
        result = workspace.solve()
        actual = str(result.x.device)
        wanted = str(torch.device(self.config.device))
        if actual != wanted or str(result.y.device) != wanted:
            raise RuntimeError(f"tensor placement mismatch requested={wanted} observed={actual}")
        self.observed_device = actual
        return (result.x.cpu().numpy().copy(), result.y.cpu().numpy().copy(),
                result.status, result.status == "solved", result.iterations)


class OSQPRunner:
    def __init__(self, problem, eps_abs, eps_rel, max_iterations):
        self.problem = problem
        self.config = dict(eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iterations,
                           verbose=False, polishing=False, warm_starting=False,
                           rho=0.1, adaptive_rho=True, adaptive_rho_interval=50)

    def prepare(self):
        import osqp
        from scipy import sparse

        validate_problem(self.problem)
        P, q, A, l, u = (v.copy() for v in arrays(self.problem))
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(P)), q=q, A=sparse.csc_matrix(A),
                     l=l, u=u, **self.config)
        return solver

    def solve(self, workspace):
        # OSQP retains adapted rho even with warm_starting=False. Restore its
        # declared initial rho inside timing; keep within-call adaptation enabled.
        workspace.update_settings(rho=self.config["rho"])
        result = workspace.solve(raise_error=True)
        return (np.asarray(result.x).copy(), np.asarray(result.y).copy(),
                result.info.status, result.info.status_val == 1, result.info.iter)


class ClarabelRunner:
    def __init__(self, problem, eps_abs, eps_rel, max_iterations, threads):
        self.problem = problem
        self.config = dict(tol_gap_abs=eps_abs, tol_gap_rel=eps_rel,
                           tol_feas=eps_abs, max_iter=max_iterations, max_threads=threads,
                           reduced_tol_gap_abs=eps_abs, reduced_tol_gap_rel=eps_rel,
                           reduced_tol_feas=eps_abs, verbose=False)
        self.mapping = None

    def prepare(self):
        import clarabel
        from scipy import sparse

        validate_problem(self.problem)
        P, q, _, _, _ = arrays(self.problem)
        matrix, b, indices, signs, count = clarabel_form(self.problem)
        settings = clarabel.DefaultSettings()
        for key, value in self.config.items():
            if not hasattr(settings, key):
                raise ValueError(f"Clarabel settings field unavailable: {key}")
            setattr(settings, key, value)
        cones = []
        if count:
            cones.append(clarabel.ZeroConeT(int(count)))
        if len(b) > count:
            cones.append(clarabel.NonnegativeConeT(int(len(b) - count)))
        self.mapping = indices, signs
        return clarabel.DefaultSolver(sparse.csc_matrix(np.triu(P)), q.copy(),
                                     sparse.csc_matrix(matrix), b, cones, settings)

    def solve(self, workspace):
        result = workspace.solve()
        indices, signs = self.mapping
        y = np.zeros(self.problem.l.numel())
        np.add.at(y, indices, np.asarray(result.z) * signs)
        return (np.asarray(result.x).copy(), y, str(result.status),
                str(result.status) == "Solved", result.iterations)


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def thread_pools(limit: int | None = None) -> list[dict]:
    """Read/control loaded OpenBLAS pools without an optional helper dependency."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return [{"status": "UNVERIFIED", "reason": "proc maps unavailable"}]
    paths = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                    if "openblas" in line and "/" in line})
    pools = []
    for path in paths:
        library = ctypes.CDLL(path)
        observed = None
        for prefix in ("openblas", "scipy_openblas"):
            for suffix in ("", "64_", "_64_"):
                getter = getattr(library, f"{prefix}_get_num_threads{suffix}", None)
                setter = getattr(library, f"{prefix}_set_num_threads{suffix}", None)
                if getter is not None and setter is not None:
                    getter.restype = ctypes.c_int
                    setter.argtypes = [ctypes.c_int]
                    if limit is not None:
                        setter(limit)
                    observed = getter()
                    break
            if observed is not None:
                break
        pools.append({"library": Path(path).name, "threads": observed,
                      "status": "MEASURED" if observed is not None else "UNVERIFIED"})
    return pools


def cpu_stat() -> dict | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.is_file():
        return None
    return {line.split()[0]: int(line.split()[1]) for line in path.read_text().splitlines()}


def process_threads() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("Threads:"):
            return int(line.split()[1])
    raise RuntimeError("proc status missing Threads")


class Telemetry:
    def __init__(self, devices):
        self.devices = [d for d in devices if d.startswith("cuda")]
        self.samples, self.error = [], None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        try:
            while not self.stop.is_set():
                sample = {"at": now(), "cpu_stat": cpu_stat(), "process_threads": process_threads()}
                sample["cuda"] = {d: {
                    "allocated_bytes": torch.cuda.memory_allocated(d),
                    "reserved_bytes": torch.cuda.memory_reserved(d),
                } for d in self.devices}
                self.samples.append(sample)
                self.stop.wait(0.1)
        except Exception:
            self.error = traceback.format_exc()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"telemetry failure: {self.error}")
        if not self.samples:
            raise RuntimeError("empty telemetry")


def synchronize(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"count": len(values), "samples_s": values,
            "p50_s": float(np.percentile(values, 50)), "p95_s": float(np.percentile(values, 95))}


def safe_config(config) -> dict:
    values = asdict(config) if not isinstance(config, dict) else config.copy()
    return {key: str(value) if isinstance(value, torch.dtype) else value
            for key, value in values.items()}


def arm(identifier, solver, device, runner=None, reason=None, control_of=None):
    config = safe_config(runner.config) if runner is not None else None
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() if config else None
    return {"id": identifier, "solver": solver, "device": device, "config": config,
            "config_hash": digest, "control_of": control_of, "modes": {},
            "status": "UNVERIFIED" if reason else "PENDING", "reason": reason}


def error_record(error: Exception, coordinate: dict, trace: str) -> dict:
    return {"at": now(), "coordinate": coordinate, "exception_type": type(error).__name__,
            "first_cause": str(error), "traceback": trace}


def measure(runner, row, mode, repeat, position, expected, args, workspace):
    coordinate = dict(arm=row["id"], mode=mode, repeat=repeat, order_position=position)
    device = row["device"]
    cpu_before, stat_before = time.process_time(), cpu_stat()
    synchronize(device)
    start = time.perf_counter()
    if mode == "cold":
        workspace = runner.prepare()
        synchronize(device)
    setup_end = time.perf_counter()
    x, y, status, solved, iterations = runner.solve(workspace)
    synchronize(device)
    end = time.perf_counter()
    sample = {
        **coordinate, "at": now(), "wall_s": end - start,
        "setup_s": setup_end - start if mode == "cold" else None,
        "solve_host_result_s": end - setup_end, "process_cpu_s": time.process_time() - cpu_before,
        "cpu_stat_before": stat_before, "cpu_stat_after": cpu_stat(),
        "process_threads": process_threads(), "solver_status": str(status), "iterations": int(iterations),
        "observed_tensor_device": getattr(runner, "observed_device", None),
        "host_solution": {"x": x.tolist(), "y": y.tolist()}
        if np.isfinite(x).all() and np.isfinite(y).all() else None,
        "quality": check_quality(runner.problem, x, y, expected, args.eps_abs, args.eps_rel),
    }
    sample["verdict"] = "PASS" if solved and sample["quality"]["verdict"] == "PASS" else "FAIL"
    return sample


def execute_mode(rows, runners, mode, expected, args, report):
    workspaces = {}
    for row in rows:
        result = {"status": "UNVERIFIED" if row["status"] == "UNVERIFIED" else "PENDING",
                  "samples": [], "latency": None, "setup_once_s": None}
        row["modes"][mode] = result
        if row["status"] in ("UNVERIFIED", "FAIL"):
            result["status"] = row["status"]
            result["reason"] = row.get("reason")
            continue
        if mode == "prepared":
            coordinate = dict(arm=row["id"], mode=mode, repeat=None, phase="prepare")
            try:
                synchronize(row["device"])
                start = time.perf_counter()
                workspaces[row["id"]] = runners[row["id"]].prepare()
                synchronize(row["device"])
                result["setup_once_s"] = time.perf_counter() - start
            except Exception as error:
                failure = error_record(error, coordinate, traceback.format_exc())
                report["failures"].append(failure)
                row.update(status="FAIL", reason=failure["first_cause"])
                result.update(status="FAIL", failure=failure)
    # Sequential latency pairs deliberately share one process/host and clock.
    for repeat in range(args.repeats):
        ordered = list(rows) if repeat % 2 == 0 else list(reversed(rows))
        report["orders"].append({"mode": mode, "repeat": repeat, "arms": [r["id"] for r in ordered]})
        for position, row in enumerate(ordered):
            result = row["modes"][mode]
            if row["status"] in ("UNVERIFIED", "FAIL"):
                continue
            coordinate = dict(arm=row["id"], mode=mode, repeat=repeat, order_position=position)
            try:
                sample = measure(runners[row["id"]], row, mode, repeat, position,
                                 expected, args, workspaces.get(row["id"]))
                result["samples"].append(sample)
                if sample["verdict"] != "PASS":
                    raise RuntimeError(f"solver status={sample['solver_status']}; "
                                       f"independent quality rejected: {sample['quality']}")
                if problem_hash(runners[row["id"]].problem) != report["problem_hash"]:
                    raise RuntimeError("frozen problem mutated")
            except Exception as error:
                failure = error_record(error, coordinate, traceback.format_exc())
                report["failures"].append(failure)
                row.update(status="FAIL", reason=failure["first_cause"])
                result.update(status="FAIL", failure=failure)
            write(args.artifact_dir / "partial.json", report)
    for row in rows:
        result = row["modes"][mode]
        result["latency"] = stats([s["wall_s"] for s in result["samples"] if s["verdict"] == "PASS"])
        if result["status"] == "PENDING":
            result["status"] = "PASS" if len(result["samples"]) == args.repeats else "FAIL"


def control_summary(rows, repeats):
    by_id = {r["id"]: r for r in rows}
    controls = []
    for row in rows:
        if row["control_of"] is None:
            continue
        original = by_id[row["control_of"]]
        control = {"arm": original["id"], "control": row["id"],
                   "identical_config_control": row["config_hash"] == original["config_hash"],
                   "modes": {}}
        for mode in ("cold", "prepared"):
            a, b = original["modes"][mode], row["modes"][mode]
            if a["status"] == b["status"] == "PASS" and len(a["samples"]) == len(b["samples"]) == repeats:
                deltas = [abs(x["wall_s"] - y["wall_s"]) for x, y in zip(a["samples"], b["samples"], strict=True)]
                control["modes"][mode] = {
                    "status": "PASS", "paired_abs_delta_s": deltas,
                    "empirical_noise_floor_s": float(np.percentile(deltas, 95)),
                }
            else:
                control["modes"][mode] = {"status": "UNVERIFIED", "empirical_noise_floor_s": None}
        controls.append(control)
    return controls


def check_result(path: Path) -> dict:
    """Read real disk evidence; reject incomplete PASS or a fabricated zero floor."""
    report = json.loads(path.read_text())
    if report.get("verdict") != "PASS" or report.get("exit_code") != 0:
        raise ValueError("comparison result is not accepted PASS")
    config = report["configuration"]
    repeats = config["repeats"]
    if repeats < 3 or not report.get("arms"):
        raise ValueError("empty/incomplete comparison")
    saved = np.load(path.parent / "problem.npz", allow_pickle=False)
    problem = QPProblem(*(torch.from_numpy(saved[k]) for k in ("P", "q", "A", "l", "u")))
    if problem_hash(problem) != report["problem_hash"]:
        raise ValueError("frozen problem hash mismatch")
    witness = check_quality(problem, saved["expected"], saved["witness_y"], saved["expected"],
                            config["eps_abs"], config["eps_rel"])
    if witness["verdict"] != "PASS":
        raise ValueError("frozen witness rejected")
    checked = 0
    by_id = {r["id"]: r for r in report["arms"]}
    if len(by_id) != len(report["arms"]):
        raise ValueError("duplicate arm identifiers")
    for row in report["arms"]:
        if row["status"] not in ("PASS", "UNVERIFIED"):
            raise ValueError(f"unaccepted arm: {row['id']}")
        if row["status"] == "PASS":
            actual = row["config"]
            digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
            if digest != row["config_hash"]:
                raise ValueError("configuration hash mismatch")
            names = ("tol_gap_abs", "tol_gap_rel") if row["solver"] == "clarabel" else ("eps_abs", "eps_rel")
            if actual[names[0]] != config["eps_abs"] or actual[names[1]] != config["eps_rel"]:
                raise ValueError("solver eps mismatch")
        for mode in ("cold", "prepared"):
            measured = row["modes"][mode]
            if row["status"] == "UNVERIFIED":
                if measured["status"] != "UNVERIFIED" or measured["samples"] or measured["latency"] is not None:
                    raise ValueError("UNVERIFIED arm contains accepted timings")
                continue
            samples = measured["samples"]
            if measured["status"] != "PASS" or len(samples) != repeats:
                raise ValueError("incomplete measured samples")
            if sorted(s["repeat"] for s in samples) != list(range(repeats)):
                raise ValueError("duplicate/missing repeat")
            for sample in samples:
                if sample["verdict"] != "PASS" or sample["quality"]["verdict"] != "PASS":
                    raise ValueError("rejected numerical sample")
                if not math.isfinite(sample["wall_s"]) or sample["wall_s"] <= 0:
                    raise ValueError("invalid measured wall")
                if row["solver"] == "openopt" and sample["observed_tensor_device"] != row["device"]:
                    raise ValueError("observed tensor placement mismatch")
                solution = sample["host_solution"]
                quality = check_quality(problem, np.asarray(solution["x"]), np.asarray(solution["y"]),
                                        saved["expected"], config["eps_abs"], config["eps_rel"])
                if quality != sample["quality"] or quality["verdict"] != "PASS":
                    raise ValueError("original-coordinate quality evidence mismatch")
                checked += 1
            if measured["latency"] != stats([s["wall_s"] for s in samples]):
                raise ValueError("latency statistics mismatch")
    for mode in ("cold", "prepared"):
        orders = [o for o in report["orders"] if o["mode"] == mode]
        forward = [r["id"] for r in report["arms"]]
        if len(orders) != repeats:
            raise ValueError("incomplete order schedule")
        for repeat, order in enumerate(orders):
            if order["repeat"] != repeat or order["arms"] != (forward if repeat % 2 == 0 else forward[::-1]):
                raise ValueError("order alternation mismatch")
    controls = control_summary(report["arms"], repeats)
    if not any(r["status"] == "PASS" and r["solver"] == "openopt" for r in report["arms"]):
        raise ValueError("no accepted openopt arm")
    if not controls or controls != report["controls"]:
        raise ValueError("identical controls/noise floor mismatch")
    for control in controls:
        original, twin = by_id[control["arm"]], by_id[control["control"]]
        if original["config"] != twin["config"] or not control["identical_config_control"]:
            raise ValueError("control configuration mismatch")
    telemetry = json.loads((path.parent / "telemetry.json").read_text())
    if not telemetry:
        raise ValueError("empty telemetry evidence")
    return {"verdict": "PASS", "read_at": now(), "problem_hash": report["problem_hash"],
            "checked_samples": checked, "arms": {r["id"]: r["status"] for r in report["arms"]},
            "latency": {r["id"]: {mode: r["modes"][mode]["latency"] for mode in ("cold", "prepared")}
                        for r in report["arms"] if r["status"] == "PASS"},
            "quality_max": {key: max(s["quality"][key] for r in report["arms"] if r["status"] == "PASS"
                                     for m in r["modes"].values() for s in m["samples"])
                            for key in ("primal_residual", "dual_residual",
                                        "normal_cone_residual", "relative_objective_gap")},
            "controls": controls, "reference_coverage": report["reference_coverage"],
            "hardware_host": report["hardware"]["host"], "torch_threads": report["hardware"]["torch_threads"],
            "thread_pools": report["thread_pools"]}


def run_comparison(args, report: dict) -> None:
    torch.set_num_threads(args.cpu_threads)
    report["thread_pools"] = thread_pools(args.cpu_threads)
    problem, expected, witness_y = generate(args.size, args.seed)
    report["problem_validation"] = validate_problem(problem)
    report["witness"] = check_quality(problem, expected, witness_y, expected, args.eps_abs, args.eps_rel)
    if report["witness"]["verdict"] != "PASS":
        raise RuntimeError("known-KKT generator witness failed")
    report["problem_hash"] = problem_hash(problem)
    report["dimensions"] = {"variables": args.size, "constraints": problem.l.numel()}
    np.savez(args.artifact_dir / "problem.npz",
             **dict(zip(("P", "q", "A", "l", "u"), arrays(problem), strict=True)),
             expected=expected, witness_y=witness_y)
    report["hardware"] = {
        "host": platform.node(), "node": os.environ.get("OPENOPT_NODE"), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        if Path("/sys/fs/cgroup/cpu.max").is_file() else None,
        "cgroup_membership": Path("/proc/self/cgroup").read_text().strip(),
        "cuda_runtime": torch.version.cuda, "gpu": {},
    }
    rows, runners = [], {}
    report.update(arms=rows, failures=[], orders=[])
    for device in args.devices:
        identifier = f"openopt:{device}:{args.variant}"
        try:
            config = openopt_config(device, args.variant, args.eps_abs, args.eps_rel, args.max_iterations)
            if device.startswith("cuda"):
                report["hardware"]["gpu"][device] = torch.cuda.get_device_name(device)
            for suffix in ("", ":control"):
                runner = OpenoptRunner(problem, config)
                rows.append(arm(identifier + suffix, "openopt", device, runner,
                                control_of=identifier if suffix else None))
                runners[identifier + suffix] = runner
        except Exception as error:
            failure = error_record(error, {"arm": identifier, "phase": "config"}, traceback.format_exc())
            report["failures"].append(failure)
            row = arm(identifier, "openopt", device)
            row.update(status="FAIL", reason=failure["first_cause"], failure=failure)
            rows.append(row)
    for module, runner_type in (("osqp", OSQPRunner), ("clarabel", ClarabelRunner)):
        if not available(module) or not available("scipy"):
            rows.append(arm(module, module, "cpu", reason="SDK unavailable (solver or scipy)"))
        else:
            # Imports and thread-pool initialization are outside model-cold timing
            # for every backend; model validation/conversion/setup is inside it.
            try:
                importlib.import_module(module)
                importlib.import_module("scipy.sparse")
                extra = [args.cpu_threads] if module == "clarabel" else []
                for suffix in ("", ":control"):
                    runner = runner_type(problem, args.eps_abs, args.eps_rel, args.max_iterations, *extra)
                    rows.append(arm(module + suffix, module, "cpu", runner,
                                    control_of=module if suffix else None))
                    runners[module + suffix] = runner
            except Exception as error:
                failure = error_record(error, {"arm": module, "phase": "import"}, traceback.format_exc())
                report["failures"].append(failure)
                row = arm(module, module, "cpu")
                row.update(status="FAIL", reason=failure["first_cause"], failure=failure)
                rows.append(row)
    report["commercial"] = {
        name: {"status": "UNVERIFIED", "sdk_available": available(name),
               "reason": "matched commercial adapter not executed; no license probe" if available(name)
               else "SDK unavailable"}
        for name in ("gurobipy", "cplex", "mosek")
    }
    report["software"] = {}
    for name in ("numpy", "torch", "scipy", "osqp", "clarabel"):
        report["software"][name] = importlib.metadata.version(name) if available(name) else None
    report["thread_pools"] = thread_pools(args.cpu_threads)
    write(args.artifact_dir / "partial.json", report)
    telemetry = Telemetry([r["device"] for r in rows if r["status"] != "FAIL"])
    try:
        with telemetry:
            for mode in ("cold", "prepared"):
                execute_mode(rows, runners, mode, expected, args, report)
    finally:
        write(args.artifact_dir / "telemetry.json", telemetry.samples)
    for row in rows:
        if row["status"] == "PENDING":
            row["status"] = "PASS" if all(m["status"] == "PASS" for m in row["modes"].values()) else "FAIL"
    report["controls"] = control_summary(rows, args.repeats)
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["reference_coverage"] = {
        "status": "PASS" if all(r["status"] == "PASS" for r in rows if r["solver"] in ("osqp", "clarabel"))
        and {r["solver"] for r in rows if r["status"] == "PASS"} >= {"osqp", "clarabel"}
        else "UNVERIFIED",
        "commercial": "UNVERIFIED",
    }
    report["comparison"] = {
        "same_host": True, "scope": "model-cold / prepared only; no process-cold or universal ranking",
        "gpu_utilization": "UNVERIFIED; GPU coordinator must attach runtime profiler evidence",
        "eligible_arms": [r["id"] for r in rows if r["status"] == "PASS"],
        "unverified_arms": [r["id"] for r in rows if r["status"] == "UNVERIFIED"],
        "failed_arms": [r["id"] for r in rows if r["status"] == "FAIL"],
        "ranking": "UNVERIFIED",
        "ranking_reason": "raw paired latencies only; commercial qualification unavailable",
    }
    if report["failures"] or any(r["status"] == "FAIL" for r in rows):
        raise RuntimeError(f"solver comparison failures={len(report['failures'])}; see independent arm records")
    if not any(r["status"] == "PASS" and r["solver"] == "openopt" for r in rows):
        raise RuntimeError("no measured openopt arm")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact-dir", type=Path)
    result.add_argument("--check-result", type=Path, help="read-only disk acceptance of a completed result")
    result.add_argument("--size", type=int, default=64)
    result.add_argument("--seed", type=int, default=20261010)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--devices", nargs="+", choices=("cpu", "cuda:0"), default=["cpu"])
    result.add_argument("--variant", choices=("baseline", "adaptive"), default="baseline")
    result.add_argument("--eps-abs", type=float, default=1e-7)
    result.add_argument("--eps-rel", type=float, default=1e-7)
    result.add_argument("--max-iterations", type=int, default=5000)
    result.add_argument("--cpu-threads", type=int, default=1)
    return result


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.check_result is not None:
        try:
            checked = check_result(args.check_result)
        except Exception as error:
            print(f"OPENOPT_COMPARISON_CHECK verdict=FAIL error={type(error).__name__}: {error}", file=sys.stderr)
            return 1
        print(json.dumps(checked, indent=2, allow_nan=False), flush=True)
        return 0
    if args.artifact_dir is None:
        cli.error("--artifact-dir is required for execution")
    if args.size < 3 or args.repeats < 3 or args.cpu_threads < 1 or args.max_iterations < 1:
        cli.error("size/repeats >=3; cpu-threads/max-iterations >=1")
    if not all(math.isfinite(v) and v > 0 for v in (args.eps_abs, args.eps_rel)):
        cli.error("eps must be finite and positive")
    if len(set(args.devices)) != len(args.devices):
        cli.error("duplicate devices")
    output = args.artifact_dir.expanduser().resolve()
    if output.is_relative_to(ROOT):
        cli.error("artifacts must be outside worktree")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        cli.error("artifact directory must be new or empty; preserve previous evidence")
    output.mkdir(parents=True, exist_ok=True)
    args.artifact_dir = output
    report = {
        "schema_version": 1, "verdict": "RUNNING", "started_at": now(),
        "argv": argv if argv is not None else sys.argv[1:],
        "python_executable": sys.executable,
        "source_hashes": {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                          for relative in ("scripts/benchmark_solver_comparison.py",
                                           "scripts/benchmark_general_qp.py", "src/openopt/qp.py")},
        "configuration": {**vars(args), "artifact_dir": str(output)},
        "qualification": "synthetic dense convex QP; missing SDK/license is UNVERIFIED, never a win",
        "timing_contract": {
            "cold": "full PSD validation + copies/conversion + setup + solve + host x/y, synchronized",
            "prepared": "setup_once_s separate; fresh numerical start per solve + host x/y, synchronized",
            "warmups": 0, "process_import_and_device_initialization": "excluded equally",
            "independent_quality": "outside timed region; original coordinates, identical eps",
        },
    }
    start = time.perf_counter()
    try:
        run_comparison(args, report)
        report["verdict"] = "PASS"
    except Exception as error:
        report["verdict"] = "FAIL"
        report["error"] = error_record(error, {"phase": "experiment"}, traceback.format_exc())
        failures = report.get("failures", [])
        write(output / "failure.json", {
            "first_cause": failures[0] if failures else report["error"],
            "errors": failures + [report["error"]],
        })
    report.update(finished_at=now(), wall_s=time.perf_counter() - start)
    report["exit_code"] = 0 if report["verdict"] == "PASS" else 1
    write(output / "result.json", report)
    verdict = (f"OPENOPT_SOLVER_COMPARISON verdict={report['verdict']} "
               f"n={args.size} wall_s={report['wall_s']:.3f} at={report['finished_at']}")
    (output / "execution.log").write_text(verdict + "\n", encoding="utf-8")
    print(f"{verdict} result={output / 'result.json'}", flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
