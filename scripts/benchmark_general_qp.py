"""Paired dense convex QP experiment with synthetic known-KKT witnesses.

Cold timing includes validation, copies, factorization, iterations and host
results. Prepared timing excludes model setup, but includes output transfer.
Repetitions are a sequential latency experiment, not independent batch units.
Every quality result is recomputed on CPU, independently of solver status.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import tempfile

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from openopt import QPConfig, QPProblem, prepare_qp, solve_qp


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generate(n: int, seed: int = 20261010) -> tuple[QPProblem, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n)) / np.sqrt(n)
    P = np.eye(n) + B.T @ B
    m = max(3, n // 4)
    A = np.concatenate([np.eye(n), rng.normal(size=(m, n)) / np.sqrt(n)])
    x = rng.uniform(-0.5, 0.5, n)
    Ax = A @ x
    l, u = Ax - 0.5, Ax + 0.5
    y = np.zeros(A.shape[0])
    # Plant active upper/lower rows and equalities; stationarity defines q.
    active = np.arange(0, A.shape[0], 7)
    u[active] = Ax[active]
    y[active] = rng.uniform(0.2, 0.8, len(active))
    active = np.arange(3, A.shape[0], 7)
    l[active] = Ax[active]
    y[active] = -rng.uniform(0.2, 0.8, len(active))
    l[-3:], u[-3:] = Ax[-3:], Ax[-3:]
    y[-3:] = rng.normal(size=3)
    q = -P @ x - A.T @ y
    p = QPProblem(*[torch.from_numpy(array) for array in (P, q, A, l, u)])
    return p, x, y


def quality(p: QPProblem, x: np.ndarray, y: np.ndarray, expected: np.ndarray) -> dict:
    P, q, A, l, u = [t.numpy() for t in (p.P, p.q, p.A, p.l, p.u)]
    Ax = A @ x
    projected = np.clip(Ax, l, u)
    primal = float(np.max(np.abs(Ax - projected)))
    dual = float(np.max(np.abs(P @ x + q + A.T @ y)))
    normal = float(np.max(np.abs(projected - np.clip(projected + y, l, u))))
    objective = float(0.5 * x @ P @ x + q @ x)
    optimum = float(0.5 * expected @ P @ expected + q @ expected)
    gap = abs(objective - optimum) / max(1, abs(optimum))
    error = float(np.max(np.abs(x - expected)))
    passed = primal <= 5e-6 and dual <= 5e-6 and normal <= 5e-6 and gap <= 1e-6 and error <= 1e-5
    return {
        "primal_residual": primal, "dual_residual": dual, "normal_cone_residual": normal,
        "objective": objective, "relative_objective_gap": gap, "solution_error": error,
        "verdict": "PASS" if passed else "FAIL",
    }


def cpu_stat() -> dict:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.is_file():
        return {"available": False}
    return {line.split()[0]: int(line.split()[1]) for line in path.read_text().splitlines()}


class Telemetry:
    def __init__(self, output: Path):
        self.output = output
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            before = time.perf_counter()
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            )
            self.samples.append({
                "at": datetime.now(timezone.utc).isoformat(), "monotonic_s": before,
                "rc": r.returncode, "gpu": r.stdout.strip() if r.returncode == 0 else None,
                "error": r.stderr.strip() if r.returncode else None, "cpu": cpu_stat(),
            })
            self.stop.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        write(self.output / "telemetry.json", self.samples)


def latency(values: list[float]) -> dict:
    return {
        "samples_s": values, "count": len(values),
        "p50_s": float(np.percentile(values, 50)), "p95_s": float(np.percentile(values, 95)),
    }


def time_openopt(p, config, expected, repeats, prepared):
    workspace = prepare_qp(p, config) if prepared else None
    timings, qualities, iterations = [], [], []
    for repeat in range(repeats + 1):
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = workspace.solve() if prepared else solve_qp(p, config)
        x = result.x.cpu().numpy()
        y = result.y.cpu().numpy()
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        measured = quality(p, x, y, expected)
        measured["solver_status"] = result.status
        if result.status != "solved" or measured["verdict"] != "PASS":
            raise RuntimeError(f"openopt quality failure device={config.device} mode={prepared}: {measured}")
        if repeat:
            timings.append(elapsed)
            qualities.append(measured)
            iterations.append(result.iterations)
    return {"latency": latency(timings), "quality": qualities, "iterations": iterations, "warmups": 1}


def paired_ab(p, expected, repeats):
    configs = [
        QPConfig(device="cuda:0", linear_solver="cholesky"),
        QPConfig(device="cuda:0", linear_solver="triangular"),
        QPConfig(device="cuda:0", linear_solver="cholesky"),
    ]
    arms = ("baseline", "triangular", "identical_control")
    results = {}
    for prepared in (False, True):
        workspaces = [prepare_qp(p, c) if prepared else None for c in configs]
        times = {arm: [] for arm in arms}
        qualities = {arm: [] for arm in arms}
        for repeat in range(repeats + 1):
            order = range(3) if repeat % 2 == 0 else reversed(range(3))
            for index in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = workspaces[index].solve() if prepared else solve_qp(p, configs[index])
                x, y = result.x.cpu().numpy(), result.y.cpu().numpy()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                measured = quality(p, x, y, expected)
                if measured["verdict"] != "PASS" or result.status != "solved":
                    raise RuntimeError(f"A/B quality failure arm={arms[index]}: {measured}")
                if repeat:
                    times[arms[index]].append(elapsed)
                    qualities[arms[index]].append(measured)
        mode = "prepared" if prepared else "cold"
        results[mode] = {arm: {"latency": latency(times[arm]), "quality": qualities[arm]} for arm in arms}
    return results


def paired_devices(p, expected, repeats):
    configs = [QPConfig(device="cpu"), QPConfig(device="cuda:0")]
    results = {}
    for prepared in (False, True):
        spaces = [prepare_qp(p, c) if prepared else None for c in configs]
        timings = {"cpu": [], "cuda:0": []}
        checks = {"cpu": [], "cuda:0": []}
        for repeat in range(repeats + 1):
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            for index in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                r = spaces[index].solve() if prepared else solve_qp(p, configs[index])
                x, y = r.x.cpu().numpy(), r.y.cpu().numpy()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                measured = quality(p, x, y, expected)
                if r.status != "solved" or measured["verdict"] != "PASS":
                    raise RuntimeError(f"paired device quality failure: {measured}")
                if repeat:
                    timings[configs[index].device].append(elapsed)
                    checks[configs[index].device].append(measured)
        results["prepared" if prepared else "cold"] = {
            device: {"latency": latency(times), "quality": checks[device]}
            for device, times in timings.items()
        }
    return results


def reference_osqp(p, expected, repeats):
    import osqp
    from scipy import sparse

    arrays = [t.numpy() for t in (p.P, p.q, p.A, p.l, p.u)]
    P, q, A, l, u = arrays
    config = dict(
        eps_abs=1e-7, eps_rel=1e-7, max_iter=5000, verbose=False,
        polishing=False, warm_starting=False,
    )

    def prepare():
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u, **config)
        return solver

    results = {}
    for prepared in (False, True):
        workspace = prepare() if prepared else None
        timings, qualities = [], []
        for repeat in range(repeats + 1):
            start = time.perf_counter()
            solver = workspace if prepared else prepare()
            solved = solver.solve(raise_error=True)
            elapsed = time.perf_counter() - start
            measured = quality(p, solved.x, solved.y, expected)
            if solved.info.status_val != 1 or measured["verdict"] != "PASS":
                raise RuntimeError(f"OSQP quality failure: {measured}")
            if repeat:
                timings.append(elapsed)
                qualities.append(measured)
        results["prepared" if prepared else "cold"] = {"latency": latency(timings), "quality": qualities}
    return results


def reference_clarabel(p, expected, repeats):
    import clarabel
    from scipy import sparse

    P, q, A, l, u = [t.numpy() for t in (p.P, p.q, p.A, p.l, p.u)]
    equality = l == u
    inequalities = ~equality
    combined = np.concatenate([A[equality], A[inequalities], -A[inequalities]])
    b = np.concatenate([u[equality], u[inequalities], -l[inequalities]])
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-8
    cones = [
        clarabel.ZeroConeT(int(equality.sum())),
        clarabel.NonnegativeConeT(2 * int(inequalities.sum())),
    ]
    times, qualities = [], []
    for repeat in range(repeats + 1):
        start = time.perf_counter()
        solver = clarabel.DefaultSolver(
            sparse.csc_matrix(np.triu(P)), q, sparse.csc_matrix(combined), b, cones, settings,
        )
        result = solver.solve()
        elapsed = time.perf_counter() - start
        z = np.array(result.z)
        y = np.zeros(len(l))
        count = int(equality.sum())
        other = int(inequalities.sum())
        y[equality] = z[:count]
        y[inequalities] = z[count:count + other] - z[count + other:]
        measured = quality(p, np.array(result.x), y, expected)
        if str(result.status) != "Solved" or measured["verdict"] != "PASS":
            raise RuntimeError(f"Clarabel quality failure: {measured}")
        if repeat:
            times.append(elapsed)
            qualities.append(measured)
    return {"cold": {"latency": latency(times), "quality": qualities}, "prepared": "UNVERIFIED"}


def commercial_probe() -> dict:
    # Never print exception text from license APIs (it may include endpoints).
    found = {}
    for name in ("gurobipy", "cplex", "mosek"):
        available = importlib.util.find_spec(name) is not None
        found[name] = {
            "status": "UNVERIFIED",
            "reason": "matched benchmark not executed" if available else "SDK unavailable",
            "sdk_available": available,
        }
    return found


def experiment(args, report):
    output = args.artifact_dir
    if "fuse.hf3fs" not in Path("/proc/self/mounts").read_text():
        raise RuntimeError("shared mount missing")
    if args.reference_only:
        saved = np.load(args.problem)
        p = QPProblem(*[torch.from_numpy(saved[key]) for key in ("P", "q", "A", "l", "u")])
        expected = saved["expected"]
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("one CUDA GPU required")
        if "RTX PRO 4500" not in torch.cuda.get_device_name(0):
            raise RuntimeError("RTX PRO 4500 required")
        p, expected, witness_y = generate(args.size)
        witness = quality(p, expected, witness_y, expected)
        if witness["verdict"] != "PASS":
            raise RuntimeError("known-KKT generator witness invalid")
        report["witness"] = witness
        np.savez(
            output / "problem.npz", P=p.P.numpy(), q=p.q.numpy(), A=p.A.numpy(),
            l=p.l.numpy(), u=p.u.numpy(), expected=expected, witness_y=witness_y,
        )
    hasher = hashlib.sha256()
    for tensor in (p.P, p.q, p.A, p.l, p.u):
        hasher.update(tensor.numpy().tobytes())
    report["problem_hash"] = hasher.hexdigest()
    report["dimensions"] = {"variables": p.q.numel(), "constraints": p.A.shape[0]}
    report["software"] = {
        name: importlib.metadata.version(name) if importlib.util.find_spec(name) else None
        for name in ("numpy", "torch", "scipy", "osqp", "clarabel")
    }
    report["hardware"] = {
        "node": os.environ.get("OPENOPT_NODE"), "cpu_threads": torch.get_num_threads(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if not args.reference_only else None,
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        if Path("/sys/fs/cgroup/cpu.max").is_file() else None,
        "cgroup_membership": Path("/proc/self/cgroup").read_text().strip(),
    }
    report["commercial"] = commercial_probe()
    write(output / "partial.json", report)
    config = QPConfig(eps_abs=1e-7, eps_rel=1e-7)
    if not args.reference_only:
        telemetry = Telemetry(output)
        with telemetry:
            report["openopt"] = {}
            for device in ("cpu", "cuda:0"):
                c = QPConfig(**{**asdict(config), "device": device})
                report["openopt"][device] = {}
                for mode in ("cold", "prepared"):
                    report["openopt"][device][mode] = time_openopt(
                        p, c, expected, args.repeats, mode == "prepared",
                    )
                    write(output / "partial.json", report)
            report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["paired_ab"] = paired_ab(p, expected, args.repeats)
            report["paired_devices"] = paired_devices(p, expected, args.repeats)
        if not telemetry.samples or any(s["rc"] != 0 for s in telemetry.samples):
            raise RuntimeError("GPU telemetry missing or failed")
        nsys = shutil.which("nsys")
        if not nsys:
            raise RuntimeError("Nsight Systems required for GPU bottleneck evidence")
        # Nsight's mmap/database scratch requires local shared memory, not
        # network FUSE. Only scratch is local; all evidence is copied to 3FS.
        with tempfile.TemporaryDirectory(prefix="openopt-nsys-", dir="/dev/shm") as temp:
            local = Path(temp)
            env = os.environ.copy()
            env["TMPDIR"] = str(local)
            with (output / "nsys.log").open("x") as log:
                result = subprocess.run([
                    nsys, "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                    "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                    "--output", str(local / "nsight"), sys.executable, str(Path(__file__).resolve()),
                    "--profile-only", "--problem", str(output / "problem.npz"),
                    "--artifact-dir", str(output), "--linear-solver", "triangular",
                ], env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            report["nsight_capture_rc"] = result.returncode
            if result.returncode or not (local / "nsight.nsys-rep").is_file():
                raise RuntimeError(f"Nsight capture failed rc={result.returncode}; see nsys.log")
            shutil.copyfile(local / "nsight.nsys-rep", output / "nsight.nsys-rep")
            with (output / "nsys-stats.csv").open("x") as log:
                result = subprocess.run([
                    nsys, "stats", "--report=cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum",
                    "--format=csv", str(local / "nsight.nsys-rep"),
                ], env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"Nsight stats failed rc={result.returncode}; see nsys-stats.csv")
            if (local / "nsight.sqlite").is_file():
                shutil.copyfile(local / "nsight.sqlite", output / "nsight.sqlite")
        report["nsight"] = {"status": "PASS", "report": "nsight.nsys-rep", "stats": "nsys-stats.csv"}
    report["references"] = {}
    for module, runner in (("osqp", reference_osqp), ("clarabel", reference_clarabel)):
        if importlib.util.find_spec(module):
            report["references"][module] = runner(p, expected, args.repeats)
        else:
            report["references"][module] = {"status": "UNVERIFIED", "reason": "SDK unavailable"}
        write(output / "partial.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--linear-solver", choices=("cholesky", "triangular"), default="cholesky")
    parser.add_argument("--problem", type=Path)
    args = parser.parse_args()
    if args.profile_only:
        saved = np.load(args.problem)
        p = QPProblem(*[torch.from_numpy(saved[key]) for key in ("P", "q", "A", "l", "u")])
        workspace = prepare_qp(p, QPConfig(device="cuda:0", linear_solver=args.linear_solver))
        workspace.solve()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        start = time.perf_counter()
        result = workspace.solve()
        result.x.cpu()
        result.y.cpu()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        torch.cuda.cudart().cudaProfilerStop()
        write(args.artifact_dir / "profile-wall.json", {
            "wall_s": elapsed, "linear_solver": args.linear_solver,
            "scope": "profiled prepared solve including host results",
        })
        return 0
    if args.size < 3 or args.repeats < 3 or (args.reference_only and not args.problem):
        parser.error("size/repeats >=3; reference-only requires a frozen problem")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "verdict": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic dense convex QP only; no commercial or multi-GPU solver qualification",
        "timing": "cold includes validation/H2D/factor/solve/D2H; prepared excludes setup; one untimed warmup",
        "reference_only": args.reference_only,
    }
    start = time.perf_counter()
    try:
        experiment(args, report)
        report["verdict"] = "PASS"
    except Exception as error:
        report["verdict"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        (args.artifact_dir / "failure.log").write_text(traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["wall_s"] = time.perf_counter() - start
    write(args.artifact_dir / "result.json", report)
    verdict = f"OPENOPT_QP_BENCH verdict={report['verdict']} n={args.size} wall_s={report['wall_s']:.3f}"
    (args.artifact_dir / "execution.log").write_text(verdict + "\n")
    print(verdict + f" result={args.artifact_dir / 'result.json'}", flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
