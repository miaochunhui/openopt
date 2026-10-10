"""Bounded RTX PRO 4500 qualification for optional QP research paths.

Independent known-KKT checks are outside latency. Cold includes model assembly,
transfer, validation, setup, solve and host outputs. Prepared excludes setup.
Sparse comparisons have declared caller PSD assumptions, not dense validation
parity. No commercial ranking, broad-corpus or multi-GPU claim is made.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from benchmark_general_qp import Telemetry, generate, latency, quality, write
from openopt.qp import QPConfig, QPProblem, prepare_qp, solve_qp


def diagonal_problem(n: int, seed: int = 47):
    rng = np.random.default_rng(seed)
    diagonal = np.linspace(1, 2, n)
    x = rng.uniform(-0.5, 0.5, n)
    y = np.zeros(n)
    y[::3] = 0.3
    y[1::3] = -0.3
    l, u = x - 0.5, x + 0.5
    u[::3], l[1::3] = x[::3], x[1::3]
    q = -diagonal * x - y
    problem = QPProblem(*(torch.from_numpy(a) for a in (
        np.diag(diagonal), q, np.eye(n), l, u,
    )))
    return problem, x, y


def check(problem, result, expected):
    if result.x.device.type != "cuda" or result.y.device.type != "cuda":
        raise RuntimeError("explicit CUDA output was not device resident")
    measured = quality(problem, result.x.cpu().numpy(), result.y.cpu().numpy(), expected)
    measured["solver_status"] = result.status
    measured["host_x"], measured["host_y"] = result.x.cpu().tolist(), result.y.cpu().tolist()
    if result.status != "solved" or measured["verdict"] != "PASS":
        raise RuntimeError(f"independent CUDA quality failed: {measured}")
    return measured


def correctness(output: Path) -> dict:
    from openopt.batch import BatchQPProblem, solve_batch_qp
    from openopt.operators import OperatorQPConfig, OperatorQPProblem, solve_operator_qp

    checks = {}
    problem, expected, witness = generate(24)
    np.savez(output / "problem-dense.npz",
             **{key: getattr(problem, key).numpy() for key in ("P", "q", "A", "l", "u")},
             expected=expected, witness_y=witness)
    if quality(problem, expected, witness, expected)["verdict"] != "PASS":
        raise RuntimeError("invalid independent witness")
    for adaptive in (False, True):
        c = QPConfig(device="cuda:0", adaptive_rho=adaptive)
        checks[f"dense_adaptive_{adaptive}"] = check(problem, solve_qp(problem, c), expected)
    batch = BatchQPProblem.stack([problem, problem, problem])
    result = solve_batch_qp(batch, QPConfig(device="cuda:0"))
    if result.x.device.type != "cuda" or result.iterations.device.type != "cuda":
        raise RuntimeError("batch outputs are not device resident")
    for i in range(3):
        measured = quality(problem, result.x[i].cpu().numpy(), result.y[i].cpu().numpy(), expected)
        measured["solver_status"] = result.status[i]
        measured["host_x"], measured["host_y"] = result.x[i].cpu().tolist(), result.y[i].cpu().tolist()
        if result.status[i] != "solved" or measured["verdict"] != "PASS":
            raise RuntimeError(f"batch quality failed item={i}: {measured}")
        checks[f"batch_{i}"] = measured
    simple, x, _ = diagonal_problem(32)
    _, _, simple_y = diagonal_problem(32)
    np.savez(output / "problem-sparse.npz",
             **{key: getattr(simple, key).numpy() for key in ("P", "q", "A", "l", "u")},
             expected=x, witness_y=simple_y)
    for layout in ("coo", "csr"):
        converter = lambda t: t.to_sparse_coo() if layout == "coo" else t.to_sparse_csr()
        sparse = OperatorQPProblem(
            converter(simple.P), simple.q, converter(simple.A), simple.l, simple.u,
            assume_psd=True,
        )
        result = solve_operator_qp(sparse, OperatorQPConfig(device="cuda:0"))
        checks[f"sparse_{layout}"] = check(simple, result, x)
        if result.convexity_certified or not result.experimental:
            raise RuntimeError("sparse path misrepresented caller PSD assumption")
    log_path = output / "pytest.log"
    xml = output / "pytest.xml"
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")])
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with log_path.open("x") as log:
        completed = subprocess.run([
            sys.executable, "-m", "pytest", "-q", "-ra",
            "-o", f"cache_dir={output / 'pytest-cache'}",
            "--basetemp", str(output / "pytest-tmp"), f"--junitxml={xml}",
            "tests/test_general_qp_cuda.py", "tests/test_qp_batch.py",
            "tests/test_qp_operators.py", "tests/test_qp_campaign_tools.py",
        ], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    from xml.etree import ElementTree
    suites = ElementTree.parse(xml).getroot().findall("testsuite") if xml.is_file() else []
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ("tests", "errors", "failures", "skipped")}
    if completed.returncode or not counts["tests"] or any(counts[k] for k in ("errors", "failures", "skipped")):
        raise RuntimeError(f"CUDA regression rc={completed.returncode} counts={counts}; {log_path}")
    return {
        "checks": checks, "tests": counts, "dtype": "float64",
        "eps_abs": 1e-7, "eps_rel": 1e-7,
        "problem_files": {"dense": "problem-dense.npz", "sparse": "problem-sparse.npz"},
    }


def performance(mode: str, n: int, repeats: int, profile: bool = False, output: Path | None = None):
    from openopt.batch import BatchQPProblem, prepare_batch_qp
    from openopt.operators import OperatorQPProblem, OperatorQPConfig, prepare_operator_qp

    problem, expected, witness = diagonal_problem(n) if mode == "operators" else generate(n)
    arrays = tuple(getattr(problem, key).numpy() for key in ("P", "q", "A", "l", "u"))
    digest = hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
    if output is not None and not profile:
        np.savez(output / "problem.npz",
                 **dict(zip(("P", "q", "A", "l", "u"), arrays, strict=True)),
                 expected=expected, witness_y=witness)
    if quality(problem, expected, witness, expected)["verdict"] != "PASS":
        raise RuntimeError("performance witness invalid")
    config = QPConfig(device="cuda:0")
    batch_size = 8

    def prepare_arm(arm):
        if arm in ("baseline", "control"):
            if mode == "batch":
                return [prepare_qp(problem, config) for _ in range(batch_size)]
            return prepare_qp(problem, config)
        if mode == "adaptive":
            return prepare_qp(problem, QPConfig(device="cuda:0", adaptive_rho=True))
        if mode == "batch":
            return prepare_batch_qp(BatchQPProblem.stack([problem] * batch_size), config)
        sparse = OperatorQPProblem(
            problem.P.to_sparse_coo(), problem.q, problem.A.to_sparse_coo(),
            problem.l, problem.u, assume_psd=True,
        )
        return prepare_operator_qp(sparse, OperatorQPConfig(device="cuda:0"))

    def execute(space):
        if isinstance(space, list):
            # This sequential arm is the intentional same-workload baseline.
            return [p.solve() for p in space]
        return space.solve()

    def host_and_validate(result, validate):
        if isinstance(result, list):
            xy = [(r.x.cpu().numpy(), r.y.cpu().numpy(), r.status) for r in result]
        elif mode == "batch" and result.x.ndim == 2:
            xy = [(x, y, status) for x, y, status in zip(
                result.x.cpu().numpy(), result.y.cpu().numpy(), result.status, strict=True)]
        else:
            xy = [(result.x.cpu().numpy(), result.y.cpu().numpy(), result.status)]
        if validate:
            measured = [quality(problem, x, y, expected) for x, y, _ in xy]
            for item, (x, y, status) in zip(measured, xy, strict=True):
                item["solver_status"] = status
                item["host_x"], item["host_y"] = x.tolist(), y.tolist()
            if any(status != "solved" or q["verdict"] != "PASS"
                   for (_, _, status), q in zip(xy, measured, strict=True)):
                raise RuntimeError(f"performance arm failed independent quality: {measured}")
            return measured
        return xy

    if profile:
        space = prepare_arm("candidate")
        host_and_validate(execute(space), True)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        start = time.perf_counter()
        result = execute(space)
        host_and_validate(result, False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        torch.cuda.cudart().cudaProfilerStop()
        return {"profiled_prepared_e2e_s": elapsed}
    data = {}
    for prepared in (False, True):
        arms = ("baseline", "candidate", "control")
        spaces = {arm: prepare_arm(arm) for arm in arms} if prepared else {}
        times, qualities = {a: [] for a in arms}, {a: [] for a in arms}
        for repeat in range(repeats + 1):
            for arm in (arms if repeat % 2 == 0 else tuple(reversed(arms))):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = execute(spaces[arm] if prepared else prepare_arm(arm))
                host_and_validate(result, False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                measured = host_and_validate(result, True)
                if repeat:
                    times[arm].append(elapsed)
                    qualities[arm].extend(measured)
        data["prepared" if prepared else "cold"] = {
            a: {"latency": latency(times[a]), "quality": qualities[a]} for a in arms}
    return {
        "problem_hash": digest, "measurements": data, "batch_size": batch_size if mode == "batch" else 1,
        "dtype": "float64", "eps_abs": config.eps_abs, "eps_rel": config.eps_rel,
        "control_order": "forward/reverse alternating; one untimed warmup",
        "input_generation": "excluded; model assembly, validation and transfers are included in cold",
        "batch_population": "eight copies of one same-shape problem" if mode == "batch" else None,
        "arm_config": {
            "baseline": {"rho": 1.0, "adaptive_rho": False, "linear_solver": "cholesky"},
            "control": {"rho": 1.0, "adaptive_rho": False, "linear_solver": "cholesky"},
            "candidate": {"rho": 1.0, "adaptive_rho": mode == "adaptive",
                          "linear_solver": "CG" if mode == "operators" else "cholesky"},
        },
        "validation_parity": mode != "operators",
        "scope": "sparse assumes PSD and includes dense-to-sparse conversion; dense validates full spectrum"
        if mode == "operators" else "same dense mathematical workload and validation",
    }


def profile_run(args) -> dict:
    from summarize_qp_benchmark import nsight_summary
    nsys = shutil.which("nsys")
    if not nsys:
        raise RuntimeError("Nsight Systems missing")
    with tempfile.TemporaryDirectory(prefix="qp3-profile-", dir="/dev/shm") as temp:
        local = Path(temp)
        env = os.environ.copy()
        env["TMPDIR"] = str(local)
        with (args.artifact_dir / "nsys.log").open("x") as log:
            result = subprocess.run([
                nsys, "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                "--output", str(local / "profile"), sys.executable, str(Path(__file__).resolve()),
                "--artifact-dir", str(args.artifact_dir), "--mode", args.mode,
                "--size", str(args.size), "--profile-only",
            ], env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Nsight capture rc={result.returncode}; nsys.log")
        shutil.copyfile(local / "profile.nsys-rep", args.artifact_dir / "profile.nsys-rep")
        with (args.artifact_dir / "nsys-stats.csv").open("x") as log:
            result = subprocess.run([
                nsys, "stats", "--report=cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum",
                "--format=csv", str(local / "profile.nsys-rep"),
            ], env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Nsight stats rc={result.returncode}")
    summary = nsight_summary(args.artifact_dir / "nsys-stats.csv")
    summary.update(json.loads((args.artifact_dir / "profile-wall.json").read_text()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("correctness", "adaptive", "batch", "operators"), default="correctness")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    if args.size < 3 or args.repeats < 3:
        parser.error("size/repeats must be >=3")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {"verdict": "RUNNING", "mode": args.mode, "size": args.size,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    start = time.perf_counter()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA device required")
        gpu = torch.cuda.get_device_name(0)
        if "RTX PRO 4500" not in gpu:
            raise RuntimeError("formal measurement requires RTX PRO 4500")
        report["hardware"] = {
            "node": os.environ.get("OPENOPT_NODE"), "gpu": gpu, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        }
        if args.profile_only:
            write(args.artifact_dir / "profile-wall.json", performance(args.mode, args.size, 3, True))
            return 0
        with Telemetry(args.artifact_dir):
            report["result"] = correctness(args.artifact_dir) if args.mode == "correctness" else performance(
                args.mode, args.size, args.repeats, output=args.artifact_dir)
        if args.mode != "correctness":
            report["nsight"] = profile_run(args)
        samples = json.loads((args.artifact_dir / "telemetry.json").read_text())
        if not samples or any(s["rc"] != 0 for s in samples):
            raise RuntimeError("telemetry missing or failed")
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["verdict"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["verdict"] = "FAIL"
        (args.artifact_dir / "failure.log").write_text(traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["wall_s"] = time.perf_counter() - start
    write(args.artifact_dir / "result.json", report)
    line = f"QP_CAMPAIGN_GPU verdict={report['verdict']} mode={args.mode} wall_s={report['wall_s']:.3f}"
    (args.artifact_dir / "execution.log").write_text(line + "\n")
    print(line, flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
