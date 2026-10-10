"""One-card actual CUDA solve preflight. Never prints license configuration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.artifact_dir
    report = {"verdict": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat()}
    start = time.perf_counter()
    try:
        mounts = Path("/proc/self/mounts").read_text()
        if "fuse.hf3fs" not in mounts:
            raise RuntimeError("shared FUSE mount missing; refusing local artifacts")
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA GPU is required")
        name = torch.cuda.get_device_name(0)
        if "RTX PRO 4500" not in name:
            raise RuntimeError(f"wrong GPU model: {name}")
        report.update({
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "node": os.environ.get("OPENOPT_NODE"), "uid": os.getuid(),
            "arch_list": torch.cuda.get_arch_list(),
            "tools": {tool: shutil.which(tool) for tool in ("nsys", "ncu", "nvidia-smi")},
            "modules": {module: importlib.util.find_spec(module) is not None for module in (
                "numpy", "scipy", "osqp", "clarabel", "highspy", "gurobipy", "cplex", "mosek", "pytest",
            )},
            "commercial_license_variable_present": {
                variable: variable in os.environ for variable in (
                    "MOSEKLM_LICENSE_FILE", "GRB_LICENSE_FILE", "ILOG_LICENSE_FILE",
                )
            },
        })
        torch.cuda.synchronize()
        report["load_s"] = time.perf_counter() - start
        solve_start = time.perf_counter()
        matrix = torch.eye(64, dtype=torch.float64, device="cuda")
        rhs = torch.arange(64, dtype=torch.float64, device="cuda")
        x = torch.linalg.solve(matrix, rhs)
        torch.cuda.synchronize()
        error = float((x - rhs).abs().max().cpu())
        if error != 0 or not x.is_cuda:
            raise RuntimeError(f"CUDA solve failed: max_error={error}")
        report["cuda_linear_solve"] = {"max_error": error, "device": str(x.device)}
        report["solve_s"] = time.perf_counter() - solve_start
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"nvidia-smi failed rc={result.returncode}")
        report["gpu_driver"] = result.stdout.strip()
        report["verdict"] = "PASS"
    except Exception as error:
        report["verdict"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        (output / "failure.log").write_text(traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["wall_s"] = time.perf_counter() - start
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "execution.log").write_text(
        f"OPENOPT_GPU_PROBE verdict={report['verdict']} wall_s={report['wall_s']:.3f}\n"
    )
    print(f"OPENOPT_GPU_PROBE verdict={report['verdict']} result={output / 'result.json'}", flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
