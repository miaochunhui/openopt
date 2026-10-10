"""Verify the standalone foundation with external artifacts and fail-loud logs.

This is a correctness gate, not the old OpenPO performance qualification.
No source repository, inherited result files or cluster access is required.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SMOKE_TESTS = [
    "tests/test_openopt_api.py", "tests/test_public_api.py",
    "tests/test_solver.py", "tests/test_constraints.py",
    "tests/test_device_contract.py", "tests/test_projection_correction.py",
]
DEPENDENCIES = {
    "numpy": "2.0", "torch": "2.4", "pytest": "8",
    "scipy": "1.15", "osqp": "1.0", "clarabel": "0.10", "highspy": "1.15",
}


def run_logged(command: list[str], artifact_dir: Path, name: str, env: dict[str, str]) -> None:
    log_path = artifact_dir / f"{name}.log"
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"command={command!r}\n")
        log.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
        log.write(f"\nreturncode={result.returncode}\n")
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-16:]
    print(f"VERIFY_STEP name={name} rc={result.returncode} log={log_path}", flush=True)
    for line in tail:
        print(line[:500], flush=True)
    if result.returncode:
        raise RuntimeError(f"{name} failed rc={result.returncode}; full log={log_path}")


def versions() -> dict[str, str]:
    from packaging.version import Version

    found = {}
    for name, minimum in DEPENDENCIES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"missing {name}; install this repository's .[test] dependencies"
            ) from error
        if Version(actual) < Version(minimum):
            raise RuntimeError(f"unsupported {name}={actual}; minimum={minimum}")
        found[name] = actual
    return found


def run(mode: str, artifact_root: Path, require_cuda: bool, targets: list[str] | None = None) -> Path:
    root = artifact_root.expanduser().resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("verification artifacts must be outside the repository")
    root.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=f"{mode}-", dir=root))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), str(ROOT / "benchmarks")]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TMPDIR"] = str(artifacts)
    env["TRITON_CACHE_DIR"] = str(artifacts / "triton-cache")
    report: dict[str, object] = {
        "schema_version": 1, "mode": mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "python_executable": sys.executable,
        "artifact_dir": str(artifacts), "verdict": "RUNNING",
        "qualification": "correctness only; no performance claim",
        "targets": targets or (SMOKE_TESTS if mode == "smoke" else ["tests"]),
        "coverage": "explicit_targets" if targets else mode,
    }
    try:
        report["dependencies"] = versions()
        run_logged(
            [sys.executable, "scripts/migrate_openpo.py", "--provenance"],
            artifacts, "snapshot", env,
        )
        sys.path.insert(0, str(ROOT / "src"))
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_runtime"] = torch.version.cuda
        report["hardware"] = {
            "cpu_count": os.cpu_count(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if report["cuda_available"] else [],
        }
        if require_cuda and not report["cuda_available"]:
            raise RuntimeError("CUDA verification requested but CUDA is unavailable")
        targets = targets or (SMOKE_TESTS if mode == "smoke" else ["tests"])
        run_logged(
            [
                sys.executable, "-m", "pytest", "-q", "-ra",
                "-o", f"cache_dir={artifacts / 'pytest-cache'}",
                "--basetemp", str(artifacts / "pytest-tmp"),
                f"--junitxml={artifacts / 'pytest.xml'}", *targets,
            ],
            artifacts, "pytest", env,
        )
        from xml.etree import ElementTree

        suites = ElementTree.parse(artifacts / "pytest.xml").getroot()
        counts = {
            name: sum(int(suite.attrib.get(name, 0)) for suite in suites.findall("testsuite"))
            for name in ("tests", "failures", "errors", "skipped")
        }
        report["tests"] = counts
        if not counts["tests"] or counts["failures"] or counts["errors"]:
            raise RuntimeError(f"test artifact did not pass: {counts}")
        run_logged(
            [sys.executable, "examples/portfolio_qp.py", "--device", "cpu"],
            artifacts, "example-cpu", env,
        )
        if require_cuda:
            run_logged(
                [sys.executable, "examples/portfolio_qp.py", "--device", "cuda:0"],
                artifacts, "example-cuda", env,
            )
        report["verdict"] = "PASS"
    except Exception as error:
        report["verdict"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (artifacts / "verification.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        print(
            f"OPENOPT_VERIFY verdict={report['verdict']} mode={mode} "
            f"report={artifacts / 'verification.json'}", flush=True,
        )
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("smoke", "tests"), default="tests")
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=Path("/hf3fs-jd/data/tmp/openopt-verification"),
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--target", action="append", help="explicit pytest target for failure reproduction")
    args = parser.parse_args(argv)
    try:
        run(args.mode, args.artifact_dir, args.require_cuda, args.target)
    except Exception as error:
        print(f"OPENOPT_VERIFY error={type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
