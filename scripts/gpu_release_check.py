"""Strict one-card CUDA regression with external JUnit and resource evidence.

Only explicit test targets are accepted. This is correctness evidence, not
commercial qualification or a latency/speedup experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def junit_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise ValueError("JUnit completion artifact missing")
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    counts = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if counts["tests"] <= 0 or len(cases) != counts["tests"]:
        raise ValueError(f"empty or incomplete JUnit execution: {counts}")
    if any(counts[key] for key in ("failures", "errors", "skipped")):
        raise ValueError(f"CUDA tests failed or skipped: {counts}")
    if any(list(case) for case in cases if any(
        child.tag in ("failure", "error", "skipped") for child in case
    )):
        raise ValueError("JUnit cases contradict successful suite totals")
    return counts


def validate_targets(targets: list[str]) -> None:
    if not targets:
        raise ValueError("at least one explicit CUDA test target is required")
    for target in targets:
        file = target.split("::", 1)[0]
        path = ROOT / file
        if not file.startswith("tests/test_") or not file.endswith(".py"):
            raise ValueError(f"invalid test target: {target}")
        if not path.resolve().is_relative_to(ROOT / "tests") or not path.is_file():
            raise ValueError(f"test target outside tests or absent: {target}")
        if path.is_symlink():
            raise ValueError(f"symlink test target rejected: {target}")


def run(args: argparse.Namespace) -> Path:
    output_root = args.artifact_dir.resolve()
    if not output_root.is_relative_to(Path("/3fs/research3fs/data/tmp")):
        raise ValueError("GPU evidence must be outside Git on shared 3FS tmp")
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="cuda-check-", dir=output_root))
    report = {
        "verdict": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
        "targets": args.target, "selection": args.expression,
        "scope": "one-card numerical CUDA regression only; no performance qualification",
        "source_sha256": {},
    }
    start = time.perf_counter()
    try:
        validate_targets(args.target)
        import torch
        from benchmark_general_qp import Telemetry

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA device is required")
        gpu = torch.cuda.get_device_name(0)
        if "RTX PRO 4500" not in gpu:
            raise RuntimeError("formal CUDA checks require RTX PRO 4500")
        report["hardware"] = {
            "gpu": gpu, "node": os.environ.get("OPENOPT_NODE"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        }
        sources = sorted((ROOT / "src/openopt").glob("*.py"))
        sources += [ROOT / t.split("::", 1)[0] for t in args.target]
        sources += [Path(__file__)]
        report["source_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        }
        env = os.environ.copy()
        env.update(
            PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "src")]),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
        )
        xml = output / "pytest.xml"
        command = [
            sys.executable, "-m", "pytest", "-q", "-ra",
            "-o", f"cache_dir={output / 'pytest-cache'}",
            "--basetemp", str(output / "pytest-tmp"), f"--junitxml={xml}",
            *args.target,
        ]
        if args.expression:
            command += ["-k", args.expression]
        with Telemetry(output):
            with (output / "pytest.log").open("x") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                log.write(f"\nreturncode={result.returncode}\n")
        report["returncode"] = result.returncode
        report["tests"] = junit_counts(xml)
        if result.returncode:
            raise RuntimeError(f"CUDA pytest rc={result.returncode}; see pytest.log")
        samples = json.loads((output / "telemetry.json").read_text())
        if not samples or any(sample["rc"] != 0 for sample in samples):
            raise RuntimeError("GPU telemetry absent or failed")
        if not all(sample.get("cpu", {}).get("usage_usec") is not None for sample in samples):
            raise RuntimeError("pod CPU cgroup telemetry missing")
        report["telemetry_samples"] = len(samples)
        report["verdict"] = "PASS"
    except Exception as error:
        report["verdict"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        (output / "failure.log").write_text(traceback.format_exc())
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["wall_s"] = time.perf_counter() - start
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        (output / "result.json").write_text(text)
        # The coordinator expects result.json directly under its assigned case.
        (output_root / "result.json").write_text(text)
        line = (
            f"OPENOPT_RELEASE_CUDA verdict={report['verdict']} "
            f"tests={report.get('tests', {}).get('tests', 'NA')} evidence={output}"
        )
        (output / "execution.log").write_text(line + "\n")
        print(line, flush=True)
    if report["verdict"] != "PASS":
        raise RuntimeError(f"CUDA correctness failed; evidence={output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--expression", default="")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"OPENOPT_RELEASE_CUDA error={error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
