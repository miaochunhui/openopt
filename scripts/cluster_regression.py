"""Run independent CUDA API checks in a one-GPU Job with external evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    import torch

    if not torch.cuda.is_available() or "RTX PRO 4500" not in torch.cuda.get_device_name(0):
        raise RuntimeError("formal CUDA regression requires RTX PRO 4500")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")])
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    log_path = args.artifact_dir / "pytest.log"
    xml_path = args.artifact_dir / "pytest.xml"
    with log_path.open("x") as log:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "-q", "-ra",
            "--basetemp", str(args.artifact_dir / "pytest-tmp"),
            "-o", f"cache_dir={args.artifact_dir / 'pytest-cache'}",
            f"--junitxml={xml_path}", str(root / "tests/test_general_qp_cuda.py"),
            str(root / "tests/test_gpu_execution.py"),
        ], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    counts = {}
    if xml_path.is_file():
        suites = ElementTree.parse(xml_path).getroot().findall("testsuite")
        counts = {name: sum(int(s.attrib.get(name, 0)) for s in suites)
                  for name in ("tests", "failures", "errors", "skipped")}
    passed = (
        result.returncode == 0 and counts.get("tests", 0) > 0
        and not any(counts.get(name, 0) for name in ("failures", "errors", "skipped"))
    )
    report = {
        "verdict": "PASS" if passed else "FAIL", "returncode": result.returncode,
        "tests": counts, "node": os.environ.get("OPENOPT_NODE"),
        "gpu": torch.cuda.get_device_name(0), "at": datetime.now(timezone.utc).isoformat(),
    }
    (args.artifact_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"OPENOPT_CUDA_REGRESSION verdict={report['verdict']} tests={counts} log={log_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
