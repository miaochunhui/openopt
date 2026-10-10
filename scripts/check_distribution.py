"""Build and validate a wheel without importing this checkout's source tree."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def validate_wheel_license(wheel: zipfile.ZipFile, expected: bytes) -> None:
    metadata_paths = [
        name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_paths) != 1:
        raise ValueError("wheel must contain exactly one distribution METADATA")
    path = metadata_paths[0]
    metadata = BytesParser().parsebytes(wheel.read(path))
    if metadata.get_all("License-Expression") != ["0BSD"]:
        raise ValueError("wheel must declare License-Expression: 0BSD")
    if metadata.get_all("License-File") != ["LICENSE"]:
        raise ValueError("wheel must declare exactly License-File: LICENSE")
    license_path = path.rsplit("/", 1)[0] + "/licenses/LICENSE"
    if license_path not in wheel.namelist() or wheel.read(license_path) != expected:
        raise ValueError("wheel license payload is absent or differs from LICENSE")
    if not expected.strip():
        raise ValueError("empty source license")


def installed_probe(target):
    sys.path.insert(0, str(target))
    import openopt
    import openpo
    import torch

    for module in (openopt, openpo):
        if not Path(module.__file__).resolve().is_relative_to(target):
            raise ValueError(f"package loaded outside isolated wheel install: {module.__name__}")
    for name in openpo.__all__:
        if getattr(openopt, name) is not getattr(openpo, name):
            raise ValueError(f"reexport identity mismatch: {name}")
    problem = openopt.PortfolioProblem(
        alpha=torch.tensor([0.01, 0.02], dtype=torch.float64),
        risk=openopt.FactorRiskModel(
            exposures=torch.zeros((2, 1), dtype=torch.float64),
            factor_variance=torch.ones(1, dtype=torch.float64),
            specific_variance=torch.ones(2, dtype=torch.float64),
        ),
        lower=torch.zeros(2, dtype=torch.float64),
        upper=torch.ones(2, dtype=torch.float64),
        risk_aversion=2.0,
    )
    result = openopt.solve(problem, openopt.SolveConfig(iterations=300, tolerance=1e-9))
    if not result.converged or not torch.isfinite(result.weights).all():
        raise ValueError("isolated wheel solve did not converge")
    if float((result.weights.sum(dim=1) - 1.0).abs().max()) > 1e-6:
        raise ValueError("isolated wheel budget constraint failed")
    print(f"WHEEL_IMPORT verdict=PASS api_objects={len(openpo.__all__)} version={openopt.__version__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("/hf3fs-jd/data/tmp/openopt-distribution"))
    parser.add_argument("--installed", type=Path)
    args = parser.parse_args()
    if args.installed:
        installed_probe(args.installed.resolve())
        return
    base = args.artifact_dir.resolve()
    if base.is_relative_to(ROOT):
        raise ValueError("distribution artifacts must be outside the repository")
    base.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="wheel-", dir=base))
    target = artifacts / "installed"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(artifacts)
    report = {"verdict": "RUNNING", "at": datetime.now(timezone.utc).isoformat()}
    try:
        with (artifacts / "distribution.log").open("x") as log:
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                 "--wheel-dir", str(artifacts), str(ROOT)], env=env, stdout=log,
                stderr=subprocess.STDOUT, check=True,
            )
            wheels = list(artifacts.glob("*.whl"))
            if len(wheels) != 1:
                raise ValueError(f"expected exactly one wheel: {wheels}")
            with zipfile.ZipFile(wheels[0]) as wheel:
                validate_wheel_license(wheel, (ROOT / "LICENSE").read_bytes())
                names = set(wheel.namelist())
                expected = {path.relative_to(ROOT / "src").as_posix()
                            for path in (ROOT / "src").rglob("*.py")}
                if not expected or not expected.issubset(names):
                    raise ValueError(f"wheel missing modules: {sorted(expected - names)}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                 "--target", str(target), str(wheels[0])], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
            subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--installed", str(target)],
                cwd=artifacts, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        report.update(
            verdict="PASS", wheel=wheels[0].name, modules=len(expected),
            license_expression="0BSD", license_payload_verified=True,
        )
    except Exception as error:
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (artifacts / "distribution.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"OPENOPT_DISTRIBUTION verdict={report['verdict']} report={artifacts / 'distribution.json'}")


if __name__ == "__main__":
    main()
