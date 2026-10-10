"""Frozen CPU-only RSOC review; run this file --artifact-dir PATH --group full.

Decimal original-coordinate references do not call a solver or project SOC.
Failed tests remain hard failures. Receipts pin actual source and log bytes;
they are not CUDA execution, remote attestation or commercial qualification.
"""

from decimal import Decimal, localcontext
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from openopt.conic import (
    ConicProblem, RSOC, SOC, Zero, check_cone_feasibility, check_conic_candidate,
)
from openopt.native_conic import project_cones

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def verified_loaded_root():
    import openopt.conic as conic
    import openopt.native_conic as native

    assert Path(conic.__file__).resolve() == ROOT / "src/openopt/conic.py"
    assert Path(native.__file__).resolve() == ROOT / "src/openopt/native_conic.py"


def decimal_values(value):
    return [Decimal.from_float(float(item)) for item in value]


def original_projection(value, rotated=True):
    """1100 digits retain squared differences across the float64 exponent span."""
    with localcontext() as context:
        context.prec = 1100
        items = decimal_values(value)
        if not rotated:
            head, tail = items[0], items[1:]
            radius = sum(item * item for item in tail).sqrt()
            if head >= radius:
                return items
            if -head >= radius:
                return [Decimal(0)] * len(items)
            weight = (radius + head) / (2 * radius)
            return [(radius + head) / 2, *[weight * item for item in tail]]
        a, b, *tail = items
        squared = sum(item * item for item in tail)
        if a >= 0 and b >= 0 and squared <= 2 * a * b:
            return items
        if a <= 0 and b <= 0 and squared <= 2 * a * b:
            return [Decimal(0)] * len(items)
        radius = ((a - b) ** 2 + 2 * squared).sqrt()
        head = a + b
        weight = ((squared - 2 * a * b) / (radius * (radius - head))
                  if head < 0 else (radius + head) / (2 * radius))
        large = weight * (radius + abs(a - b)) / 2
        small = weight * weight * squared / (2 * large)
        heads = [large, small] if a >= b else [small, large]
        return [*heads, *[weight * item for item in tail]]


def assert_coordinates(actual, expected):
    # Relative checks must not hide a lost small head behind the largest input.
    precision = Decimal("2e-12" if actual.dtype == torch.float64 else "3e-5")
    for index, (observed, reference) in enumerate(zip(decimal_values(actual), expected)):
        represented = float(torch.tensor(float(reference), dtype=actual.dtype))
        if reference == 0 or represented == 0:
            assert observed == 0, (index, observed, reference)
        else:
            assert abs(observed - reference) <= precision * abs(reference), (
                index, observed, reference,
            )


def assert_scaled_moreau(value, positive, negative):
    with localcontext() as context:
        context.prec = 1100
        original, p, n = map(decimal_values, (value, positive, negative))
        scale = max(abs(item) for item in original)
        tolerance = Decimal("3e-12" if value.dtype == torch.float64 else "4e-5")
        assert max(abs(x - a + b) / scale for x, a, b in zip(original, p, n)) <= tolerance
        products = [a / scale * (b / scale) for a, b in zip(p, n)]
        assert abs(sum(products)) <= tolerance * max(Decimal(1), sum(abs(x) for x in products))
        for cone_value in (p, n):
            assert cone_value[0] >= 0 and cone_value[1] >= 0
            squared = sum((x / scale) ** 2 for x in cone_value[2:])
            head_product = 2 * (cone_value[0] / scale) * (cone_value[1] / scale)
            assert squared - head_product <= tolerance * max(squared, head_product, Decimal("1e-1000"))


def test_reference_oracle_known_solution_and_lost_head_decoy():
    with localcontext() as context:
        context.prec = 1100
        reference = original_projection([0, 0, 4])
        root = Decimal(2).sqrt()
        assert max(abs(reference[i] - root) for i in (0, 1)) < Decimal("1e-1090")
        assert reference[2] == 2
    with pytest.raises(AssertionError):
        assert_coordinates(torch.tensor([0., 5e119, 5e29], dtype=torch.float64),
                           original_projection([-1e300, 0, 1e210]))


GEOMETRY = [
    [0, 0, 4], [-1, 2, 3], [3, 1, -2], [-2, -3, 1],
    [0, 7, 0], [1, 2, 2], [-1, -1, 2], [1, -3, 2, -4],
    [-1, -1, math.sqrt(2) * (1 + 2 ** -12)],
    [0, 0, 1e30, 1e30], [-1e30, 0, 1e17],
    [1e-30, -2e-30, 3e-30], [1e30, 1e30, 0],
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", GEOMETRY)
def test_geometry_original_projection_and_scaled_moreau(dtype, value):
    original = torch.tensor(value, dtype=dtype)
    before = original.clone()
    cone = (RSOC(len(value)),)
    positive = project_cones(original, cone)
    negative = project_cones(-original, cone, dual=True)
    assert_coordinates(positive, original_projection(original))
    assert_coordinates(negative, original_projection(-original))
    assert_scaled_moreau(original, positive, negative)
    torch.testing.assert_close(original, before, atol=0, rtol=0)
    torch.testing.assert_close(project_cones(positive, cone), positive, atol=0, rtol=3e-6)


@pytest.mark.parametrize("dtype,value", [
    (torch.float64, [-1e300, 1e-100, 0]),
    (torch.float32, [-1e30, 1e-30, 0]),
    (torch.float64, [-1e300, 0, 1e100]),
    (torch.float32, [-1e30, 0, 1e-10]),
    (torch.float64, [-1, -1, math.sqrt(2) * (1 + 2 ** -40)]),
])
def test_risk_representable_outputs_survive_weight_or_head_underflow(dtype, value):
    original = torch.tensor(value, dtype=dtype)
    actual = project_cones(original, (RSOC(len(value)),))
    assert_coordinates(actual, original_projection(original))


@pytest.mark.parametrize("dtype,value", [
    (torch.float64, [0, 0, 1.3e308, 1.3e308]),
    (torch.float64, [-1e300, 0, 1e210]),
    (torch.float64, [1.3e308, 1.3e308, 1.3e308, 1.3e308]),
    (torch.float32, [0, 0, 3e38, 3e38]),
])
def test_geometry_exponent_restoration_and_overflowing_norm(dtype, value):
    original = torch.tensor(value, dtype=dtype)
    cone = (RSOC(len(value)),)
    positive, negative = project_cones(original, cone), project_cones(-original, cone)
    assert_coordinates(positive, original_projection(original))
    assert_coordinates(negative, original_projection(-original))
    assert_scaled_moreau(original, positive, negative)


@pytest.mark.parametrize("dual", [False, True])
def test_risk_original_checker_ignores_only_insignificant_tail_without_underflow_exception(dual):
    value = [1e300, 1e300, 1e300, 1e-100]
    with localcontext() as context:
        context.prec = 1100
        a, b, *tail = decimal_values(value)
        assert 2 * a * b > sum(item * item for item in tail)
    assert check_cone_feasibility(value, (RSOC(4),), dual=dual).passed


@pytest.mark.parametrize("value,expected", [
    ([-1, 1e20, 0], False), ([0, 1e20, 1e8], False),
    ([1e-4, 1e20, 1e8], True), ([1.3e308] * 4, True),
    ([1, 1, 2], False), ([1, 2, 2], True),
])
@pytest.mark.parametrize("dual", [False, True])
def test_candidate_original_inequality_independent_of_rotated_transform(value, expected, dual):
    with localcontext() as context:
        context.prec = 1100
        a, b, *tail = decimal_values(value)
        assert (a >= 0 and b >= 0 and 2 * a * b >= sum(x * x for x in tail)) is expected
    assert check_cone_feasibility(value, (RSOC(len(value)),), dual=dual, atol=1e-12, rtol=0).passed is expected


def kkt_fixture():
    root = math.sqrt(2)
    problem = ConicProblem(
        np.zeros((3, 3)), np.array([1., 1., 0.]),
        np.vstack(([0., 0., 1.], -np.eye(3))), np.array([2., 0., 0., 0.]),
        (Zero(1), RSOC(3)),
    )
    return problem, np.array([root, root, 2.]), np.array([0., root, root, 2.]), np.array([-root, 1., 1., -root])


@pytest.mark.parametrize("mutation,check", [
    ("primal", "original_slack_cone"), ("slack", "primal_equation"),
    ("dual", "dual_cone"), ("stationarity", "stationarity"),
])
def test_candidate_false_success_rejects_primal_slack_and_dual(mutation, check):
    problem, x, s, z = kkt_fixture()
    assert check_conic_candidate(problem, x, s, z, atol=1e-12, rtol=1e-12).passed
    if mutation == "primal":
        x[:2] = 0
    elif mutation == "slack":
        s[1:] = [2, 2, 2]
    elif mutation == "dual":
        z[1:] *= -1
    else:
        z[0] *= -1
    evidence = check_conic_candidate(problem, x, s, z, atol=1e-12, rtol=1e-12)
    assert not evidence.passed and not evidence.checks[check]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", [[3, 1, 2], [-3, 1, 2], [0, 3, 4], [2, -3, 4], [1e30, 2e30, 0]])
def test_soc_default_matches_independent_lorentz_projection(dtype, value):
    original = torch.tensor(value, dtype=dtype)
    assert_coordinates(project_cones(original, (SOC(len(value)),)), original_projection(original, rotated=False))


def main():
    """Bounded batches with exact source inventory and terminal JUnit counts."""
    import argparse
    from datetime import datetime, timezone
    import hashlib
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import time
    import xml.etree.ElementTree as ET

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--group", choices=("risk", "geometry", "candidate", "soc", "full"), default="full")
    args = parser.parse_args()
    target = args.artifact_dir.resolve()
    if target.is_relative_to(ROOT) or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("external artifacts and explicit CUDA_VISIBLE_DEVICES=-1 required")
    target.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=f"{args.group}-", dir=target))
    from scripts.verify import source_snapshot, check_source_consistency
    before = source_snapshot(ROOT)
    started, tick = datetime.now(timezone.utc).isoformat(), time.monotonic()
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-ra", str(Path(__file__).resolve()),
               "-o", f"cache_dir={artifacts / 'cache'}", "--basetemp", str(artifacts / "tmp"),
               f"--junitxml={artifacts / 'pytest.xml'}"]
    if args.group != "full":
        command += ["-k", f"test_{args.group}"]
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TMPDIR=str(artifacts))
    with (artifacts / "pytest.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nreturncode={result.returncode}\n")
    elapsed = time.monotonic() - tick
    after = source_snapshot(ROOT)
    suites = ET.parse(artifacts / "pytest.xml").getroot().findall("testsuite")
    counts = {name: sum(int(s.attrib.get(name, 0)) for s in suites)
              for name in ("tests", "failures", "errors", "skipped")}
    consistency = check_source_consistency(before, after)
    okay = result.returncode == 0 and counts["tests"] > 0 and not any(counts[k] for k in ("failures", "errors", "skipped"))
    okay = okay and consistency["verdict"] == "PASS" and elapsed < 180
    report = dict(schema_version=1, verdict="PASS" if okay else "FAIL", group=args.group,
                  qualification="independent CPU numerical review only; not CUDA qualification",
                  started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=elapsed, budget_seconds=180, command=command,
                  returncode=result.returncode, tests=counts, source=before,
                  source_after=after, source_consistency=consistency,
                  artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (artifacts / "pytest.log", artifacts / "pytest.xml")})
    receipt = artifacts / "receipt.json"
    receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pin = hashlib.sha256(receipt.read_bytes()).hexdigest()
    print(f"RSOC_INDEPENDENT verdict={report['verdict']} counts={counts} elapsed={elapsed:.2f}s receipt={receipt} sha256={pin}", flush=True)
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
