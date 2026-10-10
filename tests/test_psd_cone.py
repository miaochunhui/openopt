"""Actual bounded PSD arithmetic controls; CPU is never CUDA qualification."""

from fractions import Fraction
from itertools import combinations, permutations
import math
import os
from pathlib import Path

import pytest
import torch

from openopt.psd_cone import (
    CHECKER_VERSION, MAX_ORDER, PSDProjectionRejected, check_psd_projection,
    project_psd, smat, svec,
)

DTYPES = [torch.float32, torch.float64]


def tensor(rows, dtype=torch.float64, device="cpu"):
    return torch.tensor(rows, dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["zero", "positive", "negative", "mixed", "rank-one"])
def test_analytic_projection_and_original_moreau(dtype, case):
    values = {
        "zero": ([[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]),
        "positive": ([[2., 1.], [1., 2.]], [[2., 1.], [1., 2.]]),
        "negative": ([[-2., 1.], [1., -2.]], [[0., 0.], [0., 0.]]),
        "mixed": ([[0., 1.], [1., 0.]], [[.5, .5], [.5, .5]]),
        "rank-one": ([[1., -1.], [-1., 1.]], [[1., -1.], [-1., 1.]]),
    }
    original, expected = (tensor(rows, dtype) for rows in values[case])
    before = original.clone()
    result = project_psd(original, device="cpu", dtype=dtype)
    tolerance = 128 * torch.finfo(dtype).eps
    torch.testing.assert_close(result.matrix, expected, atol=0, rtol=tolerance)
    assert result.matrix.data_ptr() != original.data_ptr()
    assert result.matrix.dtype == dtype and result.matrix.device.type == "cpu"
    assert torch.equal(original, before)
    assert result.evidence.passed and all(result.evidence.checks.values())
    assert result.evidence.checker == CHECKER_VERSION
    assert result.evidence.projected_psd["principal_minor_count"] == 3
    opposite = project_psd(-original, device="cpu", dtype=dtype)
    torch.testing.assert_close(original, result.matrix - opposite.matrix,
                               atol=tolerance, rtol=tolerance)
    assert check_psd_projection(original, result.matrix, device="cpu", dtype=dtype).passed


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", range(1, MAX_ORDER + 1))
def test_svec_order_inverse_and_frobenius_metric(dtype, order):
    a = torch.arange(order * order, dtype=dtype).reshape(order, order)
    x = a + a.T
    y = (a + a.T) / 4 + torch.eye(order, dtype=dtype)
    before = x.clone()
    packed = svec(x, device="cpu", dtype=dtype)
    coefficient = Fraction(float(tensor(math.sqrt(2), dtype)))
    expected = tensor([float(Fraction(float(x[i, j])) * (1 if i == j else coefficient))
                       for j in range(order) for i in range(j, order)], dtype)
    torch.testing.assert_close(packed, expected, atol=0, rtol=0)
    torch.testing.assert_close(smat(packed, order, device="cpu", dtype=dtype),
                               x, atol=0, rtol=4 * torch.finfo(dtype).eps)
    torch.testing.assert_close(
        packed @ svec(y, device="cpu", dtype=dtype), (x * y).sum(),
        atol=0, rtol=8 * torch.finfo(dtype).eps,
    )
    assert torch.equal(x, before)


@pytest.mark.parametrize("dtype", DTYPES)
def test_non_diagonal_three_order_analytic_householder_projection(dtype):
    h = [[Fraction(int(i == j)) - Fraction(2, 3) for j in range(3)] for i in range(3)]
    def reconstruct(diagonal):
        return [[float(9 * sum(h[i][k] * diagonal[k] * h[j][k] for k in range(3)))
                 for j in range(3)] for i in range(3)]

    x = tensor(reconstruct([1, -2, 3]), dtype)
    expected = tensor(reconstruct([1, 0, 3]), dtype)
    result = project_psd(x, device="cpu", dtype=dtype)
    torch.testing.assert_close(result.matrix, expected, atol=0,
                               rtol=128 * torch.finfo(dtype).eps)
    assert result.evidence.passed
    assert result.evidence.projected_psd["principal_minor_count"] == 7


@pytest.mark.parametrize("dtype,exponent", [
    (torch.float32, -100), (torch.float32, 100),
    (torch.float64, -600), (torch.float64, 600),
])
def test_homogeneity_near_zero_and_huge_original_scales(dtype, exponent):
    scale = math.ldexp(1., exponent)
    x = tensor([[0., scale], [scale, 0.]], dtype)
    expected = tensor([[scale / 2, scale / 2], [scale / 2, scale / 2]], dtype)
    result = project_psd(x, device="cpu", dtype=dtype)
    torch.testing.assert_close(result.matrix, expected, atol=0,
                               rtol=128 * torch.finfo(dtype).eps)
    assert result.evidence.passed
    assert result.evidence.atol == 0
    assert len(result.evidence.component_ratios) == 2


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_membership_preserves_huge_and_small_coordinates(dtype):
    big, small = ((1e30, 1e-30) if dtype == torch.float32 else (1e300, 1e-300))
    x = torch.diag(tensor([big, small, 0.], dtype))
    result = project_psd(x, device="cpu", dtype=dtype)
    assert torch.equal(result.matrix, x)
    assert result.evidence.projected_psd["exact_psd"]
    assert result.evidence.residual_psd["exact_psd"]


def test_near_maximum_projection_avoids_original_norm_overflow():
    x = tensor([[0., 1.3e308], [1.3e308, 0.]])
    result = project_psd(x, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(result.matrix, torch.full_like(x, 6.5e307),
                               atol=0, rtol=128 * torch.finfo(torch.float64).eps)
    assert result.evidence.passed


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mutation", ["negative-small", "wrong-zero", "wrong-identity", "excess-positive"])
def test_wrong_original_candidate_rejected(dtype, mutation):
    x = torch.diag(tensor([-1., 2.], dtype))
    candidates = {
        "negative-small": torch.diag(tensor([-1e-20, 2.], dtype)),
        "wrong-zero": torch.zeros_like(x),
        "wrong-identity": x,
        "excess-positive": torch.diag(tensor([1., 2.], dtype)),
    }
    before = candidates[mutation].clone()
    evidence = check_psd_projection(x, candidates[mutation], device="cpu", dtype=dtype)
    assert not evidence.passed
    assert torch.equal(candidates[mutation], before)
    assert any(not passed for passed in evidence.checks.values())


def test_all_principal_minors_not_only_leading_and_signed_underflow():
    for x in (torch.diag(tensor([0., -1.])),
              tensor([[1e-300, 2e-300], [2e-300, 1e-300]])):
        evidence = check_psd_projection(x, x.clone(), device="cpu", dtype=torch.float64)
        assert not evidence.passed
        assert -1 in evidence.projected_psd["original_minor_signs"]
        assert not evidence.projected_psd["exact_psd"]


@pytest.mark.parametrize("rows", [
    [[0., 1., -2.], [1., 0., 3.], [-2., 3., -1.]],
    [[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]],
    [[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]],
])
def test_direct_minors_against_independent_leibniz_oracle(rows):
    matrix = tensor(rows)
    evidence = check_psd_projection(matrix, matrix, device="cpu", dtype=torch.float64, rtol=0)
    signs = []
    for size in range(1, 4):
        for subset in combinations(range(3), size):
            determinant = Fraction(0)
            for permutation in permutations(range(size)):
                inversions = sum(permutation[i] > permutation[j]
                                 for i in range(size) for j in range(i + 1, size))
                term = Fraction((-1) ** inversions)
                for i in range(size):
                    term *= Fraction(rows[subset[i]][subset[permutation[i]]])
                determinant += term
            signs.append(int(determinant > 0) - int(determinant < 0))
    assert tuple(signs) == evidence.projected_psd["original_minor_signs"]
    assert len(signs) == 7


def test_large_positive_cannot_hide_small_negative_diagonal():
    x = torch.diag(tensor([1e300, -1.]))
    evidence = check_psd_projection(x, x, device="cpu", dtype=torch.float64)
    assert not evidence.passed
    assert not evidence.projected_psd["numerical_psd"]


def test_rounded_zero_kernel_is_not_exact_psd_certification():
    u = 1 + math.ldexp(1., -52)
    v = 1 + math.ldexp(1., -51)
    x = tensor([[1., u], [u, v]])
    assert Fraction(v) - Fraction(u) ** 2 < 0
    evidence = check_psd_projection(x, x, device="cpu", dtype=torch.float64)
    assert evidence.passed
    assert evidence.projected_psd["numerical_psd"]
    assert not evidence.projected_psd["exact_psd"]
    assert evidence.projected_psd["original_minor_signs"] == (1, 1, -1)
    strict = check_psd_projection(x, x, device="cpu", dtype=torch.float64, rtol=0)
    assert not strict.passed
    assert "not SDP or exact certification" in evidence.qualification


def test_checker_never_reprojects_or_uses_numeric_spectrum(monkeypatch):
    import openopt.psd_cone as module

    def forbidden(*args, **kwargs):
        raise AssertionError("original checker used projection/spectrum/solve")

    x, y = tensor([[0., 1.], [1., 0.]]), tensor([[.5, .5], [.5, .5]])
    for name in ("eigh", "eigvalsh", "eig", "solve", "cholesky", "det"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    monkeypatch.setattr(module, "project_psd", forbidden)
    assert check_psd_projection(x, y, device="cpu", dtype=torch.float64, rtol=0).passed


@pytest.mark.parametrize("mutation", ["zero-spectrum", "negative-input", "nonorthogonal"])
def test_corrupt_spectral_proposal_fails_original_gate(monkeypatch, mutation):
    def altered(x):
        if mutation == "zero-spectrum":
            return torch.zeros(2, dtype=x.dtype), torch.eye(2, dtype=x.dtype)
        if mutation == "negative-input":
            return tensor([1., 1.], x.dtype), torch.eye(2, dtype=x.dtype)
        return tensor([-1., 1.], x.dtype), torch.ones((2, 2), dtype=x.dtype)

    monkeypatch.setattr(torch.linalg, "eigh", altered)
    with pytest.raises(PSDProjectionRejected) as rejected:
        project_psd(tensor([[0., 1.], [1., 0.]]), device="cpu", dtype=torch.float64)
    assert not rejected.value.evidence.passed


def test_spectral_error_is_not_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate native factorization failure")

    monkeypatch.setattr(torch.linalg, "eigh", fail)
    with pytest.raises(RuntimeError, match="deliberate native"):
        project_psd(tensor([[0., 1.], [1., 0.]]), device="cpu", dtype=torch.float64)


def test_normalization_lost_nonzero_coordinate_is_unsupported():
    x = torch.diag(tensor([1e300, -1e300, 1e-300]))
    with pytest.raises(NotImplementedError, match="nonzero original coordinate"):
        project_psd(x, device="cpu", dtype=torch.float64)


@pytest.mark.parametrize("order", range(1, MAX_ORDER + 1))
def test_complete_minor_denominator_at_supported_orders(order):
    x = torch.diag(torch.arange(1, order + 1, dtype=torch.float64))
    evidence = check_psd_projection(x, x, device="cpu", dtype=torch.float64, rtol=0)
    assert evidence.passed
    assert evidence.projected_psd["exact_psd"]
    assert len(evidence.projected_psd["original_minor_signs"]) == (1 << order) - 1
    assert len(evidence.residual_psd["original_minor_signs"]) == (1 << order) - 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", range(2, MAX_ORDER + 1))
def test_native_spectral_projection_at_all_supported_mixed_orders(dtype, order):
    diagonal = torch.arange(1, order + 1, dtype=dtype)
    diagonal[0] = -diagonal[0]
    x = torch.diag(diagonal)
    expected = torch.diag(diagonal.clamp_min(0))
    result = project_psd(x, device="cpu", dtype=dtype)
    assert "torch.linalg.eigh" in result.algorithm
    torch.testing.assert_close(result.matrix, expected, atol=0, rtol=0)
    assert result.evidence.passed


@pytest.mark.parametrize("mutation", [
    "non-square", "batch", "empty", "order", "asymmetric", "nan", "inf",
    "integer", "complex", "half", "sparse", "grad", "not-tensor",
])
def test_strict_shape_finite_real_and_dtype(mutation):
    x = {
        "non-square": torch.zeros((2, 3), dtype=torch.float64),
        "batch": torch.zeros((1, 2, 2), dtype=torch.float64),
        "empty": torch.zeros((0, 0), dtype=torch.float64),
        "order": torch.eye(MAX_ORDER + 1, dtype=torch.float64),
        "asymmetric": tensor([[1., 1.], [0., 1.]]),
        "nan": tensor([[float("nan")]]),
        "inf": tensor([[float("inf")]]),
        "integer": torch.eye(2, dtype=torch.int64),
        "complex": torch.eye(2, dtype=torch.complex128),
        "half": torch.eye(2, dtype=torch.float16),
        "sparse": torch.eye(2, dtype=torch.float64).to_sparse(),
        "grad": torch.eye(2, dtype=torch.float64, requires_grad=True),
        "not-tensor": [[1.]],
    }[mutation]
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        project_psd(x, device="cpu", dtype=torch.float64)


@pytest.mark.parametrize("device", ["cuda", "cpu:0", "cuda:-1", "cuda:01", "mps", None])
def test_explicit_device_spelling(device):
    with pytest.raises(ValueError, match="explicit"):
        project_psd(tensor([[1.]]), device=device, dtype=torch.float64)


def test_explicit_cuda_unavailable_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        project_psd(tensor([[1.]]), device="cuda:0", dtype=torch.float64)


@pytest.mark.parametrize("name,value", [
    ("atol", -1), ("atol", True), ("rtol", 1), ("rtol", -1), ("rtol", float("nan")),
])
def test_invalid_tolerance(name, value):
    with pytest.raises(ValueError):
        project_psd(tensor([[1.]]), device="cpu", dtype=torch.float64, **{name: value})


def test_candidate_shape_and_svec_length_and_order():
    with pytest.raises(ValueError, match="shape"):
        check_psd_projection(tensor([[1.]]), torch.eye(2, dtype=torch.float64),
                             device="cpu", dtype=torch.float64)
    for order in (True, 0, MAX_ORDER + 1):
        with pytest.raises(ValueError, match="order"):
            smat(tensor([1.]), order, device="cpu", dtype=torch.float64)
    with pytest.raises(ValueError, match="length"):
        smat(tensor([1., 2.]), 2, device="cpu", dtype=torch.float64)
    with pytest.raises(ValueError, match="device/dtype"):
        svec(tensor([[1.]], torch.float32), device="cpu", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="svec overflow"):
        svec(tensor([[0., 1.3e308], [1.3e308, 0.]]), device="cpu", dtype=torch.float64)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(os.environ.get("OPENOPT_RUN_PSD_CUDA_TESTS") != "1",
                    reason="CUDA opt-in absent; local CPU is not GPU qualification")
def test_cuda_actual_projection_placement_and_original_check(dtype):
    if not torch.cuda.is_available():
        pytest.fail("explicit PSD CUDA target has no CUDA runtime; no fallback/qualification")
    x = tensor([[0., 1.], [1., 0.]], dtype, "cuda:0")
    result = project_psd(x, device="cuda:0", dtype=dtype)
    assert result.matrix.device == x.device and result.matrix.dtype == dtype
    assert result.evidence.passed
    assert result.evidence.device == "cuda:0"
    with pytest.raises(ValueError, match="device/dtype"):
        project_psd(x.cpu(), device="cuda:0", dtype=dtype)


def test_explicit_cuda_test_without_runtime_is_failure_not_skip(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(pytest.fail.Exception, match="no CUDA runtime"):
        test_cuda_actual_projection_placement_and_original_check(torch.float64)


def main(argv=None):
    """Explicit safe scan/review CLI; all logs external, no GPU submission."""
    import argparse
    from datetime import datetime, timezone
    import hashlib
    import importlib.util
    import json
    import os
    import re
    import subprocess
    import sys
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    owned = ("src/openopt/psd_cone.py", "tests/test_psd_cone.py", "docs/research/psd-cone.md")
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args(argv)
    out = args.artifact_dir.resolve()
    if not out.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve()) or out.is_relative_to(root):
        raise ValueError("external shared artifact directory required")
    out.mkdir(parents=True, exist_ok=False)
    from scripts.verify import check_source_consistency, source_snapshot

    before = source_snapshot(root)
    receipt = {"schema_version": 1, "source": before, "owned": list(owned),
               "started_at": datetime.now(timezone.utc).isoformat(),
               "qualification": "CPU bounded primitive only; CUDA/SDP/commercial UNVERIFIED"}
    rc = 1
    try:
        if args.scan_only:
            spec = importlib.util.spec_from_file_location(
                "psd_existing_scan", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
            scanner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scanner)
            needles = scanner.read_needles(SimpleNamespace(
                secret_hex=None, secrets_file=None, no_needle=False,
                needle_from=str(Path.home() / ".config/dsh-n600-token")))
            if not needles:
                raise ValueError("existing literal scan material missing; no scan acceptance")
            count, _ = scanner.scan_literal([str(root / name) for name in owned], needles)
            shape = re.compile("IDENTIFIED" + r""" BY|password["']?\s*[:=]\s*["'][^"']{6,}""",
                               re.IGNORECASE)
            shape_paths = [name for name in owned if shape.search((root / name).read_text())]
            if count or shape_paths:
                raise ValueError("scan hits; values withheld")
            canary = out / "synthetic-scan-control.txt"
            anchor = "psd-sensitivity-" + "not-a-credential"
            canary.write_text(anchor)
            seen, _ = scanner.scan_literal([str(canary)], [anchor])
            canary.write_text("cleared synthetic sensitivity control\n")
            cleared, _ = scanner.scan_literal([str(canary)], [anchor])
            if seen != 1 or cleared != 0:
                raise ValueError("literal scanner failed positive/negative sensitivity control")
            receipt["scan"] = {"literal_hits": count, "shape_hit_paths": shape_paths,
                               "values_disclosed": False,
                               "sensitivity_seen": seen, "sensitivity_cleared": cleared,
                               "scan_tool_sha256": hashlib.sha256(
                                   Path(spec.origin).read_bytes()).hexdigest()}
        else:
            command = [sys.executable, "-m", "pytest", "-q", "-ra",
                       "-o", f"cache_dir={out / 'cache'}", "--basetemp", str(out / "tmp"),
                       f"--junitxml={out / 'pytest.xml'}", str(root / owned[1])]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=str(root / "src") + os.pathsep + str(root))
            env.pop("OPENOPT_RUN_PSD_CUDA_TESTS", None)
            receipt["command"] = command
            receipt["runtime"] = {
                "python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "dtype_selection": [str(dtype) for dtype in DTYPES],
                "environment": {name: env[name] for name in (
                    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH")},
                "gpu_qualification": "UNVERIFIED", "OPENOPT_RUN_PSD_CUDA_TESTS": "unset",
            }
            collection_command = [sys.executable, "-m", "pytest", "--collect-only", "-q",
                                  "-o", f"cache_dir={out / 'collection-cache'}", str(root / owned[1])]
            collected = subprocess.run(collection_command, cwd=root, env=env,
                                       capture_output=True, timeout=30, check=False)
            (out / "collection.log").write_bytes(collected.stdout + collected.stderr)
            if collected.returncode:
                raise RuntimeError("actual PSD collection failed")
            expected = [line.split("::", 1)[1] for line in collected.stdout.decode().splitlines()
                        if line.startswith("tests/test_psd_cone.py::")]
            if not expected or len(expected) != len(set(expected)):
                raise ValueError("empty/duplicate actual collection")
            process = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                     timeout=170, check=False)
            (out / "pytest.log").write_bytes(process.stdout + process.stderr)
            receipt["pytest_exit_code"] = process.returncode
            if process.returncode:
                raise RuntimeError("actual PSD CPU tests failed; see pytest.log")
            from xml.etree import ElementTree

            suites = ElementTree.parse(out / "pytest.xml").getroot()
            receipt["counts"] = {key: sum(int(suite.attrib[key]) for suite in suites.findall("testsuite"))
                                 for key in ("tests", "failures", "errors", "skipped")}
            receipt["membership"] = [node.attrib["name"] for node in suites.iter("testcase")]
            if not receipt["counts"]["tests"] or len(receipt["membership"]) != receipt["counts"]["tests"]:
                raise ValueError("empty/incomplete actual JUnit selection")
            if len(receipt["membership"]) != len(set(receipt["membership"])) or set(
                    receipt["membership"]) != set(expected):
                raise ValueError("JUnit differs from exact actual collection")
            if receipt["counts"]["failures"] or receipt["counts"]["errors"]:
                raise ValueError("JUnit retains failed tests")
            skips = [node.attrib["name"] for node in suites.iter("testcase")
                     if node.find("skipped") is not None]
            if sorted(skips) != sorted([
                    "test_cuda_actual_projection_placement_and_original_check[dtype0]",
                    "test_cuda_actual_projection_placement_and_original_check[dtype1]"]):
                raise ValueError("unexpected skip membership, not CPU acceptance")
            receipt["collection_membership"] = expected
            receipt["skip_membership"] = skips
            receipt["exact_membership"] = True
        receipt["owned_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                   for name in owned}
        receipt["source_consistency"] = check_source_consistency(before, source_snapshot(root))
        if receipt["source_consistency"]["verdict"] != "PASS":
            raise RuntimeError("source changed during CPU review")
        receipt["verdict"], rc = "PASS", 0
    except Exception as error:
        receipt.update(verdict="FAIL", first_cause={"type": type(error).__name__, "message": str(error)})
    receipt.update(exit_code=rc, finished_at=datetime.now(timezone.utc).isoformat())
    path = out / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    line = f"PSD_PRIMITIVE verdict={receipt['verdict']} qualification=UNVERIFIED receipt={path}"
    (out / "review.log").write_text(line + f"\nexit_code={rc}\n")
    print(line)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
