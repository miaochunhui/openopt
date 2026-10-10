"""Independent bounded PSD review; CPU controls are not actual CUDA evidence.

Only this test and its companion document are writable. The CLI creates fresh
external tempfile evidence, never reads a historical GPU archive, and does not
alter source, permissions, dependencies or cluster state.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import importlib.util
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET

import pytest
import torch

import openopt.psd_cone as production

ROOT = Path(__file__).resolve().parents[1]
OWNER = Path("/data/dev/openopt-wt-release4-20261010-native-psd")
OWNER_COMMIT = "2278294820a133c23d4f3fa632bd51ec24789463"
BASE = "73613fce9faafdd622baf69dcf1ecb64b4cfa6f7"
PINS = {
    "src/openopt/psd_cone.py": "1500ffe3c2348e4d3c93c4ad3b41452cd16096083f31c22e5a7735205a0ee6e0",
    "tests/test_psd_cone.py": "929288faef0b6f70c3f120e26ad686cc601428b17af8b305ddd77ec1febc0358",
    "docs/research/psd-cone.md": "1307ec311bf62aa26da0dc5aca895337f294482d71ec2f6b9f66a57e4c1c69e8",
}
DOC = "docs/research/psd-cone-independent.md"
EXTERNAL = Path("/hf3fs-jd/data/tmp/openopt-release-20261010")
DTYPES = [torch.float32, torch.float64]
CHECKS = (
    "projected_psd_with_local_tolerance", "moreau_residual_psd_with_local_tolerance",
    "moreau_component_orthogonality", "moreau_trace_orthogonality",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                            timeout=15, check=False)
    assert result.returncode == 0, f"Git query failed rc={result.returncode}"
    return result.stdout


def source_state():
    from scripts.verify import source_snapshot
    for name, digest in PINS.items():
        expected = git(OWNER, "show", OWNER_COMMIT + ":" + name)
        assert hashlib.sha256(expected).hexdigest() == digest
        assert (ROOT / name).read_bytes() == (OWNER / name).read_bytes() == expected
    assert Path(production.__file__).resolve() == ROOT / "src/openopt/psd_cone.py"
    owner, reviewer = source_snapshot(OWNER), source_snapshot(ROOT)
    assert owner["clean"] and owner["commit"] == OWNER_COMMIT, "owner source drift"
    return {"owner": owner, "reviewer": reviewer}


def save(path, value):
    path = Path(path)
    assert path.is_absolute() and ".." not in path.parts
    assert path.resolve().is_relative_to(EXTERNAL.resolve()), "external artifacts only"
    assert not any(path.resolve().is_relative_to(root) for root in (ROOT, OWNER))
    for item in (path, *path.parents):
        if item == EXTERNAL or item == EXTERNAL.resolve():
            break
        assert not item.is_symlink(), "output symlink forbidden"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


@pytest.fixture(autouse=True)
def source_and_observation(request, tmp_path):
    before = source_state()
    assert tmp_path.resolve().is_relative_to(EXTERNAL.resolve())
    yield
    assert source_state() == before, "test modified pinned sources"
    observation = tmp_path / "observation.json"
    if not observation.exists():
        save(observation, {"nodeid": request.node.nodeid, "source_unchanged": True})
    request.node.user_properties.extend([
        ("independent_observation_path", str(observation)),
        ("independent_observation_sha256", sha(observation)),
    ])


def rational(matrix):
    """Preserve represented coordinates; no production checker helpers."""
    return [[F.from_float(float(value)) for value in row]
            for row in matrix.detach().cpu().tolist()]


def determinant(matrix):
    """Signed permutation sum via subset DP, independent of pivot elimination."""
    n = len(matrix)
    partial = {0: F(1)}
    for row in range(n):
        next_row = {}
        for mask, value in partial.items():
            for col in range(n):
                if mask & (1 << col):
                    continue
                inversions = (mask >> (col + 1)).bit_count()
                term = value * matrix[row][col] * (-1 if inversions % 2 else 1)
                target = mask | (1 << col)
                next_row[target] = next_row.get(target, F(0)) + term
        partial = next_row
    return partial[(1 << n) - 1]


def minors(matrix, shifts=None):
    n = len(matrix)
    shifts = [F(0)] * n if shifts is None else shifts
    records = []
    for size in range(1, n + 1):
        for indices in combinations(range(n), size):
            value = determinant([[matrix[i][j] + (shifts[i] if i == j else 0)
                                  for j in indices] for i in indices])
            records.append((indices, value))
    assert len(records) == (1 << n) - 1
    return records


def sign(value):
    return int(value > 0) - int(value < 0)


def multiply(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(len(a))), F(0))
             for j in range(len(a))] for i in range(len(a))]


def independent(original, candidate, atol=0, rtol=None):
    x, y = rational(original), rational(candidate)
    n = len(x)
    a = F(float(atol))
    r = F(float(128 * torch.finfo(original.dtype).eps if rtol is None else rtol))
    g = [[y[i][j] - x[i][j] for j in range(n)] for i in range(n)]
    details = []
    for matrix in (y, g):
        exact = minors(matrix)
        shifted = minors(matrix, [a + r * abs(matrix[i][i]) for i in range(n)])
        details.append({
            "original_minor_signs": tuple(sign(value) for _, value in exact),
            "shifted_minor_signs": tuple(sign(value) for _, value in shifted),
            "exact_psd": all(value >= 0 for _, value in exact),
            "numerical_psd": all(value >= 0 for _, value in shifted),
            "principal_minor_count": (1 << n) - 1,
        })
    product = multiply(y, g)
    limits = [[r * sum((abs(y[i][k] * g[k][j]) + abs(y[i][k] * x[k][j]) +
                       abs(x[i][k] * g[k][j]) for k in range(n)), F(0))
               + a * sum((abs(y[i][k]) + abs(g[k][j]) for k in range(n)), F(0)) + n * a * a
               for j in range(n)] for i in range(n)]
    trace = abs(sum((product[i][i] for i in range(n)), F(0)))
    checks = dict(zip(CHECKS, [
        details[0]["numerical_psd"], details[1]["numerical_psd"],
        all(abs(product[i][j]) <= limits[i][j] for i in range(n) for j in range(n)),
        trace <= sum((limits[i][i] for i in range(n)), F(0)),
    ]))
    return {"checks": checks, "passed": all(checks.values()), "projected": details[0],
            "residual": details[1], "product": product, "limits": limits,
            "trace": trace, "trace_limit": sum((limits[i][i] for i in range(n)), F(0)),
            "atol": a, "rtol": r}


def display_ratio(value, limit):
    if value == 0:
        return "0x0.0p+0"
    if limit == 0:
        return "unbounded: positive residual with zero tolerance"
    try:
        return float(value / limit).hex()
    except OverflowError:
        return "above binary64 display range"


def crosscheck(x, y, atol=0, rtol=None):
    oracle = independent(x, y, atol, rtol)
    before_x, before_y = x.clone(), y.clone()
    evidence = production.check_psd_projection(x, y, device="cpu", dtype=x.dtype,
                                              atol=atol, rtol=rtol)
    assert torch.equal(x, before_x) and torch.equal(y, before_y)
    assert evidence.passed is oracle["passed"] and evidence.checks == oracle["checks"]
    assert evidence.projected_psd == oracle["projected"]
    assert evidence.residual_psd == oracle["residual"]
    assert evidence.atol == float(oracle["atol"]) and evidence.rtol == float(oracle["rtol"])
    expected_ratios = tuple(tuple(display_ratio(abs(p), limit)
                                 for p, limit in zip(row, limits))
                            for row, limits in zip(oracle["product"], oracle["limits"]))
    assert evidence.component_ratios == expected_ratios
    assert evidence.trace_ratio == display_ratio(oracle["trace"], oracle["trace_limit"])
    assert evidence.order == len(x) and evidence.device == "cpu" and evidence.dtype == str(x.dtype)
    assert evidence.verification_device.startswith("cpu:")
    assert "not SDP or exact certification" in evidence.qualification
    return oracle


def tensor(rows, dtype=torch.float64):
    return torch.tensor(rows, dtype=dtype, device="cpu")


def householder_pair(order, dtype):
    h = [[F(int(i == j)) - F(2, order) for j in range(order)] for i in range(order)]
    diagonal = [(-1 if k % 2 == 0 else 1) * (k + 1) for k in range(order)]
    def form(values):
        rows = [[order * order * sum((h[i][k] * values[k] * h[j][k]
                                     for k in range(order)), F(0))
                 for j in range(order)] for i in range(order)]
        assert all(value.denominator == 1 for row in rows for value in row)
        return tensor([[float(value) for value in row] for row in rows], dtype)
    return form(diagonal), form([max(value, 0) for value in diagonal])


@pytest.mark.parametrize("order", range(1, 7))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("family", ["dense-mixed", "rank-one", "nonleading", "zero"])
def test_all_principal_minors_exact_orders_1_to_6(order, dtype, family, tmp_path):
    if family == "dense-mixed":
        x, _ = householder_pair(order, dtype)
    elif family == "rank-one":
        vector = [(-1) ** i * (i + 1) for i in range(order)]
        x = tensor([[a * b for b in vector] for a in vector], dtype)
    elif family == "nonleading":
        x = torch.diag(tensor([0] * (order - 1) + [-1], dtype))
    else:
        x = torch.zeros((order, order), dtype=dtype)
    oracle = crosscheck(x, x.clone(), rtol=0)
    if family == "nonleading":
        assert oracle["passed"] is False
        assert -1 in oracle["projected"]["original_minor_signs"]
    save(tmp_path / "observation.json", {"order": order, "family": family,
                                        "principal_minors_per_matrix": (1 << order) - 1,
                                        "all_signs": oracle["projected"]["original_minor_signs"]})


@pytest.mark.parametrize("order", range(1, 7))
@pytest.mark.parametrize("dtype", DTYPES)
def test_analytic_moreau_exact_original_coordinates(order, dtype, tmp_path):
    x, expected = householder_pair(order, dtype)
    oracle = crosscheck(x, expected, rtol=0)
    assert oracle["passed"] and oracle["trace"] == 0
    assert all(value == 0 for row in oracle["product"] for value in row)
    assert oracle["projected"]["exact_psd"] and oracle["residual"]["exact_psd"]
    save(tmp_path / "observation.json", {"order": order, "exact_moreau": True,
                                        "principal_minors_per_matrix": (1 << order) - 1})


@pytest.mark.parametrize("order", range(1, 7))
@pytest.mark.parametrize("dtype", DTYPES)
def test_actual_native_projection_original_moreau(order, dtype, tmp_path):
    x, expected = householder_pair(order, dtype)
    before = x.clone()
    result = production.project_psd(x, device="cpu", dtype=dtype)
    oracle = crosscheck(x, result.matrix)
    assert oracle["passed"]
    assert result.evidence == production.check_psd_projection(
        x, result.matrix, device="cpu", dtype=dtype)
    actual = rational(result.matrix)
    e = rational(expected)
    relative = F(128 * torch.finfo(dtype).eps)
    assert all(abs(actual[i][j] - e[i][j]) <= relative * sum(map(abs, e[i]), F(0))
               for i in range(order) for j in range(order))
    assert torch.equal(x, before) and result.matrix.data_ptr() != x.data_ptr()
    assert result.matrix.device.type == "cpu" and result.matrix.dtype == dtype
    if order > 1:
        assert "torch.linalg.eigh" in result.algorithm
    save(tmp_path / "observation.json", {"order": order, "actual_native_cpu": True,
                                        "all_original_checks": oracle["checks"],
                                        "exact_projected_psd": oracle["projected"]["exact_psd"],
                                        "gpu_execution": "UNVERIFIED"})


@pytest.mark.parametrize("mutation", [
    "wrong-zero", "wrong-identity", "extra-positive", "wrong-residual-sign",
    "nonleading-negative", "component-not-trace", "huge-hides-small",
])
def test_wrong_candidates_exactly_rejected(mutation, tmp_path):
    x = torch.diag(tensor([-1., 2.]))
    y = {
        "wrong-zero": torch.zeros_like(x),
        "wrong-identity": x.clone(),
        "extra-positive": torch.diag(tensor([1., 2.])),
        "wrong-residual-sign": torch.diag(tensor([0., 1.])),
        "nonleading-negative": torch.diag(tensor([0., -1.])),
        "component-not-trace": tensor([[1., 1.], [1., 1.]]),
        "huge-hides-small": torch.diag(tensor([1e300, -1.])),
    }[mutation]
    if mutation == "component-not-trace":
        x = tensor([[1., -1.], [-1., 1.]])
        y = torch.eye(2, dtype=torch.float64)
    elif mutation == "huge-hides-small":
        x = y.clone()
    oracle = crosscheck(x, y)
    assert not oracle["passed"], "FALSE_ACCEPT original-coordinate candidate"
    if mutation == "component-not-trace":
        assert oracle["checks"]["moreau_trace_orthogonality"] is True
        assert oracle["checks"]["moreau_component_orthogonality"] is False
    save(tmp_path / "observation.json", {"mutation": mutation, "rejected": True,
                                        "failed": [name for name, value in oracle["checks"].items()
                                                   if not value]})


@pytest.mark.parametrize("exponent", [-1000, -600, 600, 1023])
def test_extreme_signed_products_and_minors_no_float_loss(exponent, tmp_path):
    scale = math.ldexp(1., exponent)
    x = tensor([[0, scale], [scale, 0]])
    y = tensor([[scale / 2, scale / 2], [scale / 2, scale / 2]])
    oracle = crosscheck(x, y, rtol=0)
    assert oracle["passed"] and oracle["trace"] == 0
    original_minors = minors(rational(x))
    assert original_minors[-1][1] == -F.from_float(scale) ** 2 < 0
    save(tmp_path / "observation.json", {"exponent": exponent, "exact_moreau": True,
                                        "original_negative_determinant": True})


@pytest.mark.parametrize("atol,rtol", [(0, 0), (0, 2 ** -40), (2 ** -30, 0), (2 ** -30, 2 ** -40)])
def test_local_tolerance_boundary_full_evidence(atol, rtol):
    x = tensor([[-1., 0], [0, 2.]])
    y = tensor([[2 ** -32, 0], [0, 2.]])
    crosscheck(x, y, atol=atol, rtol=rtol)


def test_rounded_negative_minor_never_exact_certification(tmp_path):
    u, v = 1 + 2 ** -52, 1 + 2 ** -51
    x = tensor([[1, u], [u, v]])
    oracle = crosscheck(x, x.clone())
    assert determinant(rational(x)) == -F(1, 2 ** 104)
    assert oracle["passed"] and oracle["projected"]["exact_psd"] is False
    assert crosscheck(x, x.clone(), rtol=0)["passed"] is False
    save(tmp_path / "observation.json", {"exact_psd": False, "numerical_only": True})


def test_checker_independent_from_all_spectral_oracles(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("checker used numerical spectrum or producer")
    for name in ("eigh", "eigvalsh", "eig", "det", "solve", "cholesky", "svd"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    monkeypatch.setattr(production, "project_psd", forbidden)
    x, y = householder_pair(6, torch.float64)
    assert crosscheck(x, y, rtol=0)["passed"]


@pytest.mark.parametrize("mutation", ["zero", "fake-orthogonal", "nan", "shape", "native-error"])
def test_actual_producer_rejects_bad_spectrum(monkeypatch, mutation, tmp_path):
    def changed(matrix):
        if mutation == "native-error":
            raise RuntimeError("independent native failure")
        if mutation == "zero":
            return tensor([0., 0.]), torch.eye(2, dtype=torch.float64)
        if mutation == "fake-orthogonal":
            return tensor([-1., 1.]), torch.ones((2, 2), dtype=torch.float64)
        if mutation == "nan":
            return tensor([float("nan"), 1.]), torch.eye(2, dtype=torch.float64)
        return tensor([1.]), torch.eye(2, dtype=torch.float64)
    monkeypatch.setattr(torch.linalg, "eigh", changed)
    x = tensor([[0., 1.], [1., 0.]])
    before = x.clone()
    with pytest.raises((production.PSDProjectionRejected, RuntimeError, ValueError)):
        production.project_psd(x, device="cpu", dtype=torch.float64)
    assert torch.equal(x, before)
    save(tmp_path / "observation.json", {"mutation": mutation, "rejected": True})


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", range(1, 7))
def test_svec_column_order_frobenius_not_roundtrip_oracle(dtype, order):
    x = tensor([[i + j + 1 for j in range(order)] for i in range(order)], dtype)
    packed = production.svec(x, device="cpu", dtype=dtype)
    coefficient = F.from_float(float(tensor(math.sqrt(2), dtype)))
    expected = tensor([float(F.from_float(float(x[i, j])) *
                            (1 if i == j else coefficient))
                       for j in range(order) for i in range(j, order)], dtype)
    assert torch.equal(packed, expected)
    restored = production.smat(packed, order, device="cpu", dtype=dtype)
    xr, sr, pr = rational(x), rational(restored), [F.from_float(float(value)) for value in packed]
    eps = F.from_float(torch.finfo(dtype).eps)
    assert all(abs(xr[i][j] - sr[i][j]) <= 4 * eps * abs(xr[i][j])
               for i in range(order) for j in range(order))
    exact_norm = sum((value * value for row in xr for value in row), F(0))
    packed_norm = sum((value * value for value in pr), F(0))
    assert abs(packed_norm - exact_norm) <= 8 * eps * exact_norm


@pytest.mark.parametrize("device", [None, "cuda", "cpu:0", "cuda:01", "cuda:-1", "mps"])
def test_bad_explicit_device_rejected(device):
    with pytest.raises(ValueError):
        production.project_psd(tensor([[1]]), device=device, dtype=torch.float64)


def test_explicit_cuda_unavailable_is_failure_not_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for entry, args in ((production.project_psd, (tensor([[1]]),)),
                        (production.svec, (tensor([[1]]),)),
                        (production.smat, (tensor([1]), 1)),
                        (production.check_psd_projection, (tensor([[1]]), tensor([[1]])))):
        with pytest.raises(RuntimeError, match="no CPU fallback"):
            entry(*args, device="cuda:0", dtype=torch.float64)


@pytest.mark.parametrize("name,value", [
    ("atol", True), ("atol", -1), ("atol", float("inf")),
    ("rtol", False), ("rtol", -1), ("rtol", 1), ("rtol", float("nan")),
])
def test_bool_nonfinite_invalid_tolerance_rejected(name, value):
    with pytest.raises(ValueError):
        production.check_psd_projection(tensor([[1]]), tensor([[1]]),
                                        device="cpu", dtype=torch.float64, **{name: value})


@pytest.mark.parametrize("invalid", ["asymmetry", "batch", "order7", "bool", "grad", "nonfinite"])
def test_original_coordinates_never_silently_normalized(invalid):
    matrix = {
        "asymmetry": tensor([[1., 1.], [0., 1.]]),
        "batch": torch.zeros((1, 2, 2), dtype=torch.float64),
        "order7": torch.eye(7, dtype=torch.float64),
        "bool": torch.eye(2, dtype=torch.bool),
        "grad": torch.eye(2, dtype=torch.float64, requires_grad=True),
        "nonfinite": tensor([[float("inf")]]),
    }[invalid]
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        production.project_psd(matrix, device="cpu", dtype=torch.float64)


def test_unrepresentable_normalization_refuses_not_clip():
    matrix = torch.diag(tensor([1e300, -1e300, 1e-300]))
    with pytest.raises(NotImplementedError, match="nonzero original coordinate"):
        production.project_psd(matrix, device="cpu", dtype=torch.float64)


def test_source_default_api_and_original_owner_tests_unchanged(tmp_path):
    names = git(ROOT, "ls-tree", "-r", "--name-only", BASE, "--", "src").decode().splitlines()
    assert names
    for name in names:
        assert (ROOT / name).read_bytes() == git(ROOT, "show", BASE + ":" + name)
    import openopt
    assert not hasattr(openopt, "project_psd")
    assert not hasattr(openopt, "svec")
    save(tmp_path / "observation.json", {"preexisting_source_modules_unchanged": len(names),
                                        "public_default_api_unchanged": True,
                                        "owner_commit": OWNER_COMMIT, "owner_sha256": PINS})


def junit(path):
    document = ET.parse(path).getroot()
    assert document.tag == "testsuites" and len(document) == 1
    suite = document[0]
    cases = suite.findall("testcase")
    target = "tests/" + Path(__file__).name
    nodes = [target + "::" + case.get("name") for case in cases]
    counts = {name: int(suite.get(name)) for name in ("tests", "failures", "errors", "skipped")}
    counts["passed"] = counts["tests"] - sum(counts[name] for name in ("failures", "errors", "skipped"))
    assert counts["tests"] == len(cases) and counts["tests"] > 0
    assert counts["failures"] == sum(case.find("failure") is not None for case in cases)
    assert counts["errors"] == sum(case.find("error") is not None for case in cases)
    assert counts["skipped"] == sum(case.find("skipped") is not None for case in cases)
    failures, observed = [], {}
    for node, case in zip(nodes, cases):
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            failures.append({"nodeid": node, "reason": problem.get("message"), "trace": problem.text})
        props = case.findall("properties")
        if props:
            assert len(props) == 1 and len(props[0]) == 2
            items = {item.get("name"): item.get("value") for item in props[0]}
            assert set(items) == {"independent_observation_path", "independent_observation_sha256"}
            observation = Path(items["independent_observation_path"])
            assert observation.resolve().is_relative_to(path.parent.resolve() / "tmp")
            assert not observation.is_symlink() and sha(observation) == items["independent_observation_sha256"]
            observed[node] = {"path": str(observation), "sha256": sha(observation),
                              "value": json.loads(observation.read_bytes())}
        elif problem is None and case.find("skipped") is None:
            raise AssertionError(f"passing node lacks observation: {node}")
    if counts["passed"] == counts["tests"]:
        assert set(observed) == set(nodes), "complete observation node membership"
        observed_paths = [Path(item["path"]).resolve() for item in observed.values()]
        assert len(observed_paths) == len(set(observed_paths))
        assert set(observed_paths) == {
            item.resolve() for item in (path.parent / "tmp").rglob("observation.json")}
        exact = [item["value"] for node, item in observed.items()
                 if "::test_analytic_moreau_exact_original_coordinates[" in node]
        assert len(exact) == 12 and {item["order"] for item in exact} == set(range(1, 7))
        assert all(item["exact_moreau"] is True and item["principal_minors_per_matrix"] ==
                   (1 << item["order"]) - 1 for item in exact)
    return nodes, counts, failures, observed, float(suite.get("time"))


def inventory(output):
    return {path.relative_to(output).as_posix(): sha(path)
            for path in output.rglob("*") if path.is_file() and not path.is_symlink()
            and path.name != "receipt.json"}


def run(argv=None):
    """Fresh external CPU audit or caller-pinned read-only terminal check."""
    parser = argparse.ArgumentParser(description=run.__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=EXTERNAL)
    parser.add_argument("--check-receipt", type=Path)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_receipt:
        assert args.receipt_sha256 and sha(args.check_receipt) == args.receipt_sha256
        receipt = json.loads(args.check_receipt.read_bytes())
        output = args.check_receipt.parent
        assert receipt["source"] == receipt["source_after"] == source_state()
        assert all(item["clean"] for item in receipt["source"].values())
        assert all(item["verdict"] == "PASS" for item in receipt["source_consistency"].values())
        assert type(receipt["gpu_jobs_submitted"]) is int and receipt["gpu_jobs_submitted"] == 0
        assert receipt["CUDA"] == receipt["SDP"] == receipt["commercial"] == "UNVERIFIED"
        assert inventory(output) == receipt["artifacts"]
        nodes, counts, failures, observed, _ = junit(output / "pytest.xml")
        assert sorted(nodes) == receipt["collection"] == receipt["junit_membership"]
        assert counts == receipt["counts"] and failures == receipt["failures"]
        assert observed == receipt["observations"]
        assert not receipt["duplicates"] and not receipt["missing"] and not receipt["extra"]
        assert receipt["verdict"] in ("PASS", "FAIL")
        if receipt["verdict"] == "PASS":
            assert receipt["returncode"] == receipt["collection_returncode"] == 0
            assert counts["passed"] == len(nodes) and counts["skipped"] == 0
        else:
            assert receipt["first_failure"]
        print(f"PSD_INDEPENDENT integrity=PASS acceptance={receipt['verdict']} "
              f"counts={counts} CUDA=UNVERIFIED receipt={args.check_receipt}")
        return 0 if receipt["verdict"] == "PASS" else 1
    assert args.artifact_dir.resolve() == EXTERNAL.resolve(), "existing external parent only"
    output = Path(tempfile.mkdtemp(prefix="psd-independent-", dir=args.artifact_dir))
    before = source_state()
    receipt = {"schema_version": 1, "verdict": "FAIL", "source": before,
               "started_at": datetime.now(timezone.utc).isoformat(), "first_failure": None,
               "qualification": "bounded CPU matrix projection review only",
               "gpu_jobs_submitted": 0, "CUDA": "UNVERIFIED", "SDP": "UNVERIFIED",
               "commercial": "UNVERIFIED", "performance": "UNVERIFIED"}
    try:
        if args.scan_only:
            spec = importlib.util.spec_from_file_location(
                "psd_independent_scan", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
            scanner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scanner)
            needles = scanner.read_needles(argparse.Namespace(
                secret_hex=None, secrets_file=None, no_needle=False,
                needle_from=str(Path.home() / ".config/dsh-n600-token")))
            assert needles, "nonempty in-memory credential scan required"
            paths = [str(Path(__file__).resolve()), str(ROOT / DOC)]
            literal, _ = scanner.scan_literal(paths, needles)
            pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                                 r"""(?i:password|token|secret)\s*["']?\s*[:=]\s*["'][^"'\s]{6,}["']""")
            shape = sum(bool(pattern.search(Path(path).read_text())) for path in paths)
            assert literal == shape == 0
            receipt.update(scan={"literal_hits": literal, "shape_hits": shape, "paths": paths},
                           verdict="PASS")
        else:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TMPDIR=str(output),
                       PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT))
            env.pop("OPENOPT_RUN_PSD_CUDA_TESTS", None)
            target = "tests/" + Path(__file__).name
            collection_command = [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q",
                                  target, "-o", f"cache_dir={output / 'cache'}"]
            with (output / "collection.log").open("x") as log:
                collected = subprocess.run(collection_command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=30, check=False)
            nodes = [line for line in (output / "collection.log").read_text().splitlines()
                     if line.startswith(target + "::")]
            assert collected.returncode == 0 and nodes and len(nodes) == len(set(nodes))
            receipt.update(collection=sorted(nodes), collection_returncode=collected.returncode)
            command = [sys.executable, "-B", "-m", "pytest", "-q", "-ra", target,
                       "--basetemp", str(output / "tmp"), "--junitxml", str(output / "pytest.xml"),
                       "-o", f"cache_dir={output / 'cache'}"]
            with (output / "pytest.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=170, check=False)
            membership, counts, failures, observed, seconds = junit(output / "pytest.xml")
            duplicates = sorted(node for node, count in Counter(membership).items() if count != 1)
            missing = sorted(set(nodes) - set(membership))
            extra = sorted(set(membership) - set(nodes))
            receipt.update(command=command, returncode=result.returncode, junit_membership=sorted(membership),
                           counts=counts, failures=failures, observations=observed,
                           duplicates=duplicates, missing=missing, extra=extra, pytest_seconds=seconds)
            okay = (result.returncode == 0 and not duplicates and not missing and not extra
                    and counts["passed"] == len(nodes) and counts["skipped"] == 0)
            receipt["verdict"] = "PASS" if okay else "FAIL"
            receipt["first_failure"] = failures[0] if failures else None if okay else "count/membership gate"
    except Exception as error:
        receipt["first_failure"] = {"type": type(error).__name__, "reason": str(error)}
        save(output / "failure.json", {"trace": traceback.format_exc()})
    finally:
        from scripts.verify import check_source_consistency, source_snapshot
        after = {"owner": source_snapshot(OWNER), "reviewer": source_snapshot(ROOT)}
        receipt["source_after"] = after
        receipt["source_consistency"] = {name: check_source_consistency(before[name], after[name])
                                         for name in before}
        if before != after or not all(item["clean"] for item in before.values()):
            receipt["verdict"] = "FAIL"
            receipt["first_failure"] = receipt["first_failure"] or "clean/source gate"
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    receipt["artifacts"] = inventory(output)
    save(output / "receipt.json", receipt)
    print(json.dumps({"receipt": str(output / "receipt.json"), "sha256": sha(output / "receipt.json"),
                      "verdict": receipt["verdict"], "counts": receipt.get("counts"),
                      "first_failure": receipt["first_failure"]}))
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
