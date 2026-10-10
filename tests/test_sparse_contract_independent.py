"""Independent bounded CPU sparse contract review, with immutable external evidence.

Use --run --artifact-dir DIR for actual tests and --audit-artifacts RUN_DIR
for a no-solve byte/oracle replay. A complete evidence audit can PASS while
the production contract review remains FAIL. No production helper is an oracle.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OWNER_ENV = ("OPENOPT_SPARSE_OWNER_ROOT", "OPENOPT_SPARSE_OWNER_COMMIT", "OPENOPT_SPARSE_OPERATOR_SHA256")
if __name__ == "__main__":
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--owner-root")
    bootstrap.add_argument("--owner-commit")
    bootstrap.add_argument("--operator-sha256")
    bootstrap.add_argument("--owner-source-sha256")
    bootstrap.add_argument("--reviewer-commit")
    bootstrap.add_argument("--reviewer-source-sha256")
    options, _ = bootstrap.parse_known_args()
    supplied = (options.owner_root, options.owner_commit, options.operator_sha256)
    if any(value is not None for value in supplied):
        for key, value in zip(OWNER_ENV, supplied):
            if value is not None:
                os.environ[key] = value
    for field, key in (
        ("owner_source_sha256", "OPENOPT_SPARSE_OWNER_SOURCE_SHA256"),
        ("reviewer_commit", "OPENOPT_SPARSE_REVIEWER_COMMIT"),
        ("reviewer_source_sha256", "OPENOPT_SPARSE_REVIEWER_SOURCE_SHA256"),
    ):
        value = getattr(options, field)
        if value is not None:
            os.environ[key] = value


def git_bytes(root, *args, input_bytes=None):
    process = subprocess.run(["git", "-C", str(root), *args], input=input_bytes,
                             capture_output=True, timeout=30, check=False)
    if process.returncode:
        raise ValueError(f"source Git query failed rc={process.returncode}")
    return process.stdout


def source_snapshot(root):
    """Bind all actual members, independently of the subject's verifier."""
    root = Path(root)
    names = sorted(set(git_bytes(
        root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    ).decode().split("\0")) - {""})
    hashes = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"unsafe source member: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not hashes:
        raise ValueError("empty source membership")
    return {"commit": git_bytes(root, "rev-parse", "HEAD").decode().strip(),
            "tree": git_bytes(root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "clean": git_bytes(root, "status", "--porcelain", "--untracked-files=all") == b"",
            "sha256": hashes}


def source_pin(snapshot):
    payload = json.dumps(snapshot["sha256"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def committed_members(root, snapshot):
    """One batch reads Git blobs; clean/skip-worktree labels alone are not pins."""
    commit = snapshot["commit"]
    names = sorted(set(git_bytes(root, "ls-files", "-z").decode().split("\0")) - {""})
    queries = "".join(f"{commit}:{name}\n" for name in names).encode()
    stream = io.BytesIO(git_bytes(root, "cat-file", "--batch", input_bytes=queries))
    if set(names) != set(snapshot["sha256"]):
        raise ValueError("committed/current source membership mismatch")
    for name in names:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"committed blob unavailable: {name}")
        payload = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("invalid Git blob framing")
        if hashlib.sha256(payload).hexdigest() != snapshot["sha256"][name]:
            raise ValueError(f"committed/current source bytes mismatch: {name}")
    if stream.read():
        raise ValueError("extra Git blob output")


def bound_source(root, commit, pin=None):
    if not isinstance(commit, str) or not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("caller source commit must be full lowercase hex")
    snapshot = source_snapshot(root)
    if snapshot["commit"] != commit:
        raise ValueError("caller-pinned source full commit mismatch")
    if not snapshot["clean"]:
        raise ValueError("caller-pinned source must be clean")
    committed_members(root, snapshot)
    if pin is not None and (not re.fullmatch("[0-9a-f]{64}", pin)
                            or source_pin(snapshot) != pin):
        raise ValueError("caller-pinned source membership SHA mismatch")
    return snapshot


def owner_selection():
    supplied = [os.environ.get(key) for key in OWNER_ENV]
    if not any(value is not None for value in supplied):
        return ROOT, None, None
    if not all(supplied):
        raise ValueError("caller owner root/commit/operator SHA must be supplied together")
    root, commit, pin = supplied
    path = Path(root)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("owner-root must be a canonical absolute local path")
    if not re.fullmatch("[0-9a-f]{40}", commit) or not re.fullmatch("[0-9a-f]{64}", pin):
        raise ValueError("caller owner commit/SHA must be full lowercase hex")
    module = path / "src/openopt/operators.py"
    if module.is_symlink() or not module.resolve(strict=True).is_relative_to(path):
        raise ValueError("owner operators source cannot escape or be a symlink")
    if hashlib.sha256(module.read_bytes()).hexdigest() != pin:
        raise ValueError("caller-pinned owner operators bytes mismatch")
    bound_source(path, commit, os.environ.get("OPENOPT_SPARSE_OWNER_SOURCE_SHA256"))
    return path, commit, pin


SELECTION_ERROR = None
try:
    OWNER_ROOT, OWNER_COMMIT, OPERATOR_PIN = owner_selection()
except Exception as error:
    if __name__ != "__main__":
        raise
    SELECTION_ERROR = error
    OWNER_ROOT, OWNER_COMMIT, OPERATOR_PIN = ROOT, None, None
sys.path[:0] = [str(OWNER_ROOT / "src"), str(ROOT)]

import numpy as np
import pytest
import torch

import openopt.operators as operators

TEST = "tests/test_sparse_contract_independent.py"
DOC = "docs/research/sparse-contract-independent.md"
BASE = "c9616c4199e3cd9859dd007cd752862baec7c127"
REPAIR_BASE = "204934bc36284a40b8407b4e2fb68fa0d88184f0"
PINS = {
    "src/openopt/operators.py": "4d9a57fafc3dc1a77d7c61ffcd9e4dbdba15a020c310ddd78f8a37ff5b0615e5",
    "src/openopt/qp.py": "955d741317e5de1c5743f6e6bc65406ba14560f9a8b4c989afe6d11a1fc58e40",
    "src/openopt/presolve.py": "c19096e5eff19f29fdf24388ababbe3daa103d5abb45f24bbeec945a3ab98101",
    "tests/test_qp_operators.py": "0fa731c8fa8b0eb640b109897555ce6aefde0cc6631e7d98b8ad36f7fd9ebb23",
    "tests/test_qp_precondition.py": "1be22a1554192aada140a002c7d5a1307f88085d844ad408d962ab3c90710336",
    "tests/test_qp_presolve.py": "ae18e936dbbb4abea276860964d37bb5642abbc2af00e4408f864fe8f8b28af5",
    "scripts/verify.py": "4b90e4962ebc4e96697744f04e0f91dc1ff9b50a5634a8f24e323cfe5ebc03b5",
}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
LAYOUTS = ("dense", "coo", "csr", "matrix_free")
EPS = Fraction.from_float(1e-7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="ascii") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    assert json.loads(Path(path).read_bytes()) == value


def tensor_bytes(value):
    array = value.detach().numpy().copy(order="C")
    raw = array.tobytes()
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "hex": raw.hex(), "sha256": hashlib.sha256(raw).hexdigest()}


def decoded(value):
    raw = bytes.fromhex(value["hex"])
    if hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("original tensor byte hash mismatch")
    dtype = np.dtype(value["dtype"])
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("review supports only represented real float32/float64")
    array = np.frombuffer(raw, dtype=dtype).reshape(value["shape"])
    return array


def freeze(model):
    return {key: tensor_bytes(getattr(model, key)) for key in ("P", "q", "A", "l", "u")}


def input_checks(original, working):
    """Exact original validation and selected cast losses, not a solver oracle."""
    lo, hi = decoded(original["l"]), decoded(original["u"])
    zero_rows = [
        i for i, row in enumerate(decoded(original["A"]))
        if np.isfinite(row).all() and all(v == 0 for v in row) and (lo[i] > 0 or hi[i] < 0)
    ]
    losses = {}
    for key, value in working.items():
        before, after = decoded(original[key]), decoded(value)
        losses[key] = {
            "finite_to_nonfinite": np.flatnonzero(np.isfinite(before) & ~np.isfinite(after)).tolist(),
            "nonzero_to_zero": np.flatnonzero(np.isfinite(before) & (before != 0) & (after == 0)).tolist(),
        }
    return {"original_reversed_rows": np.flatnonzero(lo > hi).tolist(),
            "exact_infeasible_zero_rows": zero_rows, "cast_losses": losses}


def f(value):
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


def dot(left, right):
    assert len(left) == len(right)
    return sum((f(a) * f(b) for a, b in zip(left, right)), Fraction(0))


def threshold(*values):
    return EPS + EPS * max((abs(f(v)) for v in values), default=Fraction(0))


def original_checks(original, x, y):
    """Six componentwise gates on original bytes, without core check/projection helpers."""
    P, q, A, l, u = [decoded(original[key]) for key in ("P", "q", "A", "l", "u")]
    px = [dot(row, x) for row in P]
    ax = [dot(row, x) for row in A]
    aty = [dot(column, y) for column in A.T]
    rows, support = [], Fraction(0)
    for index, (value, lo, hi, dual) in enumerate(zip(ax, l, u, y)):
        low = None if np.isneginf(lo) else f(lo)
        high = None if np.isposinf(hi) else f(hi)
        multiplier = f(dual)
        row_tol = threshold(value, *([low] if low is not None else []),
                            *([high] if high is not None else []))
        projected = max(value, low) if low is not None else value
        projected = min(projected, high) if high is not None else projected
        shifted = value + multiplier
        normal_projection = max(shifted, low) if low is not None else shifted
        normal_projection = min(normal_projection, high) if high is not None else normal_projection
        endpoint = high if multiplier > 0 else low if multiplier < 0 else Fraction(0)
        sign_ok = endpoint is not None
        product = None if not sign_ok else multiplier * endpoint
        if product is not None:
            support += product
        comp = None if not sign_ok else abs(multiplier * value - product)
        comp_tol = threshold(multiplier * value, *([product] if product is not None else []))
        rows.append({
            "index": index, "Ax": str(value), "violation": str(abs(value - projected)),
            "tolerance": str(row_tol), "normal_residual": str(abs(value - normal_projection)),
            "feasible": abs(value - projected) <= row_tol,
            "normal": abs(value - normal_projection) <= row_tol,
            "support_available": sign_ok,
            "complementarity": comp is not None and comp <= comp_tol,
            "complementarity_residual": None if comp is None else str(comp),
            "complementarity_tolerance": str(comp_tol),
        })
    columns = [{
        "index": j, "residual": str(abs(p + f(c) + a)),
        "tolerance": str(threshold(p, c, a)),
        "passed": abs(p + f(c) + a) <= threshold(p, c, a),
    } for j, (p, c, a) in enumerate(zip(px, q, aty))]
    quadratic, linear = dot(x, px), dot(q, x)
    gap = abs(quadratic + linear + support)
    gap_tol = threshold(quadratic, linear, support)
    gates = {
        "feasibility": all(row["feasible"] for row in rows),
        "stationarity": all(column["passed"] for column in columns),
        "normal_cone": all(row["normal"] for row in rows),
        "infinite_bound_sign": all(row["support_available"] for row in rows),
        "complementarity": all(row["complementarity"] for row in rows),
        "gap": all(row["support_available"] for row in rows) and gap <= gap_tol,
    }
    return {"passed": all(gates.values()), "gates": gates, "rows": rows, "columns": columns,
            "gap": str(gap), "gap_tolerance": str(gap_tol),
            "objective": str(quadratic / 2 + linear)}


def global_tolerance_interval(original, x):
    """Bound the core's unseen projected z by its original finite box."""
    A, l, u = [decoded(original[key]) for key in ("A", "l", "u")]
    if not np.isfinite(l).all() or not np.isfinite(u).all():
        raise ValueError("global tolerance interval requires finite original bounds")
    ax_norm = max(map(abs, (dot(row, x) for row in A)), default=Fraction(0))
    z_min = max((Fraction(0) if lo <= 0 <= hi else min(abs(f(lo)), abs(f(hi)))
                 for lo, hi in zip(l, u)), default=Fraction(0))
    z_max = max((max(abs(f(lo)), abs(f(hi))) for lo, hi in zip(l, u)), default=Fraction(0))
    return {"minimum": str(threshold(ax_norm, z_min)),
            "maximum": str(threshold(ax_norm, z_max))}


def problem(P, q, A, l, u):
    return operators.OperatorQPProblem(
        torch.tensor(P, dtype=torch.float64), torch.tensor(q, dtype=torch.float64),
        torch.tensor(A, dtype=torch.float64).reshape(len(l), len(q)),
        torch.tensor(l, dtype=torch.float64), torch.tensor(u, dtype=torch.float64),
        assume_psd=True,
    )


def represented(matrix, kind):
    if kind == "coo":
        return matrix.to_sparse_coo()
    if kind == "csr":
        return matrix.to_sparse_csr()
    if kind == "matrix_free":
        return operators.MatrixFreeOperator(
            tuple(matrix.shape), "cpu", matrix.dtype,
            lambda v: matrix @ v, lambda v: matrix.T @ v,
        )
    return matrix


def sparse_problem(model, kind):
    return operators.OperatorQPProblem(
        represented(model.P, kind), model.q, represented(model.A, kind),
        model.l, model.u, assume_psd=True,
    )


def config_dict(config):
    return {field.name: (str(getattr(config, field.name)) if field.name == "dtype"
                         else getattr(config, field.name)) for field in fields(config)}


def observe(model, kind, config, expected, category, request, tmp_path, *, solve=True):
    before = freeze(model)
    value = {"nodeid": request.node.nodeid, "category": category, "layout": kind,
             "expected": expected, "original": before, "config": config_dict(config),
             "preparation_calls": 1, "solver_calls": 0, "solver_returns": 0}
    try:
        workspace = operators.prepare_operator_qp(sparse_problem(model, kind), config)
    except (TypeError, ValueError, RuntimeError) as error:
        value.update(outcome="REJECTED", reason=f"{type(error).__name__}: {error}")
    else:
        value["working_vectors"] = {key: tensor_bytes(getattr(workspace, key))
                                    for key in ("q", "l", "u")}
        value["outcome"] = "PREPARED"
        if solve:
            value["solver_calls"] = 1
            try:
                result = workspace.solve()
            except (TypeError, ValueError, RuntimeError) as error:
                value.update(outcome="REJECTED", reason=f"{type(error).__name__}: {error}")
            else:
                value["solver_returns"] = 1
                value.update(
                    outcome="RETURNED",
                    result={field.name: (getattr(result, field.name).tolist()
                                        if field.name in ("x", "y") else getattr(result, field.name))
                            for field in fields(result)},
                    result_device={"x": str(result.x.device), "y": str(result.y.device)},
                )
                value["oracle"] = original_checks(before, value["result"]["x"], value["result"]["y"])
                if category == "zero-row-infeasible":
                    value["global_primal_tolerance"] = global_tolerance_interval(before, value["result"]["x"])
    value["input_checks"] = input_checks(before, value.get("working_vectors", {}))
    value["after"] = freeze(model)
    path = tmp_path / "observation.json"
    write_json(path, value)
    request.node.user_properties.extend([
        ("observation_path", str(path)), ("observation_sha256", sha(path)),
    ])
    assert before == value["after"], "caller original model bytes changed"
    assert matches(value), value
    return value


def matches(value):
    if value["expected"] == "REJECTED":
        return value["outcome"] == "REJECTED"
    if value["expected"] == "NOT_SOLVED":
        return value["outcome"] == "REJECTED" or (
            value["outcome"] == "RETURNED" and value["result"]["status"] != "solved")
    if value["expected"] == "SOLVED":
        return (value["outcome"] == "RETURNED" and value["result"]["status"] == "solved"
                and value["oracle"]["passed"] and value["result_device"] == {"x": "cpu", "y": "cpu"})
    raise ValueError("unknown independent expected outcome")


def forbid_dense(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sparse review path densified or factored a dense Gram")
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)


@pytest.mark.parametrize("kind", LAYOUTS)
def test_actual_cpu_positive_has_independent_original_six_gates(kind, request, tmp_path, monkeypatch):
    model = problem([[2., 0.], [0., 4.]], [-1., 8.],
                    [[1., 0.], [0., 1.], [1., 1.]], [0., -1., -5.], [0., 1., 5.])
    assert original_checks(freeze(model), [0., -1.], [1., -4., 0.])["passed"]
    if kind != "dense":
        forbid_dense(monkeypatch)
    value = observe(model, kind, operators.OperatorQPConfig(max_iterations=250),
                    "SOLVED", "cpu-positive", request, tmp_path)
    assert value["result"]["convexity_certified"] is False
    assert abs(f(value["result"]["objective"]) - f(-6.)) <= Fraction(1, 10**6)


@pytest.mark.parametrize("kind", ("dense", "coo", "csr"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_tensor_matvec_adjoint_owned_bytes_and_duplicate_coo(kind, dtype, monkeypatch):
    indices = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 1, 0]])
    values = torch.tensor([2., -1., 3., 4., -2.], dtype=dtype)
    matrix = torch.sparse_coo_tensor(indices, values, (3, 2))
    if kind == "dense":
        matrix = torch.tensor([[1., 3.], [0., 4.], [-2., 0.]], dtype=dtype)
    elif kind == "csr":
        matrix = matrix.coalesce().to_sparse_csr()
    op = operators.TensorOperator(matrix)
    forbid_dense(monkeypatch)
    x, y = torch.tensor([2., -1.], dtype=dtype), torch.tensor([3., -2., 1.], dtype=dtype)
    expected_ax = [dot(row, x.tolist()) for row in ((1, 3), (0, 4), (-2, 0))]
    expected_aty = [dot(column, y.tolist()) for column in ((1, 0, -2), (3, 4, 0))]
    if matrix.layout == torch.strided:
        matrix.fill_(19.)
    else:
        (matrix._values() if kind == "coo" else matrix.values()).fill_(19.)
    assert list(map(f, op.matvec(x).tolist())) == expected_ax
    assert list(map(f, op.rmatvec(y).tolist())) == expected_aty
    assert dot(y.tolist(), expected_ax) == dot(x.tolist(), expected_aty)
    assert op.column_squared_norms().tolist() == [5., 25.]


@pytest.mark.parametrize("kind", ("dense", "coo", "csr"))
@pytest.mark.parametrize("preconditioned", (False, True))
def test_actual_cg_matches_fraction_analytic_system(kind, preconditioned, monkeypatch):
    K = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64)
    rhs = torch.tensor([1., 2.], dtype=torch.float64)
    op = operators.TensorOperator(represented(K, kind))
    before = tensor_bytes(rhs)
    preconditioner = (operators.JacobiPreconditioner(torch.tensor([4., 3.], dtype=K.dtype))
                      if preconditioned else None)
    forbid_dense(monkeypatch)
    result = operators.conjugate_gradient(op, rhs, preconditioner=preconditioner)
    expected = [Fraction(1, 11), Fraction(7, 11)]
    actual = list(map(f, result.x.tolist()))
    assert max(abs(x - y) for x, y in zip(actual, expected)) <= Fraction(1, 10**12)
    residual = [f(b) - dot(row, actual) for row, b in zip(((4, 1), (1, 3)), (1, 2))]
    assert sum(r * r for r in residual) <= f(result.tolerance) ** 2
    assert tensor_bytes(rhs) == before and result.x.device == torch.device("cpu")


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize("scale", (1., 1e9), ids=("ordinary", "cross-scale"))
def test_analytically_infeasible_zero_row_never_solved(kind, scale, request, tmp_path):
    model = problem([[1.]], [-scale], [[1.], [0.]], [scale, 1.], [scale, 2.])
    # Exact Farkas contradiction: A.T*[0,-1]=0; support=-1 < 0.
    assert dot([1., 0.], [0., -1.]) == 0 and dot(model.l.tolist(), [0., -1.]) == -1
    observe(model, kind, operators.OperatorQPConfig(max_iterations=100),
            "NOT_SOLVED", "zero-row-infeasible", request, tmp_path)


@pytest.mark.parametrize("kind", ("dense", "coo", "csr"))
@pytest.mark.parametrize("case", ("reversed-collapse", "finite-lower-overflow", "finite-upper-overflow"))
def test_invalid_original_or_unrepresentable_bounds_reject_before_working_cast(
    kind, case, request, tmp_path,
):
    low, high = {
        "reversed-collapse": (1. + 1e-12, 1.),
        "finite-lower-overflow": (-1e100, 1.),
        "finite-upper-overflow": (-1., 1e100),
    }[case]
    if case == "reversed-collapse":
        assert f(low) > f(high)
    else:
        assert math.isfinite(low) and math.isfinite(high)
    model = problem([[1.]], [0.], [[1.]], [low], [high])
    observe(model, kind, operators.OperatorQPConfig(dtype=torch.float32),
            "REJECTED", "cast-original-bounds", request, tmp_path, solve=False)


@pytest.mark.parametrize("kind", ("dense", "coo", "csr"))
def test_positive_zero_row_bounds_cannot_be_cast_to_zero_without_rejection(kind, request, tmp_path):
    model = problem([[1.]], [0.], [[0.]], [1e-100], [2e-100])
    assert f(model.l.item()) > 0 and dot([0.], [17.]) == 0
    observe(model, kind, operators.OperatorQPConfig(dtype=torch.float32, max_iterations=25),
            "REJECTED", "cast-zero-row", request, tmp_path)


@pytest.mark.parametrize("field", ("P", "q", "A", "l"))
def test_nonfinite_original_input_is_loud(field, request, tmp_path):
    model = problem([[1.]], [0.], [[1.]], [-1.], [1.])
    getattr(model, field).flatten()[0] = float("nan")
    observe(model, "coo", operators.OperatorQPConfig(), "REJECTED",
            "nonfinite-negative", request, tmp_path, solve=False)


@pytest.mark.parametrize("kind", ("coo", "csr"))
def test_invalid_sparse_indices_reject_before_math(kind):
    if kind == "coo":
        matrix = torch.sparse_coo_tensor(torch.tensor([[0], [2]]),
                                         torch.ones(1, dtype=torch.float64), (1, 1))
    else:
        matrix = torch.sparse_csr_tensor(torch.tensor([0, 1]), torch.tensor([2]),
                                         torch.ones(1, dtype=torch.float64), (1, 1))
    with pytest.raises(RuntimeError):
        operators.TensorOperator(matrix)


@pytest.mark.parametrize("bad", ("adjoint", "nonlinear"))
def test_deterministic_invalid_callbacks_reject(bad):
    model = problem([[1.]], [0.], [[1.]], [-1.], [1.])
    A = operators.MatrixFreeOperator(
        (1, 1), "cpu", torch.float64,
        (lambda v: v * v) if bad == "nonlinear" else (lambda v: v),
        (lambda v: v) if bad == "nonlinear" else (lambda v: 2 * v),
    )
    p = operators.OperatorQPProblem(model.P, model.q, A, model.l, model.u, assume_psd=True)
    with pytest.raises(ValueError, match="probe"):
        operators.prepare_operator_qp(p)


@pytest.mark.parametrize("case", ("wrong-sign-infinite", "wrong-primal", "wrong-dual"))
def test_fraction_oracle_negative_controls(case):
    model = problem([[1.]], [0.], [[1.]], [0.], [float("inf")])
    x, y = (0., 1.) if case == "wrong-sign-infinite" else (1., 0.) if case == "wrong-dual" else (-1., 0.)
    result = original_checks(freeze(model), [x], [y])
    assert not result["passed"]
    if case == "wrong-sign-infinite":
        assert not result["gates"]["infinite_bound_sign"]
        assert not result["gates"]["gap"]


def membership(selected, actual):
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("empty/duplicate live collection")
    if len(actual) != len(set(actual)) or set(selected) != set(actual):
        raise ValueError("collection/JUnit bidirectional membership mismatch")
    return {"missing": [], "extra": [], "duplicates": []}


@pytest.mark.parametrize("case", ("drop", "extra", "duplicate", "coordinated-drop"))
def test_collection_negative_controls(case):
    live, selected, actual = ["a", "b"], ["a", "b"], ["a", "b"]
    if case == "extra":
        actual.append("c")
    elif case == "duplicate":
        actual.append("a")
    else:
        actual.pop()
        if case == "coordinated-drop":
            selected.pop()
    assert live == ["a", "b"]
    with pytest.raises(ValueError):
        membership(live if case == "coordinated-drop" else selected, actual)


def loaded_binding():
    path = Path(operators.__file__).resolve(strict=True)
    owner = source_snapshot(OWNER_ROOT)
    pin = OPERATOR_PIN or owner["sha256"]["src/openopt/operators.py"]
    assert path == OWNER_ROOT / "src/openopt/operators.py" and sha(path) == pin
    for function in (operators.prepare_operator_qp, operators.solve_operator_qp):
        assert Path(function.__code__.co_filename).resolve() == path
        assert function.__module__ == operators.__name__
    return {"path": str(path), "sha256": sha(path)}


def test_actual_source_and_cpu_thread_boundary(monkeypatch):
    owner = source_snapshot(OWNER_ROOT)
    assert bound_source(OWNER_ROOT, OWNER_COMMIT or owner["commit"]) == owner
    assert loaded_binding()["sha256"] == (OPERATOR_PIN or owner["sha256"]["src/openopt/operators.py"])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert all(os.environ[key] == "1" for key in THREADS)
    assert torch.get_num_threads() == 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        operators.OperatorQPConfig(device="cuda:0").validate()


def environment(directory):
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(OWNER_ROOT / "src"), str(ROOT)]),
               PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               CUDA_VISIBLE_DEVICES="-1", TMPDIR=str(directory))
    env.update({key: "1" for key in THREADS})
    return env


def immutable_math():
    """Exact original function/decorator text, not merely equal test counts."""
    baseline = git_bytes(ROOT, "show", f"{REPAIR_BASE}:{TEST}").decode()
    current = (ROOT / TEST).read_text()

    def blocks(text):
        lines = text.splitlines(keepends=True)
        result = {}
        for node in ast.parse(text).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name == "test_actual_source_and_cpu_thread_boundary":
                continue
            if node.name.startswith("test_") or node.name in (
                "tensor_bytes", "decoded", "freeze", "input_checks", "f", "dot", "threshold",
                "original_checks", "global_tolerance_interval", "problem", "represented",
                "sparse_problem", "config_dict", "observe", "matches", "forbid_dense", "membership",
            ):
                start = min([node.lineno] + [item.lineno for item in node.decorator_list])
                result[node.name] = "".join(lines[start - 1:node.end_lineno])
        return result

    original, actual = blocks(baseline), blocks(current)
    if original != actual:
        raise ValueError("original mathematical functions/assertions changed")
    return {"verdict": "PASS", "functions": len(original),
            "sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()}


def scope_check(base):
    if not isinstance(base, str) or not re.fullmatch("[0-9a-f]{40}", base):
        raise ValueError("scope base must be a full Git commit")
    paths = set(git_bytes(ROOT, "diff", "--name-only", base, "HEAD").decode().splitlines())
    if not paths or paths != {TEST, DOC}:
        raise ValueError("development write scope differs from exact test/doc paths")
    return {"base": base, "paths": sorted(paths), "verdict": "PASS"}


def caller_binding(development=False):
    if SELECTION_ERROR is not None:
        raise SELECTION_ERROR
    if not all(os.environ.get(key) for key in OWNER_ENV):
        raise ValueError("explicit caller owner root/commit/operator SHA required")
    reviewer_commit = os.environ.get("OPENOPT_SPARSE_REVIEWER_COMMIT")
    reviewer_pin = os.environ.get("OPENOPT_SPARSE_REVIEWER_SOURCE_SHA256")
    owner_pin = os.environ.get("OPENOPT_SPARSE_OWNER_SOURCE_SHA256")
    if not reviewer_pin or not owner_pin:
        raise ValueError("explicit caller reviewer/owner source membership SHA required")
    owner = bound_source(OWNER_ROOT, OWNER_COMMIT, owner_pin)
    if development:
        reviewer = source_snapshot(ROOT)
        if reviewer["commit"] != reviewer_commit or source_pin(reviewer) != reviewer_pin:
            raise ValueError("development caller reviewer identity mismatch")
    else:
        reviewer = bound_source(ROOT, reviewer_commit, reviewer_pin)
    return {"reviewer_root": str(ROOT), "reviewer_commit": reviewer_commit,
            "reviewer_source_sha256": reviewer_pin, "owner_root": str(OWNER_ROOT),
            "owner_commit": OWNER_COMMIT, "owner_source_sha256": owner_pin,
            "operator_sha256": OPERATOR_PIN, "development": development,
            "math": immutable_math()}


def external_directory(path, *, existing=False):
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError("evidence requires absolute external path without traversal")
    path = raw.resolve(strict=existing)
    if (not path.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve())
            or any(path.is_relative_to(repo) for repo in (ROOT, OWNER_ROOT))):
        raise ValueError("evidence must be external to reviewer/owner on authorized 3FS tmp")
    return path


def terminal_failure(directory, error, stage, name="receipt.json"):
    """Never replace a prior receipt or turn a harness exception into acceptance."""
    frame = traceback.extract_tb(error.__traceback__)
    last = frame[-1] if frame else None
    report = {"schema_version": 2, "checked_at": datetime.now(timezone.utc).isoformat(),
              "evidence_verdict": "FAIL", "review_verdict": "UNVERIFIED",
              "accept_sparse_contract": False, "returncode": 2, "stage": stage,
              "first_cause": {"type": type(error).__name__, "message": str(error),
                              "file": last.filename if last else None,
                              "line": last.lineno if last else None},
              "historical_fail_requalified": False, "audit_solver_calls": 0}
    write_json(directory / name, report)
    with (directory / (Path(name).stem + "-failure.log")).open("x") as stream:
        stream.write("".join(traceback.format_exception(type(error), error, error.__traceback__)))
        stream.write("\nreturncode=2\n")
    print(f"SPARSE_REVIEW evidence=FAIL review=UNVERIFIED stage={stage} "
          f"cause={type(error).__name__} receipt={directory / name}", flush=True)
    return report


def collect(directory, name):
    command = [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q",
               "-o", f"cache_dir={directory / (name + '-cache')}", TEST]
    result = subprocess.run(command, cwd=ROOT, env=environment(directory),
                            capture_output=True, text=True, timeout=45, check=False)
    with (directory / (name + ".log")).open("x", encoding="utf-8") as log:
        log.write(result.stdout + result.stderr + f"\nreturncode={result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"independent collection failed rc={result.returncode}")
    nodes = [line.strip() for line in result.stdout.splitlines() if line.startswith(TEST + "::")]
    membership(nodes, nodes)
    return nodes


def audit_artifacts(directory, name="replay.json", require_clean=True):
    directory = external_directory(directory, existing=True)
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("receipt-name must be a simple new JSON filename")
    if (directory / name).exists():
        raise ValueError("immutable receipt already exists")
    try:
        return audit_body(directory, name, require_clean)
    except Exception as error:
        return terminal_failure(directory, error, "audit", name)


def audit_body(directory, name, require_clean):
    """No solves: replay original bytes, Fraction metrics, membership and source."""
    from scripts.verify import check_source_consistency
    binding = caller_binding(development=not require_clean)
    started = time.monotonic()
    current = source_snapshot(ROOT)
    owner_current = source_snapshot(OWNER_ROOT)
    execution = json.loads((directory / "execution.json").read_bytes())
    assert execution["schema_version"] == 2, "historical v1 execution not requalified"
    assert execution["caller_binding"] == binding, "execution caller identity mismatch"
    artifacts = Path(execution["verification_path"]).parent.resolve(strict=True)
    assert artifacts.is_relative_to(directory) and not artifacts.is_symlink()
    verification = json.loads((artifacts / "verification.json").read_bytes())
    tree = ET.parse(artifacts / "pytest.xml").getroot()
    cases = list(tree.iter("testcase"))
    actual = [c.get("classname", "").replace(".", "/") + ".py::" + c.get("name", "") for c in cases]
    selected = collect(directory, Path(name).stem + "-collection")
    old_collection = json.loads((directory / "collection.json").read_bytes())
    differences = membership(selected, actual)
    assert selected == old_collection["nodeids"]
    counts = {"tests": len(cases), **{
        key: sum(c.find(tag) is not None for c in cases)
        for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))
    }}
    for suite in tree.iter("testsuite"):
        leaves = suite.findall("testcase")
        if not suite.findall("testsuite"):
            assert int(suite.get("tests")) == len(leaves)
            assert all(int(suite.get(key)) == sum(c.find(tag) is not None for c in leaves)
                       for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")))
    source = check_source_consistency(execution["source_before"], current)
    assert source["verdict"] == "PASS", "reviewer full source changed"
    assert check_source_consistency(verification["source"], current)["verdict"] == "PASS", "verifier source changed"
    assert execution["source_after"] == current
    assert execution["owner_selection"] == {
        "root": str(OWNER_ROOT), "caller_commit": OWNER_COMMIT, "caller_operator_sha256": OPERATOR_PIN}
    owner_consistency = check_source_consistency(execution["owner_before"], owner_current)
    assert owner_consistency["verdict"] == "PASS" and execution["owner_after"] == owner_current, "owner full source changed"
    if OWNER_COMMIT is not None:
        assert owner_current["clean"] and owner_current["commit"] == OWNER_COMMIT
        assert owner_current["sha256"]["src/openopt/operators.py"] == OPERATOR_PIN
    if require_clean:
        assert current["clean"] and execution["source_before"]["clean"] and verification["source"]["clean"]
    if execution["development_scope"] is not None:
        assert scope_check(execution["development_scope"]["base"]) == execution["development_scope"]
    assert verification["targets"] == [TEST] and verification["coverage"] == "explicit_targets"
    assert execution["returncode"] == (1 if counts["failures"] or counts["errors"] else 0)
    assert verification["verdict"] == ("FAIL" if execution["returncode"] else "PASS")
    assert 0 < execution["wall_seconds"] <= 180
    observations, mismatches = [], []
    for case in cases:
        props = {p.get("name"): p.get("value") for p in case.findall("properties/property")}
        if "observation_path" not in props:
            continue
        path = Path(props["observation_path"]).resolve(strict=True)
        assert path.is_relative_to(artifacts) and sha(path) == props["observation_sha256"]
        value = json.loads(path.read_bytes())
        node = case.get("classname").replace(".", "/") + ".py::" + case.get("name")
        assert value["nodeid"] == node and value["original"] == value["after"]
        for record in value["original"].values():
            decoded(record)
        assert value["input_checks"] == input_checks(value["original"], value.get("working_vectors", {}))
        if value["outcome"] == "RETURNED":
            recomputed = original_checks(value["original"], value["result"]["x"], value["result"]["y"])
            assert value["oracle"] == recomputed
            if value["category"] == "zero-row-infeasible":
                assert value["global_primal_tolerance"] == global_tolerance_interval(
                    value["original"], value["result"]["x"])
        matched = matches(value)
        assert (case.find("failure") is not None) is (not matched)
        if not matched:
            mismatches.append(node)
        observations.append({"path": str(path), "sha256": sha(path), "value": value})
    failed_nodes = [node for node, c in zip(actual, cases) if c.find("failure") is not None]
    assert set(failed_nodes) == set(mismatches), "unexpected harness failure is not a production finding"
    assert not counts["errors"] and not counts["skipped"]
    assert Counter(o["value"]["category"] for o in observations) == {
        "cpu-positive": 4, "zero-row-infeasible": 8, "cast-original-bounds": 9,
        "cast-zero-row": 3, "nonfinite-negative": 4,
    }
    receipt = {
        "schema_version": 2, "checked_at": datetime.now(timezone.utc).isoformat(),
        "evidence_verdict": "PASS", "review_verdict": "FAIL" if mismatches else "PASS",
        "accept_sparse_contract": not bool(mismatches), "source": current,
        "owner_source": owner_current, "owner_source_consistency": owner_consistency,
        "owner_selection": execution["owner_selection"],
        "source_consistency": source, "loaded": loaded_binding(), "tests": counts,
        "caller_binding": binding, "development_scope": execution["development_scope"],
        "passes": counts["tests"] - counts["failures"], "membership": differences,
        "collection_nodeids": selected, "junit_nodeids": actual,
        "collection_sha256": sha(directory / "collection.json"), "junit_sha256": sha(artifacts / "pytest.xml"),
        "verification_sha256": sha(artifacts / "verification.json"),
        "execution_sha256": sha(directory / "execution.json"),
        "wall_seconds": execution["wall_seconds"], "audit_seconds": time.monotonic() - started,
        "solver_calls": sum(o["value"]["solver_calls"] for o in observations),
        "solver_returns": sum(o["value"]["solver_returns"] for o in observations),
        "standalone_cg_cases": sum("::test_actual_cg_matches_fraction_analytic_system[" in node
                                   for node in actual), "audit_solver_calls": 0,
        "mismatches": mismatches, "observations": observations, "scope": [TEST, DOC],
        "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                    "torch_num_threads": torch.get_num_threads(),
                    "thread_environment": {key: os.environ.get(key) for key in THREADS}},
        "gpu_execution": "UNVERIFIED", "commercial_qualification": False,
        "historical_fail_requalified": False,
        "returncode": 1 if mismatches else 0,
    }
    write_json(directory / name, receipt)
    print(f"SPARSE_REVIEW evidence=PASS review={receipt['review_verdict']} "
          f"tests={counts} solver_calls={receipt['solver_calls']} receipt={directory / name}", flush=True)
    return receipt


def run(artifact_root, development=False, scope_base=None):
    root = external_directory(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="review-", dir=root))
    stage = "caller_binding"
    try:
        binding = caller_binding(development)
        source_before = source_snapshot(ROOT)
        owner_before = source_snapshot(OWNER_ROOT)
        scope = scope_check(scope_base) if scope_base else None
        stage = "collection"
        nodes = collect(directory, "collection")
        write_json(directory / "collection.json", {"nodeids": nodes, "source": source_before})
        command = [sys.executable, "-B", "scripts/verify.py", "tests", "--artifact-dir", str(directory),
                   "--target", TEST]
        stage = "pytest"
        start = time.monotonic()
        with (directory / "run.log").open("x", encoding="utf-8") as log:
            log.write(f"command={command!r}\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, env=environment(directory),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=170, check=False)
            log.write(f"\nreturncode={result.returncode}\n")
        reports = list(directory.glob("tests-*/verification.json"))
        assert len(reports) == 1, "verifier terminal JSON missing or ambiguous"
        write_json(directory / "execution.json", {
            "schema_version": 2, "caller_binding": binding, "development_scope": scope,
            "command": command, "returncode": result.returncode, "wall_seconds": time.monotonic() - start,
            "verification_path": str(reports[0]), "source_before": source_before,
            "source_after": source_snapshot(ROOT), "development": development,
            "owner_selection": {"root": str(OWNER_ROOT), "caller_commit": OWNER_COMMIT,
                                "caller_operator_sha256": OPERATOR_PIN},
            "owner_before": owner_before, "owner_after": source_snapshot(OWNER_ROOT),
        })
    except Exception as error:
        terminal_failure(directory, error, stage)
        return 2
    receipt = audit_artifacts(directory, "receipt.json", require_clean=not development)
    return receipt["returncode"]


def selection_args():
    return ["--owner-root", str(OWNER_ROOT), "--owner-commit", OWNER_COMMIT,
            "--operator-sha256", OPERATOR_PIN,
            "--owner-source-sha256", os.environ["OPENOPT_SPARSE_OWNER_SOURCE_SHA256"],
            "--reviewer-commit", os.environ["OPENOPT_SPARSE_REVIEWER_COMMIT"],
            "--reviewer-source-sha256", os.environ["OPENOPT_SPARSE_REVIEWER_SOURCE_SHA256"]]


def entry_controls(artifact_root, completed=None, development=False):
    """Actual no-solve CLI negatives; retain stderr/rc and prove no run artifacts."""
    from scripts.verify import check_source_consistency
    caller_binding(development)
    root = external_directory(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="entry-", dir=root))
    before = source_snapshot(ROOT)
    common = selection_args()
    variants = {
        "partial": (common[:2], "must be supplied together"),
        "wrong-sha": ([*common[:5], "0" * 64, *common[6:]], "operators bytes mismatch"),
        "wrong-commit": ([*common[:3], "0" * 40, *common[4:]], "full commit mismatch"),
        "wrong-owner-membership": ([*common[:7], "0" * 64, *common[8:]], "membership SHA mismatch"),
        "wrong-reviewer-membership": ([*common[:-1], "0" * 64], "reviewer identity mismatch"
                                      if development else "membership SHA mismatch"),
        "wrong-scope": ([*common, "--scope-base", BASE], "write scope differs"),
    }
    results = {}
    for name, (selection, reason) in variants.items():
        target = directory / name
        command = [sys.executable, "-B", str(ROOT / TEST), "--run", "--artifact-dir", str(target), *selection]
        if development:
            command.append("--development")
        child_env = environment(directory)
        for key in (*OWNER_ENV, "OPENOPT_SPARSE_OWNER_SOURCE_SHA256",
                    "OPENOPT_SPARSE_REVIEWER_COMMIT", "OPENOPT_SPARSE_REVIEWER_SOURCE_SHA256"):
            child_env.pop(key, None)
        result = subprocess.run(command, cwd=ROOT, env=child_env,
                                capture_output=True, text=True, timeout=30, check=False)
        log = directory / (name + ".log")
        with log.open("x", encoding="utf-8") as stream:
            stream.write(f"command={command!r}\n" + result.stdout + result.stderr
                         + f"\nreturncode={result.returncode}\n")
        receipts = list(target.glob("review-*/receipt.json"))
        assert len(receipts) == 1 and result.returncode == 2
        terminal = json.loads(receipts[0].read_bytes())
        assert terminal["evidence_verdict"] == "FAIL" and terminal["accept_sparse_contract"] is False
        assert reason in terminal["first_cause"]["message"], terminal
        assert not list(target.glob("review-*/tests-*")), "invalid identity reached pytest/solve"
        results[name] = {"returncode": result.returncode, "reason": reason,
                         "log": str(log), "log_sha256": sha(log), "solver_calls": 0,
                         "receipt": str(receipts[0]), "receipt_sha256": sha(receipts[0])}
    if completed is not None:
        completed = external_directory(completed, existing=True)
        baseline = json.loads((completed / "receipt.json").read_bytes())
        assert baseline["evidence_verdict"] == baseline["review_verdict"] == "PASS"
        for name in ("drop-collection", "coordinated-drop", "wrong-owner-snapshot"):
            target = directory / name
            shutil.copytree(completed, target)
            execution = json.loads((target / "execution.json").read_bytes())
            old_artifacts = Path(execution["verification_path"]).parent
            artifacts = target / old_artifacts.name
            execution["verification_path"] = str(artifacts / "verification.json")
            collection = json.loads((target / "collection.json").read_bytes())
            if name == "wrong-owner-snapshot":
                execution["owner_before"]["sha256"]["src/openopt/operators.py"] = "0" * 64
            else:
                collection["nodeids"].pop()
                if name == "coordinated-drop":
                    tree = ET.parse(artifacts / "pytest.xml")
                    leaf = next(s for s in tree.iter("testsuite") if s.findall("testcase"))
                    leaf.remove(leaf.findall("testcase")[-1])
                    leaf.set("tests", str(len(leaf.findall("testcase"))))
                    tree.write(artifacts / "pytest.xml")
            # These are new mutation copies; old original observations/receipts stay immutable.
            (target / "execution.json").write_text(json.dumps(execution))
            (target / "collection.json").write_text(json.dumps(collection))
            command = [sys.executable, "-B", str(ROOT / TEST), "--audit-artifacts", str(target),
                       "--receipt-name", "negative.json", *common]
            if development:
                command.append("--development")
            result = subprocess.run(command, cwd=ROOT, env=environment(directory),
                                    capture_output=True, text=True, timeout=45, check=False)
            log = directory / (name + ".log")
            with log.open("x") as stream:
                stream.write(result.stdout + result.stderr + f"\nreturncode={result.returncode}\n")
            receipt = target / "negative.json"
            terminal = json.loads(receipt.read_bytes())
            assert result.returncode == 2 and terminal["evidence_verdict"] == "FAIL"
            assert terminal["stage"] == "audit" and not terminal["accept_sparse_contract"]
            results[name] = {"returncode": result.returncode, "receipt": str(receipt),
                             "receipt_sha256": sha(receipt), "log": str(log),
                             "log_sha256": sha(log), "solver_calls": 0,
                             "first_cause": terminal["first_cause"]}
    after = source_snapshot(ROOT)
    consistency = check_source_consistency(before, after)
    assert consistency["verdict"] == "PASS"
    path = directory / "receipt.json"
    write_json(path, {"schema_version": 2, "verdict": "PASS",
                      "checked_at": datetime.now(timezone.utc).isoformat(),
                      "source": before, "source_consistency": consistency,
                      "controls": results, "solver_calls": 0})
    print(f"SPARSE_ENTRY_CONTROLS verdict=PASS controls={len(results)} receipt={path}", flush=True)
    return 0


def describe_source(owner_root=None):
    snapshot = source_snapshot(ROOT)
    value = {"reviewer_root": str(ROOT), "reviewer_commit": snapshot["commit"],
             "reviewer_source_sha256": source_pin(snapshot), "clean": snapshot["clean"],
             "math": immutable_math()}
    if owner_root is not None:
        owner_root = Path(owner_root).resolve(strict=True)
        owner = source_snapshot(owner_root)
        committed_members(owner_root, owner)
        value.update(owner_root=str(owner_root), owner_commit=owner["commit"],
                     owner_source_sha256=source_pin(owner),
                     operator_sha256=owner["sha256"]["src/openopt/operators.py"])
    print(json.dumps(value, sort_keys=True, indent=2), flush=True)
    return 0


def diagnose_historical(run_directory, output):
    """Read old red bytes, never replay or promote a schema-v1 failed run."""
    run_directory = external_directory(run_directory, existing=True)
    output = external_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="history-", dir=output))
    execution_path = run_directory / "execution.json"
    execution = json.loads(execution_path.read_bytes())
    verification_path = Path(execution["verification_path"])
    assert verification_path.is_relative_to(run_directory)
    verification = json.loads(verification_path.read_bytes())
    junit = verification_path.parent / "pytest.xml"
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    collection_path = run_directory / "collection.json"
    nodes = json.loads(collection_path.read_bytes())["nodeids"]
    actual = [c.get("classname").replace(".", "/") + ".py::" + c.get("name") for c in cases]
    membership(nodes, actual)
    counts = {"tests": len(cases), **{
        key: sum(c.find(tag) is not None for c in cases)
        for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))
    }}
    assert counts == {"tests": 52, "failures": 1, "errors": 0, "skipped": 0}
    failed = [c.get("name") for c in cases if c.find("failure") is not None]
    assert failed == ["test_actual_source_and_cpu_thread_boundary"]
    assert execution["returncode"] == 1 and verification["verdict"] == "FAIL"
    mismatches = {name: {"historical": pin, "actual": execution["source_after"]["sha256"].get(name)}
                  for name, pin in PINS.items() if execution["source_after"]["sha256"].get(name) != pin}
    assert mismatches and not (run_directory / "receipt.json").exists()
    path = directory / "receipt.json"
    write_json(path, {"schema_version": 2, "checked_at": datetime.now(timezone.utc).isoformat(),
                      "evidence_verdict": "HISTORICAL_HARNESS_FAIL_RETAINED",
                      "review_verdict": "UNVERIFIED", "accept_sparse_contract": False,
                      "pytest_counts": counts, "pytest_returncode": execution["returncode"],
                      "failed_nodes": failed, "terminal_audit_returncode": "NOT_RECORDED",
                      "reported_first_cause": "historical reviewer PINS assertion; obsolete scope gate also incompatible",
                      "historical_pin_mismatches": mismatches,
                      "inputs": {str(p): sha(p) for p in (
                          execution_path, verification_path, junit, collection_path)},
                      "historical_fail_requalified": False, "solver_calls": 0})
    print(f"SPARSE_HISTORY verdict=FAIL_RETAINED receipt={path}", flush=True)
    return 0


def scan_source(artifact_root):
    """Existing authorized literal scanner, without printing credential values."""
    import importlib.util
    root = external_directory(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "sparse_safe_scanner", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(
        secret_hex=None, secrets_file=None, no_needle=False,
        needle_from=str(Path.home() / ".config/dsh-n600-token")))
    if not needles:
        raise ValueError("authorized credential scan needle unavailable")
    paths = [str(ROOT / TEST), str(ROOT / DOC)]
    literal, _ = scanner.scan_literal(paths, needles)
    pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                         r"(?i:password|token|secret)\s*[\"\x27]?\s*[:=]\s*[\"\x27][^\"\x27\s]{6,}[\"\x27]")
    shape = sum(bool(pattern.search(Path(path).read_text())) for path in paths)
    report = {"verdict": "PASS" if literal == shape == 0 else "FAIL",
              "literal_hits": literal, "shape_hits": shape, "files": {p: sha(p) for p in paths}}
    write_json(root / "source-security-scan.json", report)
    print(f"SPARSE_SCOPE_SCAN verdict={report['verdict']} literal={literal} shape={shape}", flush=True)
    return 0 if report["verdict"] == "PASS" else 1


def main_body(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit-artifacts", type=Path)
    mode.add_argument("--entry-controls", action="store_true")
    mode.add_argument("--describe-source", action="store_true")
    mode.add_argument("--historical-diagnosis", type=Path)
    mode.add_argument("--scan-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--receipt-name", default="replay.json")
    parser.add_argument("--owner-root", type=Path, help="explicit clean local owner worktree")
    parser.add_argument("--owner-commit", help="caller full 40-hex owner commit")
    parser.add_argument("--operator-sha256", help="caller full 64-hex operators source hash")
    parser.add_argument("--owner-source-sha256")
    parser.add_argument("--reviewer-commit")
    parser.add_argument("--reviewer-source-sha256")
    parser.add_argument("--scope-base", help="optional explicit development diff base, not an execution ancestry rule")
    parser.add_argument("--control-run", type=Path, help="new schema-v2 completed run for coordinated no-solve negatives")
    args = parser.parse_args(argv)
    if args.describe_source:
        return describe_source(args.owner_root)
    if args.historical_diagnosis:
        if args.artifact_dir is None:
            parser.error("historical diagnosis requires artifact-dir")
        return diagnose_historical(args.historical_diagnosis, args.artifact_dir)
    if args.scan_only:
        return scan_source(args.artifact_dir)
    if args.entry_controls:
        if args.artifact_dir is None:
            parser.error("--entry-controls requires --artifact-dir")
        return entry_controls(args.artifact_dir, args.control_run, args.development)
    if args.run:
        if args.artifact_dir is None:
            parser.error("--run requires --artifact-dir")
        return run(args.artifact_dir, args.development, args.scope_base)
    if Path(args.receipt_name).name != args.receipt_name or not args.receipt_name.endswith(".json"):
        parser.error("receipt-name must be a simple new JSON filename")
    receipt = audit_artifacts(args.audit_artifacts, args.receipt_name, require_clean=not args.development)
    return receipt["returncode"]


def main(argv=None):
    try:
        return main_body(argv)
    except Exception as error:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--artifact-dir", type=Path)
        parser.add_argument("--audit-artifacts", type=Path)
        options, _ = parser.parse_known_args(argv)
        selected = options.artifact_dir or options.audit_artifacts
        if selected is None:
            print(f"SPARSE_ENTRY evidence=FAIL cause={type(error).__name__}: {error}", flush=True)
            return 2
        try:
            root = external_directory(selected)
        except Exception:
            root = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/sparse-portable-repair/entry-failures").resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="command-failure-", dir=root))
        terminal_failure(directory, error, "entry")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
