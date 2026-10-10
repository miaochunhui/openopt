"""Frozen original-model numerical evidence, not a solver or speed benchmark.

run requires an explicit one-card RTX PRO 4500 CUDA placement. cpu-smoke is
deliberately a different scope. seal/replay are CPU-only and never run a solver,
test factory or source snapshot. Caller pins, full source membership and raw
array bytes are mandatory; a seal is integrity binding, not execution acceptance.
CUDA runs require independently pre-pinned models. Only an external coordinator
audit can qualify execution; numerical replay always leaves it UNVERIFIED.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
from uuid import UUID

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ARTIFACT_ROOT = Path("/hf3fs-jd/data/tmp").resolve()
RUNNER = "scripts/gpu_release_proof.py"
GPU_SCOPE = "pre-pinned original-model numerical replay; CUDA execution UNVERIFIED"
CPU_SCOPE = "CPU smoke only; GPU UNVERIFIED"
NUMERICAL_PASS = "SEMANTIC_NUMERICAL_PASS"
FIELDS = ("P", "q", "A", "l", "u")
CASES = (
    "scaling_basic", "scaling_ill_conditioned", "jacobi_coo", "jacobi_csr",
    "gap_spd", "gap_lp", "gap_singular", "infeasible", "unbounded",
)


class ProofError(ValueError):
    """A rejected piece of evidence with its first coordinate."""


def require(condition, coordinate, reason):
    if not condition:
        raise ProofError(f"{coordinate}: {reason}")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def parse(data, coordinate):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, coordinate, f"duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value):
        raise ProofError(f"{coordinate}: nonfinite JSON {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def pin(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "sha256", "caller-pinned SHA256 required")
    return value


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), path, "regular file required")
    data = path.read_bytes()
    require(data, path, "empty evidence")
    return data


def external(path):
    path = Path(path)
    require(path.is_absolute(), path, "absolute 3FS artifact path required")
    require(not path.is_symlink(), path, "artifact symlink rejected")
    resolved = path.resolve()
    require(resolved.is_relative_to(ARTIFACT_ROOT), path, "outside shared 3FS tmp")
    return resolved


def relative(value):
    require(isinstance(value, str) and value, "path", "nonempty relative path required")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value,
            value, "unsafe relative path")
    return value


def at(root, name):
    root = Path(root).resolve()
    path = root / relative(name)
    require(path.resolve().is_relative_to(root), path, "path escapes evidence root")
    require(not any(p.is_symlink() for p in (path, *path.parents) if p != root.parent),
            path, "symlink evidence component rejected")
    return path


def write(path, value):
    with Path(path).open("xb") as output:
        output.write(encode(value))


def descriptor(root, path):
    data = read(path)
    return {"path": Path(path).relative_to(root).as_posix(),
            "sha256": sha(data), "bytes": len(data)}


def bound(root, item):
    require(isinstance(item, dict) and set(item) == {"path", "sha256", "bytes"},
            "descriptor", "exact path/hash/bytes descriptor required")
    path = at(root, item["path"])
    data = read(path)
    require(type(item["bytes"]) is int and item["bytes"] == len(data),
            path, "byte count mutation")
    require(sha(data) == pin(item["sha256"]), path, "byte hash mutation")
    return path


def source_map(root):
    root = Path(root).resolve()
    paths = sorted((root / "src").rglob("*.py")) + [root / RUNNER]
    require(paths and (root / RUNNER).is_file(), root, "producer source missing")
    result = {}
    for path in paths:
        require(not path.is_symlink() and path.resolve().is_relative_to(root),
                path, "source symlink or escape")
        result[path.relative_to(root).as_posix()] = sha(read(path))
    require(all(f"src/openopt/{name}.py" in result
                for name in ("qp", "operators", "scaling", "certificates")),
            root, "required core source missing")
    return result


def same_sources(expected, actual, coordinate):
    delta = {"missing": sorted(set(expected) - set(actual)),
             "extra": sorted(set(actual) - set(expected)),
             "changed": sorted(k for k in set(expected) & set(actual)
                               if expected[k] != actual[k])}
    require(not any(delta.values()), coordinate, f"bidirectional source difference {delta}")


def freeze_sources(root, output):
    sources = source_map(root)
    for name, expected in sources.items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = read(Path(root) / name)
        require(sha(data) == expected, name, "source changed while snapshotting")
        with target.open("xb") as stream:
            stream.write(data)
    return sources


def imported_source_check():
    for name, module in sorted(sys.modules.items()):
        if name == "openopt" or name.startswith("openopt."):
            path = getattr(module, "__file__", None)
            require(path and Path(path).resolve().is_relative_to(ROOT / "src/openopt"),
                    name, "executed module is outside the frozen source root")


def save_array(root, label, array):
    array = np.ascontiguousarray(array)
    require(array.dtype.str in ("<f4", "<f8", "<i8"), label, "unsupported raw dtype")
    path = root / "arrays" / (label + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    return descriptor(root, path)


def save_tensor(root, label, tensor):
    import torch

    require(isinstance(tensor, torch.Tensor), label, "real observed tensor required")
    tensor = tensor.detach()
    record = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
              "device": str(tensor.device), "layout": str(tensor.layout), "components": {}}
    if tensor.layout == torch.strided:
        pieces = {"values": tensor}
    elif tensor.layout == torch.sparse_coo:
        coo = tensor.coalesce()
        pieces = {"indices": coo.indices(), "values": coo.values()}
    elif tensor.layout == torch.sparse_csr:
        pieces = {"crow_indices": tensor.crow_indices(), "col_indices": tensor.col_indices(),
                  "values": tensor.values()}
    else:
        raise ProofError(f"{label}: unsupported tensor layout {tensor.layout}")
    for name, value in pieces.items():
        record["components"][name] = save_array(root, label + "-" + name, value.cpu().numpy())
    return record


def load_tensor(root, record, coordinate):
    require(isinstance(record, dict) and set(record) == {
        "shape", "dtype", "device", "layout", "components"}, coordinate, "incomplete tensor")
    shape = record["shape"]
    require(isinstance(shape, list) and len(shape) in (1, 2)
            and all(type(n) is int and n >= 0 for n in shape),
            coordinate, "invalid tensor dimensions")
    require(record["dtype"] in ("torch.float32", "torch.float64")
            and record["device"] in ("cpu", "cuda:0"), coordinate, "invalid dtype/device")
    layout = record["layout"]
    names = {
        "torch.strided": {"values"},
        "torch.sparse_coo": {"indices", "values"},
        "torch.sparse_csr": {"crow_indices", "col_indices", "values"},
    }
    require(layout in names and set(record["components"]) == names[layout],
            coordinate, "unknown/incomplete sparse storage")
    arrays = {}
    for name, item in record["components"].items():
        path = bound(root, item)
        with path.open("rb") as stream:
            arrays[name] = np.load(stream, allow_pickle=False)
            require(stream.read() == b"", path, "trailing bytes after NPY array")
        wanted = "<i8" if name != "values" else (
            "<f8" if record["dtype"] == "torch.float64" else "<f4")
        require(arrays[name].dtype.str == wanted, coordinate, "raw dtype differs from metadata")
    values = arrays["values"]
    require(not np.isnan(values).any(), coordinate, "NaN raw data")
    if layout == "torch.strided":
        require(values.shape == tuple(shape), coordinate, "dense raw shape mismatch")
        return values
    require(len(shape) == 2 and values.ndim == 1 and np.isfinite(values).all(),
            coordinate, "invalid sparse values")
    rows, columns = shape
    if layout == "torch.sparse_coo":
        indices = arrays["indices"]
        require(indices.shape == (2, values.size), coordinate, "invalid COO index dimensions")
        row, col = indices
    else:
        crow, col = arrays["crow_indices"], arrays["col_indices"]
        require(crow.shape == (rows + 1,) and col.shape == values.shape
                and crow[0] == 0 and crow[-1] == values.size
                and np.all(np.diff(crow) >= 0), coordinate, "invalid CSR row pointers")
        row = np.repeat(np.arange(rows), np.diff(crow))
    require(np.all((row >= 0) & (row < rows) & (col >= 0) & (col < columns)),
            coordinate, "sparse index out of range")
    keys = row * max(columns, 1) + col
    require(np.all(np.diff(keys) > 0), coordinate, "noncanonical/duplicate sparse indices")
    # Host-only spectral replay for these tiny frozen models, not GPU-core densification.
    dense = np.zeros(shape, dtype=values.dtype)
    dense[row, col] = values
    return dense


def tensor_hash(record):
    value = {k: v for k, v in record.items() if k != "device"}
    value["components"] = {k: {"sha256": v["sha256"], "bytes": v["bytes"]}
                           for k, v in value["components"].items()}
    return sha(encode(value))


def model_hash(model):
    return sha(encode({name: tensor_hash(model[name]) for name in FIELDS}))


def load_expectations(path, expected_sha256, current_root):
    """Read caller-trusted pre-dispatch arrays, never reconstruct a model."""
    path = external(path)
    data = read(path)
    require(sha(data) == pin(expected_sha256), path, "caller expectations pin mismatch")
    saved = parse(data, path)
    require(isinstance(saved, dict) and set(saved) == {
        "schema_version", "source_sha256", "cases",
    } and type(saved["schema_version"]) is int and saved["schema_version"] == 1,
            path, "unknown/incomplete pre-dispatch expectations")
    cases = saved["cases"]
    require(isinstance(cases, dict) and cases and set(cases) <= set(CASES),
            path, "expectations case denominator mismatch")
    require(isinstance(saved["source_sha256"], dict) and saved["source_sha256"],
            path, "expectations source membership missing")
    same_sources(saved["source_sha256"], source_map(current_root), "expectations/current source")
    for case, entry in saved["cases"].items():
        require(isinstance(entry, dict) and set(entry) == {"model", "model_sha256"}
                and isinstance(entry["model"], dict) and set(entry["model"]) == set(FIELDS)
                and pin(entry["model_sha256"]) == model_hash(entry["model"]),
                case, "pre-dispatch model descriptor/digest mutation")
        for name, record in entry["model"].items():
            require(record["device"] == "cpu" and record["dtype"] == "torch.float64",
                    case + "/" + name, "expectations must be reviewed CPU float64 models")
            load_tensor(path.parent, record, case + "/expected/" + name)
    return saved, {"path": str(path), "sha256": expected_sha256, "bytes": len(data)}


def match_expected_model(model, expected, case):
    require(model_hash(model) == expected["model_sha256"]
            and all(tensor_hash(model[name]) == tensor_hash(expected["model"][name])
                    for name in FIELDS),
            case, "original model differs from caller pre-dispatch expectation")


def policy(case, device):
    from openopt.operators import OperatorQPConfig
    from openopt.qp import QPConfig
    from openopt.scaling import ScalingConfig

    require(case in CASES, case, "unknown case")
    if case.startswith("jacobi_"):
        config = OperatorQPConfig(
            device=device, eps_abs=1e-9, eps_rel=1e-9, cg_eps_abs=1e-12,
            cg_eps_rel=1e-12, cg_preconditioner="jacobi",
        )
        return config, None, config.eps_abs, config.eps_rel
    config = QPConfig(
        device=device, dual_gap_termination=case.startswith("gap_"),
        direction_termination=case in ("infeasible", "unbounded"),
        max_iterations=2000,
    )
    scaling = None
    if case.startswith("scaling_"):
        ill = case.endswith("ill_conditioned")
        config = QPConfig(
            device=device, eps_abs=1e-10, eps_rel=1e-10, max_iterations=20000,
            adaptive_rho=ill, adaptive_rho_interval=25,
            adaptive_rho_tolerance=2., rho_min=1e-10,
        )
        scaling = ScalingConfig(iterations=30, check_atol=1e-7, check_rtol=1e-7)
    atol, rtol = config.eps_abs, config.eps_rel
    if case in ("infeasible", "unbounded"):
        atol, rtol = config.direction_atol, config.direction_rtol
    if scaling is not None:
        atol, rtol = scaling.check_atol, scaling.check_rtol
    return config, scaling, atol, rtol


def config_dict(config):
    value = asdict(config)
    value["dtype"] = str(value["dtype"])
    return value


def make_problem(case):
    """Explicit synthetic coverage models. No supplied optimum, ray or base."""
    import torch
    from openopt.qp import QPProblem

    if case == "scaling_ill_conditioned":
        values = (
            [[4e-6, 1.], [1., 2e6]], [-.0065, 1000.],
            [[2., 0.], [0., 3.], [.001, 1000.], [0., 0.]],
            [-np.inf, -.0025, .5, -np.inf], [2000., np.inf, .5, np.inf],
        )
    elif case == "scaling_basic":
        values = (
            [[4., 1.], [1., 2.]], [-6.5, 1.],
            [[2., 0.], [0., 3.], [1., 1.], [0., 0.]],
            [-np.inf, -2.5, .5, -np.inf], [2., np.inf, .5, np.inf],
        )
    elif case.startswith("jacobi_"):
        values = ([[4., 1.], [1., 3.]], [-1., 2.], np.eye(2), [-1., -1.], [1., 1.])
    elif case == "gap_spd":
        n = 24
        P = np.diag(np.arange(1., n + 1.))
        P += np.diag(np.full(n - 1, .1), 1) + np.diag(np.full(n - 1, .1), -1)
        values = (P, np.sin(np.arange(n)) * 3, np.eye(n), -np.ones(n), np.ones(n))
    elif case == "gap_lp":
        values = (np.zeros((3, 3)), [1., -2., .5], np.eye(3), [0., 0., -1.], [2., 3., 1.])
    elif case == "gap_singular":
        values = (np.diag([2., 0., 4.]), [-1., -2., 1.], np.eye(3),
                  [-1., -1., -1.], [1., 1., 1.])
    elif case == "infeasible":
        values = (np.eye(2), [0., 0.], [[1., 0.], [1., 0.], [0., 1.]],
                  [0., 2., -1.], [1., 3., 1.])
    elif case == "unbounded":
        values = (np.diag([1., 0.]), [0., -1.],
                  [[1., 0.], [0., 1.], [0., -1.], [0., 2.]],
                  [0., 2., -np.inf, -np.inf], [0., np.inf, -2., np.inf])
    else:
        raise ProofError(f"{case}: unknown model")
    return QPProblem(*[torch.tensor(v, dtype=torch.float64) for v in values])


def norm(value):
    return float(np.max(np.abs(value))) if value.size else 0.


def original_checks(problem, vectors, status, atol, rtol):
    """Independent NumPy equations; no solver/checker status or factor is reused."""
    require(type(atol) is float and type(rtol) is float
            and 0 < atol <= 1e-7 and 0 <= rtol <= 1e-7,
            "tolerances", "invalid or overbroad original-coordinate gate")
    P, q, A, l, u = (np.asarray(problem[k], dtype=np.float64) for k in FIELDS)
    n = q.size
    require(q.shape == (n,) and n > 0 and P.shape == (n, n)
            and A.ndim == 2 and A.shape[1] == n and l.shape == u.shape == (A.shape[0],),
            "model", "invalid original dimensions")
    require(all(np.isfinite(v).all() for v in (P, q, A))
            and not np.isnan(l).any() and not np.isnan(u).any()
            and not np.isposinf(l).any() and not np.isneginf(u).any() and np.all(l <= u),
            "model", "invalid original values/bounds")
    with np.errstate(all="raise"):
        require(norm(P - P.T) <= 64 * np.finfo(float).eps * n * norm(P),
                "model", "asymmetric original curvature")
        eig, V = np.linalg.eigh(.5 * P + .5 * P.T)
        spectral = 64 * np.finfo(float).eps * n * norm(eig)
        require(eig[0] >= -spectral and np.all((eig == 0) | (eig > spectral)),
                "model", "negative or unresolved original PSD spectrum")

        def vector(name, size):
            result = np.asarray(vectors[name], dtype=np.float64)
            require(result.shape == (size,) and np.isfinite(result).all(),
                    name, "invalid original vector")
            return result

        def feasibility(x):
            Ax = A @ x
            lower, upper = np.isfinite(l), np.isfinite(u)
            violation = np.r_[np.maximum(l[lower] - Ax[lower], 0),
                              np.maximum(Ax[upper] - u[upper], 0)]
            limits = atol + rtol * np.r_[np.maximum(abs(l[lower]), abs(Ax[lower])),
                                        np.maximum(abs(u[upper]), abs(Ax[upper]))]
            return Ax, norm(violation), bool(np.all(violation <= limits))

        def support(y):
            pos, neg = y > 0, y < 0
            require(np.isfinite(u[pos]).all() and np.isfinite(l[neg]).all(),
                    "support", "multiplier/direction has infinite box support")
            terms = np.r_[u[pos] * y[pos], l[neg] * y[neg]]
            return float(np.sum(terms)), terms

        x, y = vector("x", n), vector("y", l.size)
        Ax, primal_residual, feasible = feasibility(x)
        if status == "infeasible":
            d = vector("infeasibility_direction", l.size)
            require(norm(d) > 0, "ray", "zero infeasibility direction")
            d = d / norm(d)
            box, terms = support(d)
            residual = norm(A.T @ d)
            checks = {"nonzero_direction": True, "annihilation": residual <= atol,
                      "bound_sign": True,
                      "strict_separation": box < -(atol + rtol * np.sum(abs(terms)))}
            metrics = {"annihilation": residual, "box_support": box}
        elif status == "unbounded":
            d = vector("unbounded_direction", n)
            require(norm(d) > 0 and "feasible_base" in vectors,
                    "ray", "nonzero generated ray and actual feasible base required")
            d = d / norm(d)
            base = vector("feasible_base", n)
            _, base_residual, base_feasible = feasibility(base)
            Ad, Pd = A @ d, P @ d
            lo, hi = np.isfinite(l), np.isfinite(u)
            both = lo & hi
            positive_component = norm((V.T @ d)[eig > 0])
            slope = float(q @ d)
            checks = {
                "nonzero_direction": True,
                "zero_curvature": norm(Pd) <= atol
                and positive_component <= 64 * np.finfo(float).eps * n,
                "recession": bool(np.all(abs(Ad[both]) <= atol)
                                  and np.all(Ad[lo & ~hi] >= 0)
                                  and np.all(Ad[hi & ~lo] <= 0)),
                "strict_descent": slope < -(atol + rtol * np.sum(abs(q * d))),
                "feasible_base": base_feasible,
            }
            metrics = {"zero_curvature": norm(Pd), "positive_eigenspace": positive_component,
                       "descent_slope": slope, "base_feasibility": base_residual}
        else:
            require(status == "solved", "status", "unqualified solver status")
            Px, ATy = P @ x, A.T @ y
            stationarity = Px + q + ATy
            stationarity_limits = atol + rtol * np.maximum.reduce([abs(Px), abs(q), abs(ATy)])
            box, _ = support(y)
            pos, neg = y > 0, y < 0
            comp = np.r_[y[pos] * (u[pos] - Ax[pos]), y[neg] * (l[neg] - Ax[neg])]
            comp_limits = atol + rtol * np.r_[
                np.maximum(abs(y[pos] * u[pos]), abs(y[pos] * Ax[pos])),
                np.maximum(abs(y[neg] * l[neg]), abs(y[neg] * Ax[neg]))]
            b = q + ATy
            coords = V.T @ b
            inverse = np.zeros_like(coords)
            inverse[eig > 0] = coords[eig > 0] / eig[eig > 0]
            inverse = V @ inverse
            range_residual = norm(b - P @ inverse)
            primal = float(.5 * x @ Px + q @ x)
            dual = float(-.5 * b @ inverse - box)
            gap = primal - dual
            gap_limit = atol + rtol * max(abs(primal), abs(dual))
            checks = {
                "primal_feasibility": feasible,
                "stationarity": bool(np.all(abs(stationarity) <= stationarity_limits)),
                "bound_sign": True,
                "complementarity": bool(np.all(abs(comp) <= comp_limits)),
                "dual_objective_available": range_residual <= atol + rtol * norm(b),
                "duality_gap": abs(gap) <= gap_limit,
            }
            metrics = {"primal_feasibility": primal_residual, "stationarity": norm(stationarity),
                       "complementarity": norm(comp), "primal_objective": primal,
                       "dual_objective": dual, "duality_gap": gap,
                       "range_residual": range_residual}
        require(all(math.isfinite(float(v)) for v in metrics.values()),
                "original checks", "nonfinite independent arithmetic")
        return {"passed": all(checks.values()),
                "checks": {key: bool(value) for key, value in checks.items()}, "metrics": metrics}


def placement(tensor, device, coordinate, layout=None):
    import torch

    require(tensor.device == torch.device(device) and tensor.dtype == torch.float64,
            coordinate, "observed placement/dtype differs from explicit request")
    if layout is not None:
        require(tensor.layout == layout, coordinate, "sparse layout changed")
    return {"device": str(tensor.device), "dtype": str(tensor.dtype),
            "layout": str(tensor.layout)}


@contextmanager
def observe_workspace(device, observations, root, label, models):
    import openopt.qp as qp

    original = qp.prepare_qp

    def observed(*args, **kwargs):
        workspace = original(*args, **kwargs)
        observations.update({name: placement(getattr(workspace, name), device, name)
                             for name in (*FIELDS, "factor")})
        models.update({name: save_tensor(root, label + "-resident-" + name, getattr(workspace, name))
                       for name in FIELDS})
        return workspace

    qp.prepare_qp = observed
    try:
        yield
    finally:
        qp.prepare_qp = original


def trace_termination(workspace):
    """Observe unchanged checkpoint arguments/results, not substitute witnesses."""
    import torch

    original = workspace._termination_check
    previous_x = torch.zeros_like(workspace.q)
    previous_y = torch.zeros_like(workspace.l)
    trace = {}
    current = {}

    def observed(kind, iteration, rho, *vectors):
        nonlocal previous_x, previous_y
        # The unchanged solve method owns the actual checkpoint/iterate locals.
        # Capture them, rather than reconstruct y via previous_y + dy rounding.
        frame = sys._getframe(1)
        require(frame.f_locals.get("self") is workspace, kind, "unexpected checkpoint caller")
        evidence = original(kind, iteration, rho, *vectors)
        if kind == "infeasible":
            current["dy"] = vectors[0].detach().clone()
            current["infeasible_evidence"] = evidence.to_dict()
        if kind == "unbounded":
            dx, x = vectors
            trace.clear()
            trace.update(iteration=iteration, rho=rho,
                         tensors={"previous_x": previous_x.clone(),
                                  "previous_y": previous_y.clone(),
                                  "current_x": x.detach().clone(),
                                  "current_y": frame.f_locals["y"].detach().clone(),
                                  "dx": dx.detach().clone(), "dy": current["dy"]},
                         infeasible_evidence=current["infeasible_evidence"],
                         unbounded_evidence=evidence.to_dict())
            previous_x = x.detach().clone()
            previous_y = frame.f_locals["y"].detach().clone()
        del frame
        return evidence

    workspace._termination_check = observed
    return trace


def execute_case(case, root, device, *, expected_model=None):
    import torch
    import openopt.operators as operators
    from openopt.certificates import (
        check_primal_infeasibility, check_qp_candidate, check_unbounded_direction,
    )
    from openopt.qp import prepare_qp
    from openopt.scaling import solve_scaled_qp

    config, scaling, atol, rtol = policy(case, device)
    problem = make_problem(case)
    if case.startswith("jacobi_"):
        kind = torch.sparse_coo if case == "jacobi_coo" else torch.sparse_csr
        matrices = [getattr(problem, k).to_sparse() for k in ("P", "A")]
        if kind == torch.sparse_csr:
            matrices = [v.to_sparse_csr() for v in matrices]
        problem = SimpleNamespace(P=matrices[0], q=problem.q, A=matrices[1],
                                  l=problem.l, u=problem.u)
    model = {name: save_tensor(root, case + "-original-" + name, getattr(problem, name))
             for name in FIELDS}
    if expected_model is not None:
        match_expected_model(model, expected_model, case)
    observed, extensions, resident_model = {}, {}, {}
    start = time.perf_counter()
    if case.startswith("jacobi_"):
        operator_problem = operators.OperatorQPProblem(
            problem.P, problem.q, problem.A, problem.l, problem.u, assume_psd=True)
        workspace = operators.prepare_operator_qp(operator_problem, config)
        observed.update({name: placement(getattr(workspace, name), device, name)
                         for name in ("q", "l", "u")})
        for name in ("P", "A"):
            matrix = getattr(workspace, name).operator.matrix
            observed[name] = placement(matrix, device, name, kind)
            resident_model[name] = save_tensor(root, case + "-resident-" + name, matrix)
        for name in ("q", "l", "u"):
            resident_model[name] = save_tensor(
                root, case + "-resident-" + name, getattr(workspace, name))
        diagonal = workspace.preconditioner.diagonal
        observed["jacobi"] = placement(diagonal, device, "jacobi")
        extensions["jacobi_diagonal"] = save_tensor(root, case + "-jacobi", diagonal)
        inner_calls = []
        original_cg = operators.conjugate_gradient

        def recorded_cg(system, rhs, **kwargs):
            label = case + f"-cg-{len(inner_calls):04d}"
            rhs_owned = rhs.detach().clone()
            initial = kwargs["initial"].detach().clone()
            frame = sys._getframe(1)
            require(frame.f_locals.get("self") is workspace, label, "unexpected CG caller")
            outer_y = frame.f_locals["y"].detach().clone()
            outer_z = frame.f_locals["z"].detach().clone()
            outer_iteration = frame.f_locals["iteration"]
            del frame
            inner = original_cg(system, rhs, **kwargs)
            placement(inner.x, device, label)
            residual = float(torch.linalg.vector_norm(rhs_owned - system.matvec(inner.x)))
            require(residual == inner.residual and residual <= inner.tolerance,
                    label, "actual unpreconditioned residual rejected")
            inner_calls.append({
                "rhs": save_tensor(root, label + "-rhs", rhs_owned),
                "initial": save_tensor(root, label + "-initial", initial),
                "x": save_tensor(root, label + "-x", inner.x),
                "outer_y": save_tensor(root, label + "-outer-y", outer_y),
                "outer_z": save_tensor(root, label + "-outer-z", outer_z),
                "outer_iteration": outer_iteration,
                "iterations": inner.iterations, "residual": inner.residual,
                "tolerance": inner.tolerance, "observed_device": str(inner.x.device),
            })
            return inner

        operators.conjugate_gradient = recorded_cg
        try:
            result = workspace.solve()
        finally:
            operators.conjugate_gradient = original_cg
        extensions.update(inner_calls=inner_calls, system={"rho": config.rho, "sigma": config.sigma})
    elif scaling is not None:
        with observe_workspace(device, observed, root, case, resident_model):
            result = solve_scaled_qp(problem, config, scaling)
        extensions["scaling"] = {
            "D": save_tensor(root, case + "-D", result.scaling.variable_scale),
            "E": save_tensor(root, case + "-E", result.scaling.constraint_scale),
            "c": result.scaling.objective_scale,
            "x_hat": save_tensor(root, case + "-xhat", result.scaled_result.x),
            "y_hat": save_tensor(root, case + "-yhat", result.scaled_result.y),
            "raw_status": result.scaled_result.status,
        }
    else:
        workspace = prepare_qp(problem, config)
        observed.update({name: placement(getattr(workspace, name), device, name)
                         for name in (*FIELDS, "factor")})
        resident_model.update({
            name: save_tensor(root, case + "-resident-" + name, getattr(workspace, name))
            for name in FIELDS})
        trace = trace_termination(workspace) if config.direction_termination else None
        result = workspace.solve()
        if trace:
            extensions["checkpoint"] = {
                **{key: value for key, value in trace.items() if key != "tensors"},
                "tensors": {key: save_tensor(root, case + "-checkpoint-" + key, tensor)
                            for key, tensor in trace["tensors"].items()},
            }
    if device == "cuda:0":
        torch.cuda.synchronize(0)
    observed.update({name: placement(getattr(result, name), device, name) for name in ("x", "y")})
    expected = case if case in ("infeasible", "unbounded") else "solved"
    vectors = {name: save_tensor(root, case + "-" + name, getattr(result, name))
               for name in ("x", "y")}
    for name in ("infeasibility_direction", "unbounded_direction", "feasible_base"):
        value = getattr(result, name, None)
        if value is not None:
            vectors[name] = save_tensor(root, case + "-" + name, value)
    host_problem = SimpleNamespace(**{name: load_tensor(root, model[name], name) for name in FIELDS})
    host_vectors = {name: load_tensor(root, value, name) for name, value in vectors.items()}
    if result.status == "infeasible":
        independent = check_primal_infeasibility(
            host_problem, host_vectors["infeasibility_direction"], atol=atol, rtol=rtol)
    elif result.status == "unbounded":
        independent = check_unbounded_direction(
            host_problem, host_vectors["unbounded_direction"],
            feasible_x=host_vectors["feasible_base"], atol=atol, rtol=rtol)
    else:
        independent = check_qp_candidate(
            host_problem, host_vectors["x"], host_vectors["y"], atol=atol, rtol=rtol)
    second = original_checks(vars(host_problem), host_vectors, result.status, atol, rtol)
    record = {
        "case": case, "status": result.status, "expected_status": expected,
        "dtype": "torch.float64", "device": device, "config": config_dict(config),
        "scaling_config": asdict(scaling) if scaling is not None else None,
        "atol": atol, "rtol": rtol, "model": model, "model_sha256": model_hash(model),
        "resident_model": resident_model,
        "vectors": vectors, "observed": observed, "extensions": extensions,
        "iterations": result.iterations, "objective": result.objective,
        "independent_original": independent.to_dict(), "numpy_original": second,
        "wall_s": time.perf_counter() - start,
        "qualification": "synthetic numerical coverage only; no exact/commercial/speed claim",
    }
    # Preserve a negative solve before fail-fast; never drop it from the run.
    write(root / (case + ".json"), record)
    require(result.status == expected and independent.passed and second["passed"],
            case, f"original-coordinate qualification rejected status={result.status}")
    return descriptor(root, root / (case + ".json"))


def cpu_stat(*, process_only=False):
    if process_only:
        used = resource.getrusage(resource.RUSAGE_SELF)
        return {"usage_usec": int((used.ru_utime + used.ru_stime) * 1e6),
                "scope": "current process rusage; not Pod cgroup"}
    path = Path("/sys/fs/cgroup/cpu.stat")
    values = dict(line.split() for line in path.read_text().splitlines())
    require("usage_usec" in values, path, "CPU cgroup usage missing")
    return {key: int(value) for key, value in values.items()}


class Telemetry:
    """Actual runtime samples; all failed reads remain fatal, not zero defaults."""

    def __init__(self, gpu):
        self.gpu, self.samples = gpu, []
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        item = {"at": now(), "cpu": cpu_stat(process_only=not self.gpu)}
        if self.gpu:
            command = [
                "nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
            run = subprocess.run(command, capture_output=True, text=True, check=False)
            item.update(returncode=run.returncode, stdout=run.stdout.strip(), stderr=run.stderr.strip())
        self.samples.append(item)

    def loop(self):
        while not self.stop.wait(.1):
            try:
                self.sample()
            except Exception as error:
                self.samples.append({"at": now(), "error": f"{type(error).__name__}: {error}"})
                return

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.stop.set()
        self.thread.join()
        self.sample()


def hardware(device):
    import torch

    result = {
        "requested_device": device, "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__, "python": sys.version,
        "cpu_quota": None, "cpu_scope": "current process rusage; not Pod cgroup",
        "node": os.environ.get("OPENOPT_NODE"), "pod": os.environ.get("HOSTNAME"),
        "gpu": None, "gpu_count": 0, "cuda_capability": None, "gpu_uuid": None,
    }
    if device == "cuda:0":
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "hardware", "exactly one visible CUDA device required; no fallback")
        result.update(cpu_quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
                      cpu_scope="Pod cgroup v2")
        result.update(gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(),
                      cuda_capability=list(torch.cuda.get_device_capability(0)))
        uuid = torch.cuda.get_device_properties(0).uuid
        require(isinstance(uuid, str), "hardware", "CUDA device UUID unavailable")
        result["gpu_uuid"] = "GPU-" + str(UUID(uuid.removeprefix("GPU-")))
        require("RTX PRO 4500" in result["gpu"] and result["node"] and result["pod"],
                "hardware", "actual RTX PRO 4500, node and Pod identity required")
    return result


def telemetry_check(samples, device, *, gpu_uuid=None, gpu_name=None):
    require(isinstance(samples, list) and len(samples) >= 2, "telemetry", "runtime samples missing")
    for sample in samples:
        require("error" not in sample and type(sample.get("cpu", {}).get("usage_usec")) is int,
                "telemetry", "failed/absent CPU cgroup sample")
        if device == "cuda:0":
            require(type(sample.get("returncode")) is int and sample["returncode"] == 0,
                    "telemetry", "nvidia-smi failed")
            rows = sample["stdout"].splitlines()
            require(len(rows) == 1, "telemetry", "one-card identity required")
            values = [value.strip() for value in rows[0].split(",")]
            require(len(values) == 6 and values[0] == gpu_uuid and values[1] == gpu_name
                    and re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", values[0])
                    and "RTX PRO 4500" in values[1], "telemetry", "CUDA/telemetry card identity mismatch")
            utilization, memory, total = (float(v) for v in values[2:5])
            require(0 <= utilization <= 100 and 0 <= memory <= total and total > 0,
                    "telemetry", "invalid measured utilization/memory")
    require(samples[-1]["cpu"]["usage_usec"] >= samples[0]["cpu"]["usage_usec"],
            "telemetry", "cgroup usage moved backwards")


def run_cases(artifact_root, cases, device, *, cpu_smoke=False,
              expectations=None, expectations_sha256=None):
    require(device == ("cpu" if cpu_smoke else "cuda:0"),
            "device", "run explicitly requires cuda:0; CPU uses cpu-smoke only")
    require(cases and len(set(cases)) == len(cases) and set(cases) <= set(CASES),
            "cases", "empty, duplicate or unknown selection")
    parent = external(artifact_root)
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="cpu-smoke-" if cpu_smoke else "cuda-proof-", dir=parent))
    report = {"schema_version": 1, "verdict": "RUNNING", "scope": CPU_SCOPE if cpu_smoke else GPU_SCOPE,
              "started_at": now(), "device": device, "requested_cases": cases, "cases": {},
              "source_sha256": {}, "hardware": None, "expectations": None,
              "numerical_verdict": "RUNNING", "execution_verdict": "UNVERIFIED"}
    start = time.perf_counter()
    code, telemetry = 0, Telemetry(not cpu_smoke)
    try:
        report["source_sha256"] = freeze_sources(ROOT, output)
        require(cpu_smoke or expectations is not None, "expectations",
                "formal CUDA run requires caller pre-dispatch expectations")
        require((expectations is None) == (expectations_sha256 is None),
                "expectations", "path and caller pin must be supplied together")
        expected = None
        if expectations is not None:
            expected, report["expectations"] = load_expectations(
                expectations, expectations_sha256, ROOT)
            require(set(cases) <= set(expected["cases"]),
                    "expectations", "requested case absent from pre-dispatch expectations")
        report["hardware"] = hardware(device)
        if not cpu_smoke:
            runtime = {
                "device": device, "gpu_uuid": report["hardware"]["gpu_uuid"],
                "gpu_name": report["hardware"]["gpu"], "gpu_count": report["hardware"]["gpu_count"],
                "node": report["hardware"]["node"], "pod": report["hardware"]["pod"],
            }
            print("OPENOPT_GPU_PROOF_RUNTIME " + json.dumps(
                runtime, sort_keys=True, allow_nan=False), flush=True)
        with telemetry:
            for case in cases:
                report["cases"][case] = execute_case(
                    case, output, device,
                    expected_model=expected["cases"][case] if expected is not None else None)
                print(f"OPENOPT_GPU_PROOF_CASE case={case} completed_at={now()}", flush=True)
        telemetry_check(telemetry.samples, device,
                        gpu_uuid=report["hardware"]["gpu_uuid"], gpu_name=report["hardware"]["gpu"])
        if expected is not None:
            current_expected, current_descriptor = load_expectations(
                expectations, expectations_sha256, ROOT)
            require(expected == current_expected and report["expectations"] == current_descriptor,
                    "expectations", "pre-dispatch expectations changed during solve")
        same_sources(report["source_sha256"], source_map(ROOT), "sources after solves")
        imported_source_check()
        report.update(verdict="CPU_SMOKE_PASS" if cpu_smoke else NUMERICAL_PASS,
                      numerical_verdict="PASS")
    except Exception as error:
        code = 1
        report.update(verdict="FAIL", numerical_verdict="FAIL",
                      error=f"{type(error).__name__}: {error}")
        with (output / "failure.log").open("x") as stream:
            stream.write(traceback.format_exc())
    finally:
        write(output / "telemetry.json", telemetry.samples)
        report["telemetry"] = descriptor(output, output / "telemetry.json")
        report.update(finished_at=now(), wall_s=time.perf_counter() - start, returncode=code)
        write(output / "result.json", report)
        line = (f"OPENOPT_GPU_PROOF verdict={report['verdict']} at={report['finished_at']} "
                f"result={output / 'result.json'} result_sha256={sha(read(output / 'result.json'))}")
        with (output / "execution.log").open("x") as stream:
            stream.write(line + f"\nreturncode={code}\n")
        print(line, flush=True)
    require(code == 0, output / "failure.log", report.get("error", "run failed"))
    return output / "result.json"


def compare_summary(saved, actual, coordinate):
    require(set(saved) == set(actual) and saved["passed"] == actual["passed"]
            and saved["checks"] == actual["checks"]
            and set(saved["metrics"]) == set(actual["metrics"]),
            coordinate, "independent check mutation")
    for key, value in actual["metrics"].items():
        require(type(saved["metrics"][key]) in (float, int)
                and math.isclose(saved["metrics"][key], value, rel_tol=1e-12, abs_tol=1e-12),
                coordinate + "/" + key, "saved metric differs from recomputation")


def replay_case(root, item, device, *, expected_model=None):
    saved = parse(read(bound(root, item)), item["path"])
    case = saved["case"]
    require(case in CASES and saved["device"] == device and saved["dtype"] == "torch.float64",
            case, "case/dtype/device mismatch")
    # Config validation uses constructors only; no solver/device validation is invoked.
    config, scaling, atol, rtol = policy(case, device)
    require(saved["config"] == config_dict(config) and saved["atol"] == atol and saved["rtol"] == rtol
            and saved["scaling_config"] == (asdict(scaling) if scaling is not None else None),
            case, "frozen acceptance policy/config mutation")
    require(set(saved["model"]) == set(FIELDS)
            and saved["model_sha256"] == model_hash(saved["model"]),
            case, "original model hash mutation")
    if expected_model is not None:
        match_expected_model(saved["model"], expected_model, case)
    problem = {name: load_tensor(root, saved["model"][name], case + "/" + name) for name in FIELDS}
    require(set(saved["resident_model"]) == set(FIELDS), case, "resident model incomplete")
    resident = {name: load_tensor(root, saved["resident_model"][name], case + "/resident/" + name)
                for name in FIELDS}
    for name in FIELDS:
        require(saved["resident_model"][name]["device"] == device
                and saved["resident_model"][name]["dtype"] == "torch.float64",
                case + "/" + name, "resident model placement mutation")
        if scaling is None:
            require(tensor_hash(saved["resident_model"][name]) == tensor_hash(saved["model"][name]),
                    case + "/" + name, "actual resident model differs from original bytes")
    vectors = {name: load_tensor(root, tensor, case + "/" + name)
               for name, tensor in saved["vectors"].items()}
    expected = case if case in ("infeasible", "unbounded") else "solved"
    require(saved["status"] == saved["expected_status"] == expected
            and type(saved["iterations"]) is int and 0 < saved["iterations"] <= config.max_iterations,
            case, "failed/unknown/incomplete solve cannot be qualified")
    names = {"x", "y"}
    if case == "infeasible":
        names.add("infeasibility_direction")
    if case == "unbounded":
        names.update(("unbounded_direction", "feasible_base"))
    require(set(vectors) == names, case, "missing or silently added result vector")
    for name, record in saved["vectors"].items():
        wanted = device if name in ("x", "y") else "cpu"
        require(record["device"] == wanted and record["dtype"] == "torch.float64"
                and record["layout"] == "torch.strided", case + "/" + name,
                "result export placement mutation")
    observed_names = set(FIELDS) | {"x", "y"}
    observed_names.add("jacobi" if case.startswith("jacobi_") else "factor")
    require(set(saved["observed"]) == observed_names, case, "workspace placement scope mutation")
    for name, observed in saved["observed"].items():
        layout = ("torch.sparse_coo" if case == "jacobi_coo" else "torch.sparse_csr") \
            if case.startswith("jacobi_") and name in ("P", "A") else "torch.strided"
        require(observed == {"device": device, "dtype": "torch.float64", "layout": layout},
                case + "/observed/" + name, "actual workspace placement mutation")
    actual = original_checks(problem, vectors, expected, atol, rtol)
    require(actual["passed"], case, f"original numerical replay rejected {actual}")
    compare_summary(saved["numpy_original"], actual, case + "/numpy_original")
    independent = saved["independent_original"]
    require(independent["passed"] is True and independent["checks"] == actual["checks"],
            case, "independent checker disagrees with original replay")
    for name, value in actual["metrics"].items():
        require(name in independent["metrics"]
                and type(independent["metrics"][name]) in (float, int)
                and math.isclose(independent["metrics"][name], value, rel_tol=1e-12, abs_tol=1e-12),
                case + "/independent_original/" + name, "saved independent metric mutation")
    if expected == "solved":
        require(math.isclose(saved["objective"], actual["metrics"]["primal_objective"],
                             rel_tol=1e-12, abs_tol=1e-12),
                case, "objective differs from frozen original model")
    extension = saved["extensions"]
    if scaling is not None:
        require(set(extension) == {"scaling"}, case, "scaling extension mutation")
        value = extension["scaling"]
        D, E, xhat, yhat = (load_tensor(root, value[k], case + "/" + k)
                           for k in ("D", "E", "x_hat", "y_hat"))
        require(D.shape == problem["q"].shape and E.shape == problem["l"].shape
                and np.isfinite(D).all() and np.isfinite(E).all()
                and np.all(D > 0) and np.all(E > 0) and 0 < value["c"] < np.inf
                and value["raw_status"] == "solved", case, "invalid scaling factors/status")
        for key in ("D", "E", "x_hat", "y_hat"):
            require(value[key]["device"] == device, case, "scaling resident placement mutation")
        require(np.array_equal(vectors["x"], D * xhat)
                and np.array_equal(vectors["y"], (E * yhat) * (1 / value["c"])),
                case, "original x/y restoration differs from raw solve")
        scaled = {"P": ((problem["P"] * D[:, None]) * D[None, :]) * value["c"],
                  "q": (problem["q"] * D) * value["c"],
                  "A": (problem["A"] * D[None, :]) * E[:, None],
                  "l": problem["l"] * E, "u": problem["u"] * E}
        # Ruiz iterates multiple products, so only machine-arithmetic rounding
        # is allowed here, not a relaxation of the original numerical gates.
        for name in FIELDS:
            require(np.allclose(resident[name], scaled[name], atol=0,
                                rtol=256 * np.finfo(float).eps),
                    case + "/" + name, "actual scaled resident model differs from D/E/c")
    elif case.startswith("jacobi_"):
        require(set(extension) == {"jacobi_diagonal", "inner_calls", "system"},
                case, "PCG extension mutation")
        P, A = problem["P"], problem["A"]
        require(extension["system"] == {"rho": config.rho, "sigma": config.sigma},
                case, "matrix-free system configuration mutation")
        diagonal = load_tensor(root, extension["jacobi_diagonal"], case + "/jacobi")
        require(extension["jacobi_diagonal"]["device"] == device
                and np.array_equal(diagonal, np.diag(P) + config.rho * (A * A).sum(0) + config.sigma),
                case, "Jacobi diagonal differs from actual unshifted system")
        calls = extension["inner_calls"]
        require(isinstance(calls, list) and len(calls) == saved["iterations"] > 0,
                case, "incomplete inner solve trace")
        previous = np.zeros_like(problem["q"])
        maximum = 0.
        for index, call in enumerate(calls):
            rhs, initial, x = (load_tensor(root, call[k], case + f"/cg-{index}/" + k)
                               for k in ("rhs", "initial", "x"))
            outer_y, outer_z = (load_tensor(root, call[k], case + f"/cg-{index}/" + k)
                                for k in ("outer_y", "outer_z"))
            require(all(v.shape == problem["q"].shape and np.isfinite(v).all()
                        for v in (rhs, initial, x))
                    and np.array_equal(initial, previous), case, "inner warm-start trace mutation")
            require(outer_y.shape == outer_z.shape == problem["l"].shape
                    and np.isfinite(outer_y).all() and np.isfinite(outer_z).all()
                    and call["outer_iteration"] == index + 1, case, "invalid outer CG coordinates")
            expected_rhs = config.sigma * initial - problem["q"] + A.T @ (
                config.rho * outer_z - outer_y)
            require(np.allclose(rhs, expected_rhs, atol=0, rtol=64 * np.finfo(float).eps),
                    case, "saved CG rhs differs from actual outer input")
            require(call["observed_device"] == device and all(call[k]["device"] == device
                    for k in ("rhs", "initial", "x", "outer_y", "outer_z")),
                    case, "CG placement mutation")
            residual = float(np.linalg.norm(rhs - (P @ x + config.rho * A.T @ (A @ x)
                                                  + config.sigma * x)))
            tolerance = config.cg_eps_abs + config.cg_eps_rel * float(np.linalg.norm(rhs))
            require(type(call["iterations"]) is int and 0 <= call["iterations"] <= config.cg_max_iterations
                    and math.isclose(call["tolerance"], tolerance, rel_tol=1e-14, abs_tol=1e-25)
                    and math.isfinite(call["residual"]) and 0 <= call["residual"] <= call["tolerance"]
                    and residual <= tolerance, case + f"/cg-{index}",
                    "true unpreconditioned original-system residual rejected")
            maximum, previous = max(maximum, residual), x
        require(np.array_equal(previous, vectors["x"]), case, "last CG differs from solve output")
        actual["inner_calls"] = len(calls)
        actual["maximum_inner_residual"] = maximum
    elif case in ("infeasible", "unbounded"):
        require(set(extension) == {"checkpoint"}, case, "generated-direction trace missing")
        trace = extension["checkpoint"]
        require(type(trace["iteration"]) is int and trace["iteration"] == saved["iterations"]
                and (trace["iteration"] % config.check_interval == 0
                     or trace["iteration"] == config.max_iterations),
                case, "terminal checkpoint is outside frozen check interval")
        require(not config.adaptive_rho and type(trace["rho"]) is float
                and math.isfinite(trace["rho"]) and trace["rho"] == config.rho,
                case, "terminal checkpoint nonadaptive rho mutation")
        tensors = trace["tensors"]
        require(set(tensors) == {"previous_x", "previous_y", "current_x", "current_y", "dx", "dy"},
                case, "incomplete generated checkpoint vectors")
        data = {key: load_tensor(root, value, case + "/checkpoint/" + key)
                for key, value in tensors.items()}
        require(all(v.shape == (problem["q"].shape if key.endswith("x") or key == "dx"
                                else problem["l"].shape) and np.isfinite(v).all()
                    for key, v in data.items()), case, "invalid checkpoint shape/values")
        require(all(value["device"] == device for value in tensors.values()),
                case, "checkpoint computation placement mutation")
        require(np.array_equal(data["dx"], data["current_x"] - data["previous_x"])
                and np.array_equal(data["dy"], data["current_y"] - data["previous_y"])
                and np.array_equal(data["current_x"], vectors["x"])
                and np.array_equal(data["current_y"], vectors["y"]),
                case, "ray is not tied to actual returned checkpoint")
        raw = data["dy"] if case == "infeasible" else data["dx"]
        name = "infeasibility_direction" if case == "infeasible" else "unbounded_direction"
        require(norm(raw) > 0 and np.array_equal(vectors[name], raw / norm(raw)),
                case, "exported ray differs from generated checkpoint difference")
        if case == "unbounded":
            require(np.array_equal(vectors["feasible_base"], data["current_x"]),
                    case, "feasible base replaced with a planted witness")
        if trace["iteration"] <= config.check_interval:
            require(not np.any(data["previous_x"]) and not np.any(data["previous_y"]),
                    case, "first checkpoint must start from actual zero iterates")
        from openopt.certificates import check_primal_infeasibility, check_unbounded_direction

        host = SimpleNamespace(**problem)
        both = {
            "infeasible": check_primal_infeasibility(
                host, data["dy"], atol=atol, rtol=rtol).to_dict(),
            "unbounded": check_unbounded_direction(
                host, data["dx"], feasible_x=data["current_x"], atol=atol, rtol=rtol).to_dict(),
        }
        for kind, summary in both.items():
            require(encode(trace[kind + "_evidence"]) == encode(summary),
                    case + "/" + kind, "terminal full direction summary differs from recomputation")
        require(both["infeasible"]["passed"] != both["unbounded"]["passed"]
                and both[case]["passed"] is True,
                case, "terminal direction evidence is conflicting or does not select status")
    else:
        require(extension == {}, case, "unexpected gap extension")
    return {"case": case, "status": expected, **actual}


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), path, "evidence inventory symlink rejected")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = descriptor(root, path)
    require(result, root, "empty evidence inventory")
    return result


def validate_result(result_path, expected_sha256, current_root, *, allow_cpu_smoke=False,
                    expectations=None, expectations_sha256=None):
    result_path = external(result_path)
    data = read(result_path)
    require(sha(data) == pin(expected_sha256), result_path, "caller result pin mismatch")
    root = result_path.parent
    saved = parse(data, result_path)
    device = saved["device"]
    cpu = device == "cpu"
    require(not cpu or allow_cpu_smoke, "scope", "CPU smoke cannot be accepted as GPU evidence")
    require(device in ("cpu", "cuda:0")
            and saved["scope"] == (CPU_SCOPE if cpu else GPU_SCOPE)
            and saved["verdict"] == ("CPU_SMOKE_PASS" if cpu else NUMERICAL_PASS)
            and saved["numerical_verdict"] == "PASS" and saved["execution_verdict"] == "UNVERIFIED"
            and type(saved["returncode"]) is int and saved["returncode"] == 0,
            result_path, "failed/unknown execution or scope mutation")
    cases = saved["requested_cases"]
    require(isinstance(cases, list) and cases and len(cases) == len(set(cases))
            and set(cases) <= set(CASES) and set(cases) == set(saved["cases"]),
            "cases", "empty, incomplete or duplicate requested scope")
    sources = saved["source_sha256"]
    require(isinstance(sources, dict) and sources, "sources", "empty source membership")
    same_sources(sources, source_map(root / "source"), "frozen source snapshot")
    same_sources(sources, source_map(current_root), "current source")
    require((expectations is None) == (expectations_sha256 is None),
            "expectations", "path and caller pin must be supplied together")
    require(cpu or expectations is not None, "expectations",
            "CUDA replay requires independent caller pre-dispatch expectations")
    expected, trusted = None, None
    if expectations is not None:
        expected, trusted = load_expectations(expectations, expectations_sha256, current_root)
        require(set(cases) <= set(expected["cases"]),
                "expectations", "replayed case absent from trusted pre-dispatch expectations")
    require(encode(saved["expectations"]) == encode(trusted),
            "expectations", "producer expectations differ from independent caller descriptor")
    hw = saved["hardware"]
    require(hw["requested_device"] == device and hw["torch"] and hw["numpy"],
            "hardware", "runtime/placement evidence missing")
    if not cpu:
        require(type(hw["gpu_count"]) is int and hw["gpu_count"] == 1 and "RTX PRO 4500" in hw["gpu"]
                and hw["node"] and hw["pod"] and hw["cuda"]
                and hw["cpu_quota"] and hw["cpu_scope"] == "Pod cgroup v2"
                and hw["cuda_capability"] == [12, 0]
                and isinstance(hw["gpu_uuid"], str)
                and re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", hw["gpu_uuid"]),
                "hardware", "actual CUDA 4500 placement/runtime mutation")
    telemetry_check(parse(read(bound(root, saved["telemetry"])), "telemetry"), device,
                    gpu_uuid=hw["gpu_uuid"], gpu_name=hw["gpu"])
    execution = read(root / "execution.log").decode()
    require(f"OPENOPT_GPU_PROOF verdict={saved['verdict']}" in execution
            and f"result_sha256={expected_sha256}" in execution
            and execution.rstrip().endswith("returncode=0"),
            "execution", "completion marker or result pin missing")
    outcomes = []
    for name in cases:
        require(saved["cases"][name]["path"] == name + ".json", name, "case sidecar substitution")
        outcome = replay_case(root, saved["cases"][name], device,
                              expected_model=expected["cases"][name] if expected is not None else None)
        require(outcome["case"] == name, name, "selected case identity mismatch")
        outcomes.append(outcome)
    return {"verdict": "CPU_SMOKE_REPLAY_PASS" if cpu else NUMERICAL_PASS,
            "numerical_verdict": "PASS", "execution_verdict": "UNVERIFIED",
            "expectations": trusted,
            "scope": saved["scope"], "device": device, "cases": outcomes,
            "case_count": len(outcomes), "source_count": len(sources), "hardware": hw,
            "result_sha256": expected_sha256,
            "qualification": "selected synthetic numerical replay; not exact proof, performance, "
                             "commercial parity, native conic/MIP or remote attestation"}


def seal(result_path, expected_sha256, current_root, *, allow_cpu_smoke=False,
         expectations=None, expectations_sha256=None):
    outcome = validate_result(result_path, expected_sha256, current_root,
                              allow_cpu_smoke=allow_cpu_smoke, expectations=expectations,
                              expectations_sha256=expectations_sha256)
    root = external(result_path).parent
    return {"schema_version": 1, "scope": outcome["scope"], "evidence_root": str(root),
            "result": descriptor(root, external(result_path)), "files": inventory(root)}


def replay(bundle_path, expected_sha256, current_root, *, allow_cpu_smoke=False,
           expectations=None, expectations_sha256=None):
    data = read(external(bundle_path))
    require(sha(data) == pin(expected_sha256), bundle_path, "caller bundle pin mismatch")
    bundle = parse(data, bundle_path)
    require(set(bundle) == {"schema_version", "scope", "evidence_root", "result", "files"}
            and bundle["schema_version"] == 1, "bundle", "unknown/incomplete seal")
    root = external(bundle["evidence_root"])
    actual = inventory(root)
    require(bundle["files"] == actual, "inventory", "evidence added, removed or changed since seal")
    path = bound(root, bundle["result"])
    require(path == root / "result.json", "result", "sealed result substitution")
    outcome = validate_result(path, bundle["result"]["sha256"], current_root,
                              allow_cpu_smoke=allow_cpu_smoke, expectations=expectations,
                              expectations_sha256=expectations_sha256)
    require(outcome["scope"] == bundle["scope"], "bundle", "sealed scope mutation")
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "cpu-smoke", "seal", "replay"))
    parser.add_argument("--device", choices=("cuda:0", "cpu"))
    parser.add_argument("--case", action="append", choices=(*CASES, "all"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--result-sha256")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--current-root", type=Path)
    parser.add_argument("--expectations", type=Path)
    parser.add_argument("--expectations-sha256")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.action in ("run", "cpu-smoke"):
        try:
            require(args.device and args.case and not args.allow_cpu_smoke,
                    args.action, "explicit device/case required")
            require("all" not in args.case or args.case == ["all"],
                    "cases", "all cannot be combined with another selection")
            cases = list(CASES) if args.case == ["all"] else args.case
            run_cases(args.artifact_dir, cases, args.device, cpu_smoke=args.action == "cpu-smoke",
                      expectations=args.expectations, expectations_sha256=args.expectations_sha256)
        except Exception as error:
            print(f"OPENOPT_GPU_PROOF verdict=FAIL error={type(error).__name__}: {error}", flush=True)
            return 1
        return 0
    parent = external(args.artifact_dir)
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.action + "-", dir=parent))
    report, code = {"started_at": now(), "verdict": "RUNNING"}, 0
    try:
        require(args.current_root, args.action, "explicit current source root required")
        if args.action == "seal":
            require(args.result and args.result_sha256, "seal", "caller result pin required")
            require(not output.is_relative_to(external(args.result).parent),
                    "seal", "seal output cannot mutate frozen evidence tree")
            value = seal(args.result, args.result_sha256, args.current_root,
                         allow_cpu_smoke=args.allow_cpu_smoke, expectations=args.expectations,
                         expectations_sha256=args.expectations_sha256)
            write(output / "bundle.json", value)
            report.update(verdict="CPU_SMOKE_SEALED" if args.allow_cpu_smoke else "SEALED",
                          bundle=str(output / "bundle.json"),
                          bundle_sha256=sha(read(output / "bundle.json")), scope=value["scope"],
                          numerical_verdict="PASS", execution_verdict="UNVERIFIED")
        else:
            require(args.bundle and args.sha256, "replay", "caller bundle pin required")
            report.update(replay(args.bundle, args.sha256, args.current_root,
                                 allow_cpu_smoke=args.allow_cpu_smoke, expectations=args.expectations,
                                 expectations_sha256=args.expectations_sha256))
    except Exception as error:
        code = 1
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        with (output / "failure.log").open("x") as stream:
            stream.write(traceback.format_exc())
    report.update(finished_at=now(), returncode=code)
    write(output / "acceptance.json", report)
    line = (f"OPENOPT_GPU_PROOF_{args.action.upper()} verdict={report['verdict']} "
            f"at={report['finished_at']} evidence={output / 'acceptance.json'}")
    with (output / "execution.log").open("x") as stream:
        stream.write(line + f"\nreturncode={code}\n")
    print(line, flush=True)
    if code:
        print(report["error"], flush=True)
    elif "bundle_sha256" in report:
        print(f"bundle={report['bundle']} sha256={report['bundle_sha256']}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
