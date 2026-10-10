"""Reproducible synthetic convex QP/LP stress cases, not a public benchmark.

Oracles are constructed before any solve: sufficient convex KKT conditions,
Farkas separation, or a feasible recession ray with strictly negative slope.
All checking here uses NumPy, never openopt's solver or its reported residuals.
Dyadic cases have algebraically exact witnesses; verification is float64.
The original dense generator retains its floating-point planted-KKT scope.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np
import torch

from .qp import QPProblem

GENERATOR_VERSION = "synthetic-qp-corpus-v1"
FAMILIES = (
    "original_dense", "dense_spd", "illconditioned", "psd_singular",
    "equality_redundant", "lp", "infeasible", "unbounded",
)
SCOPE = (
    "Synthetic dense convex QP/LP stress cases only; not representative of "
    "a public, industrial, sparse, conic, MIP or commercial benchmark corpus."
)


def _norm(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _integer(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return value


@dataclass(frozen=True)
class Oracle:
    expected: str
    scope: str
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    certificate: np.ndarray | None = None
    ray: np.ndarray | None = None
    unique: bool = False


@dataclass(frozen=True)
class CorpusCase:
    family: str
    seed: int
    P: np.ndarray
    q: np.ndarray
    A: np.ndarray
    l: np.ndarray
    u: np.ndarray
    oracle: Oracle

    @property
    def case_id(self) -> str:
        return f"{self.family}-n{len(self.q)}-seed{self.seed}"

    @property
    def problem_hash(self) -> str:
        """Hash ordered, shape-labelled little-endian float64 problem bytes."""
        digest = hashlib.sha256()
        for name in ("P", "q", "A", "l", "u"):
            value = np.asarray(getattr(self, name), dtype="<f8", order="C")
            digest.update(f"{name}:{value.shape}:<f8:".encode("ascii"))
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def problem(self) -> QPProblem:
        # CPU copies are identical for all backends; the solver owns placement.
        return QPProblem(*[
            torch.from_numpy(getattr(self, name).copy())
            for name in ("P", "q", "A", "l", "u")
        ])

    def metadata(self) -> dict:
        return {
            "case_id": self.case_id, "class": self.family, "seed": self.seed,
            "problem_hash": self.problem_hash, "generator_version": GENERATOR_VERSION,
            "variables": len(self.q), "constraints": len(self.l), "scope": SCOPE,
            "oracle": {
                "expected_property": self.oracle.expected,
                "scope": self.oracle.scope, "unique_solution": self.oracle.unique,
                "objective": objective(self, self.oracle.x)
                if self.oracle.expected == "optimal" else None,
            },
        }


def validate_problem(case: CorpusCase) -> None:
    n = len(case.q)
    if n < 3 or case.q.shape != (n,) or case.P.shape != (n, n):
        raise ValueError("corpus requires n >= 3 and compatible P/q dimensions")
    if case.A.ndim != 2 or case.A.shape[1] != n:
        raise ValueError("A dimensions do not match q")
    if case.l.shape != (len(case.A),) or case.u.shape != case.l.shape:
        raise ValueError("bound dimensions do not match A")
    if not all(np.isfinite(v).all() for v in (case.P, case.q, case.A)):
        raise ValueError("P/q/A must be finite")
    if (np.isnan(case.l).any() or np.isnan(case.u).any()
            or np.isposinf(case.l).any() or np.isneginf(case.u).any()
            or (case.l > case.u).any()):
        raise ValueError("invalid constraint bounds")
    tolerance = 20 * np.finfo(np.float64).eps * max(1.0, _norm(case.P))
    if _norm(case.P - case.P.T) > tolerance:
        raise ValueError("P must be symmetric")
    if float(np.linalg.eigvalsh(case.P).min()) < -tolerance:
        raise ValueError("P must be positive semidefinite")


def objective(case: CorpusCase, x: np.ndarray) -> float:
    x = _vector(x, len(case.q), "x")
    value = float(0.5 * x @ case.P @ x + case.q @ x)
    if not math.isfinite(value):
        raise ValueError("nonfinite independently recomputed objective")
    return value


def quality(
    case: CorpusCase, x: np.ndarray, y: np.ndarray,
    residual_tol: float = 5e-6, gap_tol: float = 1e-6,
) -> dict:
    """Recompute absolute KKT residuals and objective; no status is trusted."""
    if not all(math.isfinite(v) and v > 0 for v in (residual_tol, gap_tol)):
        raise ValueError("quality tolerances must be finite and positive")
    x = _vector(x, len(case.q), "x")
    y = _vector(y, len(case.l), "y")
    Ax = case.A @ x
    projected = np.clip(Ax, case.l, case.u)
    primal = _norm(Ax - projected)
    dual = _norm(case.P @ x + case.q + case.A.T @ y)
    normal = _norm(projected - np.clip(projected + y, case.l, case.u))
    value = objective(case, x)
    optimum = objective(case, case.oracle.x) if case.oracle.expected == "optimal" else None
    gap = abs(value - optimum) / max(1.0, abs(optimum)) if optimum is not None else None
    metrics = (primal, dual, normal)
    if not all(math.isfinite(v) for v in metrics):
        raise ValueError("nonfinite independently recomputed residual")
    passed = max(metrics) <= residual_tol and (gap is None or gap <= gap_tol)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "primal_residual": primal, "dual_residual": dual,
        "normal_cone_residual": normal, "objective": value,
        "oracle_objective": optimum, "relative_objective_gap": gap,
        "witness_distance": _norm(x - case.oracle.x) if case.oracle.x is not None else None,
        "solution_distance_is_gate": False,
    }


def check_infeasibility(case: CorpusCase, certificate: np.ndarray, tol: float = 1e-8) -> dict:
    """Check A.T@y=0 and support_[l,u](y)<0: a Farkas contradiction."""
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("certificate tolerance must be finite and positive")
    y = _vector(certificate, len(case.l), "infeasibility certificate")
    scale = _norm(y)
    if scale == 0:
        raise ValueError("zero infeasibility certificate")
    y = y / scale
    positive, negative = y > 0, y < 0
    finite_support = bool(
        np.isfinite(case.u[positive]).all() and np.isfinite(case.l[negative]).all()
    )
    support = float(y[positive] @ case.u[positive] + y[negative] @ case.l[negative]) \
        if finite_support else None
    residual = _norm(case.A.T @ y)
    passed = finite_support and residual <= tol and support < -tol
    return {
        "verdict": "PASS" if passed else "FAIL",
        "annihilation_residual": residual, "box_support": support,
        "finite_support": finite_support, "normalization": "infinity norm",
    }


def check_unboundedness(
    case: CorpusCase, ray: np.ndarray, base: np.ndarray, tol: float = 1e-8,
) -> dict:
    """Check a feasible base, Pd=0, recession constraints and q.T@d<0."""
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("ray tolerance must be finite and positive")
    d = _vector(ray, len(case.q), "unboundedness ray")
    x = _vector(base, len(case.q), "feasible base")
    scale = _norm(d)
    if scale == 0:
        raise ValueError("zero unboundedness ray")
    d = d / scale
    Ax, Ad = case.A @ x, case.A @ d
    primal = _norm(Ax - np.clip(Ax, case.l, case.u))
    upper = np.maximum(Ad[np.isfinite(case.u)], 0.0)
    lower = np.maximum(-Ad[np.isfinite(case.l)], 0.0)
    recession = max(_norm(upper), _norm(lower))
    curvature = _norm(case.P @ d)
    slope = float(case.q @ d)
    passed = max(primal, recession, curvature) <= tol and slope < -tol
    return {
        "verdict": "PASS" if passed else "FAIL", "base_primal_residual": primal,
        "recession_residual": recession, "curvature_residual": curvature,
        "objective_slope": slope, "normalization": "infinity norm",
    }


def validate_oracle(case: CorpusCase) -> dict:
    validate_problem(case)
    oracle = case.oracle
    if oracle.expected == "optimal":
        measured = quality(case, oracle.x, oracle.y, residual_tol=1e-10, gap_tol=1e-12)
    elif oracle.expected == "infeasible":
        measured = check_infeasibility(case, oracle.certificate, tol=1e-10)
    elif oracle.expected == "unbounded":
        measured = check_unboundedness(case, oracle.ray, oracle.x, tol=1e-10)
    else:
        raise ValueError(f"unknown oracle property: {oracle.expected}")
    if measured["verdict"] != "PASS":
        raise ValueError(f"invalid independent oracle case={case.case_id}: {measured}")
    return measured


def _dyadic(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.integers(-4, 5, size=shape).astype(np.float64) / 8


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
    B = _dyadic(rng, (n, n))
    return np.eye(n) + B.T @ B


def _planted(
    family: str, seed: int, P: np.ndarray, A: np.ndarray,
    x: np.ndarray, rng: np.random.Generator,
    equalities: int = 0, unique: bool = True,
) -> CorpusCase:
    Ax = A @ x
    l, u, y = Ax - 1, Ax + 1, np.zeros(len(A))
    upper = np.arange(0, len(A) - equalities, 7)
    lower = np.arange(3, len(A) - equalities, 7)
    u[upper], y[upper] = Ax[upper], 0.5
    l[lower], y[lower] = Ax[lower], -0.5
    if equalities:
        l[-equalities:] = u[-equalities:] = Ax[-equalities:]
        y[-equalities:] = _dyadic(rng, (equalities,))
    q = -P @ x - A.T @ y
    oracle = Oracle(
        "optimal", "Algebraically exact dyadic convex KKT witness; float64 recheck",
        x=x, y=y, unique=unique,
    )
    return CorpusCase(family, seed, P, q, A, l, u, oracle)


def generate_case(family: str, n: int = 24, seed: int = 20261010) -> CorpusCase:
    _integer(n, "n", 3)
    _integer(seed, "seed", 0)
    if family not in FAMILIES:
        raise ValueError(f"unknown corpus family: {family}")
    rng = np.random.default_rng(seed)
    if family == "original_dense":
        # Preserve the original benchmark's operation order and exact bytes.
        B = rng.normal(size=(n, n)) / np.sqrt(n)
        P = np.eye(n) + B.T @ B
        A = np.concatenate([np.eye(n), rng.normal(size=(max(3, n // 4), n)) / np.sqrt(n)])
        x = rng.uniform(-0.5, 0.5, n)
        Ax = A @ x
        l, u, y = Ax - 0.5, Ax + 0.5, np.zeros(len(A))
        active = np.arange(0, len(A), 7)
        u[active], y[active] = Ax[active], rng.uniform(0.2, 0.8, len(active))
        active = np.arange(3, len(A), 7)
        l[active], y[active] = Ax[active], -rng.uniform(0.2, 0.8, len(active))
        l[-3:], u[-3:] = Ax[-3:], Ax[-3:]
        y[-3:] = rng.normal(size=3)
        q = -P @ x - A.T @ y
        case = CorpusCase(family, seed, P, q, A, l, u, Oracle(
            "optimal", "Original floating planted convex KKT; roundoff, not exact certification",
            x=x, y=y, unique=True,
        ))
    elif family in ("dense_spd", "illconditioned", "equality_redundant"):
        P = _spd(rng, n)
        if family == "illconditioned":
            spectrum = np.exp2(-np.concatenate([[0], np.linspace(25, 30, n - 1).astype(int)]))
            v = _dyadic(rng, (n,))
            v[0] = 0.5
            P = np.diag(spectrum) + np.outer(v, v)
        A = np.concatenate([np.eye(n), _dyadic(rng, (max(3, n // 4), n))])
        equalities = 0
        if family == "equality_redundant":
            E = _dyadic(rng, (max(1, n // 4), n))
            redundant = np.concatenate([E, E, 2 * E, np.zeros((1, n))])
            A = np.concatenate([A, redundant])
            equalities = len(redundant)
        case = _planted(family, seed, P, A, _dyadic(rng, (n,)), rng, equalities)
    elif family == "psd_singular":
        rank = max(1, n // 2)
        P = np.zeros((n, n))
        P[:rank, :rank] = _spd(rng, rank)
        x = _dyadic(rng, (n,))
        A = np.eye(n)
        y = np.zeros(n)
        y[0] = 0.5
        Ax = A @ x
        l, u = Ax - 1, Ax + 1
        u[0] = Ax[0]
        q = -P @ x - A.T @ y
        case = CorpusCase(family, seed, P, q, A, l, u, Oracle(
            "optimal", "Exact dyadic KKT with unconstrained-cost nullspace; nonunique optimum",
            x=x, y=y, unique=False,
        ))
    elif family == "lp":
        P, A = np.zeros((n, n)), np.concatenate([np.eye(n), np.eye(n)])
        center = _dyadic(rng, (n,))
        l, u = np.tile(center - 1, 2), np.tile(center + 1, 2)
        q = _dyadic(rng, (n,))
        q[0], q[1], q[-1] = -0.5, 0.5, 0
        x = center - np.sign(q)
        y = np.concatenate([-q, np.zeros(n)])
        case = CorpusCase(family, seed, P, q, A, l, u, Oracle(
            "optimal", "Exact bounded box LP primal/dual witness; redundant rows and zero cost",
            x=x, y=y, unique=False,
        ))
    elif family == "infeasible":
        v = _dyadic(rng, (n,))
        v[0] = 1
        A = np.concatenate([v[None], v[None], np.eye(n)])
        l, u = np.full(n + 2, -2.0), np.full(n + 2, 2.0)
        l[0], u[0], l[1], u[1] = -np.inf, 0, 1, np.inf
        certificate = np.zeros(n + 2)
        certificate[:2] = [1, -1]
        P = np.eye(n) if seed % 2 else np.zeros((n, n))
        case = CorpusCase(family, seed, P, np.zeros(n), A, l, u, Oracle(
            "infeasible", "Exact Farkas contradiction: same row <=0 and >=1",
            certificate=certificate,
        ))
    else:
        P = np.zeros((n, n))
        if seed % 2:
            P[:-1, :-1] = _spd(rng, n - 1)
        A = np.eye(n)[:-1]
        x = _dyadic(rng, (n,))
        x[-1] = 0
        q = -P @ x
        q[-1] = -1
        ray = np.zeros(n)
        ray[-1] = 1
        Ax = A @ x
        case = CorpusCase(family, seed, P, q, A, Ax - 1, Ax + 1, Oracle(
            "unbounded", "Exact feasible recession ray: Pd=Ad=0 and q'd=-1",
            x=x, ray=ray,
        ))
    validate_oracle(case)
    return case


def build_corpus(
    sizes: Iterable[int] = (8, 24, 64), seeds: Iterable[int] = (20261010,),
    families: Iterable[str] = FAMILIES,
) -> list[CorpusCase]:
    sizes, seeds, families = list(sizes), list(seeds), list(families)
    if not sizes or not seeds or not families:
        raise ValueError("empty corpus selection")
    if any(len(values) != len(set(values)) for values in (sizes, seeds, families)):
        raise ValueError("duplicate corpus selection")
    return [
        generate_case(family, n, seed)
        for family in families for n in sizes for seed in seeds
    ]
