"""Opt-in exact structural reductions with independently checked postsolve.

Only exact zero rows and +1 identity equalities are recognized. Rational
auditing admits a fixed substitution only if every transformed coefficient,
bound, shift and objective constant is exactly representable. No solver is
imported other than the project's unchanged PyTorch dense QP implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math

import torch

from .certificates import (
    NumericalEvidence, check_primal_infeasibility, check_qp_candidate,
)
from .qp import QPConfig, QPProblem, QPResult, solve_qp

__all__ = [
    "PresolveConfig", "PresolvedQP", "PresolvedQPResult",
    "presolve_qp", "solve_presolved_qp",
]

_NAMES = ("P", "q", "A", "l", "u")


@dataclass(frozen=True)
class PresolveConfig:
    enabled: bool = False
    remove_zero_rows: bool = True
    fix_identity_equalities: bool = True
    check_atol: float = 1e-7
    check_rtol: float = 1e-7

    def validate(self) -> None:
        for name in ("enabled", "remove_zero_rows", "fix_identity_equalities"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("check_atol", "check_rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if self.check_atol <= 0 or not 0 <= self.check_rtol < 1:
            raise ValueError("check_atol > 0 and 0 <= check_rtol < 1 are required")


def _tensors(p: QPProblem) -> tuple[torch.Tensor, ...]:
    if not isinstance(p, QPProblem):
        raise TypeError("problem must be QPProblem")
    values = tuple(getattr(p, name) for name in _NAMES)
    if not all(isinstance(t, torch.Tensor) for t in values):
        raise TypeError("QP inputs must be tensors")
    for name, t in zip(_NAMES, values):
        if (
            t.layout != torch.strided or t.dtype not in (torch.float32, torch.float64)
            or t.device.type not in ("cpu", "cuda")
        ):
            raise ValueError(f"{name} must be dense CPU/CUDA float32/float64 data")
    if len({t.device for t in values}) != 1:
        raise ValueError("QP inputs must share a device")
    if len({t.dtype for t in values}) != 1:
        raise ValueError("QP inputs must share a dtype")
    return values


def _validated_original(problem: QPProblem) -> QPProblem:
    values = _tensors(problem)
    # float32 -> float64 is lossless; float64 is retained before the solver cast.
    p = QPProblem(*[t.detach().to(device="cpu", dtype=torch.float64).clone() for t in values])
    if p.q.ndim != 1 or not p.q.numel():
        raise ValueError("original q must be nonempty; only reduced zero-variable models are supported")
    n = p.q.numel()
    if p.P.shape != (n, n) or p.A.ndim != 2 or p.A.shape[1] != n:
        raise ValueError("P/A dimensions must match q")
    if p.l.shape != (p.A.shape[0],) or p.u.shape != p.l.shape:
        raise ValueError("constraint bounds must match A rows")
    if not all(bool(torch.isfinite(t).all()) for t in (p.P, p.q, p.A)):
        raise ValueError("P/q/A must be finite")
    if bool(torch.isnan(p.l).any() | torch.isnan(p.u).any()):
        raise ValueError("bounds cannot contain NaN")
    if bool(torch.isposinf(p.l).any() | torch.isneginf(p.u).any()):
        raise ValueError("lower +inf or upper -inf is invalid")
    if bool((p.l > p.u).any()):
        raise ValueError("lower bound exceeds upper bound")
    if not torch.equal(p.P, p.P.T):
        raise ValueError("exact presolve requires exactly symmetric P")
    eigenvalues = torch.linalg.eigvalsh(p.P)
    norm = float(eigenvalues.abs().max())
    limit = 64 * torch.finfo(torch.float64).eps * n * norm
    if not bool(torch.isfinite(eigenvalues).all()) or not math.isfinite(limit):
        raise ValueError("nonfinite PSD validation arithmetic")
    if float(eigenvalues.min()) < -limit:
        raise ValueError("P must be positive semidefinite")
    return p


def _working(original: QPProblem, config: QPConfig) -> QPProblem:
    values = []
    for name in _NAMES:
        before = getattr(original, name)
        after = before.to(device=config.device, dtype=config.dtype).clone()
        finite = torch.isfinite(before).to(after.device)
        if not bool(torch.isfinite(after[finite]).all()):
            raise ValueError(f"nonfinite dtype conversion: {name}")
        nonzero = (before != 0).to(after.device) & finite
        if bool((nonzero & (after == 0)).any()):
            raise ValueError(f"underflow in dtype conversion: {name}")
        values.append(after)
    return QPProblem(*values)


def _index(values: list[int], p: QPProblem) -> torch.Tensor:
    return torch.tensor(values, device=p.q.device, dtype=torch.int64)


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(value)


def _sum_products(left: list[float], right: list[float]) -> Fraction:
    return sum((_fraction(a) * _fraction(b) for a, b in zip(left, right)), Fraction())


def _exact_substitution(
    original: QPProblem, current: QPProblem,
    free: list[int], fixed: list[int], values: list[float],
) -> tuple[QPProblem, torch.Tensor, float] | None:
    # Audit against pre-cast values. A lossy solver cast must not create an
    # exact identity equality or an apparently exact reduced model.
    if any(not torch.equal(getattr(original, n), getattr(current, n).to("cpu", torch.float64))
           for n in _NAMES):
        return None
    f, k = _index(free, current), _index(fixed, current)
    c = torch.tensor(values, device=current.q.device, dtype=current.q.dtype)
    products = current.P[:, k] @ c
    shifts = current.A[:, k] @ c
    q = current.q[f] + products[f]
    l = current.l.clone()
    u = current.u.clone()
    lower, upper = torch.isfinite(l), torch.isfinite(u)
    l[lower] -= shifts[lower]
    u[upper] -= shifts[upper]
    quadratic_constant = 0.5 * torch.dot(c, products[k])
    linear_constant = torch.dot(current.q[k], c)
    offset = quadratic_constant + linear_constant
    arithmetic = (products, shifts, q, l[lower], u[upper],
                  quadratic_constant, linear_constant, offset)
    if not all(bool(torch.isfinite(t).all()) for t in arithmetic):
        return None
    P0, q0, A0, l0, u0 = [getattr(original, n).tolist() for n in _NAMES]
    shifts0 = [_sum_products([row[j] for j in fixed], values) for row in A0]
    products0 = [_sum_products([row[j] for j in fixed], values) for row in P0]
    q_exact = [_fraction(q0[j]) + products0[j] for j in free]
    offset_exact = sum(
        (_fraction(v) * (products0[j] / 2 + _fraction(q0[j]))
         for j, v in zip(fixed, values)), Fraction(),
    )
    if _fraction(float(offset)) != offset_exact:
        return None
    if any(_fraction(a) != b for a, b in zip(q.tolist(), q_exact)):
        return None
    for i, shift in enumerate(shifts.tolist()):
        if _fraction(shift) != shifts0[i]:
            return None
        if math.isfinite(l0[i]) and _fraction(float(l[i])) != _fraction(l0[i]) - shifts0[i]:
            return None
        if math.isfinite(u0[i]) and _fraction(float(u[i])) != _fraction(u0[i]) - shifts0[i]:
            return None
    return QPProblem(current.P[f][:, f].clone(), q, current.A[:, f].clone(), l, u), shifts, float(offset)


def _digest(prepared: PresolvedQP) -> str:
    digest = hashlib.sha256()
    for p in (prepared.original_problem, prepared.working_problem, prepared.reduced_problem):
        for name in _NAMES:
            t = getattr(p, name)
            digest.update(f"{name}:{t.dtype}:{t.device}:{tuple(t.shape)}".encode("ascii"))
            digest.update(t.detach().cpu().contiguous().numpy().tobytes())
    for name in ("free_variables", "fixed_variables", "fixed_values", "fixed_rows",
                 "kept_rows", "removed_rows", "row_shifts"):
        t = getattr(prepared, name)
        digest.update(f"{name}:{t.dtype}:{t.device}:{tuple(t.shape)}".encode("ascii"))
        digest.update(t.detach().cpu().contiguous().numpy().tobytes())
    if prepared.infeasibility_direction is not None:
        t = prepared.infeasibility_direction
        digest.update(f"direction:{t.dtype}:{t.device}:{tuple(t.shape)}".encode("ascii"))
        digest.update(t.detach().cpu().contiguous().numpy().tobytes())
    digest.update(repr((prepared.objective_offset, prepared.notes, prepared.options)).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PresolvedQP:
    original_problem: QPProblem
    working_problem: QPProblem
    reduced_problem: QPProblem
    free_variables: torch.Tensor
    fixed_variables: torch.Tensor
    fixed_values: torch.Tensor
    fixed_rows: torch.Tensor
    kept_rows: torch.Tensor
    removed_rows: torch.Tensor
    row_shifts: torch.Tensor
    objective_offset: float
    notes: tuple[str, ...]
    options: PresolveConfig
    infeasibility_direction: torch.Tensor | None = None
    infeasibility_evidence: NumericalEvidence | None = None
    _seal: str = ""

    @property
    def is_constant(self) -> bool:
        return self.reduced_problem.q.numel() == 0

    def validate(self) -> None:
        self.options.validate()
        if not self._seal or _digest(self) != self._seal:
            raise ValueError("presolve model/mapping mutation: sealed bytes do not match")

    def _candidate(self, value: torch.Tensor, size: int, name: str) -> torch.Tensor:
        p = self.working_problem
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if (
            value.layout != torch.strided or value.device != p.q.device
            or value.dtype != p.q.dtype or value.shape != (size,)
        ):
            raise ValueError(f"{name} must match reduced shape, dtype and explicit device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        return value.detach()

    @torch.no_grad()
    def restore(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover primal variables and signed original equality multipliers.

        The selected fixed rows have coefficient +1, so their multiplier is
        minus the remaining stationarity gradient. No free-variable residual
        is repaired and no optimality assertion is made here.
        """
        self.validate()
        x = self._candidate(x, self.free_variables.numel(), "x")
        y = self._candidate(y, self.kept_rows.numel(), "y")
        p = self.working_problem
        original_x, original_y = torch.zeros_like(p.q), torch.zeros_like(p.l)
        original_x[self.free_variables] = x
        original_x[self.fixed_variables] = self.fixed_values
        original_y[self.kept_rows] = y
        if self.fixed_variables.numel():
            gradient = p.P @ original_x + p.q + p.A.T @ original_y
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("nonfinite original dual restoration arithmetic")
            original_y[self.fixed_rows] = -gradient[self.fixed_variables]
        return original_x, original_y

    def check(self, x: torch.Tensor, y: torch.Tensor) -> NumericalEvidence:
        self.validate()
        x = self._candidate(x, self.working_problem.q.numel(), "original x")
        y = self._candidate(y, self.working_problem.l.numel(), "original y")
        return check_qp_candidate(
            self.original_problem, x.to("cpu", torch.float64), y.to("cpu", torch.float64),
            atol=self.options.check_atol, rtol=self.options.check_rtol,
        )


@dataclass(frozen=True)
class PresolvedQPResult:
    x: torch.Tensor
    y: torch.Tensor
    status: str
    objective: float
    evidence: NumericalEvidence
    presolved: PresolvedQP
    reduced_result: QPResult | None
    infeasibility_direction: torch.Tensor | None = None
    infeasibility_evidence: NumericalEvidence | None = None

    @property
    def iterations(self) -> int:
        return self.reduced_result.iterations if self.reduced_result is not None else 0


@torch.no_grad()
def presolve_qp(
    problem: QPProblem, config: QPConfig = QPConfig(),
    presolve: PresolveConfig = PresolveConfig(),
) -> PresolvedQP:
    """Prepare a sealed exact reduction, disabled by default.

    Unsupported singleton coefficients, conflicting fixed equalities and
    unrepresentable substitutions remain in the model, with explicit notes.
    Removed feasible rows have exactly zero free coefficients. Contradictory
    rows are retained; a derived direction is independently checked instead.
    """
    config.validate()
    presolve.validate()
    original = _validated_original(problem)
    current = _working(original, config)
    n, m = current.q.numel(), current.l.numel()
    fixed, fixed_rows, values, notes = [], [], [], []
    if presolve.enabled and presolve.fix_identity_equalities:
        choices: dict[int, tuple[int, float]] = {}
        conflict = False
        for i in range(m):
            row = original.A[i]
            nz = torch.nonzero(row != 0).flatten()
            equality = bool(torch.isfinite(original.l[i]) & (original.l[i] == original.u[i]))
            if not equality or nz.numel() != 1:
                continue
            j = int(nz[0])
            if float(row[j]) != 1.0:
                notes.append(f"unsupported_singleton_row:{i}")
                continue
            value = float(original.l[i])
            if j in choices and choices[j][1] != value:
                conflict = True
            elif j not in choices:
                choices[j] = (i, value)
        if conflict:
            notes.append("conflicting_identity_equalities_retained")
        else:
            fixed = sorted(choices)
            fixed_rows = [choices[j][0] for j in fixed]
            values = [choices[j][1] for j in fixed]
    fixed_set = set(fixed)
    free = [j for j in range(n) if j not in fixed_set]
    transformed = current
    shifts, offset = torch.zeros_like(current.l), 0.0
    if fixed:
        substitution = _exact_substitution(original, current, free, fixed, values)
        if substitution is None:
            notes.append("unrepresentable_fixed_substitution_retained")
            free, fixed, fixed_rows, values = list(range(n)), [], [], []
        else:
            transformed, shifts, offset = substitution
    kept, removed = [], []
    contradiction = None
    for i in range(m):
        zero = bool((transformed.A[i] == 0).all())
        # Detection in the uncast original prevents a rounded coefficient
        # from masquerading as a zero. Nonzero->zero casts already fail loud.
        feasible_zero = bool(transformed.l[i] <= 0 <= transformed.u[i])
        if presolve.enabled and zero and not feasible_zero and contradiction is None:
            contradiction = i
        can_remove = (
            presolve.enabled and zero and feasible_zero
            and (presolve.remove_zero_rows or i in fixed_rows)
        )
        (removed if can_remove else kept).append(i)
    row_index = _index(kept, current)
    reduced = QPProblem(
        transformed.P.clone(), transformed.q.clone(), transformed.A[row_index].clone(),
        transformed.l[row_index].clone(), transformed.u[row_index].clone(),
    )
    direction = direction_evidence = None
    if contradiction is not None:
        direction = torch.zeros_like(current.l)
        direction[contradiction] = -1.0 if float(transformed.l[contradiction]) > 0 else 1.0
        if fixed:
            k = _index(fixed, current)
            gradient = current.A.T @ direction
            direction[_index(fixed_rows, current)] = -gradient[k]
        direction_evidence = check_primal_infeasibility(
            original, direction.to("cpu", torch.float64),
            atol=config.direction_atol, rtol=config.direction_rtol,
        )
        notes.append("zero_row_contradiction_retained")
    prepared = PresolvedQP(
        original, current, reduced, _index(free, current), _index(fixed, current),
        torch.tensor(values, device=current.q.device, dtype=current.q.dtype),
        _index(fixed_rows, current), row_index, _index(removed, current),
        shifts, offset, tuple(notes), presolve, direction, direction_evidence,
    )
    object.__setattr__(prepared, "_seal", _digest(prepared))
    return prepared


@torch.no_grad()
def solve_presolved_qp(
    problem: QPProblem, config: QPConfig = QPConfig(),
    presolve: PresolveConfig = PresolveConfig(),
) -> PresolvedQPResult:
    """Solve the reduced QP, restore, and enforce every original checker gate.

    Zero-variable reduced models use their unique empty primal and zero row
    multipliers, not the nonempty dense solver. Only independently passed
    original-model evidence can yield ``solved`` or ``infeasible_numerical``.
    Reduced iteration limits/directions are never promoted to original claims.
    """
    prepared = presolve_qp(problem, config, presolve)
    prepared.validate()
    reduced = prepared.reduced_problem
    raw = None
    if prepared.infeasibility_direction is not None:
        xhat, yhat = torch.zeros_like(reduced.q), torch.zeros_like(reduced.l)
        # Recheck instead of trusting stored/mutable evidence.
        direction_evidence = check_primal_infeasibility(
            prepared.original_problem, prepared.infeasibility_direction.to("cpu", torch.float64),
            atol=config.direction_atol, rtol=config.direction_rtol,
        )
        status = "infeasible_numerical" if direction_evidence.passed else "infeasible_unverified"
    elif prepared.is_constant:
        xhat, yhat = torch.zeros_like(reduced.q), torch.zeros_like(reduced.l)
        direction_evidence, status = None, "solved"
    else:
        raw = solve_qp(reduced, config)
        xhat, yhat = raw.x, raw.y
        direction_evidence, status = None, raw.status
        if status in ("infeasible", "unbounded"):
            status = "reduced_direction_unverified"
    x, y = prepared.restore(xhat, yhat)
    evidence = prepared.check(x, y)
    if status == "solved" and not evidence.passed:
        status = "candidate_rejected"
    return PresolvedQPResult(
        x, y, status, evidence.metrics["primal_objective"], evidence, prepared, raw,
        prepared.infeasibility_direction, direction_evidence,
    )
