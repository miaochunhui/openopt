"""QP solver drivers for MPO-T1 (docs/mpo-t1-SPEC.md section 3).

``solve_mpo_qp(qp, backend='osqp', cfg=None) -> MpoSolution`` is fail-soft by
contract: it NEVER raises because of a solver problem. Any exception, any
non-converged status maps to ``MpoSolution(status='failed', weights=None, ...)``
and the caller (``openpo.mpo.strategies``) falls back to not trading.

Status mapping
--------------
- ``'optimal'``     -- OSQP ``solved`` / Clarabel ``Solved`` / SciPy success
- ``'inaccurate'``  -- ``solved inaccurate`` (and Clarabel's ``*InsufficientProgress``
  / ``AlmostSolved``). The primal vector is still returned for inspection, but
  callers treat anything other than ``'optimal'`` as "do not trade".
- ``'failed'``      -- infeasible, unbounded, iteration limit, exception.

Backends
--------
- ``osqp`` (primary): ADMM, settings ``eps_abs = eps_rel = cfg.qp_eps``,
  ``max_iter = cfg.qp_max_iter``, ``polish=True``, ``verbose=False``.
- ``clarabel``: interior point fallback; ``l <= Ax <= u`` is converted to a
  zero cone (equality rows) plus a non-negative cone (the two-sided rows).
- ``scipy``: last-resort dense fallback via ``scipy.optimize.minimize`` with
  ``trust-constr`` and an analytic gradient/Hessian.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from openpo.mpo.config import MpoConfig
from openpo.mpo.qp import MpoQp
from openpo.mpo.types import MpoSolution

__all__ = ["solve_mpo_qp", "SUPPORTED_BACKENDS"]

SUPPORTED_BACKENDS = ("osqp", "clarabel", "scipy")

_OSQP_OPTIMAL = {"solved"}
_OSQP_INACCURATE = {"solved inaccurate", "solved_inaccurate"}


def _failed(solve_time: float, note: str, iterations: int = 0) -> MpoSolution:
    sol = MpoSolution(
        status="failed",
        weights=None,
        buys=None,
        sells=None,
        objective=None,
        solve_time=float(solve_time),
        iterations=int(iterations),
    )
    # Diagnostics are attached (not part of the frozen dataclass fields) so the
    # first-scene failure reason is visible without re-running the solve.
    setattr(sol, "note", note)
    return sol


def solve_mpo_qp(
    qp: MpoQp,
    backend: str = "osqp",
    cfg: MpoConfig | None = None,
) -> MpoSolution:
    """Solve an assembled :class:`~openpo.mpo.qp.MpoQp`, fail-soft.

    Parameters
    ----------
    qp : MpoQp
        Output of :func:`openpo.mpo.qp.build_mpo_qp`.
    backend : str
        ``'osqp'`` (default / primary), ``'clarabel'`` or ``'scipy'``.
    cfg : MpoConfig | None
        Supplies ``qp_eps`` / ``qp_max_iter``; defaults to ``MpoConfig()``.

    Returns
    -------
    MpoSolution
        ``weights``/``buys``/``sells`` are ``[J, N]`` arrays when a primal
        vector was produced, otherwise ``None``. ``solve_time`` is wall time in
        seconds for the whole call (setup + solve).
    """
    t0 = time.perf_counter()
    if cfg is None:
        cfg = MpoConfig()
    if not isinstance(qp, MpoQp):
        return _failed(time.perf_counter() - t0, f"not an MpoQp: {type(qp).__name__}")

    name = str(backend).lower()
    try:
        if name == "osqp":
            return _solve_osqp(qp, cfg, t0)
        if name == "clarabel":
            return _solve_clarabel(qp, cfg, t0)
        if name == "scipy":
            return _solve_scipy(qp, cfg, t0)
        return _failed(
            time.perf_counter() - t0,
            f"unknown backend {backend!r}; supported {SUPPORTED_BACKENDS}",
        )
    except Exception as exc:  # fail-soft: never propagate solver problems
        return _failed(
            time.perf_counter() - t0, f"{name} raised {type(exc).__name__}: {exc}"
        )


def _finish(
    qp: MpoQp,
    x: np.ndarray | None,
    status: str,
    t0: float,
    iterations: int,
    note: str = "",
) -> MpoSolution:
    solve_time = time.perf_counter() - t0
    if x is None or not np.all(np.isfinite(x)):
        return _failed(solve_time, note or f"{status}: no finite primal vector", iterations)
    weights, buys, sells = qp.unpack(x)
    sol = MpoSolution(
        status=status,
        weights=weights,
        buys=buys,
        sells=sells,
        objective=qp.objective_value(x),
        solve_time=float(solve_time),
        iterations=int(iterations),
    )
    if note:
        setattr(sol, "note", note)
    return sol


# ---------------------------------------------------------------------------
# OSQP (primary)
# ---------------------------------------------------------------------------


def _solve_osqp(qp: MpoQp, cfg: MpoConfig, t0: float) -> MpoSolution:
    import osqp

    eps = float(cfg.qp_eps)
    data = dict(
        P=sp.csc_matrix(qp.P),
        q=np.asarray(qp.q, dtype=np.float64),
        A=sp.csc_matrix(qp.A),
        l=np.asarray(qp.l, dtype=np.float64),
        u=np.asarray(qp.u, dtype=np.float64),
    )
    settings = dict(
        eps_abs=eps,
        eps_rel=eps,
        max_iter=int(cfg.qp_max_iter),
        verbose=False,
    )
    model = osqp.OSQP()
    # Solution polishing: renamed 'polish' -> 'polishing' in OSQP v1.
    try:
        model.setup(**data, polishing=True, **settings)
    except (ValueError, TypeError):
        model = osqp.OSQP()
        model.setup(**data, polish=True, **settings)
    try:
        # OSQP v1 warns unless the fail-soft behaviour is requested explicitly;
        # v0.6 has no such argument.
        res = model.solve(raise_error=False)
    except TypeError:
        res = model.solve()
    info = res.info
    raw = str(getattr(info, "status", "unknown")).strip().lower()
    iters = int(getattr(info, "iter", 0) or 0)
    if raw in _OSQP_OPTIMAL:
        status = "optimal"
    elif raw in _OSQP_INACCURATE:
        status = "inaccurate"
    else:
        return _failed(time.perf_counter() - t0, f"osqp status={raw}", iters)
    x = np.asarray(res.x, dtype=np.float64) if res.x is not None else None
    return _finish(qp, x, status, t0, iters, note=f"osqp status={raw}")


# ---------------------------------------------------------------------------
# Clarabel (interior-point fallback)
# ---------------------------------------------------------------------------


def _solve_clarabel(qp: MpoQp, cfg: MpoConfig, t0: float) -> MpoSolution:
    import clarabel

    a = sp.csc_matrix(qp.A)
    l = np.asarray(qp.l, dtype=np.float64)
    u = np.asarray(qp.u, dtype=np.float64)

    eq = np.isclose(l, u, rtol=0.0, atol=1e-12) & np.isfinite(l)
    ineq_u = np.isfinite(u) & ~eq
    ineq_l = np.isfinite(l) & ~eq

    rows = [a[eq]] if eq.any() else []
    b_parts = [u[eq]] if eq.any() else []
    cones = [clarabel.ZeroConeT(int(eq.sum()))] if eq.any() else []

    n_ineq = int(ineq_u.sum() + ineq_l.sum())
    if n_ineq:
        if ineq_u.any():
            rows.append(a[ineq_u])
            b_parts.append(u[ineq_u])
        if ineq_l.any():
            rows.append(-a[ineq_l])
            b_parts.append(-l[ineq_l])
        cones.append(clarabel.NonnegativeConeT(n_ineq))
    if not rows:  # unconstrained: Clarabel needs at least one row
        rows = [sp.csc_matrix((0, qp.n_vars))]
        b_parts = [np.zeros(0)]
        cones = [clarabel.ZeroConeT(0)]

    a_cone = sp.csc_matrix(sp.vstack(rows, format="csc"))
    b_cone = np.concatenate(b_parts)

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = float(cfg.qp_eps)
    settings.tol_gap_rel = float(cfg.qp_eps)
    settings.tol_feas = float(cfg.qp_eps)
    settings.max_iter = int(min(cfg.qp_max_iter, 1000))

    solver = clarabel.DefaultSolver(
        sp.csc_matrix(qp.P), np.asarray(qp.q, dtype=np.float64), a_cone, b_cone, cones, settings
    )
    res = solver.solve()
    raw = str(res.status)
    lowered = raw.lower()
    iters = int(getattr(res, "iterations", 0) or 0)
    if "almost" in lowered or "insufficientprogress" in lowered:
        status = "inaccurate"
    elif lowered.endswith("solved"):
        status = "optimal"
    else:
        return _failed(time.perf_counter() - t0, f"clarabel status={raw}", iters)
    x = np.asarray(res.x, dtype=np.float64) if res.x is not None else None
    return _finish(qp, x, status, t0, iters, note=f"clarabel status={raw}")


# ---------------------------------------------------------------------------
# SciPy (dense last resort)
# ---------------------------------------------------------------------------


def _solve_scipy(qp: MpoQp, cfg: MpoConfig, t0: float) -> MpoSolution:
    from scipy.optimize import LinearConstraint, minimize

    p_full = sp.csc_matrix(qp.P)
    p_full = p_full + sp.triu(p_full, k=1).T  # mirror the upper triangle
    q = np.asarray(qp.q, dtype=np.float64)

    def fun(x):
        return 0.5 * float(x @ (p_full @ x)) + float(q @ x)

    def jac(x):
        return p_full @ x + q

    def hess(x):
        return p_full

    cons = LinearConstraint(sp.csr_matrix(qp.A), qp.l, qp.u)
    x0 = np.zeros(qp.n_vars, dtype=np.float64)
    # trust-constr stops on its own gtol/xtol, which are *not* KKT tolerances:
    # measured on this QP family, gtol=1e-12 is what buys ~1e-9 agreement with
    # OSQP (gtol=1e-9 only reaches ~3e-6), so square the requested eps.
    tol = max(min(float(cfg.qp_eps) ** 2, 1e-12), 1e-16)
    res = minimize(
        fun,
        x0,
        jac=jac,
        hess=hess,
        constraints=[cons],
        method="trust-constr",
        options={
            "gtol": tol,
            "xtol": tol,
            "maxiter": int(min(cfg.qp_max_iter, 3000)),
            "verbose": 0,
        },
    )
    iters = int(getattr(res, "niter", 0) or 0)
    max_viol = float(getattr(res, "constr_violation", np.inf))
    if not res.success or not np.isfinite(max_viol):
        return _failed(
            time.perf_counter() - t0,
            f"scipy trust-constr status={getattr(res, 'status', None)}: {res.message}",
            iters,
        )
    status = "optimal" if max_viol <= 1e-8 else "inaccurate"
    return _finish(
        qp,
        np.asarray(res.x, dtype=np.float64),
        status,
        t0,
        iters,
        note=f"scipy constr_violation={max_viol:.3e}",
    )
