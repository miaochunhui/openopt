"""Axioma 051 two-stage closed-form solution (docs/mpo-t1-SPEC.md section 3).

Reference: Axioma Research Paper 051, *Multi-Period Portfolio Optimization with
Alpha Decay*, eq. 29-30. For the two-period unconstrained problem

    max_{w1, w2}  alpha1'w1 - (gamma/2) w1'S w1 - (delta/2)(w1 - w0)'S (w1 - w0)
                + E[alpha2]'w2 - (gamma/2) w2'S w2 - (delta/2)(w2 - w1)'S (w2 - w1)

the first-order conditions give (with ``a = (gamma+delta)/((gamma+delta)^2 + delta*gamma)``)

    w1 = a * S^-1 alpha1 + a * delta * w0 + a * (delta/(delta+gamma)) * S^-1 E[alpha2]

This module is the analytic reference used by ``tests/mpo/test_qp_analytic.py``
to validate the QP assembly (``cost_mode='axioma'``, all box/budget/T+1
constraints relaxed): identical objective, so agreement must be to solver
accuracy (< 1e-6).

Note the sign convention: :mod:`openpo.mpo.qp` minimises the negated
objective, and its axioma mode uses the *unscaled* per-stage risk term
``(gamma/2) w'S w`` (no ``L_j/240`` factor) exactly so that this parity holds.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

__all__ = ["closed_form_two_stage"]


def closed_form_two_stage(
    alpha1,
    alpha2_exp,
    Sigma,
    gamma: float,
    delta: float,
    w0,
) -> np.ndarray:
    """Stage-1 holdings ``w1`` of the unconstrained two-stage problem.

    Parameters
    ----------
    alpha1 : array [N]
        Stage-1 expected returns.
    alpha2_exp : array [N]
        Stage-2 expected returns as seen at decision time, ``E[alpha2]``. Any
        alpha decay must already be baked in by the caller (this function does
        not decay anything).
    Sigma : array [N, N]
        Symmetric positive definite covariance used by *both* the risk and the
        transition-cost term (Axioma's assumption).
    gamma : float
        Risk aversion (> 0).
    delta : float
        Transition (trading) cost coefficient (>= 0).
    w0 : array [N]
        Current holdings.

    Returns
    -------
    np.ndarray
        ``w1`` [N], float64.

    Raises
    ------
    ValueError
        On shape mismatch, non-finite input, non-positive ``gamma``, negative
        ``delta``, or a non-positive-definite ``Sigma``.
    """
    alpha1 = np.asarray(alpha1, dtype=np.float64).ravel()
    alpha2_exp = np.asarray(alpha2_exp, dtype=np.float64).ravel()
    w0 = np.asarray(w0, dtype=np.float64).ravel()
    sigma = np.asarray(Sigma, dtype=np.float64)
    n = alpha1.size
    if alpha2_exp.size != n or w0.size != n:
        raise ValueError(
            f"alpha1/alpha2_exp/w0 must share length, got "
            f"{alpha1.size}/{alpha2_exp.size}/{w0.size}"
        )
    if sigma.shape != (n, n):
        raise ValueError(f"Sigma must be [{n},{n}], got {sigma.shape}")
    for name, arr in (("alpha1", alpha1), ("alpha2_exp", alpha2_exp), ("w0", w0)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("Sigma contains non-finite values")

    gamma = float(gamma)
    delta = float(delta)
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative, got {delta}")

    sigma = 0.5 * (sigma + sigma.T)
    try:
        chol = cho_factor(sigma, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Sigma must be positive definite: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Sigma must be positive definite: {exc}") from exc

    # Solve Sigma X = [alpha1, alpha2_exp] once (never form an explicit inverse).
    rhs = np.column_stack((alpha1, alpha2_exp))
    sol = cho_solve(chol, rhs, check_finite=False)
    sig_inv_a1 = sol[:, 0]
    sig_inv_a2 = sol[:, 1]

    a = (gamma + delta) / ((gamma + delta) ** 2 + delta * gamma)
    w1 = a * sig_inv_a1 + a * delta * w0
    if delta > 0.0:
        w1 = w1 + a * (delta / (delta + gamma)) * sig_inv_a2
    return np.ascontiguousarray(w1, dtype=np.float64)
