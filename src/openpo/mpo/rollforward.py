"""Roll-forward return magnitude bound for MPO-T1 (Iteration 44 leg 1).

Authoritative design: ``docs/mpo-t1-rollforward-design.md``. That document's
verdict is **NO-GO BEFORE IMPLEMENTATION**: at MPO-T1 scales the roll-forward
correction cannot move net Sharpe by as much as the 24-seed A/B can resolve, so
no roll-forward arm is built. This module exists to make that verdict
*machine-checked instead of prose*: it ships the arithmetic behind the bound and
nothing else. There is deliberately NO corrected-dynamics builder here, and
nothing in this file is imported by ``qp.py`` / ``strategies.py`` /
``backtest.py``.

What roll-forward means in OUR formulation
-----------------------------------------
Axioma 051 eq(1) links its two stages by

    w^2 = (1 + r) (o) w^1 + dw^2          ((o) = elementwise)

with ``r`` the *per-asset* expected return over the stage the portfolio is held
through, and states plainly that ``r = 0`` is assumed throughout the paper. Our
SPEC section 3 dynamics are that same row with ``r = 0``::

    w^j - w^{j-1} - b^j + s^j = 0

The model-consistent estimate of ``r`` for stage ``j`` is the term structure's
own stage alpha, so the correction to the stage-``j`` dynamics right-hand side is

    delta^j = w^{j-1} (o) alpha^j        (delta^1 uses the KNOWN holdings w^0)

**The exact reduction that fixes the magnitude.** Because the trade split
``b/s`` is free, ANY weight path ``{w^j}`` feasible for the corrected dynamics is
also feasible for the frozen ones and vice versa -- only the ``b/s`` that carry
that path shift, by exactly ``delta^j``. The alpha term ``-alpha^j . w^j``, the
risk term and the box/budget/T+1 rows are therefore *identical* between the two
problems for a fixed path; the ONLY thing roll-forward changes is where the
trade-cost origin sits. Hence, with ``c = cfg.buy_cost() + cfg.sell_cost()``,

    | Obj*_rollforward - Obj*_frozen |  <=  c * ||delta||_1                (B)

(the quadratic impact term adds ``D (b+s) . delta + D delta^2 / 2``, four orders
of magnitude below the linear part at MPO-T1 turnover, see the design doc). (B)
is an upper bound on the whole economic content of the correction -- the change
in *realized* performance is second order inside it, since ``w*`` already
satisfies KKT for the frozen problem.

:func:`rollforward_magnitude_bound` evaluates (B) on a real solved path and
converts it to a net-Sharpe ceiling::

    sharpe_bound = c * ||delta||_1 * epochs_per_year / ann_vol

and the verdict is ``'negligible'`` exactly when that ceiling is below the A/B's
own measured resolution AND the largest per-name correction is below the
execution layer's no-trade band (below the band, a per-name correction cannot
change a single order at all). Both conditions are declared up front in
:data:`DEFAULT_DETECT_FLOOR_SHARPE` / :data:`DEFAULT_NO_TRADE_BAND` and both are
pinned in ``tests/mpo/test_rollforward.py``.

The calculator has discriminating power, which is the part that matters for
honesty: fed *oracle* roll-forward returns (realized stage returns instead of
the ~1/IC times smaller conditional means, which is exactly the experiment
Axioma 051's conclusion reports as "drastically improved"), the same function
returns ``'material'``. The NO-GO is a statement about our IC, not about the
mechanism.

Units: returns/weights are fractions of NAV (``1e-4 == 1 bp``), matching
``TermStructureModel.stage_alphas`` and ``MpoConfig``. Pure numpy, float64, no
I/O, no RNG, no mutation of inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "COMMITTED_ANCHORS",
    "DEFAULT_DETECT_FLOOR_SHARPE",
    "DEFAULT_EPOCHS_PER_YEAR",
    "DEFAULT_NO_TRADE_BAND",
    "ROLLFORWARD_CANDIDATES",
    "RollForwardBound",
    "cumulative_growth",
    "rollforward_drift",
    "rollforward_magnitude_bound",
    "sharpe_bound_from_l1",
]

#: The three candidate formulations weighed in the design doc, with the reason
#: each was rejected. Kept in code so a future reader cannot re-litigate a
#: candidate without reading why it died.
ROLLFORWARD_CANDIDATES = {
    "asset_drift": (
        "dynamics RHS delta^j = w^{j-1} (o) alpha^j -- Axioma 051 eq(1)'s own r, "
        "convex (deterministic RHS shift). REJECTED on magnitude: bound (B)."
    ),
    "alpha_growth": (
        "scale alpha^j by the cumulative wealth growth prod_{k<j}(1+g_k). "
        "REJECTED: g_k ~ 5e-5 at MPO-T1 scales, so this is a <1e-4 relative "
        "rescaling of alpha, i.e. observationally a gamma reparameterisation "
        "inside gamma's own identification noise (gamma has a {0.5,1,2}-scale "
        "sensitivity axis)."
    ),
    "budget_growth": (
        "scale the budget row to 1'w^j <= prod_{k<=j}(1+g_k). REJECTED as "
        "structurally inert: measured gross exposure is ~0.14 against a bound of "
        "1.0, so the budget row carries ~86% slack and moving its bound by 5e-5 "
        "changes nothing."
    ),
}

#: Paired-A/B resolution floor on annualised net Sharpe, in Sharpe units: the
#: half-width of the 24-seed bootstrap CI actually measured for a near-identical
#: arm (Iteration 40 shrinkage A/B, mpo - mpo_shrunk = +0.0078,
#: 95% CI [-0.0118, +0.0277] -> half-width 0.01975; see
#: docs/mpo-t1-shrinkage-design.md "实测判决"). An effect whose a-priori CEILING
#: is under this cannot be established by the same design, so building the arm
#: is unjustified before the floor drops.
DEFAULT_DETECT_FLOOR_SHARPE = 0.01975

#: ``cfg.min_order_nav_frac`` -- ``execution.plan_orders`` drops any order whose
#: notional is below ``max(lot_size * price, min_order_nav_frac * NAV)``. At the
#: frozen prices (~20 CNY) and NAV (1e8) the lot leg is ~2e-5, so this 2e-4 leg
#: is the binding one.
DEFAULT_NO_TRADE_BAND = 2e-4

#: 8 decision epochs per day x 240 trading days.
DEFAULT_EPOCHS_PER_YEAR = 8 * 240


def _check_2d(name: str, arr, n_stages: int | None = None, n_assets: int | None = None):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D [J, N], got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty (shape {a.shape}): nothing to bound")
    if n_stages is not None and a.shape[0] != n_stages:
        raise ValueError(
            f"{name} has {a.shape[0]} stages but alpha has {n_stages}"
        )
    if n_assets is not None and a.shape[1] != n_assets:
        raise ValueError(f"{name} has {a.shape[1]} assets but alpha has {n_assets}")
    bad = ~np.isfinite(a)
    if bad.any():
        idx = np.unravel_index(int(np.flatnonzero(bad.ravel())[0]), a.shape)
        raise ValueError(
            f"{name} must be finite everywhere: {int(bad.sum())} bad entries, first "
            f"at {idx} (value {a[idx]!r}). Halted names arrive as 0.0, never NaN."
        )
    return a


def _check_w0(w0, n_assets: int) -> np.ndarray:
    a = np.asarray(w0, dtype=np.float64)
    if a.shape != (n_assets,):
        raise ValueError(f"w0 must have shape ({n_assets},), got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("w0 must be finite everywhere (current holdings are known)")
    return a


def _check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _check_non_negative(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def rollforward_drift(alpha, w_path, w0) -> np.ndarray:
    """Per-stage roll-forward correction to the dynamics right-hand side.

    ``delta[0] = w0 * alpha[0]`` and ``delta[j] = w_path[j-1] * alpha[j]`` for
    ``j >= 1``: stage ``j`` is held through with the weights that ENTER it, so
    the weights that drift are the previous stage's (``w^0`` = current holdings
    for the execution stage, which makes ``delta[0]`` fully deterministic --
    no circular dependence on the solution).

    Parameters
    ----------
    alpha : array_like
        ``[J, N]`` per-stage expected returns (``TermStructureModel.stage_alphas``
        output; return fractions).
    w_path : array_like
        ``[J, N]`` stage-end weights of a SOLVED path (``MpoSolution.weights``).
    w0 : array_like
        ``[N]`` current holdings weights (``PortfolioSnapshot.weights``).

    Returns
    -------
    np.ndarray
        Fresh ``[J, N]`` float64 array, same units as ``alpha * w`` (fraction of
        NAV). Sign follows ``alpha``: a name with positive expected return drifts
        UP in weight, so less of it has to be bought.
    """
    a = _check_2d("alpha", alpha)
    j, n = a.shape
    w = _check_2d("w_path", w_path, n_stages=j, n_assets=n)
    prev = np.empty_like(w)
    prev[0] = _check_w0(w0, n)
    if j > 1:
        prev[1:] = w[:-1]
    return prev * a


def cumulative_growth(alpha, w_path, w0) -> tuple[np.ndarray, np.ndarray]:
    """Expected per-stage wealth growth and its cumulative product.

    ``g[j] = alpha[j] . w_entering_stage_j`` is the portfolio's expected return
    over stage ``j`` under the model's own alphas (cash earns nothing, so the
    uninvested part contributes 0); ``G[j] = prod_{k <= j} (1 + g[k])`` is the
    expected wealth multiple by the end of stage ``j``. ``G[-1] - 1`` is the
    scalar that candidate ``alpha_growth`` / ``budget_growth`` would multiply
    into the objective or the budget row.

    Returns
    -------
    (g, G) : tuple of np.ndarray
        Two ``[J]`` float64 arrays.
    """
    a = _check_2d("alpha", alpha)
    j, n = a.shape
    w = _check_2d("w_path", w_path, n_stages=j, n_assets=n)
    prev = np.empty_like(w)
    prev[0] = _check_w0(w0, n)
    if j > 1:
        prev[1:] = w[:-1]
    g = np.einsum("jn,jn->j", a, prev)
    return g, np.cumprod(1.0 + g)


@dataclass(frozen=True)
class RollForwardBound:
    """Verdict record of the roll-forward magnitude bound (design doc section 3).

    All money-like fields are fractions of NAV per DECISION EPOCH unless the name
    says ``annual``. ``verdict`` is the machine-checked judgement:
    ``'negligible'`` (= the design's NO-GO) or ``'material'``.
    """

    delta_l1: float
    """``||delta||_1`` summed over stages and names -- the total absolute weight
    the correction moves per epoch."""

    delta_max: float
    """``max |delta|`` over stages and names -- the largest single-name change."""

    delta_l1_per_stage: np.ndarray
    delta_max_per_stage: np.ndarray

    growth_per_stage: np.ndarray
    """``g[j]``, the model's expected portfolio return over stage ``j``."""

    growth_total: float
    """``prod_j (1 + g[j]) - 1``: the cumulative expected wealth growth over the
    whole horizon, i.e. the scalar candidate ``alpha_growth`` would apply."""

    cost_sum: float
    """``cfg.buy_cost() + cfg.sell_cost()`` -- the leverage factor in bound (B)."""

    obj_bound: float
    """``cost_sum * delta_l1``: bound (B), per epoch, fraction of NAV."""

    obj_bound_annual: float
    sharpe_bound: float
    """``obj_bound_annual / ann_vol`` -- ceiling on |delta net Sharpe|."""

    detect_floor_sharpe: float
    band_ratio: float
    """``delta_max / no_trade_band``; < 1 means no single-name correction can by
    itself clear ``execution.plan_orders``' minimum order size."""

    turnover_ratio: float
    """``delta_l1 / turnover_per_epoch`` -- the correction as a fraction of the
    trade vector it perturbs (NaN when turnover was not supplied)."""

    verdict: str
    notes: dict = field(default_factory=dict)

    @property
    def is_negligible(self) -> bool:
        return self.verdict == "negligible"


def _verdict(sharpe_bound: float, band_ratio: float, floor: float) -> str:
    """Committed decision rule (design doc section 3, frozen before measurement).

    ``'negligible'`` requires BOTH legs:

    1. ``sharpe_bound < detect_floor_sharpe`` -- the ceiling on the effect is
       below what the preregistered 24-seed paired A/B can resolve, so the
       experiment could not establish the effect even if it were real; and
    2. ``band_ratio < 1`` -- the largest per-name correction is under the
       execution layer's no-trade band, so it is quantised away before it can
       reach the market.

    Either leg failing gives ``'material'`` (build the arm and measure).
    """
    if sharpe_bound < floor and band_ratio < 1.0:
        return "negligible"
    return "material"


def sharpe_bound_from_l1(
    delta_l1: float,
    *,
    cost_sum: float,
    ann_vol: float,
    epochs_per_year: float = DEFAULT_EPOCHS_PER_YEAR,
) -> tuple[float, float, float]:
    """Scalar core of bound (B): ``||delta||_1 -> (obj, obj_annual, sharpe)``.

    Separated from the array API so the design doc's recorded anchors can be
    re-derived (and pinned by a test) without re-running a backtest.

    Returns
    -------
    (obj_bound, obj_bound_annual, sharpe_bound)
        Per-epoch objective ceiling (fraction of NAV), its annualisation, and the
        implied ceiling on the annualised net-Sharpe difference.
    """
    l1 = _check_non_negative("delta_l1", delta_l1)
    c = _check_non_negative("cost_sum", cost_sum)
    vol = _check_positive("ann_vol", ann_vol)
    epy = _check_positive("epochs_per_year", epochs_per_year)
    obj = c * l1
    obj_annual = obj * epy
    return obj, obj_annual, obj_annual / vol


def rollforward_magnitude_bound(
    alpha,
    w_path,
    w0,
    *,
    cost_sum: float,
    ann_vol: float,
    epochs_per_year: float = DEFAULT_EPOCHS_PER_YEAR,
    no_trade_band: float = DEFAULT_NO_TRADE_BAND,
    turnover_per_epoch: float | None = None,
    detect_floor_sharpe: float = DEFAULT_DETECT_FLOOR_SHARPE,
) -> RollForwardBound:
    """Evaluate bound (B) on one solved multi-stage path and return the verdict.

    Parameters
    ----------
    alpha, w_path, w0 :
        As in :func:`rollforward_drift` (one decision epoch).
    cost_sum : float
        ``cfg.buy_cost() + cfg.sell_cost()`` (19 bp at the frozen defaults). This
        is the ONLY channel through which roll-forward can change the objective
        (module docstring), so a cost-free world has a zero bound.
    ann_vol : float
        Annualised net return volatility of the arm being perturbed (main-axis
        measured mean: 0.0176). Converts a return ceiling into a Sharpe ceiling.
    epochs_per_year : float
        Decision epochs per year; the per-epoch bound is additive across epochs
        (worst case: every epoch's correction pulls the same way).
    no_trade_band : float
        ``cfg.min_order_nav_frac`` in NAV fractions.
    turnover_per_epoch : float, optional
        One-epoch turnover in NAV fractions, for the reported ``turnover_ratio``
        only -- it never enters the verdict.
    detect_floor_sharpe : float
        A/B resolution floor; see :data:`DEFAULT_DETECT_FLOOR_SHARPE`.

    Returns
    -------
    RollForwardBound
    """
    delta = rollforward_drift(alpha, w_path, w0)
    absd = np.abs(delta)
    l1_stage = absd.sum(axis=1)
    max_stage = absd.max(axis=1)
    l1 = float(l1_stage.sum())
    dmax = float(max_stage.max())

    g, G = cumulative_growth(alpha, w_path, w0)
    obj, obj_annual, sharpe = sharpe_bound_from_l1(
        l1, cost_sum=cost_sum, ann_vol=ann_vol, epochs_per_year=epochs_per_year
    )
    band = _check_positive("no_trade_band", no_trade_band)
    band_ratio = dmax / band
    if turnover_per_epoch is None:
        turn_ratio = float("nan")
    else:
        turn_ratio = l1 / _check_positive("turnover_per_epoch", turnover_per_epoch)
    floor = _check_positive("detect_floor_sharpe", detect_floor_sharpe)

    return RollForwardBound(
        delta_l1=l1,
        delta_max=dmax,
        delta_l1_per_stage=l1_stage,
        delta_max_per_stage=max_stage,
        growth_per_stage=g,
        growth_total=float(G[-1] - 1.0),
        cost_sum=float(cost_sum),
        obj_bound=obj,
        obj_bound_annual=obj_annual,
        sharpe_bound=sharpe,
        detect_floor_sharpe=floor,
        band_ratio=band_ratio,
        turnover_ratio=turn_ratio,
        verdict=_verdict(sharpe, band_ratio, floor),
        notes={
            "n_stages": int(delta.shape[0]),
            "n_assets": int(delta.shape[1]),
            "ann_vol": float(ann_vol),
            "epochs_per_year": float(epochs_per_year),
            "no_trade_band": band,
            "turnover_per_epoch": (
                None if turnover_per_epoch is None else float(turnover_per_epoch)
            ),
        },
    )


#: Measured anchors behind the design doc's NO-GO (docs/mpo-t1-rollforward-design.md
#: section 3). Recorded from a real 32-epoch MPO path (seed 20260821, N=300,
#: calib_days=20, eval days 20-23), per-epoch means; ``ann_vol`` and
#: ``turnover_per_epoch`` come from the 24-seed main-axis benchmark
#: (benchmark-results/mpo-t1-main-v2.json). ``ic_mult=1.5, cost_mult=0.5`` is the
#: cell where the correction is largest relative to the frozen one.
#: ``tests/mpo/test_rollforward.py::TestCommittedVerdict`` re-derives the verdict
#: from these numbers, so the NO-GO cannot silently rot.
COMMITTED_ANCHORS: dict = {
    "main": {
        "delta_l1": 5.1605e-05,  # per-epoch mean; worst single epoch 1.0880e-04
        "delta_max": 9.4543e-06,  # worst single epoch 1.8858e-05
        "growth_total": 1.5145e-05,
        "cost_sum": 1.9e-03,  # cfg.buy_cost() + cfg.sell_cost() at cost_mult=1
        "ann_vol": 1.764e-02,  # 24-seed main-axis mean
        "turnover_per_epoch": 1.0625e-02,  # 24-seed mean 8.50%/day over 8 epochs
        "sharpe_bound": 0.010672,
    },
    "ic1.5_cost0.5": {
        "delta_l1": 3.2215e-04,  # worst single epoch 4.9890e-04
        "delta_max": 2.7788e-05,  # worst single epoch 4.3041e-05
        "growth_total": 1.5755e-04,
        "cost_sum": 9.5e-04,  # cost_mult=0.5
        # Deliberately conservative: the ic=1.5 sensitivity cell's own vol
        # (3.111e-2), which IGNORES the 4.2x book expansion cost_mult=0.5 causes
        # (measured gross 0.603 vs 0.144) -- using the expanded vol would divide
        # the bound by ~4.2 again.
        "ann_vol": 3.111e-02,
        "turnover_per_epoch": 1.6989e-01,  # probe-measured 1.359/day over 8 epochs
        "sharpe_bound": 0.018888,
    },
}
