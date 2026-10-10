"""Summary metrics for a single :class:`BacktestResult` (SPEC section 5/8).

This module is the drop-in replacement that :func:`openpo.mpo.backtest.
_summarize_fn` prefers, so :func:`summarize` MUST emit the canonical metric
contract documented on ``backtest.summarize`` (consumed by
``stats.paired_stats`` and ``report.py``) with identical numeric semantics.
Extra legacy keys (``ann_return``, ``cost_bp_breakdown``, ...) are emitted
*in addition* for the older simulator tests; they never replace canonical
ones.

Annualization uses ``TRADING_DAYS_PER_YEAR = 240`` (synthetic calendar).
The ``cost_bp_breakdown`` extra is expressed in basis points of TOTAL TRADED
VALUE (成交额), not of NAV, so it is directly comparable to the per-side cost
assumptions (buy 7bp / sell 12bp at defaults); the canonical ``costs`` dict
stays in raw CNY like ``backtest.summarize``. Trading minutes elsewhere in
the project are tau differences (overnight = 0 minutes); nothing here depends
on that beyond the daily grid.
"""

from __future__ import annotations

import numpy as np

from .config import TRADING_DAYS_PER_YEAR
from .types import BacktestResult

_COST_KEYS = ("commission", "stamp_tax", "slippage", "impact_est")


def _max_drawdown(nav: np.ndarray) -> float:
    """Max peak-to-trough drawdown as a positive fraction (0 = no drawdown)."""
    peak = np.maximum.accumulate(nav)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = 1.0 - nav / peak
    return float(np.nanmax(dd)) if dd.size else 0.0


def summarize(result: BacktestResult) -> dict:
    """Flatten one backtest into a plain-scalar dict (JSON-serializable).

    Canonical keys (identical semantics to ``backtest.summarize``)
    -------------------------------------------------------------
    - ``strategy`` / ``seed`` / ``n_eval_days``: identity + evaluated days.
    - ``net_sharpe_annual``: mean/std of daily net returns * sqrt(240).
    - ``net_return_annual``: geometric annualized net return on the NAV path.
    - ``gross_return_annual``: same, with total costs added back to the
      terminal NAV (aggregate approximation -- daily cost attribution is not
      persisted in BacktestResult).
    - ``avg_daily_turnover``: mean of daily traded value / day-open NAV.
    - ``max_drawdown``: positive fraction, peak-to-trough on the NAV path.
    - ``costs``: CNY breakdown as reported by the simulator plus ``total``.
    - ``total_cost_frac_nav``: total costs / starting NAV.
    - ``rejected_orders`` / ``truncated_buys`` / ``t1_violations``: counters.
    - ``qp_time_p50`` / ``qp_time_p95``: QP wall clock percentiles in seconds
      (0.0 when the strategy reported no solves).
    - ``qp_status_counts``: solver status histogram.
    - ``final_nav``: last point of the NAV path.

    Extra keys (kept for the simulator-level tests, never a substitute)
    ------------------------------------------------------------------
    - ``n_days``: alias of ``n_eval_days``.
    - ``ann_return``: alias of ``net_return_annual``.
    - ``ann_vol``: daily return std (ddof=1) * sqrt(240).
    - ``cost_bp_breakdown``: {commission, stamp_tax, slippage, impact_est}
      in bp of total traded value (0.0 across the board if nothing traded).
    - ``qp_time_p50_s`` / ``qp_time_p95_s``: aliases of the canonical
      percentiles; ``n_qp_solves``: number of recorded QP solves.
    """
    nav = np.asarray(result.daily_nav, dtype=np.float64)
    if nav.ndim != 1 or nav.size < 2:
        raise ValueError("daily_nav must be a 1-D path with >= 2 points")
    n_days = nav.size - 1
    ann_pow = TRADING_DAYS_PER_YEAR / n_days
    rets = result.daily_returns()

    nav0 = float(nav[0])
    nav_end = max(float(nav[-1]), 1e-9)  # guard pathological blow-ups
    net_return_annual = float((nav_end / nav0) ** ann_pow - 1.0)

    costs = {k: float(v) for k, v in result.costs.items()}
    total_costs = float(sum(costs.values()))
    costs["total"] = total_costs
    gross_return_annual = float(((nav_end + total_costs) / nav0) ** ann_pow - 1.0)

    sd = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
    ann_vol = sd * float(np.sqrt(TRADING_DAYS_PER_YEAR))

    turnover = np.asarray(result.daily_turnover, dtype=np.float64)
    avg_turnover = float(turnover.mean()) if turnover.size else 0.0
    # Total traded value: prefer the simulator's exact CNY total, else rebuild
    # it from turnover * day-open NAV (identical by construction).
    traded = float(result.meta.get("traded_value_cny", np.sum(turnover * nav[:-1])))
    if traded > 0.0:
        cost_bp = {k: float(result.costs.get(k, 0.0)) / traded * 1e4 for k in _COST_KEYS}
    else:
        cost_bp = {k: 0.0 for k in _COST_KEYS}

    times = np.asarray(result.qp_solve_times, dtype=np.float64)
    qp_p50 = float(np.percentile(times, 50)) if times.size else 0.0
    qp_p95 = float(np.percentile(times, 95)) if times.size else 0.0

    return {
        # -- canonical contract -------------------------------------------
        "strategy": result.strategy,
        "seed": int(result.seed),
        "n_eval_days": int(n_days),
        "net_sharpe_annual": float(result.net_sharpe_annual()),
        "net_return_annual": net_return_annual,
        "gross_return_annual": gross_return_annual,
        "avg_daily_turnover": avg_turnover,
        "max_drawdown": _max_drawdown(nav),
        "costs": costs,
        "total_cost_frac_nav": total_costs / nav0,
        "rejected_orders": int(result.rejected_orders),
        "truncated_buys": int(result.truncated_buys),
        "t1_violations": int(result.t1_violations),
        "qp_time_p50": qp_p50,
        "qp_time_p95": qp_p95,
        "qp_status_counts": dict(result.qp_status_counts),
        "qp_failure_notes": dict(result.meta.get("qp_failure_notes", {})),
        "final_nav": float(nav[-1]),
        # -- extras (aliases + traded-value bp view) ----------------------
        "n_days": int(n_days),
        "ann_return": net_return_annual,
        "ann_vol": ann_vol,
        "cost_bp_breakdown": cost_bp,
        "qp_time_p50_s": qp_p50,
        "qp_time_p95_s": qp_p95,
        "n_qp_solves": int(times.size),
    }
