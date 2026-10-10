"""MPO-T1 statistical comparison utilities (SPEC section 8).

Seed-level paired comparisons between strategies: paired t-test, seed-level
bootstrap confidence interval, win rate, Hedges' g (paired / d_z form), and
Holm step-down correction for multi-strategy comparison families.

Inputs are the per-seed metric dicts produced by
``openpo.mpo.backtest.run_seed`` (one dict per seed mapping strategy name ->
metric summary). This module is pure numpy/scipy and has no dependency on
sibling MPO modules.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

__all__ = ["holm_correction", "paired_stats"]


def _extract_diffs(per_seed: list[dict], a: str, b: str, metric: str) -> np.ndarray:
    """Seed-aligned metric differences a - b as float64."""
    diffs = np.empty(len(per_seed), dtype=np.float64)
    for i, seed_metrics in enumerate(per_seed):
        try:
            va = seed_metrics[a][metric]
            vb = seed_metrics[b][metric]
        except KeyError as exc:
            raise KeyError(
                f"seed entry {i}: missing strategy/metric {exc} "
                f"(need per_seed[i][{a!r}/{b!r}][{metric!r}])"
            ) from exc
        diffs[i] = float(va) - float(vb)
    if not np.all(np.isfinite(diffs)):
        raise ValueError(f"non-finite {metric} differences at seeds "
                         f"{np.where(~np.isfinite(diffs))[0].tolist()}")
    return diffs


def paired_stats(
    per_seed: list[dict],
    a: str,
    b: str,
    metric: str = "net_sharpe_annual",
    n_boot: int = 10000,
    rng_seed: int = 0,
) -> dict:
    """Paired seed-level comparison of strategy ``a`` minus strategy ``b``.

    Returns a dict with:

    - ``mean_diff``: mean of per-seed differences (a - b)
    - ``t_stat`` / ``p_value``: two-sided paired t-test (df = n_seeds - 1)
    - ``ci95_low`` / ``ci95_high``: percentile bootstrap CI of the mean
      difference, resampling SEEDS with replacement (n_boot draws,
      deterministic via ``rng_seed``)
    - ``win_rate``: fraction of seeds with a > b (exact ties count 0.5)
    - ``hedges_g``: bias-corrected paired effect size
      ``J * mean_diff / sd_diff`` with ``J = 1 - 3/(4*df - 1)``
    - ``mean_a`` / ``mean_b`` / ``n_seeds`` / ``metric`` for reporting
    """
    if len(per_seed) < 2:
        raise ValueError("paired_stats needs at least 2 seeds")
    diffs = _extract_diffs(per_seed, a, b, metric)
    vals_a = np.array([float(s[a][metric]) for s in per_seed])
    vals_b = np.array([float(s[b][metric]) for s in per_seed])

    n = diffs.size
    mean_diff = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    df = n - 1

    if sd > 0.0:
        t_stat = mean_diff / (sd / math.sqrt(n))
        p_value = float(2.0 * sps.t.sf(abs(t_stat), df))
    else:
        # all differences identical: degenerate but well-defined direction
        t_stat = 0.0 if mean_diff == 0.0 else math.copysign(math.inf, mean_diff)
        p_value = 1.0 if mean_diff == 0.0 else 0.0

    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = diffs[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    win_rate = float(np.mean(diffs > 0) + 0.5 * np.mean(diffs == 0))

    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    if sd > 0.0:
        hedges_g = correction * mean_diff / sd
    else:
        hedges_g = 0.0 if mean_diff == 0.0 else math.copysign(math.inf, mean_diff)

    return {
        "strategy_a": a,
        "strategy_b": b,
        "metric": metric,
        "n_seeds": int(n),
        "mean_a": float(vals_a.mean()),
        "mean_b": float(vals_b.mean()),
        "mean_diff": mean_diff,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "win_rate": win_rate,
        "hedges_g": float(hedges_g),
    }


def holm_correction(pvals) -> np.ndarray:
    """Holm step-down multiple-testing correction.

    Returns adjusted p-values in the ORIGINAL input order:
    ``adj_(i) = min(1, max_{j <= i} (m - j + 1) * p_(j))`` over the sorted
    sequence (enforced monotone non-decreasing, clipped at 1). Rejecting
    ``adj <= alpha`` reproduces the classic Holm procedure at level alpha.
    """
    p = np.asarray(pvals, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvals must be a non-empty 1-D sequence")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("pvals must be finite and within [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted
