"""MPO-T1 backtest orchestration (SPEC section 8).

Wires the per-seed pipeline: synthetic market generation -> term-structure
calibration on the calib window -> every strategy replayed on the *same*
market over the eval window -> per-strategy metric summaries.

Project-wide conventions (repeated here on purpose):

- ``tau = day * 240 + minute``; all horizons/decays are counted in TRADING
  minutes, overnight counts as ZERO minutes.
- The synthetic calendar has no holidays, so ``trading_minutes_between(a, b)
  == b - a`` exactly; tau differences ARE trading-minute distances.
- Calibration uses days ``[0, calib_days)``; evaluation uses days
  ``[eval_start_day, n_days)`` with ``eval_start_day >= calib_days`` enforced
  (lookahead guard at the orchestration level).

Sibling modules (synth / term_structure / strategies / simulator) are
imported lazily inside :func:`run_seed` so this module and ``stats.py`` can
be imported and unit-tested before every sibling has landed.  Strategy
constructors are invoked through a signature-matching adapter fed from a
shared candidate pool, which tolerates the parameter-name variations the
SPEC prose allows (``ts_model`` vs ``model``, ``risk`` vs ``cov``, ...).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from collections.abc import Iterable

import numpy as np

from openpo.mpo.config import (
    ExperimentConfig,
    SIGNAL_HORIZONS,
    TRADING_DAYS_PER_YEAR,
)
from openpo.mpo.types import BacktestResult

__all__ = [
    "DEFAULT_STRATEGIES",
    "KNOWN_STRATEGIES",
    "OPT_IN_STRATEGIES",
    "default_fixed_lambdas",
    "run_seed",
    "simplex_grid",
    "summarize",
    "tune_fixed_weights",
]

# Canonical strategy roster (SPEC section 7). Order = reporting order.
DEFAULT_STRATEGIES: tuple[str, ...] = (
    "mpo",
    "single_30",
    "single_240",
    "single_2400",
    "fixed_weight",
    "informed",
    "oracle_mpo",
    "buy_hold",
)

# Opt-in research arms: buildable by name via ``run_seed(strategies=[...])`` but
# deliberately OUT of the default roster, so a default run does not pay for them.
# ``mpo_shrunk`` (kappa alpha shrinkage, docs/mpo-t1-shrinkage-design.md) was
# demoted here by its own Iteration 40 A/B verdict: NEUTRAL at calib_days=20 --
# main 24-seed paired mpo - mpo_shrunk = +0.008 net Sharpe, bootstrap 95% CI
# [-0.012, +0.028], p = 0.46 (preregistered "kappa wins in the low-IC regime"
# was not confirmed), so it stays available for research but earns no default slot.
# ``mpo_delayaware`` (implementation-delay surcharge,
# docs/mpo-t1-delaycost-design.md) starts here by design: leg 2 wires it as an
# opt-in A/B arm and only the preregistered 24-seed test may promote it.
OPT_IN_STRATEGIES: tuple[str, ...] = ("mpo_shrunk", "mpo_delayaware")

# Everything ``_build_strategy`` knows how to construct.
KNOWN_STRATEGIES: tuple[str, ...] = DEFAULT_STRATEGIES + OPT_IN_STRATEGIES

# single_* -> index into the (return_1_30, return_1_240, return_1_2400) axis.
_SIGNAL_INDEX = {"single_30": 0, "single_240": 1, "single_2400": 2}

# Private candidate-pool slot memoizing the kappa-shrunk term-structure model.
# Underscore-prefixed on purpose: ``_instantiate`` only reads pool keys that a
# constructor actually declares, so this can never leak into a signature.
_SHRUNK_MODEL_KEY = "_shrunk_ts_model"


# ---------------------------------------------------------------------------
# metric summary
# ---------------------------------------------------------------------------

def summarize(result: BacktestResult) -> dict:
    """Flatten a :class:`BacktestResult` into the per-strategy metric dict.

    Keys are stable (consumed by ``stats.paired_stats`` and the report):
    net_sharpe_annual, net_return_annual, gross_return_annual,
    avg_daily_turnover, max_drawdown, costs (CNY breakdown + total),
    total_cost_frac_nav, rejected_orders, truncated_buys, t1_violations,
    qp_time_p50 / qp_time_p95 (seconds), qp_status_counts, n_eval_days,
    final_nav.

    Gross return adds total costs back to terminal NAV (aggregate
    approximation -- daily cost attribution is not persisted in
    BacktestResult).
    """
    nav = np.asarray(result.daily_nav, dtype=np.float64)
    if nav.ndim != 1 or nav.size < 2:
        raise ValueError("daily_nav must be a 1-D path with >= 2 points")
    n_days = nav.size - 1
    ann_pow = TRADING_DAYS_PER_YEAR / n_days

    nav0 = float(nav[0])
    nav_end = max(float(nav[-1]), 1e-9)  # guard pathological blow-ups
    net_return_annual = (nav_end / nav0) ** ann_pow - 1.0

    costs = {k: float(v) for k, v in result.costs.items()}
    total_costs = float(sum(costs.values()))
    costs["total"] = total_costs
    gross_return_annual = ((nav_end + total_costs) / nav0) ** ann_pow - 1.0

    running_max = np.maximum.accumulate(nav)
    max_drawdown = float(np.max(1.0 - nav / running_max))

    turnover = np.asarray(result.daily_turnover, dtype=np.float64)
    avg_turnover = float(turnover.mean()) if turnover.size else 0.0

    qp_times = np.asarray(result.qp_solve_times, dtype=np.float64)
    qp_p50 = float(np.percentile(qp_times, 50)) if qp_times.size else 0.0
    qp_p95 = float(np.percentile(qp_times, 95)) if qp_times.size else 0.0

    return {
        "strategy": result.strategy,
        "seed": int(result.seed),
        "n_eval_days": int(n_days),
        "net_sharpe_annual": float(result.net_sharpe_annual()),
        "net_return_annual": float(net_return_annual),
        "gross_return_annual": float(gross_return_annual),
        "avg_daily_turnover": avg_turnover,
        "max_drawdown": max_drawdown,
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
    }


# ---------------------------------------------------------------------------
# adaptive construction (integration glue)
# ---------------------------------------------------------------------------

def _instantiate(factory, pool: dict):
    """Call ``factory`` supplying every parameter it declares from ``pool``.

    Parameters with defaults are filled only when the pool has a candidate;
    a required parameter with no candidate raises TypeError immediately with
    an actionable message (better than a positional mismatch deep inside).
    """
    sig = inspect.signature(factory)
    kwargs = {}
    missing = []
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in pool:
            kwargs[name] = pool[name]
        elif param.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise TypeError(
            f"{getattr(factory, '__name__', factory)}: no candidate for required "
            f"parameter(s) {missing}; extend the candidate pool in "
            "openpo.mpo.backtest._strategy_pool"
        )
    return factory(**kwargs)


def _strategy_pool(market, signals, ts_model, ecfg: ExperimentConfig, scfg) -> dict:
    """Shared keyword-candidate pool for strategy/TS constructors."""
    mcfg = ecfg.mpo
    cov = market.daily_cov()
    pool = {
        # term-structure model aliases
        "ts_model": ts_model,
        "model": ts_model,
        "term_structure": ts_model,
        "ts": ts_model,
        # risk aliases (daily dense covariance, SPEC section 3: dense is fine)
        "risk": cov,
        "risk_model": cov,
        "cov": cov,
        "sigma": cov,
        "sigma_d": cov,
        "daily_cov": cov,
        # raw panels
        "market": market,
        "market_panel": market,
        "panel": market,
        "signals": signals,
        "signal_panel": signals,
        "true_seg_alpha": market.true_seg_alpha,
        # configs
        "cfg": mcfg,
        "config": mcfg,
        "mpo_cfg": mcfg,
        "mcfg": mcfg,
        "scfg": scfg,
        "synth_cfg": scfg,
        "ecfg": ecfg,
        # scalars occasionally wanted by simple strategies
        "n_assets": scfg.n_assets,
        "nav": mcfg.nav_cny,
        "calib_days": mcfg.calib_days,
    }
    return pool


def _shrunk_ts_model(pool: dict):
    """The kappa-shrunk term-structure model for this pool (built at most once).

    ``shrinkage.shrink_term_structure`` estimates the IC table on the
    CALIBRATION WINDOW only (``estimate_ic_stats`` re-applies
    ``fit_term_structure``'s ``tau + b_s < calib_days*240`` no-lookahead guard),
    so the wrapper never sees an eval-window price. The result is memoized in
    the shared candidate pool: ``mpo`` and ``mpo_shrunk`` therefore share one
    ``fit_term_structure`` product (kappa is a pure outer layer -- the A/B is
    paired by construction) and the kappa fit is never repeated per strategy
    build. That matters for :func:`tune_fixed_weights`, which builds 66 grid
    strategies off one pool; it only asks for ``fixed_weight`` today, and this
    cache keeps it O(1) even if that ever changes.
    """
    cached = pool.get(_SHRUNK_MODEL_KEY)
    if cached is not None:
        return cached
    shrinkage_mod = importlib.import_module("openpo.mpo.shrinkage")
    shrunk = shrinkage_mod.shrink_term_structure(
        pool["ts_model"], pool["market"], pool["signals"], pool["cfg"]
    )
    pool[_SHRUNK_MODEL_KEY] = shrunk
    return shrunk


def _build_strategy(name: str, strategies_mod, pool: dict, lambdas: np.ndarray):
    """Construct one strategy by canonical roster name.

    ``pool`` is read-only except for the ``mpo_shrunk`` memo slot (see
    :func:`_shrunk_ts_model`); every strategy gets its own shallow copy.
    """
    p = dict(pool)
    if name == "mpo":
        return _instantiate(strategies_mod.MpoStrategy, p)
    if name == "mpo_shrunk":
        # Same MpoStrategy, same calibration, only ts_model is kappa-wrapped.
        shrunk = _shrunk_ts_model(pool)
        p.update(ts_model=shrunk, model=shrunk, term_structure=shrunk, ts=shrunk)
        return _instantiate(strategies_mod.MpoStrategy, p)
    if name == "mpo_delayaware":
        # Same MpoStrategy, same pool ts_model, same market, same everything --
        # the ONLY difference is the stage-1 delay surcharge on the QP's linear
        # costs, so the paired A/B measures that term and nothing else.
        delay_mod = importlib.import_module("openpo.mpo.delay_cost")
        p.update(delay_cost_cfg=delay_mod.DelayCostConfig(mode="signed", scale=1.0))
        return _instantiate(strategies_mod.MpoStrategy, p)
    if name in _SIGNAL_INDEX:
        k = _SIGNAL_INDEX[name]
        p.update(
            k=k,
            signal_index=k,
            signal=k,
            horizon=SIGNAL_HORIZONS[k],
            horizon_minutes=SIGNAL_HORIZONS[k],
        )
        return _instantiate(strategies_mod.SingleSignalStrategy, p)
    if name == "fixed_weight":
        lam = np.asarray(lambdas, dtype=np.float64)
        if lam.shape != (3,) or lam.min() < 0 or lam.sum() <= 0:
            raise ValueError(f"fixed_weight lambdas must be 3 non-negative weights, got {lam}")
        lam = lam / lam.sum()
        p.update(lambdas=lam, weights=lam, signal_weights=lam, lam=lam)
        return _instantiate(strategies_mod.FixedWeightStrategy, p)
    if name == "informed":
        return _instantiate(strategies_mod.InformedTradingStrategy, p)
    if name == "oracle_mpo":
        return _instantiate(strategies_mod.OracleMpoStrategy, p)
    if name == "buy_hold":
        return _instantiate(strategies_mod.BuyAndHoldStrategy, p)
    raise KeyError(f"unknown strategy {name!r}; roster: {KNOWN_STRATEGIES}")


def _summarize_fn():
    """Prefer a sibling ``openpo.mpo.metrics.summarize`` when available."""
    try:
        metrics_mod = importlib.import_module("openpo.mpo.metrics")
        fn = getattr(metrics_mod, "summarize", None)
        if callable(fn):
            return fn
    except ImportError:
        pass
    return summarize


def default_fixed_lambdas(scfg) -> np.ndarray:
    """Default FixedWeight mixing weights: lambda proportional to target ICs."""
    lam = np.asarray(scfg.target_ics, dtype=np.float64)
    return lam / lam.sum()


# ---------------------------------------------------------------------------
# per-seed pipeline
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    ecfg: ExperimentConfig,
    strategies: list[str] | None = None,
    synth_overrides: dict | None = None,
    fixed_lambdas: np.ndarray | None = None,
) -> dict:
    """Run one full seed: generate -> calibrate -> backtest every strategy.

    Parameters
    ----------
    seed : market generation seed (fed to ``synth.generate_market``).
    ecfg : frozen experiment configuration.
    strategies : subset of :data:`KNOWN_STRATEGIES` (None = all of
        :data:`DEFAULT_STRATEGIES`; :data:`OPT_IN_STRATEGIES` must be named
        explicitly).
    synth_overrides : field overrides applied to ``ecfg.synth`` via
        ``dataclasses.replace`` (sensitivity axes / micro test configs).
    fixed_lambdas : optional pre-tuned FixedWeight lambdas (output of
        :func:`tune_fixed_weights`); default = IC-proportional weights.

    Returns
    -------
    dict mapping strategy name -> metric summary dict (see :func:`summarize`).
    Every strategy sees the *same* market path and the same calibrated
    term-structure model; only the decision rule differs. ``mpo_shrunk`` is the
    kappa-shrinkage A/B arm: byte-identical to ``mpo`` except that its
    ``ts_model`` is the same fitted model wrapped by
    ``shrinkage.shrink_term_structure`` (one wrapper per seed, shared). It is
    opt-in only (see :data:`OPT_IN_STRATEGIES`): the Iteration 40 A/B came back
    NEUTRAL at ``calib_days=20`` (24-seed paired mpo - mpo_shrunk = +0.008 net
    Sharpe, 95% CI [-0.012, +0.028], p = 0.46), so it is no longer a default arm.
    ``mpo_delayaware`` is the Iteration 42 delay-cost A/B arm, also opt-in:
    byte-identical to ``mpo`` down to the shared ``ts_model``, except that the
    stage-1 buy/sell linear costs of its QP carry the per-name
    implementation-delay surcharge (``mode='signed'``, ``scale=1.0``; design
    ``docs/mpo-t1-delaycost-design.md``).
    """
    # Validate before any sibling import so misconfiguration fails fast and
    # deterministically even while sibling modules are still being built.
    scfg = ecfg.synth
    if synth_overrides:
        scfg = dataclasses.replace(scfg, **synth_overrides)
    mcfg = ecfg.mpo

    eval_start = int(ecfg.eval_start_day)
    if eval_start < mcfg.calib_days:
        raise ValueError(
            f"eval_start_day={eval_start} < calib_days={mcfg.calib_days}: "
            "evaluation would overlap the calibration window (lookahead)"
        )
    if eval_start >= scfg.n_days:
        raise ValueError(f"eval_start_day={eval_start} leaves no eval days (n_days={scfg.n_days})")

    names = list(strategies) if strategies is not None else list(DEFAULT_STRATEGIES)
    unknown = [n for n in names if n not in KNOWN_STRATEGIES]
    if unknown:
        raise KeyError(f"unknown strategies {unknown}; roster: {KNOWN_STRATEGIES}")

    synth = importlib.import_module("openpo.mpo.synth")
    term_structure = importlib.import_module("openpo.mpo.term_structure")
    strategies_mod = importlib.import_module("openpo.mpo.strategies")
    simulator_mod = importlib.import_module("openpo.mpo.simulator")

    market, signals = synth.generate_market(scfg, seed)

    # Calibrate on days [0, calib_days) only; the model is frozen afterwards.
    ts_model = _instantiate(
        term_structure.fit_term_structure,
        {
            "market": market,
            "market_panel": market,
            "panel": market,
            "signals": signals,
            "signal_panel": signals,
            "cfg": mcfg,
            "config": mcfg,
            "mpo_cfg": mcfg,
            "mcfg": mcfg,
            "scfg": scfg,
            "calib_days": mcfg.calib_days,
        },
    )

    pool = _strategy_pool(market, signals, ts_model, ecfg, scfg)
    lam = default_fixed_lambdas(scfg) if fixed_lambdas is None else np.asarray(fixed_lambdas, float)
    summarize_result = _summarize_fn()

    day_range = range(eval_start, scfg.n_days)
    out: dict[str, dict] = {}
    for name in names:
        strategy = _build_strategy(name, strategies_mod, pool, lam)
        # fresh Simulator per strategy: no state can leak across strategies
        sim = simulator_mod.Simulator(market, mcfg)
        result = sim.run(strategy, signals, day_range)
        metrics = summarize_result(result)
        metrics["strategy"] = name  # canonical roster name wins over sibling labels
        metrics["seed"] = int(seed)
        out[name] = metrics
    return out


# ---------------------------------------------------------------------------
# FixedWeight lambda tuning (simplex grid, tuning seeds only)
# ---------------------------------------------------------------------------

def simplex_grid(step: float = 0.1, dim: int = 3) -> np.ndarray:
    """All non-negative 3-vectors summing to 1 on a ``step`` lattice.

    step=0.1 -> 66 grid points. ``1/step`` must be an integer. Only dim=3 is
    supported (one lambda per signal horizon).
    """
    if dim != 3:
        raise ValueError("simplex_grid supports dim=3 only (one lambda per signal)")
    n = int(round(1.0 / step))
    if n < 1 or abs(n * step - 1.0) > 1e-9:
        raise ValueError(f"step={step} must evenly divide 1.0")
    pts = [
        (i, j, n - i - j)
        for i in range(n + 1)
        for j in range(n - i + 1)
    ]
    return np.asarray(pts, dtype=np.float64) / n


def tune_fixed_weights(
    tuning_seeds: Iterable[int] | int,
    ecfg: ExperimentConfig,
    grid_step: float = 0.1,
    synth_overrides: dict | None = None,
) -> np.ndarray:
    """Grid-search FixedWeight lambdas on dedicated tuning seeds.

    For every lattice point on the unit simplex (``grid_step`` spacing) run a
    small FixedWeight-only backtest per tuning seed and keep the point with
    the highest mean net annualized Sharpe. Tuning seeds must be disjoint
    from evaluation seeds (SPEC section 7); market generation and
    term-structure calibration are done once per seed and shared across all
    lattice points.

    ``tuning_seeds``: iterable of seeds, or an int n -> seeds
    ``tuning_seed_base .. tuning_seed_base+n-1``.
    """
    synth = importlib.import_module("openpo.mpo.synth")
    term_structure = importlib.import_module("openpo.mpo.term_structure")
    strategies_mod = importlib.import_module("openpo.mpo.strategies")
    simulator_mod = importlib.import_module("openpo.mpo.simulator")

    if isinstance(tuning_seeds, int):
        seeds = [ecfg.tuning_seed_base + i for i in range(tuning_seeds)]
    else:
        seeds = [int(s) for s in tuning_seeds]
    if not seeds:
        raise ValueError("need at least one tuning seed")

    scfg = ecfg.synth
    if synth_overrides:
        scfg = dataclasses.replace(scfg, **synth_overrides)
    mcfg = ecfg.mpo
    eval_start = int(ecfg.eval_start_day)
    day_range = range(eval_start, scfg.n_days)

    grid = simplex_grid(grid_step, dim=3)
    scores = np.zeros(len(grid), dtype=np.float64)

    for seed in seeds:
        market, signals = synth.generate_market(scfg, seed)
        ts_model = _instantiate(
            term_structure.fit_term_structure,
            {
                "market": market,
                "market_panel": market,
                "panel": market,
                "signals": signals,
                "signal_panel": signals,
                "cfg": mcfg,
                "config": mcfg,
                "mpo_cfg": mcfg,
                "mcfg": mcfg,
                "scfg": scfg,
                "calib_days": mcfg.calib_days,
            },
        )
        pool = _strategy_pool(market, signals, ts_model, ecfg, scfg)
        for gi, lam in enumerate(grid):
            if lam.sum() <= 0:  # degenerate corner guard (cannot happen on simplex)
                scores[gi] = -np.inf
                continue
            strategy = _build_strategy("fixed_weight", strategies_mod, pool, lam)
            sim = simulator_mod.Simulator(market, mcfg)
            result = sim.run(strategy, signals, day_range)
            scores[gi] += result.net_sharpe_annual()

    scores /= len(seeds)
    best = int(np.argmax(scores))
    return grid[best].copy()
