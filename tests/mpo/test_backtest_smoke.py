"""Smoke tests for MPO-T1 backtest orchestration + stats (SPEC section 8).

The end-to-end ``run_seed`` test importorskips the sibling modules (synth /
term_structure / strategies / simulator) so this file stays green while
siblings are still under parallel construction; the stats tests are
self-contained and always run.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openpo.mpo import backtest, stats
from openpo.mpo.config import ExperimentConfig, MpoConfig, SynthConfig

_SIBLINGS = (
    "openpo.mpo.synth",
    "openpo.mpo.term_structure",
    "openpo.mpo.strategies",
    "openpo.mpo.simulator",
)


# ---------------------------------------------------------------------------
# stats.paired_stats
# ---------------------------------------------------------------------------

def _make_per_seed(diffs, base=None, a="mpo", b="fixed_weight", metric="net_sharpe_annual"):
    diffs = np.asarray(diffs, dtype=float)
    base = np.zeros_like(diffs) if base is None else np.asarray(base, dtype=float)
    return [
        {a: {metric: float(base[i] + diffs[i])}, b: {metric: float(base[i])}}
        for i in range(diffs.size)
    ]


def test_paired_stats_positive_direction():
    rng = np.random.default_rng(0)
    n = 24
    base = rng.normal(1.0, 0.3, n)  # common seed-level noise, cancels in pairing
    diffs = rng.normal(0.5, 0.1, n)
    per_seed = _make_per_seed(diffs, base=base)

    out = stats.paired_stats(per_seed, "mpo", "fixed_weight", n_boot=4000, rng_seed=1)

    assert out["n_seeds"] == n
    assert out["metric"] == "net_sharpe_annual"
    assert abs(out["mean_diff"] - diffs.mean()) < 1e-12
    assert out["t_stat"] > 10.0
    assert out["p_value"] < 1e-9
    assert 0.0 < out["ci95_low"] < out["mean_diff"] < out["ci95_high"]
    assert out["win_rate"] == 1.0
    assert out["hedges_g"] > 2.0
    assert abs(out["mean_a"] - out["mean_b"] - out["mean_diff"]) < 1e-12


def test_paired_stats_antisymmetric_and_null():
    rng = np.random.default_rng(7)
    diffs = rng.normal(0.4, 0.1, 16)
    per_seed = _make_per_seed(diffs)

    fwd = stats.paired_stats(per_seed, "mpo", "fixed_weight", n_boot=1000, rng_seed=2)
    rev = stats.paired_stats(per_seed, "fixed_weight", "mpo", n_boot=1000, rng_seed=2)
    assert rev["mean_diff"] == pytest.approx(-fwd["mean_diff"])
    assert rev["t_stat"] == pytest.approx(-fwd["t_stat"])
    assert rev["p_value"] == pytest.approx(fwd["p_value"])
    assert rev["win_rate"] == pytest.approx(1.0 - fwd["win_rate"])
    assert rev["ci95_high"] < 0.0  # reversed CI sits below zero

    # true null: zero-centered differences should not be significant
    null_diffs = np.random.default_rng(11).normal(0.0, 0.2, 20)
    null = stats.paired_stats(_make_per_seed(null_diffs), "mpo", "fixed_weight",
                              n_boot=1000, rng_seed=3)
    assert null["p_value"] > 0.05
    assert null["ci95_low"] < 0.0 < null["ci95_high"]


def test_paired_stats_bootstrap_ci_coverage():
    """Seed-level percentile bootstrap should cover the true mean ~95%.

    n=12 seeds and finite bootstrap draws undercover slightly, so assert a
    conservative 82% floor over 120 replications (fast: pure numpy).
    """
    true_mean, reps, hits = 0.3, 120, 0
    for r in range(reps):
        diffs = np.random.default_rng(1000 + r).normal(true_mean, 0.2, 12)
        out = stats.paired_stats(_make_per_seed(diffs, metric="m"), "mpo",
                                 "fixed_weight", metric="m", n_boot=500, rng_seed=r)
        hits += out["ci95_low"] <= true_mean <= out["ci95_high"]
    assert hits / reps >= 0.82


def test_paired_stats_determinism_and_validation():
    per_seed = _make_per_seed(np.linspace(-0.1, 0.4, 10))
    a = stats.paired_stats(per_seed, "mpo", "fixed_weight", n_boot=500, rng_seed=5)
    b = stats.paired_stats(per_seed, "mpo", "fixed_weight", n_boot=500, rng_seed=5)
    assert a == b  # deterministic given rng_seed

    with pytest.raises(ValueError):
        stats.paired_stats(per_seed[:1], "mpo", "fixed_weight")
    with pytest.raises(KeyError):
        stats.paired_stats(per_seed, "mpo", "missing_strategy")


def test_holm_correction_known_example():
    raw = [0.01, 0.04, 0.03, 0.005]
    adj = stats.holm_correction(raw)
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert np.all(adj >= np.asarray(raw))  # never anti-conservative
    assert np.all(stats.holm_correction([0.5, 0.9, 0.7]) <= 1.0)
    with pytest.raises(ValueError):
        stats.holm_correction([0.1, 1.5])
    with pytest.raises(ValueError):
        stats.holm_correction([])


# ---------------------------------------------------------------------------
# backtest helpers (no sibling dependency)
# ---------------------------------------------------------------------------

def test_simplex_grid_properties():
    grid = backtest.simplex_grid(0.1, dim=3)
    assert grid.shape == (66, 3)
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert grid.min() >= 0.0
    assert len(np.unique(grid, axis=0)) == 66
    coarse = backtest.simplex_grid(0.5, dim=3)
    assert coarse.shape == (6, 3)
    with pytest.raises(ValueError):
        backtest.simplex_grid(0.3)


def test_default_fixed_lambdas_ic_proportional():
    scfg = SynthConfig()
    lam = backtest.default_fixed_lambdas(scfg)
    assert lam.sum() == pytest.approx(1.0)
    expected = np.array(scfg.target_ics) / np.sum(scfg.target_ics)
    assert lam == pytest.approx(expected)


def test_summarize_schema_from_synthetic_result():
    from openpo.mpo.types import BacktestResult

    nav = 1e8 * np.cumprod(1.0 + np.array([0.001, -0.002, 0.003, 0.0005]))
    result = BacktestResult(
        strategy="mpo",
        seed=3,
        daily_nav=np.concatenate([[1e8], nav]),
        daily_turnover=np.array([0.1, 0.05, 0.08, 0.02]),
        costs={"commission": 1e4, "stamp_tax": 2e4, "slippage": 3e4, "impact_est": 5e3},
        rejected_orders=2,
        truncated_buys=1,
        t1_violations=0,
        qp_solve_times=[0.01, 0.02, 0.03],
        qp_status_counts={"optimal": 3},
    )
    m = backtest.summarize(result)
    assert m["n_eval_days"] == 4
    assert m["costs"]["total"] == pytest.approx(6.5e4)
    assert m["gross_return_annual"] > m["net_return_annual"]
    assert 0.0 <= m["max_drawdown"] < 1.0
    assert m["avg_daily_turnover"] == pytest.approx(0.0625)
    assert m["qp_time_p50"] == pytest.approx(0.02)
    assert m["t1_violations"] == 0


# ---------------------------------------------------------------------------
# end-to-end run_seed (skips until sibling modules land)
# ---------------------------------------------------------------------------

_REQUIRED_METRIC_KEYS = {
    "net_sharpe_annual",
    "net_return_annual",
    "gross_return_annual",
    "avg_daily_turnover",
    "max_drawdown",
    "costs",
    "rejected_orders",
    "t1_violations",
    "qp_time_p50",
    "qp_time_p95",
}


def _micro_ecfg() -> ExperimentConfig:
    scfg = SynthConfig(n_assets=60, n_days=16)
    mcfg = dataclasses.replace(MpoConfig(), calib_days=8)
    return ExperimentConfig(synth=scfg, mpo=mcfg, n_seeds=1, eval_start_day=8)


def test_run_seed_micro_end_to_end():
    for mod in _SIBLINGS:
        pytest.importorskip(mod)

    ecfg = _micro_ecfg()
    subset = ["mpo", "fixed_weight", "buy_hold"]
    out = backtest.run_seed(3, ecfg, strategies=subset)

    assert set(out) == set(subset)
    for name, m in out.items():
        missing = _REQUIRED_METRIC_KEYS - set(m)
        assert not missing, f"{name}: missing metric keys {missing}"
        assert m["strategy"] == name
        assert m["seed"] == 3
        assert m["t1_violations"] == 0, f"{name}: T+1 violations must be 0"
        assert m["n_eval_days"] == 8
        for key in ("net_sharpe_annual", "net_return_annual", "avg_daily_turnover",
                    "max_drawdown"):
            assert np.isfinite(m[key]), f"{name}.{key} not finite"
        assert m["avg_daily_turnover"] >= 0.0
        assert m["max_drawdown"] >= 0.0
        assert all(np.isfinite(v) for v in m["costs"].values())

    # summaries must be paired_stats-consumable end to end
    ps = stats.paired_stats([out, out], "mpo", "fixed_weight", n_boot=50)
    assert ps["n_seeds"] == 2


# ---------------------------------------------------------------------------
# kappa-shrinkage A/B arm: mpo_shrunk (docs/mpo-t1-shrinkage-design.md 3)
# ---------------------------------------------------------------------------

def test_mpo_shrunk_is_opt_in_not_default_roster():
    """Iteration 40 verdict was NEUTRAL => demoted to an opt-in research arm.

    It must stay *buildable* (the module keeps the branch + the memo), but a
    default run must not pay for it.
    """
    roster = backtest.DEFAULT_STRATEGIES
    assert "mpo_shrunk" not in roster, "neutral A/B arm must not sit in the default roster"
    assert "mpo_shrunk" in backtest.OPT_IN_STRATEGIES
    assert "mpo_shrunk" in backtest.KNOWN_STRATEGIES
    assert backtest.KNOWN_STRATEGIES == roster + backtest.OPT_IN_STRATEGIES
    assert len(set(backtest.KNOWN_STRATEGIES)) == len(backtest.KNOWN_STRATEGIES)
    assert not set(roster) & set(backtest.OPT_IN_STRATEGIES)


def test_build_strategy_still_accepts_the_opt_in_arm():
    """Demotion is roster-only: ``_build_strategy`` must still know the name."""
    pool, strategies_mod, _, _ = _shrink_wiring_fixture()
    lam = backtest.default_fixed_lambdas(SynthConfig())
    built = backtest._build_strategy("mpo_shrunk", strategies_mod, pool, lam)
    assert isinstance(built, strategies_mod.MpoStrategy)


def _shrink_wiring_fixture(seed: int = 11):
    """``(pool, strategies_mod, base_model, mcfg)`` for the wiring tests.

    Real synth panel + real calibration (both < 0.1 s at this size) but NO
    simulator run: these tests are about the wiring, not the P&L.
    """
    for mod in _SIBLINGS:
        pytest.importorskip(mod)
    import importlib

    synth = importlib.import_module("openpo.mpo.synth")
    term_structure = importlib.import_module("openpo.mpo.term_structure")
    strategies_mod = importlib.import_module("openpo.mpo.strategies")

    scfg = SynthConfig(n_assets=20, n_days=10)
    mcfg = dataclasses.replace(MpoConfig(), calib_days=5)
    ecfg = ExperimentConfig(synth=scfg, mpo=mcfg, n_seeds=1, eval_start_day=5)
    market, signals = synth.generate_market(scfg, seed)
    base = term_structure.fit_term_structure(market, signals, mcfg)
    pool = backtest._strategy_pool(market, signals, base, ecfg, scfg)
    return pool, strategies_mod, base, mcfg


def test_build_strategy_mpo_shrunk_wraps_the_pools_base_model():
    from openpo.mpo.shrinkage import ShrunkTermStructureModel

    pool, strategies_mod, base, _ = _shrink_wiring_fixture()
    lam = backtest.default_fixed_lambdas(SynthConfig())

    plain = backtest._build_strategy("mpo", strategies_mod, pool, lam)
    shrunk = backtest._build_strategy("mpo_shrunk", strategies_mod, pool, lam)

    assert isinstance(shrunk, strategies_mod.MpoStrategy)
    assert plain.ts_model is base  # unshrunk arm untouched
    assert isinstance(shrunk.ts_model, ShrunkTermStructureModel)
    # SAME fitted calibration underneath => the A/B is paired by construction.
    assert shrunk.ts_model.base is base
    assert shrunk.ts_model.unshrunk() is base

    kappa = np.asarray(shrunk.ts_model.kappa)
    assert kappa.shape == base.B.shape
    assert kappa.min() >= 0.0 and kappa.max() <= 1.0
    assert kappa.min() < 1.0, "kappa table has no resolution: A/B would be empty"
    assert shrunk.ts_model.B == pytest.approx(kappa * base.B)
    # everything except the alpha model must be identical to the mpo arm
    assert np.array_equal(shrunk.Sigma_d, plain.Sigma_d)
    assert shrunk.n_days == plain.n_days
    assert shrunk.cfg is plain.cfg


def test_shrunk_model_is_built_once_per_pool(monkeypatch):
    import importlib

    pool, strategies_mod, base, _ = _shrink_wiring_fixture()
    lam = backtest.default_fixed_lambdas(SynthConfig())

    shrinkage_mod = importlib.import_module("openpo.mpo.shrinkage")
    calls = []
    real = shrinkage_mod.shrink_term_structure

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(shrinkage_mod, "shrink_term_structure", counting)

    backtest._build_strategy("mpo", strategies_mod, pool, lam)
    backtest._build_strategy("fixed_weight", strategies_mod, pool, lam)
    assert calls == []  # nobody else pays for the kappa fit

    first = backtest._build_strategy("mpo_shrunk", strategies_mod, pool, lam)
    second = backtest._build_strategy("mpo_shrunk", strategies_mod, pool, lam)
    assert len(calls) == 1, "kappa fit must be memoized per pool, not per build"
    assert first.ts_model is second.ts_model
    assert pool[backtest._SHRUNK_MODEL_KEY] is first.ts_model
    assert first is not second  # strategy objects stay independent


def test_shrunk_model_sees_only_the_calibration_window():
    pool, strategies_mod, base, mcfg = _shrink_wiring_fixture()
    lam = backtest.default_fixed_lambdas(SynthConfig())
    shrunk = backtest._build_strategy("mpo_shrunk", strategies_mod, pool, lam)

    notes = shrunk.ts_model.stats.notes
    t_calib = mcfg.calib_days * 240
    assert notes["calib_days"] == mcfg.calib_days
    assert notes["t_calib"] == t_calib
    assert notes["max_tau_touched"] < t_calib, "kappa peeked past the calib window"


def test_run_seed_micro_mpo_shrunk_vs_mpo():
    for mod in _SIBLINGS:
        pytest.importorskip(mod)

    out = backtest.run_seed(3, _micro_ecfg(), strategies=["mpo", "mpo_shrunk"])
    assert set(out) == {"mpo", "mpo_shrunk"}

    shr = out["mpo_shrunk"]
    plain = out["mpo"]
    assert not _REQUIRED_METRIC_KEYS - set(shr)
    assert shr["strategy"] == "mpo_shrunk" and shr["seed"] == 3
    assert shr["t1_violations"] == 0
    assert shr["n_eval_days"] == plain["n_eval_days"]
    for key in ("net_sharpe_annual", "net_return_annual", "avg_daily_turnover",
                "max_drawdown", "final_nav"):
        assert np.isfinite(shr[key]), f"mpo_shrunk.{key} not finite"

    # kappa < 1 somewhere => the shrunk arm must actually decide differently,
    # and damping alpha magnitude must not increase trading (design doc 3).
    assert shr["final_nav"] != plain["final_nav"]
    assert shr["avg_daily_turnover"] != plain["avg_daily_turnover"]
    assert shr["avg_daily_turnover"] < plain["avg_daily_turnover"]

    # paired stats consume the new arm with no benchmark-side change
    ps = stats.paired_stats([out, out], "mpo", "mpo_shrunk", n_boot=50)
    assert ps["n_seeds"] == 2


def test_run_seed_validation_no_siblings_needed():
    ecfg = _micro_ecfg()
    with pytest.raises(KeyError):
        backtest.run_seed(0, ecfg, strategies=["nope"])
    bad = dataclasses.replace(ecfg, eval_start_day=4)  # < calib_days=8: lookahead
    with pytest.raises(ValueError):
        backtest.run_seed(0, bad, strategies=["buy_hold"])
    empty = dataclasses.replace(ecfg, eval_start_day=16)  # no eval days left
    with pytest.raises(ValueError):
        backtest.run_seed(0, empty, strategies=["buy_hold"])
