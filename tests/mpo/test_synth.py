"""Tests for the MPO-T1 synthetic market generator (SPEC section 6).

Sizing note: every panel used here is deliberately small (the default
``SynthConfig`` is N=300 x D=100, ~2.6 s and ~0.5 GB), and each fixture is
module-scoped so a panel is generated once and shared by all tests that need
it. Total runtime is well under the 120 s budget.

Statistics note: the long-horizon (2400 trading minute) IC estimator is
intrinsically noisy on a short panel -- only ``T / 2400`` independent
non-overlapping blocks fit, so the standard error of a measured 2400-minute IC
is ~``(1/sqrt(N)) * sqrt(2400/T)``, which is of the same order as the 0.03
target itself. Tests that need a low-noise long-horizon number therefore
average over several independent seeds (``pooled`` fixture) instead of
asserting a tight band on a single panel; short-horizon numbers are asserted
per panel.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openpo.mpo import synth
from openpo.mpo.config import (
    MINUTES_PER_DAY,
    SEGMENT_BOUNDS,
    SIGNAL_HORIZONS,
    SynthConfig,
)
from openpo.mpo.types import MarketPanel, SignalPanel

SMALL = SynthConfig(n_assets=40, n_days=6)  # T = 1440: no 2400-minute window
MED = SynthConfig(n_assets=200, n_days=60)  # 12000 stock-days for event rates
POOL = SynthConfig(n_assets=100, n_days=100)  # seed-pooled IC statistics
POOL_SEEDS = (0, 1, 2, 3, 4, 5)
SMALL_SEED = 7
MED_SEED = 11


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small():
    return synth.generate_market(SMALL, SMALL_SEED)


@pytest.fixture(scope="module")
def med():
    return synth.generate_market(MED, MED_SEED)


@pytest.fixture(scope="module")
def med_low_ic():
    return synth.generate_market(dataclasses.replace(MED, ic_mult=0.5), MED_SEED)


@pytest.fixture(scope="module")
def pooled():
    """Per-seed IC / oracle-slope statistics over independent POOL panels.

    Panels are discarded as we go (each is ~180 MB) and only the [n_seed, 3]
    statistic tables are kept.
    """
    oos_ic, seg_ic, seg_slope, in_ic = [], [], [], []
    for seed in POOL_SEEDS:
        market, signals = synth.generate_market(POOL, seed)
        oos_ic.append(synth.measure_signal_ics(market, signals, POOL, offset=0.5))
        seg_ic.append(synth.measure_segment_ics(market, POOL))
        in_ic.append(market.meta["measured_ic"])
        row = []
        for s, (a, b) in enumerate(SEGMENT_BOUNDS):
            taus = synth.ic_eval_taus(market.prices.shape[0], b, max(POOL.alpha_half_lives))
            x = market.true_seg_alpha[taus][:, :, s].ravel()
            y = synth.segment_return(market.prices, taus, (a, b)).ravel()
            xc = x - x.mean()
            row.append(float(xc @ (y - y.mean()) / (xc @ xc)))
        seg_slope.append(row)
        del market, signals
    return {
        "oos_ic": np.asarray(oos_ic),
        "seg_ic": np.asarray(seg_ic),
        "seg_slope": np.asarray(seg_slope),
        "in_ic": np.asarray(in_ic),
    }


# ---------------------------------------------------------------------------
# unit-level building blocks
# ---------------------------------------------------------------------------

def test_ar1_filter_matches_naive_recursion():
    """The blocked closed-form AR(1) must equal the textbook loop exactly."""
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((97, 5))
    state = rng.standard_normal(5)
    for rho in (0.0, 0.5, 0.977159, 0.9997112):
        s = np.sqrt(1.0 - rho * rho)
        ref = np.empty_like(eps)
        prev = state.copy()
        for t in range(eps.shape[0]):
            prev = rho * prev + s * eps[t]
            ref[t] = prev
        for block in (1, 7, 32, 97, 500):
            x = eps.copy()
            synth._ar1_inplace(x, rho, state, block)
            assert np.allclose(x, ref, atol=1e-12), f"rho={rho} block={block}"


def test_ar1_filter_is_stationary_unit_variance():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60000, 3))
    rho = 0.5 ** (1.0 / 240.0)
    synth._ar1_inplace(x, rho, rng.standard_normal(3), 240)
    assert abs(x.std() - 1.0) < 0.15
    # empirical half-life: autocorrelation at lag 240 should be ~0.5
    lag = 240
    a, b = x[:-lag].ravel(), x[lag:].ravel()
    corr = np.corrcoef(a, b)[0, 1]
    assert 0.35 < corr < 0.65


def test_geometric_sum_matches_bruteforce():
    rho = np.array([0.5 ** (1.0 / hl) for hl in (30.0, 240.0, 2400.0)])
    for a, b in [(0, 30), (30, 240), (240, 2400), (0, 1)]:
        brute = np.array([sum(r ** j for j in range(a + 1, b + 1)) for r in rho])
        assert synth._geometric_sum(rho, a, b) == pytest.approx(brute, rel=1e-10)
    # segments telescope onto the cumulative horizons
    seg = sum(synth._geometric_sum(rho, a, b) for a, b in SEGMENT_BOUNDS)
    assert seg == pytest.approx(synth._geometric_sum(rho, 0, SIGNAL_HORIZONS[-1]), rel=1e-12)
    assert synth._geometric_sum(np.array([1.0]), 0, 7) == pytest.approx([7.0])


def test_u_shape_normalisation_and_peak():
    peak = 1.5
    raw = synth.intraday_shape(MINUTES_PER_DAY, peak)
    assert raw[0] == pytest.approx(peak)
    assert raw[-1] == pytest.approx(peak)
    assert raw.min() == pytest.approx(1.0, abs=1e-3)  # exact 1.0 only at mid-day
    uv = synth.u_shape_vol(MINUTES_PER_DAY, peak)
    assert np.mean(uv**2) == pytest.approx(1.0)  # daily variance preserved
    assert uv[0] / uv[MINUTES_PER_DAY // 2] == pytest.approx(peak, rel=1e-4)
    ua = synth.u_shape_amount(MINUTES_PER_DAY, peak)
    assert ua.sum() == pytest.approx(1.0)  # daily amount == ADV
    assert ua[0] / ua[MINUTES_PER_DAY // 2] == pytest.approx(peak, rel=1e-4)


def test_ic_eval_taus_invariants():
    t = 100 * MINUTES_PER_DAY
    for h in SIGNAL_HORIZONS:
        taus = synth.ic_eval_taus(t, h, 2400.0)
        assert taus.size > 0
        assert taus.min() >= 2400  # post burn-in (longest half life)
        assert taus.max() + h <= t - 1  # realised return is in-panel
        assert np.all(np.diff(taus) > 0)
    # short panel: burn-in shrinks instead of returning an empty window
    short = synth.ic_eval_taus(12 * MINUTES_PER_DAY, 2400, 2400.0)
    assert short.size > 0
    assert short.max() + 2400 <= 12 * MINUTES_PER_DAY - 1
    # impossible horizon -> empty, never a partial/invalid window
    assert synth.ic_eval_taus(6 * MINUTES_PER_DAY, 2400, 2400.0).size == 0
    # offset moves the grid but keeps it valid
    base = synth.ic_eval_taus(t, 240, 2400.0, n_max=64)
    off = synth.ic_eval_taus(t, 240, 2400.0, n_max=64, offset=0.5)
    assert off.size == base.size and not np.array_equal(off, base)
    assert off.max() + 240 <= t - 1


def test_spearman_ic_known_cases():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 20))
    assert synth.cross_sectional_spearman_ic(x, x) == pytest.approx(1.0)
    assert synth.cross_sectional_spearman_ic(x, -x) == pytest.approx(-1.0)
    # monotone transform must not change a rank correlation
    assert synth.cross_sectional_spearman_ic(x, np.exp(3.0 * x)) == pytest.approx(1.0)
    # independent draws -> ~0
    assert abs(synth.cross_sectional_spearman_ic(x, rng.standard_normal((5, 20)))) < 0.5
    flat = np.zeros((5, 20))
    assert synth.cross_sectional_spearman_ic(flat, x) == 0.0  # degenerate row guard


# ---------------------------------------------------------------------------
# panel structure
# ---------------------------------------------------------------------------

def test_shapes_dtypes_and_finiteness(small):
    market, signals = small
    n, d, m = SMALL.n_assets, SMALL.n_days, SMALL.minutes_per_day
    t = d * m
    assert isinstance(market, MarketPanel) and isinstance(signals, SignalPanel)
    assert market.n_assets == n and market.n_days == d
    n_f = 1 + SMALL.n_style_factors
    expected = {
        "prices": (t, n),
        "minute_amount": (t, n),
        "prev_close": (d, n),
        "limit_up": (t, n),
        "limit_down": (t, n),
        "halted": (t, n),
        "adv_cny": (n,),
        "beta": (n,),
        "factor_exposures": (n, n_f),
        "factor_cov_daily": (n_f, n_f),
        "specific_var_daily": (n,),
        "true_seg_alpha": (t, n, len(SEGMENT_BOUNDS)),
    }
    for name, shape in expected.items():
        arr = getattr(market, name)
        assert arr.shape == shape, f"{name}: {arr.shape} != {shape}"
        if name in ("limit_up", "limit_down", "halted"):
            assert arr.dtype == np.bool_, f"{name} dtype {arr.dtype}"
        else:
            assert arr.dtype == np.float64, f"{name} dtype {arr.dtype}"
            assert np.all(np.isfinite(arr)), f"{name} has non-finite entries"
    assert signals.signals.shape == (t, n, len(SIGNAL_HORIZONS))
    assert signals.signals.dtype == np.float64
    assert np.all(np.isfinite(signals.signals))
    assert signals.meta["columns"] == ["return_1_30", "return_1_240", "return_1_2400"]
    assert market.meta["seed"] == SMALL_SEED == signals.meta["seed"]


def test_price_and_amount_invariants(small):
    market, _ = small
    d, m = SMALL.n_days, SMALL.minutes_per_day
    px = market.prices
    assert np.all(px > 0.0)
    # prev_close[0] is the initial price level, prev_close[d] is close of d-1
    assert np.all(market.prev_close[0] >= SMALL.price_init_mean - SMALL.price_init_spread)
    assert np.all(market.prev_close[0] <= SMALL.price_init_mean + SMALL.price_init_spread)
    closes = px[m - 1 :: m]
    assert market.prev_close[1:] == pytest.approx(closes[:-1])
    assert np.all(market.minute_amount >= 0.0)
    assert np.all(market.adv_cny > 0.0)
    # ADV is lognormal around the configured median (loose: N=40 sample)
    assert 0.4 < np.median(market.adv_cny) / SMALL.adv_median_cny < 2.5
    assert np.all(market.beta >= SMALL.beta_range[0])
    assert np.all(market.beta <= SMALL.beta_range[1])
    assert market.factor_exposures[:, 0] == pytest.approx(market.beta)


def test_halt_and_limit_semantics(med):
    market, _ = med
    d, m, n = MED.n_days, MED.minutes_per_day, MED.n_assets
    halted = market.halted.reshape(d, m, n)
    px = market.prices.reshape(d, m, n)
    amount = market.minute_amount.reshape(d, m, n)
    up = market.limit_up.reshape(d, m, n)
    dn = market.limit_down.reshape(d, m, n)

    # halts are whole-day and freeze both price and volume
    assert np.all(halted.any(axis=1) == halted.all(axis=1))
    hd, ha = np.nonzero(halted[:, 0, :])
    assert hd.size > 0
    for day, asset in zip(hd[:20], ha[:20]):
        assert px[day, :, asset] == pytest.approx(market.prev_close[day, asset])
        assert np.all(amount[day, :, asset] == 0.0)
    assert not np.any(up[halted]) and not np.any(dn[halted])

    # limit flags mean the price is pinned to the band, and are exclusive
    assert not np.any(up & dn)
    cap = market.prev_close[:, None, :] * (1.0 + MED.limit_pct)
    flr = market.prev_close[:, None, :] * (1.0 - MED.limit_pct)
    assert np.allclose(px[up], np.broadcast_to(cap, px.shape)[up])
    assert np.allclose(px[dn], np.broadcast_to(flr, px.shape)[dn])
    # ... and the freeze lasts to the close of that day (breach direction frozen)
    assert np.all(up[:, -1, :][up.any(axis=1)])
    assert np.all(dn[:, -1, :][dn.any(axis=1)])
    # nothing ever trades outside the band
    assert np.all(px <= cap * (1.0 + 1e-9)) and np.all(px >= flr * (1.0 - 1e-9))


def test_limit_and_halt_frequencies(med):
    market, _ = med
    stock_days = MED.n_days * MED.n_assets
    limit_days = market.meta["limit_up_stock_days"] + market.meta["limit_down_stock_days"]
    halt_days = market.meta["halted_stock_days"]
    lim_rate = limit_days / stock_days
    halt_rate = halt_days / stock_days
    assert MED.limit_hit_rate_daily / 3.0 <= lim_rate <= MED.limit_hit_rate_daily * 3.0, (
        f"limit rate {lim_rate} vs configured {MED.limit_hit_rate_daily}"
    )
    assert MED.halt_rate_daily / 3.0 <= halt_rate <= MED.halt_rate_daily * 3.0, (
        f"halt rate {halt_rate} vs configured {MED.halt_rate_daily}"
    )
    # both directions occur
    assert market.meta["limit_up_stock_days"] > 0
    assert market.meta["limit_down_stock_days"] > 0


def test_daily_cov_is_psd_and_correctly_scaled(small):
    market, _ = small
    cov = market.daily_cov()
    assert cov.shape == (SMALL.n_assets, SMALL.n_assets)
    assert np.allclose(cov, cov.T)
    eig = np.linalg.eigvalsh(cov)
    assert eig.min() > 0.0, f"daily_cov not PD, min eigenvalue {eig.min()}"
    assert np.all(market.specific_var_daily > 0.0)
    assert np.all(np.linalg.eigvalsh(market.factor_cov_daily) >= 0.0)
    # specific vol is the configured idiosyncratic daily vol (plus tiny alpha var)
    assert np.sqrt(np.mean(market.specific_var_daily)) == pytest.approx(
        SMALL.idio_daily_vol, rel=0.05
    )
    # total daily vol: idio 2.5% + market 20%/sqrt(240) + style -> ~2.9%
    total_vol = np.sqrt(np.diag(cov))
    assert 0.02 < float(np.mean(total_vol)) < 0.05


def test_realised_daily_vol_matches_risk_model(med):
    market, _ = med
    m = MED.minutes_per_day
    closes = market.prices[m - 1 :: m]
    daily_ret = closes[1:] / closes[:-1] - 1.0
    realised = float(np.mean(np.std(daily_ret, axis=1)))  # cross-sectional dispersion
    model = float(np.mean(np.sqrt(np.diag(market.daily_cov()))))
    assert 0.02 < realised < 0.05
    assert realised == pytest.approx(model, rel=0.25)


def test_intraday_u_shape_present_in_volatility_and_volume(med):
    market, _ = med
    d, m, n = MED.n_days, MED.minutes_per_day, MED.n_assets
    px = market.prices.reshape(d, m, n)
    ret = np.empty((d, m, n))
    ret[:, 0, :] = px[:, 0, :] / market.prev_close - 1.0
    ret[:, 1:, :] = px[:, 1:, :] / px[:, :-1, :] - 1.0
    abs_ret = np.abs(ret).mean(axis=(0, 2))
    amount = market.minute_amount.reshape(d, m, n).mean(axis=(0, 2))
    mid = m // 2
    peak = MED.intraday_u_shape_peak
    for series, label in ((abs_ret, "abs return"), (amount, "amount")):
        assert series[0] > series[mid], f"{label}: no open-side U-shape"
        assert series[-1] > series[mid], f"{label}: no close-side U-shape"
        for edge in (series[0], series[-1]):
            ratio = edge / series[mid]
            assert 0.6 * peak < ratio < 1.4 * peak, f"{label} edge/mid ratio {ratio}"
        assert series[mid] == min(series[0], series[mid], series[-1])


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def _panel_arrays(market, signals):
    return [
        market.prices, market.minute_amount, market.prev_close,
        market.limit_up, market.limit_down, market.halted,
        market.adv_cny, market.beta, market.factor_exposures,
        market.factor_cov_daily, market.specific_var_daily,
        market.true_seg_alpha, signals.signals,
    ]


def test_same_seed_is_bit_identical():
    a = synth.generate_market(SMALL, 123)
    b = synth.generate_market(SMALL, 123)
    for x, y in zip(_panel_arrays(*a), _panel_arrays(*b)):
        assert np.array_equal(x, y), "same seed must be bit-identical"
    assert set(a[0].meta) == set(b[0].meta)
    for key, va in a[0].meta.items():  # NaN-aware (unmeasurable ICs are nan)
        vb = b[0].meta[key]
        if isinstance(va, list):
            assert np.array_equal(np.asarray(va, dtype=object).astype(str),
                                  np.asarray(vb, dtype=object).astype(str)), key
        else:
            assert va == vb or (va != va and vb != vb), key


def test_different_seed_differs_everywhere():
    a = synth.generate_market(SMALL, 123)
    b = synth.generate_market(SMALL, 124)
    for x, y in zip(_panel_arrays(*a), _panel_arrays(*b)):
        if x.dtype == np.bool_:
            continue  # rare-event flags may coincide (mostly all-False)
        assert not np.array_equal(x, y), "different seeds must give a different path"


def test_config_validation():
    with pytest.raises(ValueError):
        synth.generate_market(dataclasses.replace(SMALL, n_assets=1), 0)
    with pytest.raises(ValueError):
        synth.generate_market(dataclasses.replace(SMALL, n_days=0), 0)
    with pytest.raises(ValueError):
        synth.generate_market(dataclasses.replace(SMALL, minutes_per_day=1), 0)
    with pytest.raises(ValueError):
        synth.generate_market(dataclasses.replace(SMALL, alpha_half_lives=(0.0, 240.0, 2400.0)), 0)


# ---------------------------------------------------------------------------
# signal / alpha calibration
# ---------------------------------------------------------------------------

def test_signal_ic_calibration_hits_target_in_sample(med):
    market, signals = med
    target = np.asarray(MED.target_ics) * MED.ic_mult
    measured = np.asarray(market.meta["measured_ic"])
    modes = market.meta["ic_calib_mode"]
    assert modes[0] == "bisect" and modes[1] == "bisect"
    for q, mode in enumerate(modes):
        if mode != "bisect":
            continue  # long-horizon fail-soft: population closed form, see docstring
        assert measured[q] == pytest.approx(target[q], rel=0.05), (
            f"horizon {SIGNAL_HORIZONS[q]}: measured IC {measured[q]} vs target {target[q]}"
        )
    # the generator's own re-measurement must reproduce the recorded numbers
    again = synth.measure_signal_ics(market, signals, MED)
    for q, mode in enumerate(modes):
        if mode != "analytic_no_window":
            assert again[q] == pytest.approx(measured[q], abs=1e-12)


def test_signal_ic_within_50pct_of_target_out_of_sample(med, pooled):
    """Scored on a *different* tau subsample than the one used to calibrate."""
    target = np.asarray(MED.target_ics) * MED.ic_mult
    oos = synth.measure_signal_ics(*med, scfg=MED, offset=0.5)
    for q in (0, 1):  # short horizons: many independent blocks, low estimator noise
        assert 0.5 * target[q] <= oos[q] <= 1.5 * target[q], (
            f"horizon {SIGNAL_HORIZONS[q]}: out-of-sample IC {oos[q]} vs {target[q]}"
        )
    # 2400-minute horizon: average over independent seeds to beat the estimator
    # noise (see module docstring), then apply the same +-50% band.
    pooled_long = float(pooled["oos_ic"][:, 2].mean())
    tgt_long = POOL.target_ics[2] * POOL.ic_mult
    assert 0.5 * tgt_long <= pooled_long <= 1.5 * tgt_long, (
        f"pooled 2400-minute IC {pooled_long} vs target {tgt_long}"
    )
    assert np.all(pooled["oos_ic"] > 0.0)


def test_oracle_ic_is_about_double_the_target(med):
    """c_k calibration: the true conditional mean's IC bound ~ 2x target IC."""
    market, _ = med
    oracle = np.asarray(market.meta["oracle_ic_measured"])
    target = np.asarray(MED.target_ics)
    for q in (0, 1):
        assert oracle[q] == pytest.approx(synth.ORACLE_IC_MULTIPLE * target[q], rel=0.35), (
            f"horizon {SIGNAL_HORIZONS[q]}: oracle IC {oracle[q]}"
        )
    assert oracle[2] > target[2]  # noisy estimator, but must beat the observed target
    # the oracle must dominate the observed signal at every horizon
    assert np.all(oracle >= np.asarray(market.meta["measured_ic"]) - 1e-12)


def test_true_seg_alpha_predicts_realised_segment_returns(med, pooled):
    market, _ = med
    seg_ic = synth.measure_segment_ics(market, MED)
    assert seg_ic[0] > 0.05, f"segment (0,30] oracle IC too low: {seg_ic[0]}"
    assert seg_ic[1] > 0.02, f"segment (30,240] oracle IC too low: {seg_ic[1]}"
    # long segment: pooled over seeds (single-panel estimator noise ~ the signal)
    pooled_ic = pooled["seg_ic"].mean(axis=0)
    assert pooled_ic[2] > 0.02, f"segment (240,2400] pooled oracle IC too low: {pooled_ic[2]}"
    assert np.all(pooled_ic > 0.02)
    # true_seg_alpha is a conditional MEAN, so realised-on-oracle slope ~ 1
    slope = pooled["seg_slope"]
    assert np.all(slope[:, 0] > 0.7) and np.all(slope[:, 0] < 1.4)
    assert np.all(slope[:, 1] > 0.6) and np.all(slope[:, 1] < 1.5)
    assert 0.5 < float(slope[:, 2].mean()) < 1.6


def test_signals_are_raw_conditional_means_plus_noise(med):
    """signals[:, :, q] == cumsum_s(true_seg_alpha)[q] + sigma_q * N(0,1)."""
    market, signals = med
    sigma = np.asarray(market.meta["noise_sigma"])
    taus = np.arange(0, market.prices.shape[0], 37)
    resid = signals.signals[taus] - np.cumsum(market.true_seg_alpha[taus], axis=2)
    for q in range(len(SIGNAL_HORIZONS)):
        assert sigma[q] > 0.0
        assert resid[:, :, q].std() == pytest.approx(sigma[q], rel=0.05)
        assert abs(resid[:, :, q].mean()) < 0.1 * sigma[q]
    # raw, NOT cross-sectionally z-scored (return-sized numbers)
    xs_std = signals.signals[taus].std(axis=1)
    assert np.all(xs_std < 0.5)
    assert np.all(xs_std > 0.0)


def test_ic_mult_lowers_measured_ic_without_moving_the_market(med, med_low_ic):
    market, signals = med
    market_lo, signals_lo = med_low_ic
    # the ic_mult sensitivity axis must only degrade the signal, not the path
    assert np.array_equal(market.prices, market_lo.prices)
    assert np.array_equal(market.true_seg_alpha, market_lo.true_seg_alpha)
    assert np.array_equal(market.minute_amount, market_lo.minute_amount)
    hi = np.asarray(market.meta["measured_ic"])
    lo = np.asarray(market_lo.meta["measured_ic"])
    sig_hi = np.asarray(market.meta["noise_sigma"])
    sig_lo = np.asarray(market_lo.meta["noise_sigma"])
    assert np.all(sig_lo > sig_hi), f"ic_mult=0.5 must add noise: {sig_lo} vs {sig_hi}"
    for q, mode in enumerate(market_lo.meta["ic_calib_mode"]):
        assert lo[q] < hi[q], f"horizon {SIGNAL_HORIZONS[q]}: {lo[q]} !< {hi[q]}"
        if mode == "bisect":
            assert lo[q] == pytest.approx(0.5 * MED.target_ics[q], rel=0.05)
    oos_lo = synth.measure_signal_ics(market_lo, signals_lo, MED, offset=0.5)
    oos_hi = synth.measure_signal_ics(market, signals, MED, offset=0.5)
    assert np.all(oos_lo[:2] < oos_hi[:2])


def test_short_panel_falls_back_to_analytic_noise(small):
    """T = 1440 < 2400: the long horizon has no window; noise must still be sane."""
    market, signals = small
    modes = market.meta["ic_calib_mode"]
    assert modes[2] == "analytic_no_window"
    assert market.meta["ic_calib_taus"][2] == 0
    sigma = np.asarray(market.meta["noise_sigma"])
    assert np.all(sigma > 0.0) and np.all(np.isfinite(sigma))
    # short horizons still calibrate empirically on the shrunk window
    assert modes[0] == "bisect" and modes[1] == "bisect"
    measured = np.asarray(market.meta["measured_ic"])
    for q in (0, 1):
        assert measured[q] == pytest.approx(SMALL.target_ics[q] * SMALL.ic_mult, rel=0.05)
    assert np.all(np.isfinite(signals.signals))


def test_latent_alpha_cross_correlation_and_decay(med):
    """Innovation correlation is scfg.alpha_cross_corr; levels are attenuated."""
    market, _ = med
    corr = np.asarray(market.meta["level_corr_m"])
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T, atol=1e-12)
    off = corr[np.triu_indices(3, 1)]
    assert np.all(off > 0.0)
    assert np.all(off < MED.alpha_cross_corr + 0.05)
    assert np.all(np.linalg.eigvalsh(corr) > -1e-9)
    rho = np.asarray(market.meta["rho_minute"])
    assert rho == pytest.approx(0.5 ** (1.0 / np.asarray(MED.alpha_half_lives)))
    # segment coefficients grow with segment length x persistence
    seg = np.asarray(market.meta["segment_coef"])
    assert seg.shape == (3, 3)
    assert np.all(seg > 0.0)
    assert np.all(np.diff(seg[:, 2]) > 0.0)  # the 2400-minute signal keeps paying out
    assert seg[2, 0] < seg[0, 0]  # the 30-minute signal has decayed away by then


def test_meta_is_json_sized_and_complete(small):
    market, signals = small
    required = {
        "seed", "rho_minute", "c_alpha", "segment_coef", "level_corr_m", "target_ic",
        "observed_ic_target", "oracle_ic_target", "oracle_ic_measured", "measured_ic",
        "noise_sigma", "ic_calib_taus", "ic_calib_mode", "burn_in_minutes",
        "style_return_weight", "alpha_half_lives", "segment_bounds", "signal_horizons",
        "limit_up_stock_days", "limit_down_stock_days", "halted_stock_days",
    }
    missing = required - set(market.meta)
    assert not missing, f"market.meta missing {missing}"
    assert all(not isinstance(v, np.ndarray) for v in market.meta.values())
    assert market.meta["segment_bounds"] == [list(b) for b in SEGMENT_BOUNDS]
    assert market.meta["signal_horizons"] == list(SIGNAL_HORIZONS)
    assert market.meta["style_return_weight"] == synth.STYLE_RETURN_WEIGHT
    assert set(signals.meta) >= {"seed", "columns", "noise_sigma", "measured_ic"}
