"""Tests for robust per-horizon alpha shrinkage (docs/mpo-t1-shrinkage-design.md).

Covers ``openpo.mpo.shrinkage``:

1. the committed kappa formula ``kappa = IC^2 / (IC^2 + Var(IC))`` -- degenerate,
   certain and typical cells, monotonicity, loud validation;
2. the IC estimator (epoch-level and panel-level, incl. the no-lookahead guard
   and the overlapping-window deflation);
3. ``ShrunkTermStructureModel`` interface parity with ``TermStructureModel``
   (same shapes, ``kappa == 1`` reproduces the wrapped model BITWISE,
   ``kappa == 0`` zeroes the alphas, the wrapped model is never mutated);
4. a seeded synthetic recovery study: with a WEAK signal the shrunk alphas have
   materially lower MSE against the known true alphas than the unshrunk ones,
   while a STRONG signal is left essentially untouched.

The panels are built here by hand (no ``openpo.mpo.synth`` dependency): a
persistent per-asset latent drives every minute's expected return, and the
three signal columns observe that latent with increasing noise, so the true IC
ordering is ``k=0 > k=1 > k=2`` by construction.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from openpo.mpo.config import EPOCH_OFFSETS, MINUTES_PER_DAY, SEGMENT_BOUNDS, MpoConfig
from openpo.mpo.shrinkage import (
    IcStats,
    ShrunkTermStructureModel,
    estimate_ic_stats,
    ic_stats_from_epochs,
    kappa_from_ic,
    overlap_multipliers,
    shrink_term_structure,
    shrinkage_kappa,
)
from openpo.mpo.stages import build_stage_grid
from openpo.mpo.term_structure import (
    TermStructureModel,
    fit_term_structure,
    zscore_cross_section,
)
from openpo.mpo.types import MarketPanel, SignalPanel

ONE_SEGMENT = ((0, 30),)  # single-segment bounds for the array-level studies


def _cfg(**kw) -> MpoConfig:
    return dataclasses.replace(MpoConfig(), **kw)


# ---------------------------------------------------------------------------
# synthetic builders
# ---------------------------------------------------------------------------

def _make_panel(
    n_days: int = 14,
    n_assets: int = 40,
    drift: float = 6e-5,
    noise: float = 1.5e-4,
    signal_noise: tuple[float, float, float] = (0.2, 1.0, 3.0),
    seed: int = 0,
):
    """Panel whose per-asset latent persists over the whole window.

    Every minute's expected return is ``drift * a[i]`` with ``a`` a fixed
    standard-normal cross-section, so all three segments have a POSITIVE
    population IC. Signal column ``k`` observes ``a`` with noise
    ``signal_noise[k]``, hence a decreasing true IC in ``k``.
    """
    rng = np.random.default_rng(seed)
    n_min = n_days * MINUTES_PER_DAY
    a = rng.standard_normal(n_assets)
    ret = drift * a[None, :] + noise * rng.standard_normal((n_min, n_assets))
    prices = 20.0 * np.cumprod(1.0 + ret, axis=0)

    sig = np.empty((n_min, n_assets, 3), dtype=np.float64)
    for k, sn in enumerate(signal_noise):
        sig[:, :, k] = a[None, :] + sn * rng.standard_normal((n_min, n_assets))

    market = MarketPanel(
        n_assets=n_assets,
        n_days=n_days,
        prices=prices,
        minute_amount=np.full((n_min, n_assets), 1e6),
        prev_close=prices[::MINUTES_PER_DAY].copy(),
        limit_up=np.zeros((n_min, n_assets), dtype=bool),
        limit_down=np.zeros((n_min, n_assets), dtype=bool),
        halted=np.zeros((n_min, n_assets), dtype=bool),
        adv_cny=np.full(n_assets, 8e7),
        beta=np.ones(n_assets),
        factor_exposures=np.ones((n_assets, 1)),
        factor_cov_daily=np.full((1, 1), 4e-4),
        specific_var_daily=np.full(n_assets, 6e-4),
    )
    return market, SignalPanel(signals=sig)


def _epoch_sample(rho: float, n_epochs: int, n_assets: int, rng, n_sig: int = 1):
    """``(z[E,N,K], r[E,N,1])`` with ``corr(z[:,k], r) = rho`` per epoch.

    ``r = beta * z[:, 0] + eps``, ``beta = rho / sqrt(1 - rho^2)``, ``eps``
    standard normal -- so the population Pearson IC of column 0 is exactly
    ``rho`` and the true coefficient is ``beta``.
    """
    beta = float(rho) / np.sqrt(max(1.0 - float(rho) ** 2, 1e-12))
    z = rng.standard_normal((n_epochs, n_assets, n_sig))
    r = beta * z[:, :, :1] + rng.standard_normal((n_epochs, n_assets, 1))
    return z, r, beta


# ---------------------------------------------------------------------------
# 0. overlap deflation
# ---------------------------------------------------------------------------

class TestOverlapMultipliers:
    def test_frozen_defaults(self):
        m = overlap_multipliers()
        assert m.shape == (3,)
        # (0,30] fits one epoch; (30,240] spans 7; (240,2400] spans 72.
        assert np.allclose(m, [1.0, 7.0, 72.0])

    def test_short_segment_floors_at_one(self):
        assert overlap_multipliers(((0, 10),), epoch_spacing=30)[0] == 1.0

    @pytest.mark.parametrize("bad", [0, -30, np.nan])
    def test_bad_spacing_raises(self, bad):
        with pytest.raises(ValueError):
            overlap_multipliers(SEGMENT_BOUNDS, epoch_spacing=bad)

    def test_degenerate_segment_raises(self):
        with pytest.raises(ValueError):
            overlap_multipliers(((30, 30),))


# ---------------------------------------------------------------------------
# 1. the committed kappa formula
# ---------------------------------------------------------------------------

class TestKappaFormula:
    def test_typical_cell_equal_signal_and_noise_halves_the_alpha(self):
        # IC = 0.05 with SE = 0.05 (t = 1) -> kappa = 1/2 exactly.
        k = kappa_from_ic(np.array([0.05]), np.array([0.05**2]))
        assert k.shape == (1,)
        assert k[0] == pytest.approx(0.5, abs=1e-15)

    @pytest.mark.parametrize("t", [0.25, 0.5, 1.0, 2.0, 5.0, 20.0])
    def test_matches_t2_over_one_plus_t2(self, t):
        ic = 0.04
        var = (ic / t) ** 2
        assert kappa_from_ic(np.array([ic]), np.array([var]))[0] == pytest.approx(
            t**2 / (1.0 + t**2), rel=1e-12
        )

    def test_certain_estimate_gives_exactly_one(self):
        k = kappa_from_ic(np.array([0.03, -0.9]), np.zeros(2))
        assert np.array_equal(k, np.ones(2))

    def test_no_sample_gives_exactly_zero(self):
        # The documented degenerate encoding: ic = 0, var = +inf.
        k = kappa_from_ic(np.array([0.0, 0.05]), np.array([np.inf, np.inf]))
        assert np.array_equal(k, np.zeros(2))

    def test_zero_ic_gives_zero_even_with_zero_variance(self):
        assert kappa_from_ic(np.array([0.0]), np.array([0.0]))[0] == 0.0

    def test_sign_agnostic(self):
        pos = kappa_from_ic(np.array([0.06]), np.array([4e-4]))
        neg = kappa_from_ic(np.array([-0.06]), np.array([4e-4]))
        assert pos[0] == neg[0]

    def test_monotone_decreasing_in_noise(self):
        var = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1.0])
        k = kappa_from_ic(np.full(6, 0.05), var)
        assert np.all(np.diff(k) < 0.0)
        assert k[0] > 0.99 and k[-1] < 0.01

    def test_monotone_increasing_in_ic(self):
        ic = np.array([0.0, 0.01, 0.02, 0.05, 0.2])
        k = kappa_from_ic(ic, np.full(5, 1e-3))
        assert np.all(np.diff(k) > 0.0)

    def test_correlation_above_one_is_clipped(self):
        k = kappa_from_ic(np.array([2.0]), np.array([1.0]))
        assert k[0] == pytest.approx(0.5, rel=1e-12)

    def test_range_is_the_unit_interval(self):
        rng = np.random.default_rng(1)
        ic = rng.uniform(-1, 1, 500)
        var = 10 ** rng.uniform(-8, 2, 500)
        k = kappa_from_ic(ic, var)
        assert np.all((k >= 0.0) & (k <= 1.0))
        assert np.all(np.isfinite(k))

    @pytest.mark.parametrize(
        "ic, var",
        [
            (np.zeros(3), np.zeros(2)),          # shape mismatch
            (np.array([np.nan]), np.array([1.0])),  # nan IC
            (np.array([0.1]), np.array([-1e-9])),   # negative variance
            (np.array([0.1]), np.array([np.nan])),  # nan variance
        ],
    )
    def test_loud_validation(self, ic, var):
        with pytest.raises(ValueError):
            kappa_from_ic(ic, var)


# ---------------------------------------------------------------------------
# 2. IC estimator (epoch level)
# ---------------------------------------------------------------------------

class TestIcStatsFromEpochs:
    def _stats(self, rho, n_epochs=40, n_assets=80, seed=7, **kw):
        rng = np.random.default_rng(seed)
        z, r, _ = _epoch_sample(rho, n_epochs, n_assets, rng)
        kw.setdefault("seg_bounds", ONE_SEGMENT)
        kw.setdefault("overlap_mult", np.ones(1))
        return ic_stats_from_epochs(z, r, **kw)

    def test_shapes_and_types(self):
        st = self._stats(0.1)
        assert isinstance(st, IcStats)
        for arr in (st.ic, st.ic_var, st.n_eff, st.n_names):
            assert arr.shape == (1, 1)
        assert st.n_epochs.shape == (1, 1)
        assert st.n_epochs[0, 0] == 40
        assert st.n_names[0, 0] == 80
        assert st.kappa().shape == (1, 1)
        assert shrinkage_kappa(st)[0, 0] == st.kappa()[0, 0]

    def test_recovers_the_population_ic(self):
        st = self._stats(0.2, n_epochs=60, n_assets=200)
        # Spearman runs ~2-3% below Pearson for gaussians; loose but pointed.
        assert st.ic[0, 0] == pytest.approx(0.2, abs=0.03)

    def test_strong_signal_barely_shrinks_weak_signal_shrinks_hard(self):
        strong = self._stats(0.30, n_epochs=40, n_assets=200).kappa()[0, 0]
        weak = self._stats(0.02, n_epochs=4, n_assets=40).kappa()[0, 0]
        assert strong > 0.95
        assert weak < 0.5
        assert weak < strong

    def test_se_and_kappa_improve_with_sample_size(self):
        """More epochs of the same signal -> smaller SE -> larger kappa.

        The SE claim is mechanical and tested on NESTED prefixes of one draw;
        kappa also depends on the (noisy) IC point estimate, so it is only
        monotone on average -- checked over 8 independent draws per size.
        """
        rng = np.random.default_rng(11)
        z, r, _ = _epoch_sample(0.05, 128, 100, rng)
        sizes = (2, 8, 32, 128)
        ses = [float(ic_stats_from_epochs(z[:e], r[:e], seg_bounds=ONE_SEGMENT,
                                         overlap_mult=np.ones(1)).ic_se()[0, 0])
               for e in sizes]
        assert all(b < a for a, b in zip(ses, ses[1:])), ses
        means = [float(np.mean([self._stats(0.05, n_epochs=e, n_assets=100,
                                           seed=100 + j).kappa()[0, 0]
                                for j in range(8)]))
                 for e in sizes]
        assert all(b > a for a, b in zip(means, means[1:])), means

    def test_noisier_ic_gives_smaller_kappa_at_equal_sample(self):
        ks = [self._stats(rho, n_epochs=16, n_assets=60, seed=5).kappa()[0, 0]
              for rho in (0.01, 0.05, 0.15, 0.4)]
        assert all(b > a for a, b in zip(ks, ks[1:])), ks

    def test_single_epoch_uses_the_analytic_floor(self):
        """E = 1 has no empirical variance; the Fisher-z floor keeps 0 < kappa < 1."""
        rng = np.random.default_rng(3)
        z, r, _ = _epoch_sample(0.9, 1, 60, rng)
        st = ic_stats_from_epochs(z, r, seg_bounds=ONE_SEGMENT, overlap_mult=np.ones(1))
        assert st.n_epochs[0, 0] == 1
        assert st.notes["v_emp"][0][0] == 0.0
        assert st.notes["v_ana"][0][0] > 0.0
        k = st.kappa()[0, 0]
        assert 0.0 < k < 1.0

    def test_variance_floor_never_claims_less_noise_than_theory(self):
        st = self._stats(0.05, n_epochs=40, n_assets=80)
        v_emp = np.asarray(st.notes["v_emp"])
        v_ana = np.asarray(st.notes["v_ana"])
        assert st.ic_var[0, 0] == pytest.approx(max(v_emp[0, 0], v_ana[0, 0]) / 40.0)

    def test_overlap_deflation_shrinks_harder(self):
        """Default deflation (m = 1/7/72) must lower kappa vs pretending m = 1."""
        rng = np.random.default_rng(9)
        z = rng.standard_normal((80, 120, 3))
        beta = 0.05
        r = beta * z[:, :, :1] + rng.standard_normal((80, 120, 1))
        r3 = np.repeat(r, 3, axis=2)  # same relation in all three segments
        honest = ic_stats_from_epochs(z, r3)
        flat = ic_stats_from_epochs(z, r3, overlap_mult=np.ones(3))
        assert np.allclose(honest.ic, flat.ic)  # point estimate unaffected
        assert np.allclose(honest.n_eff[0], flat.n_eff[0])  # segment 0: m = 1
        assert np.all(honest.ic_var[1] > flat.ic_var[1])
        assert np.all(honest.ic_var[2] > honest.ic_var[1])
        kh, kf = honest.kappa(), flat.kappa()
        assert np.allclose(kh[0], kf[0])
        assert np.all(kh[1] < kf[1])
        assert np.all(kh[2] < kh[1])

    def test_empty_cell_is_zero_kappa_and_recorded(self):
        rng = np.random.default_rng(13)
        z = rng.standard_normal((20, 50, 2))
        r = np.stack([z[:, :, 0] + rng.standard_normal((20, 50)),
                      np.full((20, 50), np.nan)], axis=2)
        st = ic_stats_from_epochs(z, r, seg_bounds=((0, 30), (30, 240)))
        assert st.n_epochs[1, 0] == 0 and st.n_epochs[1, 1] == 0
        assert np.array_equal(st.ic[1], np.zeros(2))
        assert np.all(np.isinf(st.ic_var[1]))
        assert np.array_equal(st.kappa()[1], np.zeros(2))
        assert [1, 0] in st.notes["degenerate_cells"]
        assert [1, 1] in st.notes["degenerate_cells"]
        # the usable segment is untouched by its degenerate neighbour
        assert st.n_epochs[0, 0] == 20
        assert st.kappa()[0, 0] > 0.5

    def test_epochs_below_min_names_are_dropped_not_imputed(self):
        rng = np.random.default_rng(17)
        z, r, _ = _epoch_sample(0.3, 10, 40, rng)
        r[0, 2:, 0] = np.nan  # only 2 usable names in epoch 0
        st = ic_stats_from_epochs(z, r, seg_bounds=ONE_SEGMENT, overlap_mult=np.ones(1))
        assert st.n_epochs[0, 0] == 9

    def test_t_stat_and_se_are_consistent(self):
        st = self._stats(0.1, n_epochs=30, n_assets=100)
        assert st.ic_se()[0, 0] == pytest.approx(np.sqrt(st.ic_var[0, 0]))
        assert st.t_stat()[0, 0] == pytest.approx(st.ic[0, 0] / st.ic_se()[0, 0])
        k = st.kappa()[0, 0]
        t = st.t_stat()[0, 0]
        assert k == pytest.approx(t**2 / (1.0 + t**2), rel=1e-12)

    def test_empty_cell_t_stat_is_zero(self):
        st = IcStats(
            ic=np.zeros((1, 1)), ic_var=np.full((1, 1), np.inf),
            n_epochs=np.zeros((1, 1), dtype=np.int64), n_eff=np.zeros((1, 1)),
            n_names=np.zeros((1, 1)), overlap_mult=np.ones(1),
        )
        assert st.t_stat()[0, 0] == 0.0
        assert st.kappa()[0, 0] == 0.0

    @pytest.mark.parametrize("kw", [{"overlap_mult": np.ones(2)},
                                    {"overlap_mult": np.array([0.0])},
                                    {"min_names": 2}])
    def test_loud_validation_of_options(self, kw):
        rng = np.random.default_rng(21)
        z, r, _ = _epoch_sample(0.1, 5, 30, rng)
        with pytest.raises(ValueError):
            ic_stats_from_epochs(z, r, seg_bounds=ONE_SEGMENT, **kw)

    def test_loud_validation_of_shapes(self):
        rng = np.random.default_rng(23)
        z, r, _ = _epoch_sample(0.1, 5, 30, rng)
        with pytest.raises(ValueError):
            ic_stats_from_epochs(z[0], r, seg_bounds=ONE_SEGMENT)      # 2-D z
        with pytest.raises(ValueError):
            ic_stats_from_epochs(z, r[:, :, 0], seg_bounds=ONE_SEGMENT)  # 2-D r
        with pytest.raises(ValueError):
            ic_stats_from_epochs(z, r[:3], seg_bounds=ONE_SEGMENT)     # E mismatch
        with pytest.raises(ValueError):
            ic_stats_from_epochs(z, r[:, :10], seg_bounds=ONE_SEGMENT)  # N mismatch
        with pytest.raises(ValueError):
            ic_stats_from_epochs(z, r)                                  # S mismatch
        with pytest.raises(ValueError):
            ic_stats_from_epochs(z[:0], r[:0], seg_bounds=ONE_SEGMENT)   # E == 0

    def test_shrinkage_kappa_rejects_non_stats(self):
        with pytest.raises(TypeError):
            shrinkage_kappa(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# 3. IC estimator (panel level)
# ---------------------------------------------------------------------------

class TestEstimateIcStats:
    def test_shapes_and_ordering_by_signal_noise(self):
        market, signals = _make_panel(n_days=14, n_assets=60, seed=31)
        st = estimate_ic_stats(market, signals, _cfg(calib_days=12))
        assert st.ic.shape == (3, 3)
        assert np.all(np.isfinite(st.ic))
        k = st.kappa()
        assert k.shape == (3, 3)
        assert np.all((k >= 0.0) & (k <= 1.0))
        # signal 0 is the cleanest observation of the latent, signal 2 the noisiest
        assert st.ic[0, 0] > st.ic[0, 1] > st.ic[0, 2]
        assert k[0, 0] > k[0, 2]

    def test_overlap_mult_is_the_frozen_table(self):
        market, signals = _make_panel(n_days=14, n_assets=40, seed=33)
        st = estimate_ic_stats(market, signals, _cfg(calib_days=12))
        assert np.allclose(st.overlap_mult, [1.0, 7.0, 72.0])
        assert st.notes["source"] == "estimate_ic_stats"
        assert st.notes["calib_days"] == 12

    def test_no_lookahead_past_the_calibration_window(self):
        market, signals = _make_panel(n_days=14, n_assets=40, seed=35)
        cfg = _cfg(calib_days=12)
        clean = estimate_ic_stats(market, signals, cfg)
        t_calib = 12 * MINUTES_PER_DAY
        assert clean.notes["max_tau_touched"] < t_calib
        poisoned = dataclasses.replace(market, prices=market.prices.copy())
        poisoned.prices[t_calib:] = np.nan
        poisoned_sig = SignalPanel(signals=signals.signals.copy())
        poisoned_sig.signals[t_calib:] = np.nan
        after = estimate_ic_stats(poisoned, poisoned_sig, cfg)
        assert np.array_equal(clean.ic, after.ic)
        assert np.array_equal(clean.ic_var, after.ic_var)
        assert np.array_equal(clean.n_epochs, after.n_epochs)

    def test_far_segment_needs_room_inside_the_window(self):
        """calib_days = 5 leaves no (240,2400] sample -> that row's kappa is 0."""
        market, signals = _make_panel(n_days=14, n_assets=40, seed=37)
        st = estimate_ic_stats(market, signals, _cfg(calib_days=5))
        assert np.all(st.n_epochs[2] == 0)
        assert np.array_equal(st.kappa()[2], np.zeros(3))
        assert np.all(st.n_epochs[0] > 0)

    def test_calib_days_defaults_to_cfg(self):
        market, signals = _make_panel(n_days=14, n_assets=40, seed=39)
        cfg = _cfg(calib_days=12)
        a = estimate_ic_stats(market, signals, cfg)
        b = estimate_ic_stats(market, signals, cfg, calib_days=12)
        assert np.array_equal(a.ic, b.ic)

    @pytest.mark.parametrize("bad", [0, -1, 99])
    def test_calib_days_validated(self, bad):
        market, signals = _make_panel(n_days=14, n_assets=40, seed=41)
        with pytest.raises(ValueError):
            estimate_ic_stats(market, signals, MpoConfig(), calib_days=bad)

    def test_panel_shape_mismatch_raises(self):
        market, signals = _make_panel(n_days=14, n_assets=40, seed=43)
        bad = SignalPanel(signals=signals.signals[:, :10])
        with pytest.raises(ValueError):
            estimate_ic_stats(market, bad, _cfg(calib_days=12))
        with pytest.raises(ValueError):
            estimate_ic_stats(market, SignalPanel(signals=signals.signals[:, :, 0]),
                              _cfg(calib_days=12))

    def test_kappa_table_matches_B_shape(self):
        market, signals = _make_panel(n_days=14, n_assets=40, seed=45)
        cfg = _cfg(calib_days=12)
        model = fit_term_structure(market, signals, cfg)
        st = estimate_ic_stats(market, signals, cfg)
        assert st.kappa().shape == model.B.shape


# ---------------------------------------------------------------------------
# 4. the wrapper
# ---------------------------------------------------------------------------

class TestShrunkTermStructureModel:
    def _fitted(self, seed=51):
        market, signals = _make_panel(n_days=14, n_assets=40, seed=seed)
        cfg = _cfg(calib_days=12)
        return market, signals, cfg, fit_term_structure(market, signals, cfg)

    def _grid_and_z(self, market, signals, cfg, day=12):
        tau = day * MINUTES_PER_DAY + EPOCH_OFFSETS[3]
        grid = build_stage_grid(tau, market.n_days, cfg)
        return grid, zscore_cross_section(signals.signals[tau])

    def test_interface_parity(self):
        market, signals, cfg, base = self._fitted()
        w = ShrunkTermStructureModel(base=base, kappa=np.full((3, 3), 0.5))
        for name in ("B", "rho", "seg_bounds", "decay_mult", "decay",
                     "diagnostics", "effective_rho", "stage_alphas"):
            assert hasattr(w, name), name
        assert list(inspect.signature(w.stage_alphas).parameters) == \
            list(inspect.signature(base.stage_alphas).parameters)
        assert w.B.shape == base.B.shape
        assert w.rho.shape == base.rho.shape
        assert w.seg_bounds == base.seg_bounds
        assert w.decay_mult == base.decay_mult
        assert np.array_equal(w.effective_rho(), base.effective_rho())
        assert w.unshrunk() is base

        grid, z = self._grid_and_z(market, signals, cfg)
        a_base = base.stage_alphas(z, grid)
        a_w = w.stage_alphas(z, grid)
        assert a_w.shape == a_base.shape == (grid.n_stages, market.n_assets)
        assert a_w.dtype == a_base.dtype == np.float64
        assert np.all(np.isfinite(a_w))

    def test_kappa_one_is_bitwise_identical(self):
        market, signals, cfg, base = self._fitted(seed=53)
        w = ShrunkTermStructureModel(base=base, kappa=np.ones((3, 3)))
        assert w.B.tobytes() == base.B.tobytes()
        for day in (12, 13):
            grid, z = self._grid_and_z(market, signals, cfg, day=day)
            assert w.stage_alphas(z, grid).tobytes() == base.stage_alphas(z, grid).tobytes()

    def test_kappa_zero_zeroes_the_alphas(self):
        market, signals, cfg, base = self._fitted(seed=55)
        w = ShrunkTermStructureModel(base=base, kappa=np.zeros((3, 3)))
        grid, z = self._grid_and_z(market, signals, cfg)
        alpha = w.stage_alphas(z, grid)
        assert np.array_equal(w.B, np.zeros((3, 3)))
        assert np.array_equal(alpha, np.zeros_like(alpha))
        # the fail-safe direction: the base model was NOT zero
        assert np.abs(base.stage_alphas(z, grid)).max() > 0.0

    def test_kappa_scales_alphas_linearly_per_cell(self):
        market, signals, cfg, base = self._fitted(seed=57)
        kappa = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        w = ShrunkTermStructureModel(base=base, kappa=kappa)
        manual = TermStructureModel(
            B=base.B * kappa, rho=base.rho, seg_bounds=base.seg_bounds,
            decay_mult=base.decay_mult, decay=base.decay,
        )
        grid, z = self._grid_and_z(market, signals, cfg)
        assert w.stage_alphas(z, grid).tobytes() == manual.stage_alphas(z, grid).tobytes()

    def test_uniform_kappa_scales_the_whole_alpha_vector(self):
        market, signals, cfg, base = self._fitted(seed=59)
        w = ShrunkTermStructureModel(base=base, kappa=np.full((3, 3), 0.25))
        grid, z = self._grid_and_z(market, signals, cfg)
        assert np.allclose(w.stage_alphas(z, grid), 0.25 * base.stage_alphas(z, grid),
                           rtol=0, atol=1e-18)

    def test_base_model_is_never_mutated(self):
        market, signals, cfg, base = self._fitted(seed=61)
        b_before = base.B.copy()
        notes_before = dict(base.diagnostics.notes)
        w = ShrunkTermStructureModel(base=base, kappa=np.full((3, 3), 0.3))
        _ = w.B, w.diagnostics
        assert np.array_equal(base.B, b_before)
        assert set(base.diagnostics.notes) == set(notes_before)
        assert not any(str(k).startswith("shrinkage") for k in base.diagnostics.notes)
        assert base.diagnostics is not w.diagnostics

    def test_diagnostics_carry_the_audit_trail(self):
        market, signals, cfg, base = self._fitted(seed=63)
        st = estimate_ic_stats(market, signals, cfg)
        w = ShrunkTermStructureModel(base=base, kappa=st.kappa(), stats=st)
        d = w.diagnostics
        notes = d.notes
        assert np.allclose(np.asarray(notes["shrinkage_kappa"]), st.kappa())
        assert np.allclose(np.asarray(notes["shrinkage_B_unshrunk"]), base.B)
        assert np.allclose(np.asarray(notes["shrinkage_ic"]), st.ic)
        assert notes["shrinkage_kappa_min"] == pytest.approx(st.kappa().min())
        assert notes["calib_days"] == 12  # base notes preserved
        assert np.allclose(d.coef_table, w.B)
        assert np.array_equal(d.seg_ic, base.diagnostics.seg_ic)
        assert d.n_samples == base.diagnostics.n_samples

    def test_wrapping_a_wrapper_composes_multiplicatively(self):
        market, signals, cfg, base = self._fitted(seed=65)
        inner = ShrunkTermStructureModel(base=base, kappa=np.full((3, 3), 0.5))
        outer = ShrunkTermStructureModel(base=inner, kappa=np.full((3, 3), 0.5))
        assert np.allclose(outer.B, 0.25 * base.B)

    @pytest.mark.parametrize(
        "kappa",
        [
            np.full((2, 3), 0.5),      # wrong shape
            np.full(3, 0.5),           # not a table
            0.5,                       # scalar, no implicit broadcast
            np.full((3, 3), 1.5),      # out of range
            np.full((3, 3), -0.1),     # out of range
            np.full((3, 3), np.nan),   # non-finite
        ],
    )
    def test_loud_validation_of_kappa(self, kappa):
        _, _, _, base = self._fitted(seed=67)
        with pytest.raises(ValueError):
            ShrunkTermStructureModel(base=base, kappa=kappa)

    def test_shrink_term_structure_end_to_end(self):
        market, signals, cfg, base = self._fitted(seed=69)
        w = shrink_term_structure(base, market, signals, cfg)
        assert isinstance(w, ShrunkTermStructureModel)
        assert w.stats is not None
        assert np.all(np.abs(w.B) <= np.abs(base.B) + 1e-18)
        grid, z = self._grid_and_z(market, signals, cfg)
        assert np.abs(w.stage_alphas(z, grid)).max() <= np.abs(base.stage_alphas(z, grid)).max()

    def test_shrink_term_structure_with_explicit_kappa(self):
        _, _, _, base = self._fitted(seed=71)
        w = shrink_term_structure(base, kappa=np.full((3, 3), 0.5))
        assert np.allclose(w.B, 0.5 * base.B)
        assert w.stats is None

    def test_shrink_term_structure_refuses_ambiguous_calls(self):
        market, signals, cfg, base = self._fitted(seed=73)
        with pytest.raises(ValueError):
            shrink_term_structure(base)  # no kappa, no data
        with pytest.raises(ValueError):
            shrink_term_structure(base, market, signals)  # cfg missing
        with pytest.raises(ValueError):
            shrink_term_structure(base, market, signals, cfg, kappa=np.ones((3, 3)))


# ---------------------------------------------------------------------------
# 5. seeded recovery study: does shrinkage actually lower alpha MSE?
# ---------------------------------------------------------------------------

def _mse_study(rho: float, n_epochs: int, n_assets: int, n_reps: int, seed: int):
    """Paired MSE of unshrunk vs kappa-shrunk alphas against the KNOWN truth.

    Per replication: draw a calibration sample with population IC ``rho``, fit
    the single coefficient by pooled OLS, estimate kappa on the same sample
    (exactly the leg-2 wiring), then score both alpha vectors on a fresh
    cross-section against ``alpha_true = beta * z_test``.
    """
    rng = np.random.default_rng(seed)
    se_un = 0.0
    se_sh = 0.0
    kappas = []
    for _ in range(n_reps):
        z, r, beta = _epoch_sample(rho, n_epochs, n_assets, rng)
        zf = z.reshape(-1)
        rf = r.reshape(-1)
        b_hat = float(zf @ rf / (zf @ zf))  # pooled OLS, the estimation step
        st = ic_stats_from_epochs(z, r, seg_bounds=ONE_SEGMENT, overlap_mult=np.ones(1))
        kappa = float(st.kappa()[0, 0])
        kappas.append(kappa)
        z_test = rng.standard_normal(512)
        alpha_true = beta * z_test
        se_un += float(np.mean((b_hat * z_test - alpha_true) ** 2))
        se_sh += float(np.mean((kappa * b_hat * z_test - alpha_true) ** 2))
    return se_un / n_reps, se_sh / n_reps, float(np.mean(kappas))


class TestRecovery:
    """Known-truth synthetic study (seeded, deterministic)."""

    def test_weak_signal_shrinkage_lowers_mse(self):
        mse_un, mse_sh, kbar = _mse_study(
            rho=0.03, n_epochs=8, n_assets=60, n_reps=200, seed=20260823
        )
        ratio = mse_sh / mse_un
        print(f"\nweak signal (rho=0.03, 8x60): MSE unshrunk={mse_un:.3e} "
              f"shrunk={mse_sh:.3e} ratio={ratio:.3f} mean kappa={kbar:.3f}")
        assert mse_sh < mse_un
        assert ratio < 0.85, ratio
        assert kbar < 0.6, kbar

    def test_strong_signal_is_left_alone(self):
        mse_un, mse_sh, kbar = _mse_study(
            rho=0.30, n_epochs=32, n_assets=200, n_reps=60, seed=20260824
        )
        ratio = mse_sh / mse_un
        print(f"\nstrong signal (rho=0.30, 32x200): MSE unshrunk={mse_un:.3e} "
              f"shrunk={mse_sh:.3e} ratio={ratio:.3f} mean kappa={kbar:.4f}")
        assert kbar > 0.99, kbar
        assert ratio < 1.05, ratio

    def test_mse_gain_grows_as_the_signal_weakens(self):
        ratios = []
        for i, rho in enumerate((0.02, 0.05, 0.15)):
            mse_un, mse_sh, _ = _mse_study(
                rho=rho, n_epochs=8, n_assets=60, n_reps=120, seed=777 + i
            )
            ratios.append(mse_sh / mse_un)
        print(f"\nMSE ratio by population IC (0.02/0.05/0.15): "
              f"{['%.3f' % r for r in ratios]}")
        assert ratios[0] < ratios[1] < ratios[2]
        assert ratios[0] < 1.0
