"""Tests for the MPO-T1 alpha term structure (SPEC section 2).

Covers ``openpo.mpo.term_structure`` (calibration, stage alphas, the exported
cross-sectional z-score) and ``openpo.mpo.decay`` (AR(1) persistence).

The synthetic panels are built HERE by hand (no ``openpo.mpo.synth`` import --
that module is a parallel work item) with a generator whose term structure is
known in closed form:

- Per signal ``k`` a latent AR(1) ``a_k[e]`` lives on the DECISION-EPOCH grid
  (8 epochs/day, offsets (5,35,...,215)), with epoch coefficient
  ``rho_epoch_k = 0.5 ** (30 / half_life_minutes_k)`` and unit stationary
  cross-sectional variance. The observed signal at epoch tau IS ``a[e]``.
- Consecutive epochs are exactly 30 trading minutes apart (also across the day
  boundary: ``d*240+215`` -> ``(d+1)*240+5``), so the 30-minute "blocks"
  ``(tau_e, tau_e + 30]`` tile the minute axis with NO gaps. The per-minute
  expected return of every minute in block ``e`` is ``sum_k g_k * a_k[e]``, so
  the segment ``(a, b]`` anchored at epoch ``e`` covers exactly the blocks
  ``e + a/30 .. e + b/30 - 1`` and therefore

      E[ r_seg[s] | a[e] ]  =  30 * sum_k g_k * (sum_{j in J_s} rho_epoch_k**j)
                                    * a_k[e]
      J_0 = {0}, J_1 = {1..7}, J_2 = {8..79}

  which is the ground truth ``B_true[s, k]`` (in z-score units, because ``a``
  has unit cross-sectional dispersion). Prices are the compounded minute
  returns, so the fit sees only prices -- never the latent drift.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from openpo.mpo import decay as decay_mod
from openpo.mpo.calendar import epoch_taus
from openpo.mpo.config import EPOCH_OFFSETS, MINUTES_PER_DAY, SEGMENT_BOUNDS, MpoConfig
from openpo.mpo.stages import build_stage_grid
from openpo.mpo.term_structure import (
    TELESCOPING_PRIOR,
    TermStructureModel,
    fit_term_structure,
    segment_overlap_weights,
    winsorize_cross_section,
    zscore_cross_section,
)
from openpo.mpo.types import MarketPanel, SignalPanel

N_EPOCHS_PER_DAY = len(EPOCH_OFFSETS)
EPOCH_MINUTES = decay_mod.EPOCH_MINUTES
# Block index sets of each segment, in units of 30-minute epoch blocks.
_SEG_BLOCKS = tuple(range(a // EPOCH_MINUTES, b // EPOCH_MINUTES) for a, b in SEGMENT_BOUNDS)


# ---------------------------------------------------------------------------
# synthetic panel with a closed-form term structure
# ---------------------------------------------------------------------------

def _block_index(t: int) -> int:
    """Epoch-block index owning minute ``t`` (block e = ``(tau_e, tau_e+30]``).

    Returns -1 for the handful of minutes before the very first epoch.
    """
    d, m = divmod(int(t), MINUTES_PER_DAY)
    if m <= EPOCH_OFFSETS[0]:  # 0..5 belong to the previous day's last block
        return (d - 1) * N_EPOCHS_PER_DAY + (N_EPOCHS_PER_DAY - 1)
    return d * N_EPOCHS_PER_DAY + (m - EPOCH_OFFSETS[0] - 1) // EPOCH_MINUTES


def _make_panel(
    n_days: int = 30,
    n_assets: int = 120,
    half_lives: tuple[float, float, float] = (30.0, 240.0, 2400.0),
    g: tuple[float, float, float] = (6e-5, 2e-5, 8e-6),
    noise: float = 1.5e-4,
    seed: int = 0,
):
    """Build (MarketPanel, SignalPanel, B_true[3,3], rho_epoch_true[3]).

    ``half_lives`` are TRADING-minute half-lives of the latent alphas;
    ``g`` their per-minute return loadings; ``noise`` the per-minute idio
    return sigma. See the module docstring for the closed-form B_true.
    """
    rng = np.random.default_rng(seed)
    n_min = n_days * MINUTES_PER_DAY
    n_sig = len(half_lives)
    n_ep = n_days * N_EPOCHS_PER_DAY

    rho_epoch = 0.5 ** (EPOCH_MINUTES / np.asarray(half_lives, dtype=np.float64))
    latent = np.empty((n_ep, n_assets, n_sig))
    latent[0] = rng.standard_normal((n_assets, n_sig))
    sd_eps = np.sqrt(np.maximum(1.0 - rho_epoch**2, 0.0))
    for e in range(1, n_ep):
        latent[e] = rho_epoch * latent[e - 1] + sd_eps * rng.standard_normal((n_assets, n_sig))

    gv = np.asarray(g, dtype=np.float64)
    blocks = np.array([_block_index(t) for t in range(n_min)])
    mu = np.zeros((n_min, n_assets))
    valid = blocks >= 0
    mu[valid] = latent[blocks[valid]] @ gv
    ret = mu + noise * rng.standard_normal((n_min, n_assets))
    prices = 20.0 * np.cumprod(1.0 + ret, axis=0)

    signals = np.zeros((n_min, n_assets, n_sig))
    signals[epoch_taus(n_days)] = latent  # observed only on the epoch grid

    b_true = np.array(
        [[EPOCH_MINUTES * gv[k] * sum(rho_epoch[k] ** j for j in blocks_s)
          for k in range(n_sig)]
         for blocks_s in _SEG_BLOCKS]
    )

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
    return market, SignalPanel(signals=signals), b_true, rho_epoch


def _cfg(**kw) -> MpoConfig:
    return dataclasses.replace(MpoConfig(), **kw)


# ---------------------------------------------------------------------------
# caller contract (backtest._instantiate signature matching)
# ---------------------------------------------------------------------------

class TestCallerContract:
    """The fit is constructed by name from backtest's candidate pool."""

    CANDIDATES = {
        "market", "market_panel", "panel", "signals", "signal_panel",
        "cfg", "config", "mpo_cfg", "mcfg", "scfg", "calib_days",
    }

    def test_signature_names_are_in_the_candidate_pool(self):
        params = inspect.signature(fit_term_structure).parameters
        assert set(params) <= self.CANDIDATES, (
            f"unknown parameter names {set(params) - self.CANDIDATES}; "
            "backtest._instantiate could not fill them"
        )
        required = [n for n, p in params.items() if p.default is inspect.Parameter.empty]
        assert required == ["market", "signals", "cfg"]

    def test_instantiate_through_backtest_adapter(self):
        from openpo.mpo import backtest

        market, signals, _, _ = _make_panel(n_days=14, n_assets=40, seed=3)
        cfg = _cfg(calib_days=12)
        model = backtest._instantiate(
            fit_term_structure,
            {
                "market": market, "market_panel": market, "panel": market,
                "signals": signals, "signal_panel": signals,
                "cfg": cfg, "config": cfg, "mpo_cfg": cfg, "mcfg": cfg,
                "scfg": None, "calib_days": cfg.calib_days,
            },
        )
        assert model.B.shape == (3, 3)
        assert model.rho.shape == (3,)
        assert model.seg_bounds == SEGMENT_BOUNDS
        assert model.diagnostics.coef_table.shape == (3, 3)
        assert np.all(np.isfinite(model.B))

    def test_calib_days_defaults_to_cfg(self):
        market, signals, _, _ = _make_panel(n_days=14, n_assets=40, seed=3)
        cfg = _cfg(calib_days=12)
        a = fit_term_structure(market, signals, cfg)
        b = fit_term_structure(market, signals, cfg, calib_days=12)
        assert np.array_equal(a.B, b.B)
        assert a.diagnostics.notes["calib_days"] == 12

    @pytest.mark.parametrize("bad", [0, -1, 99])
    def test_calib_days_validated(self, bad):
        market, signals, _, _ = _make_panel(n_days=14, n_assets=40, seed=3)
        with pytest.raises(ValueError):
            fit_term_structure(market, signals, MpoConfig(), calib_days=bad)


# ---------------------------------------------------------------------------
# 1. coefficient recovery
# ---------------------------------------------------------------------------

class TestCoefficientRecovery:
    """Fitted B must recover the known generator term structure (corr > 0.9)."""

    @pytest.mark.parametrize("seed", [0, 4])
    def test_b_rows_recover_truth(self, seed):
        market, signals, b_true, _ = _make_panel(n_days=30, n_assets=120, seed=seed)
        model = fit_term_structure(market, signals, _cfg(calib_days=28))

        assert model.B.shape == b_true.shape
        for s in range(3):
            corr = np.corrcoef(b_true[s], model.B[s])[0, 1]
            assert corr > 0.9, f"segment {s}: row correlation {corr:.3f} <= 0.9"
        flat = np.corrcoef(b_true.ravel(), model.B.ravel())[0, 1]
        assert flat > 0.9, f"pooled B correlation {flat:.3f} <= 0.9"
        # Magnitudes, not just shape: the strongest column of each row is right
        # to within a factor of 2 (ridge shrinkage is intentionally mild).
        for s in range(3):
            k = int(np.argmax(np.abs(b_true[s])))
            ratio = model.B[s, k] / b_true[s, k]
            assert 0.5 < ratio < 2.0, f"segment {s} col {k} scale off: {ratio:.2f}"

    def test_diagnostics_are_populated(self):
        market, signals, _, rho_true = _make_panel(n_days=24, n_assets=80, seed=1)
        cfg = _cfg(calib_days=20)
        model = fit_term_structure(market, signals, cfg)
        diag = model.diagnostics

        assert diag.coef_table.shape == (3, 3)
        assert np.array_equal(diag.coef_table, model.B)
        assert diag.seg_ic.shape == (3,)
        assert np.all(np.isfinite(diag.seg_ic))
        assert diag.seg_ic[0] > 0.1  # in-sample IC of a well-specified segment
        assert diag.rho_epoch.shape == (3,)
        assert diag.n_samples == sum(diag.notes["n_samples_per_segment"])
        assert diag.n_samples > 0
        # Segment guard: the long segment necessarily uses fewer epochs.
        n_ep = diag.notes["n_epochs_per_segment"]
        assert n_ep[0] > n_ep[2] > 0
        assert diag.notes["calib_days"] == 20
        assert diag.notes["max_tau_touched"] < diag.notes["t_calib"]

    def test_prior_is_telescoping_and_scaled(self):
        market, signals, _, _ = _make_panel(n_days=24, n_assets=80, seed=1)
        model = fit_term_structure(market, signals, _cfg(calib_days=20))
        prior = np.asarray(model.diagnostics.notes["prior_scaled"])
        scale = np.asarray(model.diagnostics.notes["seg_return_scale"])
        assert np.all(scale > 0)
        assert np.allclose(prior, TELESCOPING_PRIOR * scale[:, None])
        # Realized return scale grows with the segment length.
        assert scale[0] < scale[1] < scale[2]

    def test_huge_ridge_collapses_to_prior(self):
        """lam = ridge_lambda * n_samples: a giant lambda pins B to B0."""
        market, signals, _, _ = _make_panel(n_days=14, n_assets=60, seed=2)
        model = fit_term_structure(market, signals, _cfg(calib_days=12, ridge_lambda=1e12))
        prior = np.asarray(model.diagnostics.notes["prior_scaled"])
        assert np.allclose(model.B, prior, rtol=1e-4, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. lookahead guards
# ---------------------------------------------------------------------------

class TestLookaheadGuards:
    def test_signal_shift_collapses_segment_ic(self):
        """Misaligning signals by one epoch bar must destroy the segment IC.

        The panel has white-noise-per-epoch alpha (half-life -> 0), so the
        signal observed at tau predicts EXACTLY the (0,30] segment and nothing
        else. Replacing it with the NEXT epoch's signal (a value that is not
        knowable at tau) collapses seg_ic[0] to ~0: a fit that had been
        peeking at future signals could not lose this much.
        """
        market, signals, _, _ = _make_panel(
            n_days=14, n_assets=100, half_lives=(1e-6, 1e-6, 1e-6), seed=5
        )
        cfg = _cfg(calib_days=12)
        base = fit_term_structure(market, signals, cfg)

        taus = epoch_taus(market.n_days)
        shifted = signals.signals.copy()
        shifted[taus[:-1]] = signals.signals[taus[1:]]  # one epoch bar forward
        shifted_model = fit_term_structure(market, SignalPanel(signals=shifted), cfg)

        ic0, ic0_shift = base.diagnostics.seg_ic[0], shifted_model.diagnostics.seg_ic[0]
        assert ic0 > 0.5, f"baseline seg_ic[0]={ic0:.3f} too weak to be a guard"
        assert ic0_shift < 0.2 * ic0, (
            f"seg_ic[0] only moved {ic0:.3f} -> {ic0_shift:.3f} after a "
            "one-bar signal shift; alignment is not being enforced"
        )
        # The fitted loading on the (0,30] segment collapses with the IC.
        assert np.abs(shifted_model.B[0]).max() < 0.2 * np.abs(base.B[0]).max()

    def test_no_lookahead_past_calib_window(self):
        """Nothing at tau >= calib_days*240 may influence the fit.

        Prices and signals outside the calibration window are poisoned with
        NaN / 1e9; every fitted quantity must be bit-identical.
        """
        market, signals, _, _ = _make_panel(n_days=30, n_assets=80, seed=6)
        cfg = _cfg(calib_days=20)
        clean = fit_term_structure(market, signals, cfg)

        t_calib = cfg.calib_days * MINUTES_PER_DAY
        assert market.prices.shape[0] > t_calib  # the poison must be reachable
        poisoned_prices = market.prices.copy()
        poisoned_prices[t_calib:] = np.nan
        poisoned_prices[t_calib::7] = 1e9
        poisoned_signals = signals.signals.copy()
        poisoned_signals[t_calib:] = np.nan
        poisoned_signals[t_calib::5] = -1e9
        dirty = fit_term_structure(
            dataclasses.replace(market, prices=poisoned_prices),
            SignalPanel(signals=poisoned_signals),
            cfg,
        )

        assert np.array_equal(clean.B, dirty.B)
        assert np.array_equal(clean.rho, dirty.rho)
        assert np.array_equal(clean.diagnostics.seg_ic, dirty.diagnostics.seg_ic, equal_nan=True)
        assert np.array_equal(clean.diagnostics.rho_epoch, dirty.diagnostics.rho_epoch)
        assert clean.diagnostics.n_samples == dirty.diagnostics.n_samples
        assert np.all(np.isfinite(dirty.B))

    def test_far_segment_needs_room_inside_the_window(self):
        """With a window shorter than the long segment that segment gets no samples."""
        market, signals, _, _ = _make_panel(n_days=30, n_assets=40, seed=7)
        model = fit_term_structure(market, signals, _cfg(calib_days=9))  # 2160 < 2400
        n_ep = model.diagnostics.notes["n_epochs_per_segment"]
        assert n_ep[0] > 0 and n_ep[1] > 0
        assert n_ep[2] == 0
        assert np.all(model.B[2] == 0.0)  # untouched row, no silent garbage
        assert np.isnan(model.diagnostics.seg_ic[2])


# ---------------------------------------------------------------------------
# 3. AR(1) decay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_half_life_recovery(self):
        """Known epoch/minute persistence recovered within tolerance.

        Also pins the documented ordering choice: the Kendall correction is
        applied at the EPOCH scale. Applied to rho_minute (~1 - 2e-3) instead,
        the +4/T_eff bump would clip rho to 1 and the half-lives below would
        blow up to infinity.
        """
        half_lives = (30.0, 240.0, 2400.0)
        market, signals, _, rho_true = _make_panel(
            n_days=30, n_assets=120, half_lives=half_lives, seed=8
        )
        model = fit_term_structure(market, signals, _cfg(calib_days=28))

        rho_epoch = model.diagnostics.rho_epoch
        assert np.allclose(rho_epoch, rho_true, atol=0.03), f"{rho_epoch} vs {rho_true}"
        assert np.all(rho_epoch > 0) and np.all(rho_epoch < 1)
        assert np.all(model.rho > 0) and np.all(model.rho < 1)
        # rho is the PER-MINUTE coefficient: rho_minute**30 == rho_epoch.
        assert np.allclose(model.rho**EPOCH_MINUTES, rho_epoch, rtol=1e-9)

        hl = decay_mod.half_life_minutes(model.rho)
        assert np.all(hl < 1e6)  # not clipped to "never decays"
        for k, want in enumerate(half_lives):
            assert 0.75 * want < hl[k] < 1.35 * want, f"signal {k}: half-life {hl[k]:.0f}"
        assert hl[0] < hl[1] < hl[2]  # monotone term structure of persistence

    def test_kendall_correction_lifts_the_ols_slope(self):
        market, signals, _, _ = _make_panel(n_days=24, n_assets=80, seed=9)
        model = fit_term_structure(market, signals, _cfg(calib_days=20))
        raw = np.asarray(model.diagnostics.notes["rho_epoch_raw"])
        t_eff = model.diagnostics.notes["kendall_t_eff"]
        assert t_eff == 20 * N_EPOCHS_PER_DAY - 1
        assert np.all(model.diagnostics.rho_epoch > raw)  # bias added back
        assert np.allclose(
            model.diagnostics.rho_epoch, raw + (1 + 3 * raw) / t_eff, atol=1e-12
        )

    def test_ols_slopes_and_helpers(self):
        rng = np.random.default_rng(11)
        rho = 0.7
        n = 4000
        x = np.zeros((n, 2))
        for e in range(1, n):
            x[e] = rho * x[e - 1] + rng.standard_normal(2)
        slopes = decay_mod.ar1_ols_slopes(x)
        assert slopes.shape == (2,)
        assert np.allclose(slopes, rho, atol=0.05)

        # degenerate / too-short series -> NaN, never a bogus number
        assert np.isnan(decay_mod.ar1_ols_slopes(np.ones((50, 1)))[0])
        assert np.isnan(decay_mod.ar1_ols_slopes(np.zeros((2, 1)))[0])
        # NaN tolerance: holes are dropped pairwise
        y = x.copy()
        y[::17, 0] = np.nan
        assert abs(decay_mod.ar1_ols_slopes(y)[0] - rho) < 0.08

        assert decay_mod.rho_from_half_life(30.0) ** 30 == pytest.approx(0.5)
        assert decay_mod.half_life_minutes(0.5 ** (1 / 30)) == pytest.approx(30.0)

    def test_degenerate_signal_column_is_fail_soft(self):
        """A constant signal column yields no persistence and no NaNs."""
        market, signals, _, _ = _make_panel(n_days=14, n_assets=60, seed=12)
        raw = signals.signals.copy()
        raw[:, :, 2] = 3.0  # zero cross-sectional dispersion for every tau
        model = fit_term_structure(market, SignalPanel(signals=raw), _cfg(calib_days=12))
        assert np.all(np.isfinite(model.B))
        assert np.all(np.isfinite(model.rho))
        assert model.decay.notes["degenerate_signals"] == [2]
        # No persistence at all: rho_epoch pinned to the floor, so the
        # per-minute coefficient is its 30th root and the half-life ~ 0.
        assert model.diagnostics.rho_epoch[2] == pytest.approx(decay_mod.RHO_FLOOR)
        assert model.rho[2] == pytest.approx(decay_mod.RHO_FLOOR ** (1 / EPOCH_MINUTES))
        assert decay_mod.half_life_minutes(model.rho)[2] < 2.0

    @pytest.mark.parametrize("mult", [0.5, 1.0, 2.0, 4.0])
    def test_effective_rho_monotone_in_decay_mult(self, mult):
        rho = np.array([0.9, 0.99, 0.999])
        eff = decay_mod.effective_rho(rho, mult)
        assert np.allclose(eff, rho ** (1.0 / mult))
        assert np.all((eff > 0) & (eff < 1))
        if mult > 1:
            assert np.all(eff > rho)  # slower decay
        elif mult < 1:
            assert np.all(eff < rho)  # faster decay

    def test_decay_mult_from_cfg_reaches_the_model(self):
        market, signals, _, _ = _make_panel(n_days=14, n_assets=60, seed=13)
        base = fit_term_structure(market, signals, _cfg(calib_days=12))
        slow = fit_term_structure(market, signals, _cfg(calib_days=12, decay_mult=3.0))
        # The estimate itself is untouched; only the effective decay moves.
        assert np.array_equal(base.rho, slow.rho)
        assert np.all(slow.effective_rho() > base.effective_rho())

    def test_stage_alpha_decay_monotone_in_decay_mult(self):
        rng = np.random.default_rng(14)
        z = np.abs(rng.standard_normal((25, 3))) + 0.1  # positive z, positive alpha
        B = np.full((3, 3), 1e-3)
        rho = np.array([0.98, 0.999, 0.9999])
        grid = build_stage_grid(5, 15, MpoConfig())
        alphas = [
            TermStructureModel(B=B, rho=rho, decay_mult=m).stage_alphas(z, grid)
            for m in (0.5, 1.0, 2.0)
        ]
        for j in range(1, grid.n_stages):  # stage 0 has delta = 0 by contract
            assert np.all(alphas[0][j] < alphas[1][j])
            assert np.all(alphas[1][j] < alphas[2][j])
        assert np.allclose(alphas[0][0], alphas[2][0])  # execution stage unaffected


# ---------------------------------------------------------------------------
# 4. stage alphas
# ---------------------------------------------------------------------------

class TestStageAlphas:
    def _model(self, B=None, rho=None, decay_mult=1.0) -> TermStructureModel:
        B = np.array([[2e-3, 1e-3, 5e-4],
                      [1e-3, 2e-3, 1e-3],
                      [0.0, 1e-3, 4e-3]]) if B is None else B
        rho = np.array([0.977, 0.997, 0.9997]) if rho is None else rho
        return TermStructureModel(B=B, rho=rho, decay_mult=decay_mult)

    def test_shape_and_finiteness(self):
        rng = np.random.default_rng(20)
        n = 17
        z = rng.standard_normal((n, 3))
        grid = build_stage_grid(5, 15, MpoConfig())
        alpha = self._model().stage_alphas(z, grid)
        assert alpha.shape == (grid.n_stages, n)
        assert alpha.dtype == np.float64
        assert np.all(np.isfinite(alpha))

    def test_nan_z_is_neutralized(self):
        z = np.zeros((4, 3))
        z[1] = np.nan
        z[2] = [np.inf, 1.0, -np.inf]
        grid = build_stage_grid(35, 15, MpoConfig())
        alpha = self._model().stage_alphas(z, grid)
        assert np.all(np.isfinite(alpha))
        assert np.all(alpha[:, 1] == 0.0)  # all-NaN row -> zero alpha

    def test_execution_stage_uses_delta_zero(self):
        rng = np.random.default_rng(21)
        z = rng.standard_normal((9, 3))
        grid = build_stage_grid(65, 15, MpoConfig())
        model = self._model()
        alpha = model.stage_alphas(z, grid)

        stage0 = grid.stages[0]
        seg_w = segment_overlap_weights(grid.tau_now, stage0.start, stage0.end)
        expected = z @ (seg_w @ model.B)  # no decay factor at all
        assert np.allclose(alpha[0], expected)

        # Independent proof: changing rho cannot move the execution stage.
        fast = self._model(rho=np.array([0.5, 0.5, 0.5]))
        assert np.allclose(fast.stage_alphas(z, grid)[0], alpha[0])
        assert not np.allclose(fast.stage_alphas(z, grid)[1], alpha[1])

    def test_later_stages_apply_midpoint_decay(self):
        rng = np.random.default_rng(22)
        z = rng.standard_normal((6, 3))
        grid = build_stage_grid(5, 15, MpoConfig())
        model = self._model(decay_mult=1.5)
        alpha = model.stage_alphas(z, grid)
        rho_eff = model.effective_rho()
        for j, stage in enumerate(grid.stages[1:], start=1):
            delta = stage.midpoint_offset(grid.tau_now)
            seg_w = segment_overlap_weights(grid.tau_now, stage.start, stage.end)
            expected = z @ ((seg_w @ model.B) * rho_eff**delta)
            assert np.allclose(alpha[j], expected), f"stage {j}"

    def test_segment_outside_stage_contributes_zero(self):
        """Only the long segment loaded -> the intraday stages get exactly 0."""
        rng = np.random.default_rng(23)
        z = rng.standard_normal((11, 3))
        B = np.zeros((3, 3))
        B[2] = [1e-3, 2e-3, 3e-3]  # segment (240, 2400] only
        grid = build_stage_grid(5, 15, MpoConfig())
        alpha = self._model(B=B).stage_alphas(z, grid)
        # S1 = (5,35] and S2 = (35,239] both end before tau_now + 240.
        for j, stage in enumerate(grid.stages):
            if stage.end <= grid.tau_now + 240:
                assert np.all(alpha[j] == 0.0), f"stage {j} should be alpha-free"
        assert np.any(alpha[-1] != 0.0)  # the tail stage does carry it

    def test_overlap_weights_partition_each_segment(self):
        """Stages tile (tau, tau+2400], so each segment's weights sum to 1.

        With rho == 1 (no decay) the stage alphas must therefore add up to the
        full-horizon alpha ``z @ B.sum(axis=0)``.
        """
        rng = np.random.default_rng(24)
        z = rng.standard_normal((13, 3))
        cfg = MpoConfig()
        tau = 5
        grid = build_stage_grid(tau, 15, cfg)
        assert grid.stages[0].start == tau
        assert grid.stages[-1].end == tau + cfg.horizon_minutes  # full coverage

        model = self._model(rho=np.ones(3))
        total = model.stage_alphas(z, grid).sum(axis=0)
        assert np.allclose(total, z @ model.B.sum(axis=0))

        # And the raw weights: each segment is exactly accounted for once.
        w = np.zeros(3)
        for stage in grid.stages:
            w += segment_overlap_weights(tau, stage.start, stage.end)
        assert np.allclose(w, 1.0)

    def test_overlap_weights_helper(self):
        # exact segment -> weight 1 for that segment only
        w = segment_overlap_weights(1000, 1000, 1030)
        assert np.allclose(w, [1.0, 0.0, 0.0])
        # half of the middle segment
        w = segment_overlap_weights(1000, 1030, 1135)
        assert np.allclose(w, [0.0, 105 / 210, 0.0])
        # stage entirely before / after the segments
        assert np.allclose(segment_overlap_weights(1000, 900, 1000), 0.0)
        assert np.allclose(segment_overlap_weights(1000, 3400, 3500), 0.0)
        assert np.all(segment_overlap_weights(1000, 1000, 3400) == 1.0)

    def test_bad_z_shape_raises(self):
        grid = build_stage_grid(5, 15, MpoConfig())
        with pytest.raises(ValueError):
            self._model().stage_alphas(np.zeros((4, 2)), grid)
        with pytest.raises(ValueError):
            self._model().stage_alphas(np.zeros(4), grid)

    def test_fitted_model_feeds_the_strategy_seam(self):
        """End-to-end: fitted model -> stage_alphas on a real grid, sane scale."""
        market, signals, _, _ = _make_panel(n_days=24, n_assets=50, seed=25)
        cfg = _cfg(calib_days=20)
        model = fit_term_structure(market, signals, cfg)
        tau = 20 * MINUTES_PER_DAY + EPOCH_OFFSETS[3]
        grid = build_stage_grid(tau, market.n_days, cfg)
        z = zscore_cross_section(signals.signals[tau])
        alpha = model.stage_alphas(z, grid)
        assert alpha.shape == (grid.n_stages, market.n_assets)
        assert np.all(np.isfinite(alpha))
        # Segment returns are fractions of price: single-digit percent at most.
        assert np.abs(alpha).max() < 0.1


# ---------------------------------------------------------------------------
# 5. cross-sectional z-score (also consumed by simulator.py)
# ---------------------------------------------------------------------------

class TestZscoreCrossSection:
    def test_matrix_columns_are_standardized(self):
        rng = np.random.default_rng(30)
        x = rng.normal(5.0, 3.0, (200, 3)) * np.array([1.0, 10.0, 0.1])
        z = zscore_cross_section(x)
        assert z.shape == x.shape
        assert z.dtype == np.float64
        assert np.allclose(z.mean(axis=0), 0.0, atol=1e-12)
        assert np.allclose(z.std(axis=0), 1.0, atol=1e-12)
        # rank preserving within a column
        for k in range(3):
            assert np.array_equal(np.argsort(x[:, k]), np.argsort(z[:, k]))

    def test_vector_input_keeps_shape(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        z = zscore_cross_section(x)
        assert z.shape == (4,)
        assert z.mean() == pytest.approx(0.0)
        assert z.std() == pytest.approx(1.0)
        assert np.allclose(z, (x - x.mean()) / x.std())

    def test_columns_are_independent(self):
        x = np.column_stack([np.arange(5.0), np.arange(5.0) * 100.0 + 7.0])
        z = zscore_cross_section(x)
        assert np.allclose(z[:, 0], z[:, 1])

    def test_nan_inputs_stay_nan_and_do_not_pollute_others(self):
        x = np.array([[1.0, 1.0, 1.0],
                      [2.0, np.nan, 2.0],
                      [3.0, 3.0, np.nan],
                      [4.0, 5.0, 4.0]])
        z = zscore_cross_section(x)
        assert np.isnan(z[1, 1]) and np.isnan(z[2, 2])
        finite = np.array([1.0, 3.0, 5.0])
        assert np.allclose(z[[0, 2, 3], 1], (finite - finite.mean()) / finite.std())
        col0 = x[:, 0]
        assert np.allclose(z[:, 0], (col0 - col0.mean()) / col0.std())
        assert np.isfinite(z[[0, 1, 3], 2]).all()

    def test_infinities_are_treated_as_missing(self):
        x = np.array([[1.0], [2.0], [np.inf], [3.0], [-np.inf]])
        z = zscore_cross_section(x)
        assert np.isnan(z[2, 0]) and np.isnan(z[4, 0])
        vals = np.array([1.0, 2.0, 3.0])
        assert np.allclose(z[[0, 1, 3], 0], (vals - vals.mean()) / vals.std())

    def test_degenerate_columns_map_to_zero(self):
        x = np.column_stack([np.full(6, 2.5), np.arange(6.0)])
        z = zscore_cross_section(x)
        assert np.all(z[:, 0] == 0.0)
        assert z[:, 1].std() == pytest.approx(1.0)

    def test_all_nan_and_single_observation(self):
        z = zscore_cross_section(np.full((5, 2), np.nan))
        assert np.all(np.isnan(z))
        x = np.array([[np.nan], [7.0], [np.nan]])
        z = zscore_cross_section(x)
        assert z[1, 0] == 0.0  # single finite observation -> neutral
        assert np.isnan(z[[0, 2], 0]).all()

    def test_bad_rank_raises(self):
        with pytest.raises(ValueError):
            zscore_cross_section(np.zeros((2, 2, 2)))

    def test_simulator_picks_it_up(self):
        """simulator.py imports this symbol; keep the seam wired."""
        from openpo.mpo import simulator

        assert simulator._ts_zscore is zscore_cross_section

    def test_winsorize_helper(self):
        x = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        w = winsorize_cross_section(x, 1.0)
        mu, sd = x.mean(), x.std()
        assert w.max() == pytest.approx(mu + sd)  # the 100 is pulled in
        assert w.min() == 0.0  # already inside mu - sd (which is negative)
        assert np.array_equal(w[1:5], x[1:5])  # untouched middle
        # symmetric band, tighter sigma clips both ends
        y = np.array([-10.0, 0.0, 0.0, 10.0])
        w2 = winsorize_cross_section(y, 0.5)
        assert w2.min() == pytest.approx(-0.5 * y.std())
        assert w2.max() == pytest.approx(0.5 * y.std())
        assert np.array_equal(winsorize_cross_section(x, 0.0), x)  # disabled
        assert np.array_equal(winsorize_cross_section(np.full(4, 2.0), 3.0), np.full(4, 2.0))
        nan_in = np.array([np.nan, 1.0, 2.0, 3.0])
        assert np.isnan(winsorize_cross_section(nan_in, 2.0)[0])

    def test_winsor_sigma_caps_targets(self):
        """cfg.winsor_sigma must blunt crazy price prints in the targets.

        A/B on the mechanism itself: poison a few bars, then fit with the
        frozen +-3 sigma clip vs with clipping effectively disabled. The
        winsorized fit has to stay closer to the clean baseline.
        """
        market, signals, _, _ = _make_panel(n_days=14, n_assets=60, seed=31)
        cfg = _cfg(calib_days=12)
        base = fit_term_structure(market, signals, cfg)

        outlier = market.prices.copy()
        for stock, tau in ((0, EPOCH_OFFSETS[0] + 30), (1, 3 * MINUTES_PER_DAY + 100)):
            outlier[tau, stock] *= 20.0  # single absurd bar (fat-finger print)
        poisoned = dataclasses.replace(market, prices=outlier)
        with_winsor = fit_term_structure(poisoned, signals, cfg)
        without = fit_term_structure(poisoned, signals, _cfg(calib_days=12, winsor_sigma=1e9))

        err_w = np.abs(with_winsor.B - base.B).sum()
        err_raw = np.abs(without.B - base.B).sum()
        assert err_w < err_raw, f"winsorization did not help: {err_w:.2e} vs {err_raw:.2e}"
        assert np.all(np.isfinite(with_winsor.B))
