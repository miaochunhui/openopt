"""Leg-2 wiring of the implementation-delay surcharge (Iteration 42).

Design source: ``docs/mpo-t1-delaycost-design.md``. Leg 1 shipped the
coefficient math (``openpo.mpo.delay_cost``, covered by
``tests/mpo/test_delay_cost.py``); THIS file covers the three seams leg 2 adds
and nothing else:

1. ``qp.build_mpo_qp(..., c_buy_extra=, c_sell_extra=)`` -- the extras must move
   exactly the stage-1 ``b``/``s`` entries of ``q`` and nothing else, and the
   default (``None``) must leave the assembled problem BIT-identical, because
   every non-delay strategy keeps going through this function.
2. ``strategies.MpoStrategy(delay_cost_cfg=...)`` -- off by default; on, it must
   change decisions when and only when ``alpha_S1`` is non-zero.
3. ``backtest`` roster: ``mpo_delayaware`` buildable, opt-in, sharing the pool's
   ``ts_model`` so the A/B stays paired.

Everything here is small and hermetic (fake alpha model, tiny N) except the two
roster tests, which use a real synth panel at N=20/60.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openpo.mpo import backtest
from openpo.mpo.closed_form import closed_form_two_stage
from openpo.mpo.config import EPOCH_OFFSETS, MINUTES_PER_DAY, ExperimentConfig, MpoConfig, SynthConfig
from openpo.mpo.delay_cost import (
    DelayCostConfig,
    delay_cost_coefficients,
    effective_trade_costs,
)
from openpo.mpo.qp import build_mpo_qp
from openpo.mpo.solve import solve_mpo_qp
from openpo.mpo.stages import build_stage_grid
from openpo.mpo.strategies import MpoStrategy
from openpo.mpo.types import DecisionContext, PortfolioSnapshot, Stage, StageGrid

_SIBLINGS = (
    "openpo.mpo.synth",
    "openpo.mpo.term_structure",
    "openpo.mpo.strategies",
    "openpo.mpo.simulator",
)

BIG = 1e6
RELAXED = dict(w_min=-BIG, w_max=BIG, budget_lo=-BIG, budget_hi=BIG, b_max=BIG, s_max=BIG)


# ---------------------------------------------------------------------------
# shared tiny fixtures
# ---------------------------------------------------------------------------


def _sigma(n: int, vol: float = 0.025) -> np.ndarray:
    """Diagonal daily covariance at ``vol`` daily vol (keeps analytics readable)."""
    return np.eye(n) * vol**2


def _qp_args(n: int, cfg: MpoConfig, grid: StageGrid, w0=None, sellable=None, alpha=None):
    """Positional argument tuple of ``build_mpo_qp`` for an N-name problem."""
    sigma = _sigma(n)
    if alpha is None:
        alpha = np.full((grid.n_stages, n), 20e-4)
    if w0 is None:
        w0 = np.zeros(n)
    if sellable is None:
        sellable = np.zeros(n)
    return (
        np.asarray(alpha, dtype=np.float64),
        grid,
        np.asarray(w0, dtype=np.float64),
        np.asarray(sellable, dtype=np.float64),
        sigma,
        np.full(n, 0.8),
        np.sqrt(np.diag(sigma)),
        np.zeros(n, bool),
        np.zeros(n, bool),
        cfg,
    )


def _four_stage_grid(cfg: MpoConfig) -> StageGrid:
    grid = build_stage_grid(5, 100, cfg)
    assert grid.n_stages == 4  # guard: the stage-1-only claim needs J > 1
    return grid


def _two_stage_grid() -> StageGrid:
    """J=2, no stage ends today (T+1 rows irrelevant) -- the parity geometry."""
    return StageGrid(
        tau_now=0,
        stages=(Stage(index=0, start=0, end=240), Stage(index=1, start=240, end=480)),
        today_stage_indices=(),
    )


def _same_matrix(a, b) -> bool:
    """Bit-identical sparse comparison (values AND sparsity pattern)."""
    return (
        a.shape == b.shape
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.data, b.data)
    )


# ---------------------------------------------------------------------------
# 1. QP layer: default off == bit-identical
# ---------------------------------------------------------------------------


def test_extras_default_none_is_bit_identical():
    """Every non-delay strategy still goes through here: no drift allowed."""
    cfg = MpoConfig()
    grid = _four_stage_grid(cfg)
    n = 6
    args = _qp_args(n, cfg, grid, w0=np.full(n, 0.005), sellable=np.full(n, 0.002))

    base = build_mpo_qp(*args)
    explicit_none = build_mpo_qp(*args, c_buy_extra=None, c_sell_extra=None)

    assert np.array_equal(base.q, explicit_none.q)
    assert np.array_equal(base.l, explicit_none.l)
    assert np.array_equal(base.u, explicit_none.u)
    assert _same_matrix(base.P, explicit_none.P)
    assert _same_matrix(base.A, explicit_none.A)
    assert base.blocks == explicit_none.blocks
    assert base.objective_offset == explicit_none.objective_offset
    assert base.meta["stage1_cost_extra"] is None
    assert explicit_none.meta["stage1_cost_extra"] is None


def test_zero_extras_are_still_bit_identical_in_q():
    """``scale=0`` / all-zero coefficients must not perturb a single entry.

    ``DelayCostConfig(scale=0.0)`` is the documented "off" switch, so the arm has
    to collapse onto the baseline exactly rather than approximately.
    """
    cfg = MpoConfig()
    grid = _four_stage_grid(cfg)
    n = 5
    args = _qp_args(n, cfg, grid)
    zeros = np.zeros(n)

    base = build_mpo_qp(*args)
    zeroed = build_mpo_qp(*args, c_buy_extra=zeros, c_sell_extra=zeros)
    assert np.array_equal(base.q, zeroed.q)
    assert _same_matrix(base.P, zeroed.P)
    # but the ledger records that the surcharge path WAS taken (0.0, not None)
    assert zeroed.meta["stage1_cost_extra"] == {
        "buy_mean": 0.0, "buy_max": 0.0, "sell_mean": 0.0, "sell_max": 0.0
    }


# ---------------------------------------------------------------------------
# 2. QP layer: the exact q delta, stage 1 only
# ---------------------------------------------------------------------------


def test_extras_change_only_stage1_q_entries_by_exactly_the_coefficients():
    """The whole design in one assertion: q delta == extras on b^1/s^1, else 0.

    Design section 1 / risk 4: later stages' fills are hypothetical replans, so
    the delay correction applies to the EXECUTED stage only. If this ever leaks
    into stages 2..J the A/B stops measuring the delay term.
    """
    cfg = MpoConfig()
    grid = _four_stage_grid(cfg)
    n = 7
    rng = np.random.default_rng(42)
    args = _qp_args(n, cfg, grid, w0=np.full(n, 0.004))
    c_buy_extra = rng.uniform(0.0, 30e-4, size=n)
    c_sell_extra = rng.uniform(0.0, 30e-4, size=n)

    base = build_mpo_qp(*args)
    surch = build_mpo_qp(*args, c_buy_extra=c_buy_extra, c_sell_extra=c_sell_extra)

    delta = surch.q - base.q
    touched = np.zeros(delta.size, bool)
    touched[surch.slice_b(0)] = True
    touched[surch.slice_s(0)] = True
    # Untouched entries are EXACTLY unchanged (bitwise), not just close.
    np.testing.assert_array_equal(delta[~touched], np.zeros(int((~touched).sum())))
    # The touched ones move by exactly the coefficients (up to the ~1e-19 float
    # residue of adding a 30 bp number onto a 7 bp one and subtracting back).
    np.testing.assert_allclose(delta[surch.slice_b(0)], c_buy_extra, rtol=0, atol=1e-18)
    np.testing.assert_allclose(delta[surch.slice_s(0)], c_sell_extra, rtol=0, atol=1e-18)

    # Absolute values, not just the delta: base scalar + per-name surcharge.
    np.testing.assert_allclose(
        surch.q[surch.slice_b(0)], cfg.buy_cost() + c_buy_extra, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        surch.q[surch.slice_s(0)], cfg.sell_cost() + c_sell_extra, rtol=0, atol=0
    )
    # stages 2..J keep the untouched scalars
    for j in range(1, grid.n_stages):
        assert np.all(surch.q[surch.slice_b(j)] == cfg.buy_cost())
        assert np.all(surch.q[surch.slice_s(j)] == cfg.sell_cost())
    # the alpha block and the whole constraint/curvature side are untouched
    assert np.array_equal(surch.q[: grid.n_stages * n], base.q[: grid.n_stages * n])
    assert _same_matrix(surch.P, base.P)
    assert _same_matrix(surch.A, base.A)
    assert np.array_equal(surch.l, base.l) and np.array_equal(surch.u, base.u)

    ledger = surch.meta["stage1_cost_extra"]
    assert ledger["buy_mean"] == pytest.approx(float(np.mean(c_buy_extra)))
    assert ledger["sell_max"] == pytest.approx(float(np.max(c_sell_extra)))


def test_one_sided_extras_are_allowed():
    """Sign asymmetry is the point: a buy-only surcharge must be expressible."""
    cfg = MpoConfig()
    grid = _four_stage_grid(cfg)
    n = 4
    args = _qp_args(n, cfg, grid)
    extra = np.full(n, 13.2e-4)

    base = build_mpo_qp(*args)
    buy_only = build_mpo_qp(*args, c_buy_extra=extra)
    delta = buy_only.q - base.q
    mask = np.ones(delta.size, bool)
    mask[buy_only.slice_b(0)] = False
    np.testing.assert_array_equal(delta[mask], np.zeros(int(mask.sum())))
    np.testing.assert_allclose(delta[buy_only.slice_b(0)], extra, rtol=0, atol=1e-18)
    assert buy_only.meta["stage1_cost_extra"]["sell_mean"] == 0.0


def test_extras_match_delay_cost_effective_costs():
    """Cross-module tie: q's stage-1 costs == ``effective_trade_costs`` output.

    ``delay_cost.effective_trade_costs`` is documented as "the leg-2 wiring
    surface"; this pins that the wiring actually produces those numbers, so the
    doc example (7 bp + 15 bp = 22 bp buy) is verifiable in the solver input.
    """
    cfg = MpoConfig()
    grid = _four_stage_grid(cfg)
    alpha_s1 = np.array([15e-4, -15e-4, 0.0, 4e-4])
    n = alpha_s1.size
    dcfg = DelayCostConfig()  # mode='signed', scale=1, cap=50bp

    c_buy_extra, c_sell_extra = delay_cost_coefficients(alpha_s1, dcfg)
    want_buy, want_sell = effective_trade_costs(alpha_s1, cfg, dcfg)

    alpha = np.tile(alpha_s1, (grid.n_stages, 1))
    args = _qp_args(n, cfg, grid, alpha=alpha)
    qp = build_mpo_qp(*args, c_buy_extra=c_buy_extra, c_sell_extra=c_sell_extra)

    np.testing.assert_allclose(qp.q[qp.slice_b(0)], want_buy, rtol=0, atol=0)
    np.testing.assert_allclose(qp.q[qp.slice_s(0)], want_sell, rtol=0, atol=0)
    # the doc's worked example, in bp
    assert qp.q[qp.slice_b(0)][0] == pytest.approx(22e-4)
    assert qp.q[qp.slice_s(0)][0] == pytest.approx(12e-4)
    assert qp.q[qp.slice_b(0)][1] == pytest.approx(7e-4)
    assert qp.q[qp.slice_s(0)][1] == pytest.approx(27e-4)


# ---------------------------------------------------------------------------
# 3. QP layer: validation
# ---------------------------------------------------------------------------


def test_extras_reject_bad_shapes_and_values():
    cfg = MpoConfig()
    grid = _four_stage_grid(cfg)
    n = 5
    args = _qp_args(n, cfg, grid)

    with pytest.raises(ValueError, match=r"c_buy_extra must be a 1-D \[5\]"):
        build_mpo_qp(*args, c_buy_extra=np.zeros(n + 1))
    with pytest.raises(ValueError, match=r"c_sell_extra must be a 1-D \[5\]"):
        build_mpo_qp(*args, c_sell_extra=np.zeros((grid.n_stages, n)))
    # a scalar is rejected on purpose: "which stage" is the load-bearing detail
    with pytest.raises(ValueError, match="must be a 1-D"):
        build_mpo_qp(*args, c_buy_extra=13.2e-4)
    with pytest.raises(ValueError, match="c_buy_extra must be finite"):
        build_mpo_qp(*args, c_buy_extra=np.full(n, np.nan))
    with pytest.raises(ValueError, match="c_sell_extra must be finite"):
        build_mpo_qp(*args, c_sell_extra=np.full(n, np.inf))
    with pytest.raises(ValueError, match="c_buy_extra must be non-negative"):
        build_mpo_qp(*args, c_buy_extra=np.full(n, -1e-4))
    # a misspelling falls through to the pre-existing override typo guard
    with pytest.raises(ValueError, match="unknown build_mpo_qp overrides"):
        build_mpo_qp(*args, c_buy_extras=np.zeros(n))


def test_negative_extras_need_the_explicit_flag_and_stay_bounded():
    """``mode='exact'`` rebates are legal but must be asked for (design risk 2)."""
    cfg = MpoConfig()
    grid = _four_stage_grid(cfg)
    alpha_s1 = np.array([15e-4, -15e-4, 0.0, 4e-4])
    n = alpha_s1.size
    args = _qp_args(n, cfg, grid, alpha=np.tile(alpha_s1, (grid.n_stages, 1)))
    cb, cs = delay_cost_coefficients(alpha_s1, DelayCostConfig(mode="exact"))
    assert cb.min() < 0.0 and cs.min() < 0.0  # both sides really do go negative

    with pytest.raises(ValueError, match="allow_negative_extra=True"):
        build_mpo_qp(*args, c_buy_extra=cb, c_sell_extra=cs)

    qp = build_mpo_qp(
        *args, c_buy_extra=cb, c_sell_extra=cs, allow_negative_extra=True
    )
    # Boundedness of the wash pair: the two extras cancel exactly, so the
    # round-trip price is still the base cost -- nothing to arbitrage.
    b1 = qp.q[qp.slice_b(0)]
    s1 = qp.q[qp.slice_s(0)]
    np.testing.assert_allclose(b1 + s1, cfg.buy_cost() + cfg.sell_cost(), atol=1e-15)
    sol = solve_mpo_qp(qp, cfg.qp_backend, cfg)
    assert sol.status == "optimal", getattr(sol, "note", sol.status)


def test_extras_do_not_alias_the_callers_buffer():
    cfg = MpoConfig()
    grid = _four_stage_grid(cfg)
    n = 4
    args = _qp_args(n, cfg, grid)
    extra = np.full(n, 10e-4)
    qp = build_mpo_qp(*args, c_buy_extra=extra, c_sell_extra=extra)
    q_before = qp.q.copy()
    extra[:] = 999.0  # mutate the caller's array after the build
    assert np.array_equal(qp.q, q_before)


# ---------------------------------------------------------------------------
# 4. QP layer: axioma parity mode -- documented refusal, parity untouched
# ---------------------------------------------------------------------------


def test_axioma_mode_refuses_extras_rather_than_ignoring_them():
    """DOCUMENTED CHOICE: raise, do not ignore.

    ``cost_mode='axioma'`` replaces every linear b/s cost with the quadratic
    transition penalty, so a per-unit-traded surcharge has nowhere to land.
    Silently dropping it would present a delay-aware arm that is in fact the
    baseline -- design-doc falsification F2 ("coefficients near zero => wiring
    bug, not a verdict") exists precisely because that failure is invisible in
    the results. So the build refuses.
    """
    cfg = MpoConfig(cost_mode="axioma", t1_enabled=False)
    grid = _two_stage_grid()
    n = 4
    args = _qp_args(n, cfg, grid)
    with pytest.raises(ValueError, match="require cost_mode='production'"):
        build_mpo_qp(*args, c_buy_extra=np.zeros(n), **RELAXED)
    with pytest.raises(ValueError, match="require cost_mode='production'"):
        build_mpo_qp(*args, c_sell_extra=np.zeros(n), **RELAXED)


def test_closed_form_parity_survives_the_new_keywords():
    """The section-3 acceptance gate, re-checked with the extras defaulted off."""
    n = 10
    seed = 20260823
    rng = np.random.default_rng(seed)
    sigma = _sigma(n)
    alpha = rng.normal(scale=2e-3, size=(2, n))
    w0 = rng.uniform(0.0, 0.02, size=n)
    cfg = MpoConfig(
        cost_mode="axioma", axioma_delta=1.0, gamma=10.0, t1_enabled=False, qp_eps=1e-9
    )
    qp = build_mpo_qp(
        alpha,
        _two_stage_grid(),
        w0,
        np.zeros(n),
        sigma,
        np.full(n, 0.8),
        np.sqrt(np.diag(sigma)),
        np.zeros(n, bool),
        np.zeros(n, bool),
        cfg,
        c_buy_extra=None,
        c_sell_extra=None,
        **RELAXED,
    )
    sol = solve_mpo_qp(qp, cfg.qp_backend, cfg)
    assert sol.status == "optimal", getattr(sol, "note", sol.status)
    w1_ref = closed_form_two_stage(
        alpha[0], alpha[1], sigma, cfg.gamma, cfg.axioma_delta, w0
    )
    assert float(np.max(np.abs(sol.weights[0] - w1_ref))) < 1e-6


# ---------------------------------------------------------------------------
# 5. QP layer: the surcharge actually bites
# ---------------------------------------------------------------------------


def test_surcharge_raises_the_alpha_hurdle_for_buying():
    """Analytic hurdle: a single name buys iff alpha > c_buy + extra.

    Iteration 41's finding in one problem: the QP thinks a 10 bp alpha clears a
    7 bp buy cost, but with the 13.2 bp delay priced in the true hurdle is
    ~20 bp and the trade should not happen.
    """
    cfg = MpoConfig(w_max=1.0)
    grid = StageGrid(
        tau_now=5, stages=(Stage(index=0, start=5, end=35),), today_stage_indices=(0,)
    )
    alpha = np.array([[10e-4]])
    args = _qp_args(1, cfg, grid, alpha=alpha)

    _, plain = build_mpo_qp(*args), solve_mpo_qp(build_mpo_qp(*args), cfg.qp_backend, cfg)
    qp_s = build_mpo_qp(*args, c_buy_extra=np.array([13.2e-4]))
    surch = solve_mpo_qp(qp_s, cfg.qp_backend, cfg)

    assert plain.status == "optimal" and surch.status == "optimal"
    assert float(plain.buys[0, 0]) > 1e-3, "10bp alpha must clear the 7bp base cost"
    assert float(surch.buys[0, 0]) < 1e-9, "20.2bp hurdle must kill the same trade"

    # And a smaller surcharge only shrinks the position (monotone, not binary).
    mid = solve_mpo_qp(
        build_mpo_qp(*args, c_buy_extra=np.array([1.5e-4])), cfg.qp_backend, cfg
    )
    assert 0.0 < float(mid.buys[0, 0]) < float(plain.buys[0, 0])


def test_sell_surcharge_bites_the_sell_side_only():
    cfg = MpoConfig(w_max=1.0, t1_enabled=False)
    grid = StageGrid(
        tau_now=5, stages=(Stage(index=0, start=5, end=35),), today_stage_indices=(0,)
    )
    n = 1
    w0 = np.array([0.02])
    alpha = np.array([[-15e-4]])  # wants out
    args = _qp_args(n, cfg, grid, w0=w0, sellable=w0.copy(), alpha=alpha)

    plain = solve_mpo_qp(build_mpo_qp(*args), cfg.qp_backend, cfg)
    surch = solve_mpo_qp(
        build_mpo_qp(*args, c_sell_extra=np.array([15e-4])), cfg.qp_backend, cfg
    )
    assert plain.status == "optimal" and surch.status == "optimal"
    assert float(plain.sells[0, 0]) > 1e-4
    assert float(surch.sells[0, 0]) < float(plain.sells[0, 0])
    # a BUY-side surcharge must leave a pure sell decision alone
    buy_side = solve_mpo_qp(
        build_mpo_qp(*args, c_buy_extra=np.array([15e-4])), cfg.qp_backend, cfg
    )
    assert float(buy_side.sells[0, 0]) == pytest.approx(float(plain.sells[0, 0]), abs=1e-9)


# ---------------------------------------------------------------------------
# 6. MpoStrategy wiring
# ---------------------------------------------------------------------------


class _FixedAlphaModel:
    """Term-structure stand-in with a hand-set stage-1 / later-stage alpha.

    The delay coefficients are a pure function of ``alpha[0]``, so controlling it
    exactly is the only way to assert "different iff alpha_S1 != 0" without the
    calibration noise of a real fitted model.
    """

    def __init__(self, alpha_s1, alpha_later):
        self.alpha_s1 = np.asarray(alpha_s1, dtype=np.float64)
        self.alpha_later = np.asarray(alpha_later, dtype=np.float64)
        self.B = np.zeros((3, 3), dtype=np.float64)
        self.calls = 0

    def stage_alphas(self, z, grid):
        self.calls += 1
        out = np.tile(self.alpha_later, (grid.n_stages, 1))
        out[0] = self.alpha_s1
        return out


def _make_ctx(n: int, w0=None, sellable=None, day: int = 1, minute: int | None = None):
    minute = EPOCH_OFFSETS[0] if minute is None else minute
    w0 = np.zeros(n) if w0 is None else np.asarray(w0, dtype=np.float64)
    sellable = np.zeros(n) if sellable is None else np.asarray(sellable, dtype=np.float64)
    prices = np.full(n, 10.0)
    nav = 1e8
    snap = PortfolioSnapshot(
        tau=day * MINUTES_PER_DAY + minute,
        nav=nav,
        cash=nav * (1.0 - float(w0.sum())),
        holdings_shares=w0 * nav / prices,
        weights=w0.copy(),
        sellable_shares=sellable * nav / prices,
        sellable_weights=sellable.copy(),
    )
    return DecisionContext(
        snapshot=snap,
        z_signals=np.zeros((n, 3)),
        prices=prices,
        limit_up=np.zeros(n, bool),
        limit_down=np.zeros(n, bool),
        halted=np.zeros(n, bool),
        day=day,
        minute=minute,
    )


def _make_strategy(model, n: int, delay_cost_cfg=None, cfg=None) -> MpoStrategy:
    cfg = MpoConfig() if cfg is None else cfg
    return MpoStrategy(
        model,
        Sigma_d=_sigma(n),
        adv_frac=np.full(n, 0.8),
        cfg=cfg,
        n_days=10,
        delay_cost_cfg=delay_cost_cfg,
    )


def test_mpo_strategy_delay_cost_is_off_by_default():
    n = 8
    model = _FixedAlphaModel(np.full(n, 20e-4), np.full(n, 20e-4))
    plain = _make_strategy(model, n)
    assert plain.name == "mpo"
    assert plain.delay_cost_cfg is None
    assert plain.last_delay_extra is None
    assert plain._qp_build_kwargs(np.zeros((4, n)), _make_ctx(n)) == {}


def test_delay_cost_cfg_flips_the_roster_name_and_type_checks():
    n = 4
    model = _FixedAlphaModel(np.zeros(n), np.zeros(n))
    arm = _make_strategy(model, n, DelayCostConfig())
    assert arm.name == "mpo_delayaware"
    assert isinstance(arm.delay_cost_cfg, DelayCostConfig)
    # The classic argument swap must be loud, not a silently ignored knob.
    with pytest.raises(TypeError, match="delay_cost_cfg must be"):
        _make_strategy(model, n, MpoConfig())
    with pytest.raises(TypeError, match="delay_cost_cfg must be"):
        _make_strategy(model, n, "signed")


def test_decisions_are_identical_when_alpha_s1_is_zero():
    """No S1 alpha => no forfeited move => zero coefficients => same decision.

    This is the null leg of the wiring test: it proves the difference measured by
    the next test comes from the coefficients and not from taking a different
    code path.
    """
    n = 8
    later = np.full(n, 25e-4)
    plain = _make_strategy(_FixedAlphaModel(np.zeros(n), later), n)
    arm = _make_strategy(_FixedAlphaModel(np.zeros(n), later), n, DelayCostConfig())
    ctx = _make_ctx(n)

    w_plain = plain.decide(ctx)
    w_arm = arm.decide(ctx)
    assert w_plain is not None and w_arm is not None
    assert float(w_plain.sum()) > 1e-3, "the null leg must actually trade"
    np.testing.assert_array_equal(w_plain, w_arm)
    assert arm.last_delay_extra["c_buy_extra_mean"] == 0.0
    assert arm.last_delay_extra["c_sell_extra_mean"] == 0.0


def test_two_sigma_alpha_s1_names_change_the_decision():
    """z = 2 sigma names (~9 bp S1 alpha, design section 3) must trade less.

    Direction is preregistered (design section 4, expectation 1): the surcharge
    raises the buy hurdle, so the delay-aware arm buys strictly less of exactly
    the names whose S1 alpha is positive, and leaves zero-alpha names alone.
    """
    n = 8
    # design section 3: sd(alpha_S1) = IC * sigma_30 = 4.4 bp => 2 sigma = 8.8 bp
    alpha_s1 = np.zeros(n)
    alpha_s1[:4] = 8.8e-4
    later = np.full(n, 25e-4)

    plain = _make_strategy(_FixedAlphaModel(alpha_s1, later), n)
    arm = _make_strategy(_FixedAlphaModel(alpha_s1, later), n, DelayCostConfig())
    ctx = _make_ctx(n)

    w_plain = plain.decide(ctx)
    w_arm = arm.decide(ctx)
    assert w_plain is not None and w_arm is not None
    assert not np.array_equal(w_plain, w_arm), "2-sigma S1 alpha must move the decision"
    # Surcharged names shrink materially; zero-alpha names move only by the
    # budget coupling + OSQP tolerance (two orders of magnitude smaller), so the
    # effect is localised to the names that actually carry stage-1 alpha.
    assert np.all(w_arm[:4] < w_plain[:4] - 1e-4)
    moved_untouched = float(np.max(np.abs(w_arm[4:] - w_plain[4:])))
    moved_charged = float(np.min(np.abs(w_arm[:4] - w_plain[:4])))
    assert moved_untouched < 0.01 * moved_charged, (moved_untouched, moved_charged)
    assert float(w_arm.sum()) < float(w_plain.sum())

    ledger = arm.last_delay_extra
    assert ledger["mode"] == "signed" and ledger["scale"] == 1.0
    assert ledger["c_buy_extra_max"] == pytest.approx(8.8e-4)
    assert ledger["c_buy_extra_mean"] == pytest.approx(0.5 * 8.8e-4)
    assert ledger["c_sell_extra_max"] == 0.0  # signed mode: positive alpha => buys only
    # F2 tripwire: the coefficient the arm actually charges must be >= 1 bp here
    assert ledger["c_buy_extra_mean"] > 1e-4


def test_scale_zero_reproduces_the_baseline_decision():
    """The documented off switch collapses the arm onto plain mpo exactly."""
    n = 6
    alpha_s1 = np.full(n, 9e-4)
    later = np.full(n, 25e-4)
    ctx = _make_ctx(n)
    plain = _make_strategy(_FixedAlphaModel(alpha_s1, later), n)
    off = _make_strategy(
        _FixedAlphaModel(alpha_s1, later), n, DelayCostConfig(scale=0.0)
    )
    np.testing.assert_array_equal(plain.decide(ctx), off.decide(ctx))
    assert off.name == "mpo_delayaware"  # name follows the wiring, not the value


def test_exact_mode_asks_the_qp_for_the_negative_allowance():
    """mode='exact' must set ``allow_negative_extra`` or the build would raise."""
    n = 6
    alpha_s1 = np.linspace(-10e-4, 10e-4, n)
    later = np.full(n, 25e-4)
    arm = _make_strategy(
        _FixedAlphaModel(alpha_s1, later), n, DelayCostConfig(mode="exact")
    )
    grid = build_stage_grid(MINUTES_PER_DAY + EPOCH_OFFSETS[0], 10, arm.cfg)
    kwargs = arm._qp_build_kwargs(
        arm.ts_model.stage_alphas(np.zeros((n, 3)), grid), _make_ctx(n)
    )
    assert kwargs["allow_negative_extra"] is True
    assert float(np.min(kwargs["c_buy_extra"])) < 0.0
    # and the full decision path really solves (no fail-soft from the rebate)
    w = arm.decide(_make_ctx(n))
    assert w is not None and arm.fail_soft_count == 0

    signed = _make_strategy(_FixedAlphaModel(alpha_s1, later), n, DelayCostConfig())
    assert "allow_negative_extra" not in signed._qp_build_kwargs(
        arm.ts_model.stage_alphas(np.zeros((n, 3)), grid), _make_ctx(n)
    )


def test_flat_mode_charges_every_name_the_measured_rate():
    """Design section 2(b) fallback: a flat trade-weighted rate, both sides."""
    n = 5
    arm = _make_strategy(
        _FixedAlphaModel(np.zeros(n), np.full(n, 25e-4)),
        n,
        DelayCostConfig(mode="flat", flat_rate=13.2e-4),
    )
    grid = build_stage_grid(MINUTES_PER_DAY + EPOCH_OFFSETS[0], 10, arm.cfg)
    kwargs = arm._qp_build_kwargs(
        arm.ts_model.stage_alphas(np.zeros((n, 3)), grid), _make_ctx(n)
    )
    np.testing.assert_allclose(kwargs["c_buy_extra"], 13.2e-4)
    np.testing.assert_allclose(kwargs["c_sell_extra"], 13.2e-4)


def test_other_qp_strategies_never_pass_extras():
    """Only MpoStrategy overrides the hook; the rest of the zoo is untouched."""
    from openpo.mpo import strategies as strategies_mod

    for cls in (
        strategies_mod.SingleSignalStrategy,
        strategies_mod.FixedWeightStrategy,
        strategies_mod.InformedTradingStrategy,
        strategies_mod.OracleMpoStrategy,
    ):
        assert cls._qp_build_kwargs is strategies_mod._QpStrategy._qp_build_kwargs
    assert MpoStrategy._qp_build_kwargs is not strategies_mod._QpStrategy._qp_build_kwargs


# ---------------------------------------------------------------------------
# 7. Roster wiring
# ---------------------------------------------------------------------------


def test_mpo_delayaware_is_opt_in_not_default():
    assert "mpo_delayaware" not in backtest.DEFAULT_STRATEGIES
    assert "mpo_delayaware" in backtest.OPT_IN_STRATEGIES
    assert "mpo_delayaware" in backtest.KNOWN_STRATEGIES
    assert backtest.KNOWN_STRATEGIES == (
        backtest.DEFAULT_STRATEGIES + backtest.OPT_IN_STRATEGIES
    )
    assert len(set(backtest.KNOWN_STRATEGIES)) == len(backtest.KNOWN_STRATEGIES)


def _pool_fixture(seed: int = 11):
    """``(pool, strategies_mod)`` from a real synth panel + real calibration."""
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
    return pool, strategies_mod, base


def test_build_strategy_mpo_delayaware_shares_everything_but_the_surcharge():
    pool, strategies_mod, base = _pool_fixture()
    lam = backtest.default_fixed_lambdas(SynthConfig())

    plain = backtest._build_strategy("mpo", strategies_mod, pool, lam)
    arm = backtest._build_strategy("mpo_delayaware", strategies_mod, pool, lam)

    assert isinstance(arm, strategies_mod.MpoStrategy)
    assert arm.name == "mpo_delayaware" and plain.name == "mpo"
    # SAME fitted calibration and SAME risk/liquidity inputs => paired A/B.
    assert arm.ts_model is base and plain.ts_model is base
    assert np.array_equal(arm.Sigma_d, plain.Sigma_d)
    assert np.array_equal(arm.adv_frac, plain.adv_frac)
    assert arm.n_days == plain.n_days and arm.cfg is plain.cfg
    # ... the surcharge being the only difference, at the committed defaults
    assert plain.delay_cost_cfg is None
    assert arm.delay_cost_cfg == DelayCostConfig(mode="signed", scale=1.0)
    assert arm.delay_cost_cfg.cap == 5e-3 and arm.delay_cost_cfg.flat_rate is None


def test_mpo_delayaware_does_not_disturb_the_shrinkage_memo():
    """The two opt-in arms are independent: neither pays for the other's fit."""
    import importlib

    pool, strategies_mod, _ = _pool_fixture()
    lam = backtest.default_fixed_lambdas(SynthConfig())
    shrinkage_mod = importlib.import_module("openpo.mpo.shrinkage")
    calls = []
    real = shrinkage_mod.shrink_term_structure
    try:
        shrinkage_mod.shrink_term_structure = lambda *a, **k: (
            calls.append(1) or real(*a, **k)
        )
        backtest._build_strategy("mpo_delayaware", strategies_mod, pool, lam)
    finally:
        shrinkage_mod.shrink_term_structure = real
    assert calls == []
    assert backtest._SHRUNK_MODEL_KEY not in pool


def _micro_ecfg() -> ExperimentConfig:
    scfg = SynthConfig(n_assets=60, n_days=16)
    mcfg = dataclasses.replace(MpoConfig(), calib_days=8)
    return ExperimentConfig(synth=scfg, mpo=mcfg, n_seeds=1, eval_start_day=8)


def test_run_seed_micro_mpo_vs_mpo_delayaware():
    """End-to-end micro A/B: the arm must trade strictly less, same market.

    Design section 4 expectation 1 (turnover down) is the one directional call
    that must already hold on a single micro seed; the Sharpe verdict needs the
    preregistered 24-seed run and is deliberately NOT asserted here.
    """
    for mod in _SIBLINGS:
        pytest.importorskip(mod)

    out = backtest.run_seed(3, _micro_ecfg(), strategies=["mpo", "mpo_delayaware"])
    assert set(out) == {"mpo", "mpo_delayaware"}

    plain, arm = out["mpo"], out["mpo_delayaware"]
    assert arm["strategy"] == "mpo_delayaware" and arm["seed"] == 3
    assert arm["t1_violations"] == 0
    assert arm["n_eval_days"] == plain["n_eval_days"]
    for key in (
        "net_sharpe_annual", "net_return_annual", "avg_daily_turnover",
        "max_drawdown", "final_nav",
    ):
        assert np.isfinite(arm[key]), f"mpo_delayaware.{key} not finite"

    assert arm["avg_daily_turnover"] < plain["avg_daily_turnover"], (
        arm["avg_daily_turnover"], plain["avg_daily_turnover"]
    )
    assert arm["final_nav"] != plain["final_nav"]
    # F5: the coefficients must not make the QP harder to solve.
    assert arm["qp_status_counts"].get("optimal", 0) >= plain["qp_status_counts"].get(
        "optimal", 0
    )
