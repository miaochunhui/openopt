"""Tests for openpo.mpo.delay_cost (Iteration 42 leg 1).

Design source: docs/mpo-t1-delaycost-design.md. Leg 1 is un-wired, so every
assertion here is on the coefficient math itself:

* sign asymmetry (positive alpha penalizes BUYS, negative alpha penalizes SELLS)
* zero alpha  =>  exactly zero surcharge in every mode
* magnitude sanity against hand-computed cases, including the design doc's
  2-sigma sanity number (IC 0.05, daily vol 2.5%  =>  8.84 bp over 30 minutes)
  and the measured Iteration 41 anchor (13.2 bp per unit traded)
* purity: the coefficients are functions of the ALPHA ONLY -- no decision
  variables, no portfolio state -- which is what keeps the leg-2 QP convex
* the wash-pair invariant that keeps that QP bounded
* loud validation on every degenerate input
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import numpy as np
import pytest

from openpo.mpo.config import MpoConfig
from openpo.mpo.delay_cost import (
    DELAY_COST_MODES,
    DelayCostConfig,
    delay_cost_coefficients,
    delay_cost_telemetry,
    effective_trade_costs,
)

BP = 1e-4


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestDelayCostConfig:
    def test_defaults_are_the_committed_primary_remedy(self):
        cfg = DelayCostConfig()
        assert cfg.mode == "signed"  # design doc section 2(a), clipped variant
        assert cfg.scale == 1.0
        assert cfg.cap == 5e-3  # 50 bp breaker
        assert cfg.flat_rate is None
        assert cfg.fill_window_scale == 1.0

    def test_is_frozen(self):
        cfg = DelayCostConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.scale = 2.0  # type: ignore[misc]

    def test_is_not_mpoconfig(self):
        # The whole point of a separate dataclass: MpoConfig stays frozen by SPEC.
        assert not isinstance(DelayCostConfig(), MpoConfig)
        assert not hasattr(DelayCostConfig(), "buy_cost")

    @pytest.mark.parametrize("mode", DELAY_COST_MODES)
    def test_all_documented_modes_construct(self, mode):
        assert DelayCostConfig(mode=mode).mode == mode

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            DelayCostConfig(mode="signd")

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be >= 0"):
            DelayCostConfig(scale=-0.1)

    def test_zero_scale_is_legal_it_is_the_off_switch(self):
        assert DelayCostConfig(scale=0.0).scale == 0.0

    @pytest.mark.parametrize("cap", [0.0, -1e-4])
    def test_nonpositive_cap_raises(self, cap):
        with pytest.raises(ValueError, match="cap must be > 0"):
            DelayCostConfig(cap=cap)

    @pytest.mark.parametrize("val", [0.0, -1.0])
    def test_nonpositive_fill_window_scale_raises(self, val):
        with pytest.raises(ValueError, match="fill_window_scale must be > 0"):
            DelayCostConfig(fill_window_scale=val)

    @pytest.mark.parametrize("field,val", [("scale", math.nan), ("cap", math.inf),
                                          ("fill_window_scale", math.nan)])
    def test_non_finite_knobs_raise(self, field, val):
        with pytest.raises(ValueError, match="must be finite"):
            DelayCostConfig(**{field: val})

    def test_flat_rate_outside_flat_mode_raises(self):
        # typo guard: a flat_rate that silently does nothing is a trap
        with pytest.raises(ValueError, match="only meaningful for mode='flat'"):
            DelayCostConfig(flat_rate=13.2 * BP)

    def test_negative_flat_rate_raises(self):
        with pytest.raises(ValueError, match="flat_rate must be >= 0"):
            DelayCostConfig(mode="flat", flat_rate=-BP)

    def test_flat_rate_non_finite_raises(self):
        with pytest.raises(ValueError, match="flat_rate must be finite"):
            DelayCostConfig(mode="flat", flat_rate=math.inf)

    def test_bool_knob_raises(self):
        with pytest.raises(TypeError, match="must be a real number"):
            DelayCostConfig(scale=True)


# ---------------------------------------------------------------------------
# sign asymmetry -- the core semantics
# ---------------------------------------------------------------------------

class TestSignAsymmetry:
    def test_positive_alpha_penalizes_buys_only(self):
        cb, cs = delay_cost_coefficients(np.array([20 * BP]))
        assert cb[0] == pytest.approx(20 * BP)
        assert cs[0] == 0.0

    def test_negative_alpha_penalizes_sells_only(self):
        cb, cs = delay_cost_coefficients(np.array([-20 * BP]))
        assert cb[0] == 0.0
        assert cs[0] == pytest.approx(20 * BP)

    def test_mixed_cross_section(self):
        alpha = np.array([15 * BP, -15 * BP, 0.0, 3 * BP, -40 * BP])
        cb, cs = delay_cost_coefficients(alpha)
        np.testing.assert_allclose(cb, [15 * BP, 0.0, 0.0, 3 * BP, 0.0])
        np.testing.assert_allclose(cs, [0.0, 15 * BP, 0.0, 0.0, 40 * BP])

    def test_exactly_one_side_is_charged_per_name(self):
        rng = np.random.default_rng(20260823)
        alpha = rng.normal(scale=5 * BP, size=300)
        cb, cs = delay_cost_coefficients(alpha)
        # signed mode: the two sides never both bite on the same name
        assert np.all(cb * cs == 0.0)
        # and the charged side always matches the alpha sign
        assert np.all(cb[alpha > 0] > 0.0)
        assert np.all(cs[alpha < 0] > 0.0)
        assert np.all(cb[alpha < 0] == 0.0)
        assert np.all(cs[alpha > 0] == 0.0)

    def test_signed_mode_is_never_a_rebate(self):
        rng = np.random.default_rng(7)
        alpha = rng.normal(scale=20 * BP, size=500)
        cb, cs = delay_cost_coefficients(alpha, DelayCostConfig(mode="signed"))
        assert cb.min() >= 0.0 and cs.min() >= 0.0

    def test_sum_of_the_two_sides_is_abs_alpha(self):
        alpha = np.array([7 * BP, -7 * BP, 0.0])
        cb, cs = delay_cost_coefficients(alpha)
        np.testing.assert_allclose(cb + cs, np.abs(alpha))


# ---------------------------------------------------------------------------
# zero alpha
# ---------------------------------------------------------------------------

class TestZeroAlpha:
    @pytest.mark.parametrize("mode", DELAY_COST_MODES)
    def test_zero_alpha_gives_zero_surcharge(self, mode):
        cb, cs = delay_cost_coefficients(np.zeros(8), DelayCostConfig(mode=mode))
        # exact zeros, not "small": a halted / neutral name must cost nothing extra
        assert np.all(cb == 0.0) and np.all(cs == 0.0)

    @pytest.mark.parametrize("mode", DELAY_COST_MODES)
    def test_scale_zero_switches_the_whole_layer_off(self, mode):
        alpha = np.array([30 * BP, -30 * BP, 5 * BP])
        kw = {"flat_rate": 13.2 * BP} if mode == "flat" else {}
        cfg = DelayCostConfig(mode=mode, scale=0.0, **kw)
        cb, cs = delay_cost_coefficients(alpha, cfg)
        assert np.all(cb == 0.0) and np.all(cs == 0.0)

    def test_zero_alpha_effective_cost_equals_base_cost(self):
        mpo = MpoConfig()
        c_buy, c_sell = effective_trade_costs(np.zeros(4), mpo)
        np.testing.assert_allclose(c_buy, mpo.buy_cost())
        np.testing.assert_allclose(c_sell, mpo.sell_cost())


# ---------------------------------------------------------------------------
# magnitude sanity
# ---------------------------------------------------------------------------

class TestMagnitude:
    def test_hand_computed_fifteen_bp_case(self):
        """Design doc example: buying a +15 bp name costs 15 bp of delay."""
        cb, cs = delay_cost_coefficients(np.array([15 * BP, -15 * BP]))
        assert cb[0] * 1e4 == pytest.approx(15.0)
        assert cs[1] * 1e4 == pytest.approx(15.0)

    def test_two_sigma_sanity_number_from_the_design_doc(self):
        """IC 0.05, daily vol 2.5%, 30/240 minutes, z = 2  =>  8.84 bp.

        This is the design doc's section-3 order-of-magnitude check: the
        conditional 30-minute move of a 2-sigma name is ~9 bp, i.e. the right
        order to explain the 13.2 bp measured per unit traded.
        """
        ic, daily_vol, z = 0.05, 0.025, 2.0
        sigma_30 = daily_vol * math.sqrt(30.0 / 240.0)
        assert sigma_30 * 1e4 == pytest.approx(88.39, abs=0.02)
        alpha_2sigma = ic * sigma_30 * z
        assert alpha_2sigma * 1e4 == pytest.approx(8.84, abs=0.02)
        # ... and the coefficient reproduces exactly that number as a buy cost
        cb, cs = delay_cost_coefficients(np.array([alpha_2sigma]))
        assert cb[0] * 1e4 == pytest.approx(8.84, abs=0.02)
        assert cs[0] == 0.0
        # in the 10-20 bp band the measurement lives in, once the QP concentrates
        # its trades ~2.4 sd out in the alpha cross-section (design doc section 3)
        assert 5 * BP < cb[0] < 20 * BP

    def test_measured_iter41_anchor_dominates_the_explicit_buy_cost(self):
        """13.2 bp of delay vs the 7 bp explicit buy cost the QP models."""
        mpo = MpoConfig()
        assert mpo.buy_cost() * 1e4 == pytest.approx(7.0)
        assert mpo.sell_cost() * 1e4 == pytest.approx(12.0)
        c_buy, _ = effective_trade_costs(np.array([13.2 * BP]), mpo)
        assert c_buy[0] * 1e4 == pytest.approx(20.2)  # true hurdle ~20 bp, not 7
        assert c_buy[0] > 2 * mpo.buy_cost()

    def test_effective_cost_is_base_plus_extra(self):
        mpo = MpoConfig()
        alpha = np.array([15 * BP, -15 * BP, 0.0])
        cb_x, cs_x = delay_cost_coefficients(alpha)
        c_buy, c_sell = effective_trade_costs(alpha, mpo)
        np.testing.assert_allclose(c_buy, mpo.buy_cost() + cb_x)
        np.testing.assert_allclose(c_sell, mpo.sell_cost() + cs_x)

    def test_scale_is_linear_and_the_half_retreat_rung_halves_it(self):
        alpha = np.array([13.2 * BP, -13.2 * BP])
        full_b, full_s = delay_cost_coefficients(alpha)
        half_b, half_s = delay_cost_coefficients(alpha, DelayCostConfig(scale=0.5))
        np.testing.assert_allclose(half_b, 0.5 * full_b)
        np.testing.assert_allclose(half_s, 0.5 * full_s)
        assert half_b[0] * 1e4 == pytest.approx(6.6)

    def test_fill_window_scale_doubles_the_charge_for_a_double_length_bar(self):
        """Iter 41 arms C/D: the delay tracks the BAR length, not the stage."""
        alpha = np.array([10 * BP])
        cb1, _ = delay_cost_coefficients(alpha)
        cb2, _ = delay_cost_coefficients(alpha, DelayCostConfig(fill_window_scale=2.0))
        assert cb1[0] * 1e4 == pytest.approx(10.0)
        assert cb2[0] * 1e4 == pytest.approx(20.0)

    def test_cap_binds_a_blown_up_alpha(self):
        cfg = DelayCostConfig(cap=50 * BP)
        cb, cs = delay_cost_coefficients(np.array([500 * BP, -500 * BP]), cfg)
        assert cb[0] == pytest.approx(50 * BP)
        assert cs[1] == pytest.approx(50 * BP)

    def test_cap_does_not_touch_realistic_magnitudes(self):
        rng = np.random.default_rng(11)
        alpha = rng.normal(scale=5.5 * BP, size=300)  # measured sd of alpha_S1
        cb, cs = delay_cost_coefficients(alpha)
        assert cb.max() < 5e-3 and cs.max() < 5e-3

    def test_monotone_in_alpha_magnitude(self):
        mags = np.array([1 * BP, 5 * BP, 13.2 * BP, 40 * BP])
        cb, _ = delay_cost_coefficients(mags)
        assert np.all(np.diff(cb) > 0.0)


# ---------------------------------------------------------------------------
# exact mode + the boundedness invariant
# ---------------------------------------------------------------------------

class TestExactMode:
    def test_exact_mode_zeroes_the_net_s1_credit(self):
        """Algebraically: the two sides sum to zero, name by name."""
        rng = np.random.default_rng(3)
        alpha = rng.normal(scale=6 * BP, size=200)
        cb, cs = delay_cost_coefficients(alpha, DelayCostConfig(mode="exact"))
        np.testing.assert_allclose(cb + cs, 0.0, atol=0.0)
        np.testing.assert_allclose(cb, alpha)

    def test_exact_mode_allows_a_rebate_signed_mode_does_not(self):
        alpha = np.array([-20 * BP])
        cb_exact, _ = delay_cost_coefficients(alpha, DelayCostConfig(mode="exact"))
        cb_signed, _ = delay_cost_coefficients(alpha, DelayCostConfig(mode="signed"))
        assert cb_exact[0] < 0.0  # the rebate the committed default clips away
        assert cb_signed[0] == 0.0

    def test_exact_mode_cap_is_symmetric(self):
        cfg = DelayCostConfig(mode="exact", cap=50 * BP)
        cb, cs = delay_cost_coefficients(np.array([-500 * BP]), cfg)
        assert cb[0] == pytest.approx(-50 * BP)
        assert cs[0] == pytest.approx(50 * BP)

    @pytest.mark.parametrize("mode", ["signed", "exact"])
    def test_wash_pair_stays_penalized_so_the_qp_stays_bounded(self, mode):
        """b_i and s_i both -> inf at fixed w costs (c_buy+dx)+(c_sell+ds) > 0.

        This is the module docstring's boundedness argument: the only unbounded
        direction of a linear-cost QP is the wash pair, and its coefficient must
        stay strictly positive for every name in every mode.
        """
        mpo = MpoConfig()
        rng = np.random.default_rng(99)
        alpha = rng.normal(scale=30 * BP, size=400)
        c_buy, c_sell = effective_trade_costs(alpha, mpo, DelayCostConfig(mode=mode))
        wash = c_buy + c_sell
        assert np.all(wash > 0.0)
        if mode == "exact":
            # exactly the base round trip: the two extras cancel
            np.testing.assert_allclose(wash, mpo.buy_cost() + mpo.sell_cost())
        else:
            assert np.all(wash >= mpo.buy_cost() + mpo.sell_cost() - 1e-18)

    def test_exact_mode_can_drive_an_effective_cost_negative(self):
        mpo = MpoConfig()
        alpha = np.array([-30 * BP])  # |alpha| > buy_cost = 7 bp
        c_buy, _ = effective_trade_costs(alpha, mpo, DelayCostConfig(mode="exact"))
        assert c_buy[0] < 0.0
        tel = delay_cost_telemetry(alpha, DelayCostConfig(mode="exact"), mpo_cfg=mpo)
        assert tel["n_negative_effective"] == 1


# ---------------------------------------------------------------------------
# flat mode (fallback remedy (b))
# ---------------------------------------------------------------------------

class TestFlatMode:
    def test_flat_mode_charges_both_sides_equally(self):
        cfg = DelayCostConfig(mode="flat", flat_rate=13.2 * BP)
        cb, cs = delay_cost_coefficients(np.array([15 * BP, -15 * BP, 0.0]), cfg)
        np.testing.assert_allclose(cb, 13.2 * BP)
        np.testing.assert_allclose(cs, 13.2 * BP)

    def test_flat_mode_drops_the_per_name_alpha_dependence(self):
        cfg = DelayCostConfig(mode="flat", flat_rate=10 * BP)
        a = np.array([50 * BP, -1 * BP])
        b = np.array([-1 * BP, 50 * BP])
        np.testing.assert_allclose(
            delay_cost_coefficients(a, cfg)[0], delay_cost_coefficients(b, cfg)[0]
        )

    def test_derived_rate_is_the_name_weighted_mean_abs_alpha(self):
        alpha = np.array([10 * BP, -20 * BP, 0.0, 30 * BP])
        cb, cs = delay_cost_coefficients(alpha, DelayCostConfig(mode="flat"))
        expected = np.mean(np.abs(alpha))
        np.testing.assert_allclose(cb, expected)
        np.testing.assert_allclose(cs, expected)

    def test_derived_rate_is_a_lower_bound_vs_the_trade_weighted_measurement(self):
        """Design doc section 2(b) caliper trap: name-weighted ~3x too small.

        A realistic alpha cross-section gives mean|alpha| ~ 4.4 bp, while the
        realized TRADE-weighted delay is 13.2 bp -- because the QP only trades
        the tail. So the auto-derived flat rate under-corrects and leg 2 must
        pass flat_rate explicitly.
        """
        rng = np.random.default_rng(20260821)
        alpha = rng.normal(scale=5.53 * BP, size=3000)  # sd implied by 4.41 bp mean|.|
        derived = delay_cost_coefficients(alpha, DelayCostConfig(mode="flat"))[0][0]
        assert derived * 1e4 == pytest.approx(4.4, abs=0.3)
        assert derived < 0.5 * 13.2 * BP

    def test_explicit_flat_rate_ignores_fill_window_scale(self):
        cfg = DelayCostConfig(mode="flat", flat_rate=13.2 * BP, fill_window_scale=2.0)
        cb, _ = delay_cost_coefficients(np.array([10 * BP]), cfg)
        assert cb[0] == pytest.approx(13.2 * BP)  # a measured rate, not an alpha

    def test_derived_flat_rate_does_scale_with_fill_window(self):
        cfg = DelayCostConfig(mode="flat", fill_window_scale=2.0)
        cb, _ = delay_cost_coefficients(np.array([10 * BP, -10 * BP]), cfg)
        assert cb[0] == pytest.approx(20 * BP)

    def test_flat_mode_never_rebates(self):
        cfg = DelayCostConfig(mode="flat", flat_rate=13.2 * BP)
        cb, cs = delay_cost_coefficients(np.array([-99 * BP]), cfg)
        assert cb[0] > 0.0 and cs[0] > 0.0


# ---------------------------------------------------------------------------
# purity: coefficients are decision-variable independent
# ---------------------------------------------------------------------------

class TestPurity:
    def test_signature_admits_no_decision_variables(self):
        """Mechanical guard: no w / b / s / portfolio argument may ever appear.

        If a future edit lets the coefficient depend on the decision vector, the
        leg-2 QP objective stops being a linear cost and the convexity argument
        of the design doc (risk 2) collapses. Freeze the parameter set here.
        """
        params = list(inspect.signature(delay_cost_coefficients).parameters)
        assert params == ["alpha_s1", "cfg", "mode"]

    def test_deterministic_across_repeated_calls(self):
        rng = np.random.default_rng(5)
        alpha = rng.normal(scale=8 * BP, size=120)
        first = delay_cost_coefficients(alpha)
        for _ in range(3):
            again = delay_cost_coefficients(alpha)
            np.testing.assert_array_equal(again[0], first[0])
            np.testing.assert_array_equal(again[1], first[1])

    def test_input_is_not_mutated(self):
        alpha = np.array([15 * BP, -15 * BP, 0.0])
        before = alpha.copy()
        delay_cost_coefficients(alpha, DelayCostConfig(mode="exact"))
        np.testing.assert_array_equal(alpha, before)

    def test_outputs_do_not_alias_the_input_or_each_other(self):
        alpha = np.array([15 * BP, -15 * BP])
        for mode in DELAY_COST_MODES:
            cb, cs = delay_cost_coefficients(alpha, DelayCostConfig(mode=mode))
            assert cb is not alpha and cs is not alpha
            assert not np.shares_memory(cb, alpha)
            assert not np.shares_memory(cs, alpha)
            assert not np.shares_memory(cb, cs)
            cb[0] = 123.0  # writing an output must not disturb the other
            assert cs[0] != 123.0

    def test_independent_of_the_portfolio_state_around_it(self):
        """Same alpha, wildly different holdings/trades => identical coefficients."""
        alpha = np.array([12 * BP, -4 * BP, 0.0])
        base = delay_cost_coefficients(alpha)
        rng = np.random.default_rng(17)
        for _ in range(5):
            # simulate arbitrary decision-variable context existing in the caller
            _w0 = rng.uniform(0.0, 0.02, size=3)
            _b = rng.uniform(0.0, 0.01, size=3)
            _s = rng.uniform(0.0, 0.01, size=3)
            got = delay_cost_coefficients(alpha)
            np.testing.assert_array_equal(got[0], base[0])
            np.testing.assert_array_equal(got[1], base[1])

    def test_positively_homogeneous_in_alpha(self):
        alpha = np.array([9 * BP, -3 * BP, 0.0])
        cb1, cs1 = delay_cost_coefficients(alpha, DelayCostConfig(cap=1.0))
        cb2, cs2 = delay_cost_coefficients(3.0 * alpha, DelayCostConfig(cap=1.0))
        np.testing.assert_allclose(cb2, 3.0 * cb1)
        np.testing.assert_allclose(cs2, 3.0 * cs1)

    def test_permutation_equivariant(self):
        alpha = np.array([1 * BP, -7 * BP, 22 * BP, 0.0])
        perm = np.array([2, 0, 3, 1])
        cb, cs = delay_cost_coefficients(alpha)
        cb_p, cs_p = delay_cost_coefficients(alpha[perm])
        np.testing.assert_allclose(cb_p, cb[perm])
        np.testing.assert_allclose(cs_p, cs[perm])

    def test_returns_float64_of_the_right_shape_from_a_list(self):
        cb, cs = delay_cost_coefficients([15e-4, -15e-4])
        assert cb.dtype == np.float64 and cs.dtype == np.float64
        assert cb.shape == (2,) and cs.shape == (2,)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_two_d_alpha_raises(self):
        with pytest.raises(ValueError, match=r"alpha_s1 must be 1-D"):
            delay_cost_coefficients(np.zeros((4, 3)))

    def test_empty_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha_s1 is empty"):
            delay_cost_coefficients(np.zeros(0))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_alpha_raises_with_the_offending_index(self, bad):
        alpha = np.array([1 * BP, bad, 2 * BP])
        with pytest.raises(ValueError, match="first at index 1"):
            delay_cost_coefficients(alpha)

    def test_mpoconfig_as_cfg_raises_a_helpful_typeerror(self):
        with pytest.raises(TypeError, match="looks like an MpoConfig"):
            delay_cost_coefficients(np.zeros(3), MpoConfig())

    def test_random_object_as_cfg_raises(self):
        with pytest.raises(TypeError, match="must be a DelayCostConfig or None"):
            delay_cost_coefficients(np.zeros(3), cfg=0.5)

    def test_none_cfg_uses_the_committed_defaults(self):
        a = np.array([15 * BP])
        np.testing.assert_array_equal(
            delay_cost_coefficients(a, None)[0],
            delay_cost_coefficients(a, DelayCostConfig())[0],
        )

    def test_mode_override_beats_the_config(self):
        cfg = DelayCostConfig(mode="signed")
        cb, _ = delay_cost_coefficients(np.array([-20 * BP]), cfg, mode="exact")
        assert cb[0] == pytest.approx(-20 * BP)

    def test_bad_mode_override_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            delay_cost_coefficients(np.zeros(3), mode="nope")

    def test_overriding_away_from_flat_refuses_to_drop_flat_rate(self):
        cfg = DelayCostConfig(mode="flat", flat_rate=13.2 * BP)
        with pytest.raises(ValueError, match="refusing to silently ignore"):
            delay_cost_coefficients(np.zeros(3), cfg, mode="signed")

    def test_effective_trade_costs_rejects_swapped_arguments(self):
        with pytest.raises(TypeError, match="arguments look swapped"):
            effective_trade_costs(np.zeros(3), DelayCostConfig())

    def test_effective_trade_costs_rejects_a_costless_config(self):
        with pytest.raises(TypeError, match="buy_cost"):
            effective_trade_costs(np.zeros(3), object())

    def test_effective_trade_costs_rejects_non_finite_base_costs(self):
        class Bad:
            def buy_cost(self):
                return math.nan

            def sell_cost(self):
                return 12 * BP

        with pytest.raises(ValueError, match="base costs must be finite"):
            effective_trade_costs(np.zeros(3), Bad())

    def test_cost_mult_flows_through_the_base_cost(self):
        mpo = MpoConfig(cost_mult=2.0)
        c_buy, c_sell = effective_trade_costs(np.array([0.0]), mpo)
        assert c_buy[0] * 1e4 == pytest.approx(14.0)
        assert c_sell[0] * 1e4 == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_reports_the_resolved_knobs_and_magnitudes(self):
        alpha = np.array([15 * BP, -15 * BP, 0.0, 5 * BP])
        tel = delay_cost_telemetry(alpha)
        assert tel["mode"] == "signed"
        assert tel["scale"] == 1.0
        assert tel["cap"] == 5e-3
        assert tel["fill_window_scale"] == 1.0
        assert tel["flat_rate"] is None
        assert tel["n_assets"] == 4
        assert tel["alpha_abs_mean"] == pytest.approx(np.mean(np.abs(alpha)))
        assert tel["alpha_abs_max"] == pytest.approx(15 * BP)
        assert tel["c_buy_extra_max"] == pytest.approx(15 * BP)
        assert tel["c_sell_extra_max"] == pytest.approx(15 * BP)
        assert tel["n_capped"] == 0

    def test_f2_falsification_hook_is_visible(self):
        """Design doc F2 keys off c_buy_extra_mean < 1 bp meaning 'wiring bug'."""
        rng = np.random.default_rng(20260823)
        alpha = rng.normal(scale=5.53 * BP, size=300)
        tel = delay_cost_telemetry(alpha)
        assert tel["c_buy_extra_mean"] * 1e4 > 1.0  # healthy wiring
        dead = delay_cost_telemetry(alpha, DelayCostConfig(scale=0.0))
        assert dead["c_buy_extra_mean"] == 0.0  # what a dead wiring looks like

    def test_counts_capped_entries(self):
        cfg = DelayCostConfig(cap=10 * BP)
        tel = delay_cost_telemetry(np.array([50 * BP, -50 * BP, 1 * BP]), cfg)
        assert tel["n_capped"] == 2

    def test_effective_cost_block_only_with_mpo_cfg(self):
        alpha = np.array([13.2 * BP])
        assert "c_buy_mean" not in delay_cost_telemetry(alpha)
        tel = delay_cost_telemetry(alpha, mpo_cfg=MpoConfig())
        assert tel["base_buy_cost"] * 1e4 == pytest.approx(7.0)
        assert tel["base_sell_cost"] * 1e4 == pytest.approx(12.0)
        assert tel["c_buy_mean"] * 1e4 == pytest.approx(20.2)
        assert tel["c_sell_mean"] * 1e4 == pytest.approx(12.0)
        assert tel["n_negative_effective"] == 0

    def test_is_json_safe(self):
        import json

        blob = json.dumps(delay_cost_telemetry(np.array([15 * BP]), mpo_cfg=MpoConfig()))
        assert "c_buy_extra_mean" in blob

    def test_mode_override_is_reflected(self):
        tel = delay_cost_telemetry(np.array([-20 * BP]), mode="exact")
        assert tel["mode"] == "exact"
        assert tel["c_buy_extra_mean"] == pytest.approx(-20 * BP)
