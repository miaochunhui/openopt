"""Tests for openpo.mpo.rollforward (Iteration 44 leg 1).

Design source: docs/mpo-t1-rollforward-design.md, whose verdict is **NO-GO
BEFORE IMPLEMENTATION**. This file is the machine check on that verdict, so
beyond the usual math/validation tests it carries three load-bearing ones:

* ``TestCommittedVerdict`` re-derives the design doc's section 3 table from
  :data:`COMMITTED_ANCHORS` and asserts the verdict is ``'negligible'`` in both
  cells. If a future config change (costs, IC, vol, no-trade band) invalidates
  the NO-GO, this test goes red instead of the doc going quietly stale.
* ``TestDiscriminatingPower`` scales the very same measured drift up by the
  design doc section 0 oracle leverage (1/IC = 16.7x, i.e. realized stage
  returns instead of conditional means -- exactly the Axioma 051 conclusion's
  "drastically improved" experiment) and asserts the verdict flips to
  ``'material'``. Without it, "negligible" could be a constant-return stub.
* ``TestIsolation`` asserts nothing in the production import graph touches this
  module: the NO-GO means no wiring, and that is checkable.
"""

from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest

from openpo.mpo import rollforward as rf
from openpo.mpo.config import MpoConfig
from openpo.mpo.rollforward import (
    COMMITTED_ANCHORS,
    DEFAULT_DETECT_FLOOR_SHARPE,
    DEFAULT_EPOCHS_PER_YEAR,
    DEFAULT_NO_TRADE_BAND,
    ROLLFORWARD_CANDIDATES,
    RollForwardBound,
    cumulative_growth,
    rollforward_drift,
    rollforward_magnitude_bound,
    sharpe_bound_from_l1,
)

# Design doc section 0: the feasible roll-forward estimate is a conditional mean
# (IC * sigma); the oracle one is the realized return (~sigma). Measured ratio
# E|r| / E|alpha_S1| = 7.05e-3 / 4.22e-4 = 16.7.
ORACLE_LEVERAGE = 16.7


def _paths_with_drift(delta: np.ndarray, w_level: float = 0.02):
    """Build ``(alpha, w_path, w0)`` whose :func:`rollforward_drift` is ``delta``.

    Every entering weight is ``w_level`` (a name at ``cfg.w_max``), so
    ``alpha = delta / w_level`` inverts the drift definition exactly. Lets the
    anchor tests work from a recorded ``delta`` without re-running a backtest.
    """
    delta = np.asarray(delta, dtype=np.float64)
    j, n = delta.shape
    w_path = np.full((j, n), float(w_level))
    w0 = np.full(n, float(w_level))
    return delta / float(w_level), w_path, w0


def _delta_matching(l1: float, dmax: float, n_stages: int = 4, n_assets: int = 8):
    """A ``[J, N]`` drift with exactly the given ``||.||_1`` and ``max |.|``."""
    if l1 < dmax:
        raise ValueError("l1 must be >= dmax")
    delta = np.zeros((n_stages, n_assets), dtype=np.float64)
    delta[0, 0] = dmax
    rest = l1 - dmax
    spread = delta[0, 1:]
    if rest > 0:
        # Spread the remainder below dmax so max stays put; widen if needed.
        per = rest / spread.size
        assert per <= dmax, "not representable with this many assets"
        delta[0, 1:] = per
    return delta


# ---------------------------------------------------------------------------
# rollforward_drift
# ---------------------------------------------------------------------------

class TestDrift:
    def test_stage_indexing_uses_the_entering_weights(self):
        alpha = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        w_path = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        w0 = np.array([100.0, 200.0])
        delta = rollforward_drift(alpha, w_path, w0)
        # stage 0 drifts the KNOWN holdings; stage j >= 1 the previous stage end.
        expected = np.array([[100.0, 400.0], [30.0, 80.0], [150.0, 240.0]])
        np.testing.assert_allclose(delta, expected)

    def test_single_stage_uses_only_w0(self):
        alpha = np.array([[2.0, -3.0]])
        delta = rollforward_drift(alpha, np.array([[7.0, 7.0]]), np.array([5.0, 5.0]))
        np.testing.assert_allclose(delta, [[10.0, -15.0]])

    def test_sign_follows_alpha(self):
        alpha = np.array([[1e-3, -1e-3, 0.0]])
        delta = rollforward_drift(alpha, np.ones((1, 3)) * 0.01, np.ones(3) * 0.01)
        assert delta[0, 0] > 0  # winner drifts up -> less of it must be bought
        assert delta[0, 1] < 0
        assert delta[0, 2] == 0.0

    def test_zero_alpha_reproduces_the_frozen_dynamics(self):
        """r = 0 is the SPEC's dynamics: the correction must vanish identically."""
        delta = rollforward_drift(np.zeros((4, 5)), np.ones((4, 5)) * 0.02, np.ones(5) * 0.02)
        assert np.count_nonzero(delta) == 0

    def test_zero_holdings_give_zero_drift(self):
        alpha = np.full((2, 3), 5e-4)
        delta = rollforward_drift(alpha, np.zeros((2, 3)), np.zeros(3))
        assert np.count_nonzero(delta) == 0

    def test_does_not_mutate_or_alias_inputs(self):
        alpha = np.full((2, 3), 1e-3)
        w_path = np.full((2, 3), 0.02)
        w0 = np.full(3, 0.02)
        a_ref, w_ref, w0_ref = alpha.copy(), w_path.copy(), w0.copy()
        delta = rollforward_drift(alpha, w_path, w0)
        delta[:] = 123.0
        np.testing.assert_array_equal(alpha, a_ref)
        np.testing.assert_array_equal(w_path, w_ref)
        np.testing.assert_array_equal(w0, w0_ref)

    @pytest.mark.parametrize(
        "alpha, w_path, w0",
        [
            (np.ones(3), np.ones((1, 3)), np.ones(3)),          # alpha 1-D
            (np.ones((2, 3)), np.ones((3, 3)), np.ones(3)),     # stage mismatch
            (np.ones((2, 3)), np.ones((2, 4)), np.ones(3)),     # asset mismatch
            (np.ones((2, 3)), np.ones((2, 3)), np.ones(4)),     # w0 shape
            (np.zeros((0, 3)), np.zeros((0, 3)), np.ones(3)),   # empty
        ],
    )
    def test_shape_errors_are_loud(self, alpha, w_path, w0):
        with pytest.raises(ValueError):
            rollforward_drift(alpha, w_path, w0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_alpha_raises_with_a_locating_message(self, bad):
        alpha = np.ones((2, 3))
        alpha[1, 2] = bad
        with pytest.raises(ValueError, match=r"finite"):
            rollforward_drift(alpha, np.ones((2, 3)), np.ones(3))

    def test_non_finite_w0_raises(self):
        w0 = np.array([0.01, np.nan, 0.01])
        with pytest.raises(ValueError, match="w0 must be finite"):
            rollforward_drift(np.ones((1, 3)), np.ones((1, 3)), w0)


# ---------------------------------------------------------------------------
# cumulative_growth
# ---------------------------------------------------------------------------

class TestCumulativeGrowth:
    def test_hand_computed(self):
        alpha = np.array([[0.1, 0.2], [0.3, 0.4]])
        w_path = np.array([[1.0, 2.0], [3.0, 4.0]])
        w0 = np.array([10.0, 20.0])
        g, G = cumulative_growth(alpha, w_path, w0)
        np.testing.assert_allclose(g, [0.1 * 10 + 0.2 * 20, 0.3 * 1 + 0.4 * 2])
        np.testing.assert_allclose(G, np.cumprod(1.0 + np.array([5.0, 1.1])))

    def test_zero_alpha_means_no_growth(self):
        g, G = cumulative_growth(np.zeros((3, 4)), np.ones((3, 4)), np.ones(4))
        np.testing.assert_array_equal(g, np.zeros(3))
        np.testing.assert_allclose(G, np.ones(3))

    def test_growth_is_the_portfolio_expected_return_not_a_per_name_number(self):
        """g_j sums over names: cash (the unfunded part) contributes exactly 0."""
        alpha = np.full((1, 100), 4e-4)
        w0 = np.zeros(100)
        w0[:50] = 0.002  # gross 0.1, the rest is cash
        g, _ = cumulative_growth(alpha, np.zeros((1, 100)), w0)
        assert g[0] == pytest.approx(4e-4 * 0.1)

    def test_anchor_growth_is_five_orders_below_one(self):
        """Design doc section 2: alpha_growth is a <1e-4 relative rescaling."""
        for cell, anchor in COMMITTED_ANCHORS.items():
            assert 0.0 < anchor["growth_total"] < 1e-3, cell


# ---------------------------------------------------------------------------
# bound (B), scalar core
# ---------------------------------------------------------------------------

class TestScalarBound:
    def test_exact_arithmetic(self):
        obj, ann, sharpe = sharpe_bound_from_l1(
            2e-4, cost_sum=1e-3, ann_vol=2e-2, epochs_per_year=1000
        )
        assert obj == pytest.approx(2e-7)
        assert ann == pytest.approx(2e-4)
        assert sharpe == pytest.approx(1e-2)

    def test_cost_free_world_has_a_zero_bound(self):
        """Bound (B)'s only channel is the trade-cost origin (design doc 1)."""
        obj, ann, sharpe = sharpe_bound_from_l1(1.0, cost_sum=0.0, ann_vol=1e-2)
        assert (obj, ann, sharpe) == (0.0, 0.0, 0.0)

    def test_zero_drift_has_a_zero_bound(self):
        assert sharpe_bound_from_l1(0.0, cost_sum=1e-3, ann_vol=1e-2)[2] == 0.0

    def test_monotone_in_drift_and_cost_inverse_in_vol(self):
        base = sharpe_bound_from_l1(1e-4, cost_sum=1e-3, ann_vol=1e-2)[2]
        assert sharpe_bound_from_l1(2e-4, cost_sum=1e-3, ann_vol=1e-2)[2] > base
        assert sharpe_bound_from_l1(1e-4, cost_sum=2e-3, ann_vol=1e-2)[2] > base
        assert sharpe_bound_from_l1(1e-4, cost_sum=1e-3, ann_vol=2e-2)[2] < base

    def test_scales_linearly_in_the_l1_mass(self):
        a = sharpe_bound_from_l1(3e-5, cost_sum=1.9e-3, ann_vol=1.764e-2)[2]
        b = sharpe_bound_from_l1(3e-4, cost_sum=1.9e-3, ann_vol=1.764e-2)[2]
        assert b == pytest.approx(10.0 * a)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"delta_l1": -1e-6, "cost_sum": 1e-3, "ann_vol": 1e-2},
            {"delta_l1": np.nan, "cost_sum": 1e-3, "ann_vol": 1e-2},
            {"delta_l1": 1e-6, "cost_sum": -1e-3, "ann_vol": 1e-2},
            {"delta_l1": 1e-6, "cost_sum": 1e-3, "ann_vol": 0.0},
            {"delta_l1": 1e-6, "cost_sum": 1e-3, "ann_vol": -1e-2},
            {"delta_l1": 1e-6, "cost_sum": 1e-3, "ann_vol": np.inf, "epochs_per_year": 0},
        ],
    )
    def test_validation_is_loud(self, kwargs):
        l1 = kwargs.pop("delta_l1")
        with pytest.raises(ValueError):
            sharpe_bound_from_l1(l1, **kwargs)


# ---------------------------------------------------------------------------
# array entry point + verdict rule
# ---------------------------------------------------------------------------

class TestMagnitudeBound:
    def _bound(self, delta, **kw):
        alpha, w_path, w0 = _paths_with_drift(delta)
        kw.setdefault("cost_sum", 1.9e-3)
        kw.setdefault("ann_vol", 1.764e-2)
        return rollforward_magnitude_bound(alpha, w_path, w0, **kw)

    def test_reports_l1_and_max_per_stage(self):
        delta = np.array([[1e-5, -2e-5], [3e-5, 0.0]])
        out = self._bound(delta)
        assert isinstance(out, RollForwardBound)
        np.testing.assert_allclose(out.delta_l1_per_stage, [3e-5, 3e-5])
        np.testing.assert_allclose(out.delta_max_per_stage, [2e-5, 3e-5])
        assert out.delta_l1 == pytest.approx(6e-5)
        assert out.delta_max == pytest.approx(3e-5)

    def test_band_ratio_uses_the_execution_layer_band(self):
        out = self._bound(np.array([[1e-4]]), no_trade_band=2e-4)
        assert out.band_ratio == pytest.approx(0.5)

    def test_no_trade_band_default_matches_the_frozen_config(self):
        assert DEFAULT_NO_TRADE_BAND == MpoConfig().min_order_nav_frac

    def test_epochs_per_year_default_is_the_frozen_calendar(self):
        assert DEFAULT_EPOCHS_PER_YEAR == 8 * 240

    def test_turnover_ratio_is_nan_when_not_supplied(self):
        assert np.isnan(self._bound(np.array([[1e-6]])).turnover_ratio)

    def test_turnover_ratio_is_a_fraction_of_the_trade_vector(self):
        out = self._bound(np.array([[5e-5]]), turnover_per_epoch=1e-2)
        assert out.turnover_ratio == pytest.approx(5e-3)

    def test_verdict_negligible_needs_both_legs(self):
        # tiny drift: under the floor AND under the band
        assert self._bound(np.array([[1e-6]])).verdict == "negligible"

    def test_big_l1_flips_the_verdict_even_with_a_small_max(self):
        """Leg 1 alone can convict: many small per-name corrections still add up."""
        delta = np.full((1, 400), 5e-5)  # l1 = 2e-2, max under the band
        out = self._bound(delta)
        assert out.band_ratio < 1.0
        assert out.sharpe_bound > DEFAULT_DETECT_FLOOR_SHARPE
        assert out.verdict == "material"

    def test_band_breach_alone_flips_the_verdict(self):
        """Leg 2 alone can convict: one huge per-name correction reaches the market."""
        out = self._bound(np.array([[1e-3]]), cost_sum=0.0)
        assert out.sharpe_bound == 0.0  # leg 1 says negligible
        assert out.band_ratio >= 1.0
        assert out.verdict == "material"

    def test_floor_comparison_is_strict(self):
        """A bound exactly AT the resolution floor is not established -> material."""
        floor = 0.01
        # Exact boundary semantics: `bound == floor` must NOT count as negligible.
        assert rf._verdict(floor, 0.5, floor) == "material"
        assert rf._verdict(np.nextafter(floor, 0.0), 0.5, floor) == "negligible"
        # And the same through the full calculator (nudged just above the floor,
        # because an l1 solved to land exactly on it lands a hair below in fp).
        l1 = floor * 1.764e-2 / (1.9e-3 * DEFAULT_EPOCHS_PER_YEAR)
        out = self._bound(np.array([[l1 * 1.001]]), detect_floor_sharpe=floor)
        assert out.sharpe_bound == pytest.approx(floor, rel=2e-3)
        assert out.verdict == "material"

    def test_is_negligible_property_agrees_with_the_verdict(self):
        out = self._bound(np.array([[1e-6]]))
        assert out.is_negligible is (out.verdict == "negligible")

    def test_record_is_frozen(self):
        out = self._bound(np.array([[1e-6]]))
        with pytest.raises(Exception):
            out.delta_l1 = 1.0

    def test_notes_carry_the_reproduction_inputs(self):
        out = self._bound(np.array([[1e-6], [2e-6]]), turnover_per_epoch=1e-2)
        assert out.notes["n_stages"] == 2
        assert out.notes["n_assets"] == 1
        assert out.notes["ann_vol"] == pytest.approx(1.764e-2)
        assert out.notes["turnover_per_epoch"] == pytest.approx(1e-2)

    @pytest.mark.parametrize("bad", [{"no_trade_band": 0.0}, {"detect_floor_sharpe": -1.0},
                                     {"turnover_per_epoch": 0.0}])
    def test_validation_is_loud(self, bad):
        with pytest.raises(ValueError):
            self._bound(np.array([[1e-6]]), **bad)


# ---------------------------------------------------------------------------
# the committed NO-GO, machine checked
# ---------------------------------------------------------------------------

class TestCommittedVerdict:
    def test_anchor_table_is_complete(self):
        assert set(COMMITTED_ANCHORS) == {"main", "ic1.5_cost0.5"}
        need = {"delta_l1", "delta_max", "growth_total", "cost_sum", "ann_vol",
                "turnover_per_epoch", "sharpe_bound"}
        for cell, anchor in COMMITTED_ANCHORS.items():
            assert need <= set(anchor), cell
            assert all(v > 0 for v in anchor.values()), cell

    @pytest.mark.parametrize("cell", sorted(COMMITTED_ANCHORS))
    def test_recorded_sharpe_bound_is_reproducible(self, cell):
        """The design doc section 3 table must be the arithmetic, not a guess."""
        a = COMMITTED_ANCHORS[cell]
        _, _, sharpe = sharpe_bound_from_l1(
            a["delta_l1"], cost_sum=a["cost_sum"], ann_vol=a["ann_vol"]
        )
        assert sharpe == pytest.approx(a["sharpe_bound"], rel=1e-4)

    @pytest.mark.parametrize("cell", sorted(COMMITTED_ANCHORS))
    def test_verdict_is_negligible_in_every_measured_cell(self, cell):
        """THE NO-GO. Red here == the design doc's verdict no longer holds."""
        a = COMMITTED_ANCHORS[cell]
        delta = _delta_matching(a["delta_l1"], a["delta_max"], n_assets=64)
        alpha, w_path, w0 = _paths_with_drift(delta)
        out = rollforward_magnitude_bound(
            alpha, w_path, w0,
            cost_sum=a["cost_sum"],
            ann_vol=a["ann_vol"],
            turnover_per_epoch=a["turnover_per_epoch"],
        )
        assert out.delta_l1 == pytest.approx(a["delta_l1"])
        assert out.delta_max == pytest.approx(a["delta_max"])
        assert out.sharpe_bound == pytest.approx(a["sharpe_bound"], rel=1e-4)
        assert out.sharpe_bound < DEFAULT_DETECT_FLOOR_SHARPE
        assert out.band_ratio < 1.0
        assert out.verdict == "negligible"

    def test_detect_floor_is_the_measured_shrinkage_ci_half_width(self):
        # docs/mpo-t1-shrinkage-design.md: 24-seed CI [-0.0118, +0.0277].
        assert DEFAULT_DETECT_FLOOR_SHARPE == pytest.approx((0.0277 + 0.0118) / 2, abs=5e-5)

    def test_per_name_corrections_are_a_small_fraction_of_the_band(self):
        """Design doc section 3: 4.7% of the band on the main axis, 13.9% at the edge."""
        assert (COMMITTED_ANCHORS["main"]["delta_max"] / DEFAULT_NO_TRADE_BAND
                == pytest.approx(0.047, abs=5e-3))
        assert (COMMITTED_ANCHORS["ic1.5_cost0.5"]["delta_max"] / DEFAULT_NO_TRADE_BAND
                == pytest.approx(0.139, abs=5e-3))

    def test_correction_is_under_one_percent_of_turnover(self):
        for cell, a in COMMITTED_ANCHORS.items():
            assert a["delta_l1"] / a["turnover_per_epoch"] < 0.01, cell

    def test_edge_cell_margin_is_thin_and_that_is_recorded(self):
        """Honesty guard: the ic=1.5/cost=0.5 cell clears the floor by only ~5%."""
        s = COMMITTED_ANCHORS["ic1.5_cost0.5"]["sharpe_bound"]
        assert s < DEFAULT_DETECT_FLOOR_SHARPE
        assert 0.9 < s / DEFAULT_DETECT_FLOOR_SHARPE < 1.0

    def test_candidate_registry_records_all_three_with_reasons(self):
        assert set(ROLLFORWARD_CANDIDATES) == {"asset_drift", "alpha_growth", "budget_growth"}
        for name, why in ROLLFORWARD_CANDIDATES.items():
            assert "REJECTED" in why, name


# ---------------------------------------------------------------------------
# discriminating power: the calculator is not a constant NO-GO
# ---------------------------------------------------------------------------

class TestDiscriminatingPower:
    def _verdict_at_scale(self, scale: float) -> RollForwardBound:
        a = COMMITTED_ANCHORS["main"]
        delta = _delta_matching(a["delta_l1"] * scale, a["delta_max"] * scale, n_assets=64)
        alpha, w_path, w0 = _paths_with_drift(delta)
        return rollforward_magnitude_bound(
            alpha, w_path, w0, cost_sum=a["cost_sum"], ann_vol=a["ann_vol"]
        )

    def test_oracle_roll_forward_is_material(self):
        """Design doc section 0: 1/IC = 16.7x is the Axioma 051 oracle experiment.

        Its "drastically improved" result is an INFORMATION effect (realized
        returns instead of conditional means), and this bound says so: the same
        mechanism at 16.7x the drift clears the resolution floor easily.
        """
        out = self._verdict_at_scale(ORACLE_LEVERAGE)
        assert out.sharpe_bound > DEFAULT_DETECT_FLOOR_SHARPE
        assert out.verdict == "material"

    def test_the_flip_point_is_where_the_arithmetic_says(self):
        need = DEFAULT_DETECT_FLOOR_SHARPE / COMMITTED_ANCHORS["main"]["sharpe_bound"]
        assert need == pytest.approx(1.855, abs=0.01)  # ~1.9x drift, i.e. ~1.9x IC*cost
        assert self._verdict_at_scale(need * 0.99).verdict == "negligible"
        assert self._verdict_at_scale(need * 1.01).verdict == "material"

    def test_verdict_is_monotone_in_the_drift_scale(self):
        seen = [self._verdict_at_scale(s).verdict for s in (0.5, 1.0, 2.0, 5.0, 20.0)]
        assert seen == ["negligible", "negligible", "material", "material", "material"]


# ---------------------------------------------------------------------------
# leg-1 isolation: NO-GO means nothing is wired
# ---------------------------------------------------------------------------

class TestIsolation:
    _CONSUMERS = ("qp.py", "strategies.py", "backtest.py", "config.py", "solve.py",
                  "term_structure.py", "simulator.py")

    def test_no_production_module_imports_rollforward(self):
        pkg = pathlib.Path(inspect.getfile(rf)).parent
        for name in self._CONSUMERS:
            src = (pkg / name).read_text(encoding="utf-8")
            assert "rollforward" not in src, f"{name} references rollforward"

    def test_module_imports_nothing_from_the_mpo_package(self):
        """A pure calculator: numpy only, so it can never perturb a benchmark."""
        src = pathlib.Path(inspect.getfile(rf)).read_text(encoding="utf-8")
        assert "openpo.mpo" not in src.split('"""', 2)[2]

    def test_functions_are_pure(self):
        alpha = np.full((2, 3), 4e-4)
        w_path = np.full((2, 3), 0.02)
        w0 = np.full(3, 0.02)
        refs = (alpha.copy(), w_path.copy(), w0.copy())
        for _ in range(2):
            rollforward_magnitude_bound(
                alpha, w_path, w0, cost_sum=1.9e-3, ann_vol=1.764e-2
            )
            cumulative_growth(alpha, w_path, w0)
        for got, ref in zip((alpha, w_path, w0), refs):
            np.testing.assert_array_equal(got, ref)

    def test_repeated_calls_are_deterministic(self):
        alpha, w_path, w0 = _paths_with_drift(np.full((3, 4), 1e-6))
        outs = [
            rollforward_magnitude_bound(alpha, w_path, w0, cost_sum=1.9e-3, ann_vol=1e-2)
            for _ in range(3)
        ]
        assert {o.sharpe_bound for o in outs} == {outs[0].sharpe_bound}
        assert {o.verdict for o in outs} == {outs[0].verdict}
