"""Contract tests for the production data adapter layer (Iteration 39, P1).

These are the **本地契约腿**: they prove the adapter maps the production
``HandleScore`` callback onto the frozen MPO-T1 types without changing a single
number the strategy sees. The cluster 30-day replay leg is PENDING (no real
prediction data on this box) -- see ``adapters.CLUSTER_LEG_TODO``.

The keystone is :func:`test_adapter_decision_parity_with_direct_context`: an
adapter-fed decision must be *identical* to the simulator-path decision on
identical inputs. Everything else guards one documented decision (seq->minute
map, buy/sell merge, sentinel masks, sellable passthrough, NaN safety).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("openpo.mpo.qp")
pytest.importorskip("openpo.mpo.solve")

from openpo.mpo.adapters import (
    CLUSTER_LEG_TODO,
    HORIZON_KEYS,
    PRODUCTION_BUY_SELL_DIFF,
    PRODUCTION_COSTS,
    PRODUCTION_SCORE_SENTINEL,
    PRODUCTION_SEQ_MAX,
    PRODUCTION_SEQ_MIN,
    CallbackRecord,
    HorizonScores,
    IidUniverse,
    ProductionCosts,
    ReplayHarness,
    SnapshotAdapter,
    TradingDayIndex,
    build_score_frame,
    buy_sell_from_payload,
    is_epoch_seq,
    minute_of_seq,
    production_mpo_config,
    seq_of_minute,
    tau_of_seq,
    union_universe,
)
from openpo.mpo.calendar import day_of, is_epoch, minute_of
from openpo.mpo.config import EPOCH_OFFSETS, MINUTES_PER_DAY, MpoConfig, SynthConfig
from openpo.mpo.simulator import Simulator
from openpo.mpo.strategies import MpoStrategy
from openpo.mpo.synth import generate_market
from openpo.mpo.term_structure import fit_term_structure
from openpo.mpo.types import DecisionContext, PortfolioSnapshot

N_ASSETS = 20
N_DAYS = 4
SEED = 20260823
CALIB_DAYS = 2


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def panel():
    """Small synthetic market + signal panel (realistic generator, fast size)."""
    scfg = SynthConfig(n_assets=N_ASSETS, n_days=N_DAYS)
    return generate_market(scfg, SEED)


@pytest.fixture(scope="module")
def cfg():
    # Production costs + a calibration window that fits the tiny panel.
    return production_mpo_config(calib_days=CALIB_DAYS)


@pytest.fixture(scope="module")
def ts_model(panel, cfg):
    market, signals = panel
    return fit_term_structure(market, signals, cfg, calib_days=CALIB_DAYS)


@pytest.fixture(scope="module")
def universe():
    return IidUniverse([f"{600000 + i}.SH" for i in range(N_ASSETS)])


def make_strategy(ts_model, market, cfg):
    """A fresh strategy instance (no warm-start state shared across paths)."""
    return MpoStrategy(ts_model, market=market, cfg=cfg)


def callback_scores(raw_row: np.ndarray, sell_offset: float = 1.0) -> dict:
    """[N,3] raw signal row -> production ``{horizon: (buy, sell)}`` payloads.

    ``sell_scores`` are irrelevant to the production alpha (buy_scores is the
    alpha) so they are set to ``buy - offset``, i.e. no sentinel, positive spread.
    """
    return {
        key: (raw_row[:, k].copy(), raw_row[:, k].copy() - sell_offset)
        for k, key in enumerate(HORIZON_KEYS)
    }


# --------------------------------------------------------------------- costs
def test_production_costs_match_production_params():
    """2bp buy / 9bp all-in sell / 50bp slippage, from probe_opt_bt4_real_shapes."""
    c = PRODUCTION_COSTS
    assert c.buy_commission == pytest.approx(0.0002)  # buy_commission_rate
    assert c.sell_commission_all_in == pytest.approx(0.0009)  # sell_commission_rate
    assert c.slippage == pytest.approx(0.005)  # spread_slippage_rate
    # The 9bp is split as sell_commission + the repo's canonical 5bp stamp tax.
    assert c.sell_commission + c.stamp_tax == pytest.approx(c.sell_commission_all_in)
    assert c.buy_total == pytest.approx(52e-4)
    assert c.sell_total == pytest.approx(59e-4)


def test_production_config_replaces_only_costs():
    default = MpoConfig()
    prod = production_mpo_config()

    assert prod.buy_cost() == pytest.approx(52e-4)
    assert prod.sell_cost() == pytest.approx(59e-4)
    # 50bp vs 5bp slippage: a 10x jump, round trip 19bp -> 111bp.
    assert prod.slippage / default.slippage == pytest.approx(10.0)
    assert (prod.buy_cost() + prod.sell_cost()) / (
        default.buy_cost() + default.sell_cost()
    ) == pytest.approx(111.0 / 19.0)

    # Nothing else moved, and the frozen defaults are untouched (no mutation).
    cost_fields = {"buy_commission", "sell_commission", "stamp_tax", "slippage"}
    for f in dataclasses.fields(MpoConfig):
        if f.name not in cost_fields:
            assert getattr(prod, f.name) == getattr(default, f.name), f.name
    assert MpoConfig().slippage == pytest.approx(5e-4)
    assert MpoConfig().buy_cost() == pytest.approx(7e-4)


def test_production_config_overrides_and_custom_costs():
    prod = production_mpo_config(gamma=3.0, nav_cny=500e8)
    assert prod.gamma == 3.0 and prod.nav_cny == 500e8
    assert prod.buy_cost() == pytest.approx(52e-4)

    doubled = production_mpo_config(costs=ProductionCosts(slippage=100e-4))
    assert doubled.sell_cost() == pytest.approx(4e-4 + 5e-4 + 100e-4)


def test_cluster_leg_todo_is_documented():
    """The PENDING cluster leg must stay enumerated in code, not just in docs."""
    assert len(CLUSTER_LEG_TODO) >= 5
    joined = " ".join(CLUSTER_LEG_TODO).lower()
    for token in ("d/m/t_alpha", "seq", "signal_horizons", "scale"):
        assert token in joined

    # Iteration 45 closed the solve-wall prerequisite; the ledger must say so.
    # The item is NOT deleted -- a closed prerequisite keeps its slot with its
    # evidence pointer so a future reader can tell "measured, fine" apart from
    # "nobody looked". Gate: the entry naming the solve wall must carry both the
    # RESOLVED marker and the two levers that bought the headroom (the low-rank
    # factor assembly and the Clarabel backend), because either one alone is not
    # enough at N=5000 (see ``openpo.mpo.qp`` module docstring for the walls).
    solve_items = [t for t in CLUSTER_LEG_TODO if "solve wall" in t.lower()]
    assert len(solve_items) == 1, solve_items
    item = solve_items[0]
    assert item.startswith("RESOLVED"), item
    lowered = item.lower()
    for token in ("iteration 45", "n=5000", "low-rank", "clarabel", "risk_form"):
        assert token in lowered, token

    # ...and no OTHER item may claim to be resolved, so the ledger cannot drift
    # into "everything is fine" without a matching probe.
    resolved = [t for t in CLUSTER_LEG_TODO if t.startswith("RESOLVED")]
    assert resolved == [item]
    # The blocking prerequisite (user-side three-family split) is still open.
    assert not any(
        t.startswith("RESOLVED") for t in CLUSTER_LEG_TODO if "d/m/t_alpha" in t
    )


# ------------------------------------------------------------ seq / tau mapping
def test_seq_minute_mapping_is_identity_over_the_production_range():
    assert minute_of_seq(PRODUCTION_SEQ_MIN) == 1  # seq 1 == 9:31
    assert minute_of_seq(PRODUCTION_SEQ_MAX) == 237  # seq 237 == 14:57
    # 9:31..14:57 == 120 morning + 117 afternoon trading minutes.
    assert PRODUCTION_SEQ_MAX == 120 + 117
    for seq in range(PRODUCTION_SEQ_MIN, PRODUCTION_SEQ_MAX + 1):
        assert seq_of_minute(minute_of_seq(seq)) == seq
        assert 0 <= minute_of_seq(seq) < MINUTES_PER_DAY


@pytest.mark.parametrize("bad", [0, -1, 238, 240, 1000])
def test_out_of_range_seq_is_loud(bad):
    with pytest.raises(ValueError, match="seq must be in"):
        minute_of_seq(bad)


def test_unscored_minutes_have_no_seq():
    """Minutes 0/238/239 are unscored in production; asking for them must fail."""
    for minute in (0, 238, 239):
        with pytest.raises(ValueError, match="no production score seq"):
            seq_of_minute(minute)


def test_every_epoch_offset_is_reachable_from_a_production_seq():
    for offset in EPOCH_OFFSETS:
        seq = seq_of_minute(offset)
        assert is_epoch_seq(seq)
        assert minute_of_seq(seq) == offset
    # ... and no unscored minute is an epoch, so the 240-min grid is unaffected.
    assert max(EPOCH_OFFSETS) <= PRODUCTION_SEQ_MAX
    assert min(EPOCH_OFFSETS) >= PRODUCTION_SEQ_MIN


def test_tau_agrees_with_the_frozen_calendar():
    for day in (0, 1, 7):
        for seq in (1, 5, 35, 215, 237):
            tau = tau_of_seq(day, seq)
            assert tau == day * MINUTES_PER_DAY + seq
            assert day_of(tau) == day
            assert minute_of(tau) == minute_of_seq(seq)
            assert is_epoch(tau) == (seq in EPOCH_OFFSETS)
    with pytest.raises(ValueError, match="day index"):
        tau_of_seq(-1, 5)


def test_trading_day_index_maps_dates_to_consecutive_days():
    idx = TradingDayIndex([20260601, 20260602, 20260603, 20260608])
    assert idx.n_days == 4
    assert idx.day_of(20260601) == 0
    assert idx.day_of(20260608) == 3  # weekend gap is NOT a calendar gap
    assert idx.tau(20260602, 35) == 1 * MINUTES_PER_DAY + 35
    with pytest.raises(ValueError, match="not in the replay window"):
        idx.day_of(20260609)
    with pytest.raises(ValueError, match="ascending"):
        TradingDayIndex([20260602, 20260601])
    with pytest.raises(ValueError, match="duplicates"):
        TradingDayIndex([20260601, 20260601])


class _FakeDate:
    """Production hands ``alpha.Date`` objects exposing ``ToInteger()``."""

    def __init__(self, value: int) -> None:
        self._value = value

    def ToInteger(self) -> int:  # noqa: N802 - production API name
        return self._value


def test_production_date_objects_are_accepted():
    idx = TradingDayIndex([20260601, 20260602])
    assert idx.day_of(_FakeDate(20260602)) == 1


# ------------------------------------------------------------- payload handling
class _ScoreItem:
    def __init__(self, buy: float, sell: float) -> None:
        self.buy_score = buy
        self.sell_score = sell


def test_buy_sell_payload_forms_agree():
    buy = np.array([1.0, -2.0, 3.0])
    sell = np.array([0.5, -1.0, 2.0])

    from_tuple = buy_sell_from_payload((buy, sell))
    from_objs = buy_sell_from_payload([_ScoreItem(b, s) for b, s in zip(buy, sell)])
    from_map = buy_sell_from_payload({"buy_scores": buy, "sell_scores": sell})

    for got in (from_tuple, from_objs, from_map):
        np.testing.assert_array_equal(got[0], buy)
        np.testing.assert_array_equal(got[1], sell)


def test_blended_scoremain_payload_is_rejected_with_the_real_reason():
    with pytest.raises(ValueError, match="BLENDED"):
        buy_sell_from_payload({"ScoreMain": (np.zeros(3), np.zeros(3))})


def test_missing_score_family_is_loud():
    iids = ["a", "b"]
    partial = {HORIZON_KEYS[0]: (np.zeros(2), np.zeros(2))}
    with pytest.raises(ValueError, match="score families missing"):
        build_score_frame(20260601, 5, partial, day=0, iids=iids)


# ---------------------------------------------------------------- iid alignment
def test_union_universe_first_appearance_order_and_nan_fill():
    d = HorizonScores.from_payload(["a", "b"], (np.array([1.0, 2.0]), np.array([0.0, 0.0])))
    m = HorizonScores.from_payload(["b", "c"], (np.array([3.0, 4.0]), np.array([0.0, 0.0])))
    t = HorizonScores.from_payload(["a", "c"], (np.array([5.0, 6.0]), np.array([0.0, 0.0])))

    frame = build_score_frame(
        20260601, 35, dict(zip(HORIZON_KEYS, (d, m, t))), day=0, buy_sell_diff=None
    )

    assert frame.iids == ("a", "b", "c")
    expected = np.array(
        [
            [1.0, np.nan, 5.0],  # a: absent from m_alpha
            [2.0, 3.0, np.nan],  # b: absent from t_alpha
            [np.nan, 4.0, 6.0],  # c: absent from d_alpha
        ]
    )
    np.testing.assert_array_equal(frame.raw, expected)
    assert frame.diagnostics["n_missing_per_family"] == {
        HORIZON_KEYS[0]: 1,
        HORIZON_KEYS[1]: 1,
        HORIZON_KEYS[2]: 1,
    }
    assert frame.tau == 35 and frame.minute == 35 and frame.day == 0
    assert not frame.halted.any()  # every name is scored by at least one family


def test_unknown_iid_is_loud():
    uni = IidUniverse(["a", "b"])
    with pytest.raises(ValueError, match="absent from the universe"):
        uni.align(np.array([1.0]), ["zzz"])
    with pytest.raises(ValueError, match="duplicate iids"):
        IidUniverse(["a", "a"])
    with pytest.raises(ValueError, match="entries but"):
        uni.align(np.array([1.0, 2.0]), ["a"])


def test_name_missing_from_every_family_is_halted():
    uni = union_universe([["a", "b"], ["ghost"]])
    scores = {
        key: HorizonScores.from_payload(["a", "b"], (np.array([1.0, 2.0]), np.zeros(2)))
        for key in HORIZON_KEYS
    }
    frame = build_score_frame(20260601, 5, scores, day=0, universe=uni)
    assert frame.iids == ("a", "b", "ghost")
    assert frame.halted.tolist() == [False, False, True]
    assert frame.diagnostics["n_unscored"] == 1


# ------------------------------------------------------------ buy/sell semantics
def test_alpha_is_buy_scores_and_sell_scores_are_only_gates():
    """Production feeds ``score = buy_scores`` (opt_openpo.py:768)."""
    iids = ["a", "b", "c"]
    buy = np.array([0.3, -0.1, 0.2])
    sell = np.array([9.0, -9.0, 0.0])  # wildly different -> must not touch alpha
    scores = {key: (buy, sell) for key in HORIZON_KEYS}

    frame = build_score_frame(20260601, 5, scores, day=0, iids=iids)
    for k in range(3):
        np.testing.assert_array_equal(frame.raw[:, k], buy)

    net = build_score_frame(20260601, 5, scores, day=0, iids=iids, alpha_mode="net")
    for k in range(3):
        np.testing.assert_array_equal(net.raw[:, k], buy - sell)

    with pytest.raises(ValueError, match="alpha_mode"):
        build_score_frame(20260601, 5, scores, day=0, iids=iids, alpha_mode="mean")


def test_sentinel_scores_become_masks_not_alphas():
    iids = ["a", "b", "c"]
    s = PRODUCTION_SCORE_SENTINEL
    buy = np.array([0.5, -10_000.0, 0.1])  # b: cannot buy
    sell = np.array([0.0, 0.0, 5_000.0])  # c: cannot sell
    scores = {key: (buy, sell) for key in HORIZON_KEYS}

    frame = build_score_frame(20260601, 5, scores, day=0, iids=iids)

    assert frame.limit_up.tolist() == [False, True, False]  # limit_up == no buy
    assert frame.limit_down.tolist() == [False, False, True]  # limit_down == no sell
    # The -10000 sentinel must NOT survive into the alpha or it destroys the
    # cross-sectional z-score of the whole panel.
    assert np.isnan(frame.raw[1]).all()
    assert np.isfinite(frame.raw[0]).all() and np.isfinite(frame.raw[2]).all()
    assert abs(buy[1]) > s and sell[2] > s
    assert frame.diagnostics["n_limit_up"] == 1
    assert frame.diagnostics["n_limit_down"] == 1


def test_valid_flag_maps_to_halted():
    iids = ["a", "b"]
    scores = {key: (np.array([1.0, 2.0]), np.zeros(2)) for key in HORIZON_KEYS}
    frame = build_score_frame(
        20260601, 5, scores, day=0, iids=iids, valid=np.array([1, 0])
    )
    assert frame.halted.tolist() == [False, True]


def test_zero_trade_spread_gate_is_opt_in():
    iids = ["a", "b"]
    buy = np.array([0.10, 0.05])
    sell = np.array([0.00, -0.02])  # spread = 0.10 - (-0.02) = 0.12 <= 0.6
    scores = {key: (buy, sell) for key in HORIZON_KEYS}

    off = build_score_frame(20260601, 5, scores, day=0, iids=iids)
    assert off.gated is False  # default: gate disabled (score units unknown)

    on = build_score_frame(
        20260601, 5, scores, day=0, iids=iids, buy_sell_diff=PRODUCTION_BUY_SELL_DIFF
    )
    assert on.gated is True
    np.testing.assert_allclose(on.spread, 0.12)

    wide = build_score_frame(
        20260601,
        5,
        {key: (buy + 1.0, sell) for key in HORIZON_KEYS},
        day=0,
        iids=iids,
        buy_sell_diff=PRODUCTION_BUY_SELL_DIFF,
    )
    assert wide.gated is False  # spread 1.12 > 0.6


# --------------------------------------------------------------------- snapshot
def test_snapshot_sellable_passthrough_and_share_recovery(universe):
    n = universe.n_assets
    rng = np.random.default_rng(11)
    weights = rng.uniform(0.0, 0.9 / n, size=n)
    unlimited = weights * 0.4  # only 40% of each holding is T+1 sellable
    prices = rng.uniform(8.0, 30.0, size=n)
    nav = 1e8

    snap = SnapshotAdapter(universe).build(
        tau=245, nav=nav, weights=weights, unlimited_weights=unlimited, prices=prices
    )

    assert snap.tau == 245 and snap.nav == nav
    np.testing.assert_array_equal(snap.weights, weights)
    # sellable == unlimited_weights, verbatim (no re-marking, no cap).
    np.testing.assert_array_equal(snap.sellable_weights, unlimited)
    np.testing.assert_allclose(snap.holdings_shares, weights * nav / prices, rtol=0, atol=0)
    np.testing.assert_allclose(snap.sellable_shares, unlimited * nav / prices)
    # Cash is the un-invested residual; weights sum < 1 here.
    assert snap.cash == pytest.approx(nav * (1.0 - weights.sum()))
    assert snap.cash + float(snap.holdings_shares @ prices) == pytest.approx(nav)


def test_snapshot_reorders_callback_iids_onto_the_universe(universe):
    """Production iid order need not match the universe order; values must follow."""
    n = universe.n_assets
    shuffled = list(reversed(universe.iids))
    weights = np.linspace(0.001, 0.01, n)
    prices = np.linspace(10.0, 30.0, n)
    adapter = SnapshotAdapter(universe)

    snap = adapter.build(
        tau=5, nav=1e8, weights=weights, unlimited_weights=weights * 0.5,
        prices=prices, iids=shuffled,
    )
    np.testing.assert_array_equal(snap.weights, weights[::-1])
    np.testing.assert_allclose(snap.holdings_shares, weights[::-1] * 1e8 / prices[::-1])


def test_snapshot_partial_universe_coverage_is_loud(universe):
    """A universe name with no callback price cannot get a fabricated one."""
    subset = list(universe.iids[:5])
    with pytest.raises(ValueError, match="callback covers 5 of 20"):
        SnapshotAdapter(universe).build(
            tau=5, nav=1e8, weights=np.full(5, 0.01),
            unlimited_weights=np.full(5, 0.005), prices=np.full(5, 10.0), iids=subset,
        )


def test_snapshot_validation_is_loud(universe):
    n = universe.n_assets
    adapter = SnapshotAdapter(universe)
    ok = dict(
        tau=5,
        nav=1e8,
        weights=np.full(n, 0.01),
        unlimited_weights=np.full(n, 0.005),
        prices=np.full(n, 10.0),
    )

    with pytest.raises(ValueError, match="entries but the universe has"):
        adapter.build(**{**ok, "weights": np.full(n - 1, 0.01)})
    with pytest.raises(ValueError, match="non-finite"):
        adapter.build(**{**ok, "weights": np.full(n, np.nan)})
    with pytest.raises(ValueError, match="strictly positive"):
        adapter.build(**{**ok, "prices": np.zeros(n)})
    with pytest.raises(ValueError, match="nav must be finite and positive"):
        adapter.build(**{**ok, "nav": 0.0})
    with pytest.raises(ValueError, match="non-negative"):
        adapter.build(**{**ok, "weights": np.full(n, -0.01)})
    with pytest.raises(ValueError, match=r"sum\(weights\)"):
        adapter.build(**{**ok, "weights": np.full(n, 0.5)})
    with pytest.raises(ValueError, match="T\\+1 sellable cannot exceed"):
        adapter.build(**{**ok, "unlimited_weights": np.full(n, 0.02)})
    with pytest.raises(ValueError, match="nan_policy"):
        SnapshotAdapter(universe, nan_policy="whatever")


def test_snapshot_production_parity_knobs(universe):
    """Production's nan_to_num / clamp / renormalize are opt-in, never implicit."""
    n = universe.n_assets
    weights = np.full(n, 0.02)
    weights[0] = np.nan
    unlimited = np.full(n, 0.005)
    unlimited[0] = 0.0  # no weight -> nothing sellable either
    unlimited[1] = -1e-3  # production sees negative unlimited_weights in the wild

    strict = SnapshotAdapter(universe)
    with pytest.raises(ValueError, match="non-finite"):
        strict.build(tau=5, nav=1e8, weights=weights, unlimited_weights=unlimited,
                     prices=np.full(n, 10.0))

    prod = SnapshotAdapter(
        universe, nan_policy="zero", clip_negative=True, normalize_weights=True
    )
    snap = prod.build(
        tau=5, nav=1e8, weights=weights, unlimited_weights=unlimited,
        prices=np.full(n, 10.0),
    )
    assert snap.weights.sum() == pytest.approx(1.0)  # production weights /= sum
    assert snap.weights[0] == 0.0
    assert snap.sellable_weights.min() >= 0.0
    assert snap.cash == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------- replay harness
def _day_callbacks(market, signals, universe, day, date, nav, weights, unlimited):
    """Every production minute of one day as CallbackRecords (seq 1..237)."""
    out = []
    for seq in range(PRODUCTION_SEQ_MIN, PRODUCTION_SEQ_MAX + 1):
        tau = day * MINUTES_PER_DAY + seq
        out.append(
            CallbackRecord(
                date=date,
                seq=seq,
                iids=universe.iids,
                horizon_scores=callback_scores(signals.signals[tau]),
                weights=weights,
                unlimited_weights=unlimited,
                nav=nav,
                prices=market.prices[tau].astype(float),
            )
        )
    return out


def test_replay_fires_exactly_on_the_eight_epochs(panel, cfg, ts_model, universe):
    market, signals = panel
    n = universe.n_assets
    weights = np.full(n, 0.5 / n)
    unlimited = weights * 0.5

    harness = ReplayHarness(
        make_strategy(ts_model, market, cfg),
        universe,
        n_days=N_DAYS,
        cfg=cfg,
        day_index=TradingDayIndex([20260601, 20260602, 20260603, 20260604]),
    )
    decisions = harness.run_day(
        _day_callbacks(market, signals, universe, 1, 20260602, 1e8, weights, unlimited)
    )

    assert [d.seq for d in decisions] == list(EPOCH_OFFSETS)
    assert [d.minute for d in decisions] == list(EPOCH_OFFSETS)
    assert [d.tau for d in decisions] == [MINUTES_PER_DAY + o for o in EPOCH_OFFSETS]
    assert all(d.day == 1 for d in decisions)
    assert harness.counters["callbacks"] == PRODUCTION_SEQ_MAX
    assert harness.counters["epochs"] == len(EPOCH_OFFSETS)
    assert harness.counters["skipped_non_epoch"] == PRODUCTION_SEQ_MAX - len(EPOCH_OFFSETS)
    # Non-epoch minutes really do return None (no hidden decision).
    assert harness.handle_score(
        20260602, 6, universe.iids, callback_scores(signals.signals[246]),
        weights, unlimited, nav=1e8, prices=market.prices[246].astype(float), day=1,
    ) is None

    for d in decisions:
        # sellable passthrough survives the whole pipeline
        np.testing.assert_array_equal(d.snapshot.sellable_weights, unlimited)
        assert d.status in ("optimal", "inaccurate", "failed", "fail_soft")
        if d.target_weights is not None:
            assert d.target_weights.shape == (n,)
            assert np.all(np.isfinite(d.target_weights))


def test_replay_rejects_mismatched_universe_and_calendar(panel, cfg, ts_model, universe):
    market, _ = panel
    with pytest.raises(ValueError, match="bound to"):
        ReplayHarness(
            make_strategy(ts_model, market, cfg),
            IidUniverse(["a", "b"]),
            n_days=N_DAYS,
            cfg=cfg,
        )
    harness = ReplayHarness(
        make_strategy(ts_model, market, cfg), universe, n_days=N_DAYS, cfg=cfg
    )
    with pytest.raises(ValueError, match="day index unavailable"):
        harness.handle_score(
            20260601, 5, universe.iids,
            callback_scores(np.zeros((universe.n_assets, 3))),
            np.zeros(universe.n_assets), np.zeros(universe.n_assets),
            nav=1e8, prices=np.full(universe.n_assets, 10.0),
        )
    with pytest.raises(ValueError, match="outside the harness calendar"):
        harness.handle_score(
            20260601, 5, universe.iids,
            callback_scores(np.zeros((universe.n_assets, 3))),
            np.zeros(universe.n_assets), np.zeros(universe.n_assets),
            nav=1e8, prices=np.full(universe.n_assets, 10.0), day=N_DAYS,
        )


def test_replay_gate_returns_no_trade_without_solving(panel, cfg, ts_model, universe):
    """With the production gate enabled and a flat cross-section, no trade."""
    market, _ = panel
    n = universe.n_assets
    harness = ReplayHarness(
        make_strategy(ts_model, market, cfg),
        universe,
        n_days=N_DAYS,
        cfg=cfg,
        buy_sell_diff=PRODUCTION_BUY_SELL_DIFF,
    )
    flat = np.zeros((n, 3))
    decision = harness.handle_score(
        20260601, 5, universe.iids, callback_scores(flat, sell_offset=0.0),
        np.full(n, 0.5 / n), np.full(n, 0.25 / n),
        nav=1e8, prices=np.full(n, 10.0), day=0,
    )
    assert decision is not None
    assert decision.target_weights is None
    assert decision.status == "zero_trade_gate"
    assert decision.traded is False
    assert harness.counters["gated"] == 1
    assert harness.counters["decided"] == 0


def test_halted_names_are_nan_safe(panel, cfg, ts_model, universe):
    """NaN scores for a halted name must neutralize, never poison the panel."""
    market, signals = panel
    n = universe.n_assets
    tau = MINUTES_PER_DAY + EPOCH_OFFSETS[2]
    raw = signals.signals[tau].copy()
    raw[3] = np.nan  # halted name: no prediction at all this minute

    harness = ReplayHarness(
        make_strategy(ts_model, market, cfg), universe, n_days=N_DAYS, cfg=cfg
    )
    weights = np.full(n, 0.5 / n)
    frame = build_score_frame(
        20260602, EPOCH_OFFSETS[2], callback_scores(raw), day=1, universe=universe,
        iids=universe.iids,
    )
    assert frame.halted[3] and not frame.halted[[0, 1, 2]].any()

    snap = SnapshotAdapter(universe).build(
        tau, 1e8, weights, weights * 0.5, market.prices[tau].astype(float)
    )
    ctx = harness.build_context(frame, snap, market.prices[tau].astype(float))
    assert np.all(np.isfinite(ctx.z_signals))
    assert ctx.z_signals[3].tolist() == [0.0, 0.0, 0.0]  # neutral, not NaN
    # The other names keep a live cross-section (the NaN did not collapse it).
    assert np.abs(ctx.z_signals[np.arange(n) != 3]).max() > 0.1

    decision = harness.handle_score(
        20260602, EPOCH_OFFSETS[2], universe.iids, callback_scores(raw),
        weights, weights * 0.5, nav=1e8,
        prices=market.prices[tau].astype(float), day=1,
    )
    assert decision is not None
    if decision.target_weights is not None:
        assert np.all(np.isfinite(decision.target_weights))


# ------------------------------------------------------------------- KEY parity
def _direct_context(market, raw, tau, weights, unlimited, nav, prices, halted, cfg):
    """Rebuild the DecisionContext exactly as ``Simulator.run`` does.

    Mirrors ``src/openpo/mpo/simulator.py:226-249``: weights from holdings marked
    at the current price, sellable from the ledger, z-score of the raw signal row
    with halted names zeroed.
    """
    sim = Simulator(market, cfg)
    holdings = weights * nav / prices
    sellable_shares = unlimited * nav / prices
    snapshot = PortfolioSnapshot(
        tau=tau,
        nav=nav,
        cash=nav * (1.0 - float(weights.sum())),
        holdings_shares=holdings,
        weights=holdings * prices / nav,
        sellable_shares=sellable_shares,
        sellable_weights=sellable_shares * prices / nav,
    )
    z = sim._zscore(raw)
    z[halted] = 0.0
    return DecisionContext(
        snapshot=snapshot,
        z_signals=z,
        prices=prices,
        limit_up=np.asarray(market.limit_up[tau], dtype=bool),
        limit_down=np.asarray(market.limit_down[tau], dtype=bool),
        halted=halted,
        day=tau // MINUTES_PER_DAY,
        minute=tau % MINUTES_PER_DAY,
    )


@pytest.mark.parametrize("epoch_index", [0, 3, 7])
def test_adapter_decision_parity_with_direct_context(
    panel, cfg, ts_model, universe, epoch_index
):
    """THE contract test: same inputs -> same decision, adapter or simulator path."""
    market, signals = panel
    n = universe.n_assets
    day = 1
    seq = EPOCH_OFFSETS[epoch_index]
    tau = day * MINUTES_PER_DAY + seq
    prices = market.prices[tau].astype(float)
    raw = signals.signals[tau].copy()
    halted = np.asarray(market.halted[tau], dtype=bool)

    # The adapter derives limit masks from score sentinels only, so parity needs
    # a tau where the market has no limit hits (assert it, don't assume it).
    assert not np.asarray(market.limit_up[tau]).any()
    assert not np.asarray(market.limit_down[tau]).any()

    rng = np.random.default_rng(7 + epoch_index)
    weights = rng.uniform(0.0, 0.9 / n, size=n)
    unlimited = weights * rng.uniform(0.2, 0.9, size=n)
    nav = 1e8

    direct_ctx = _direct_context(
        market, raw, tau, weights, unlimited, nav, prices, halted, cfg
    )
    direct_target = make_strategy(ts_model, market, cfg).decide(direct_ctx)

    harness = ReplayHarness(
        make_strategy(ts_model, market, cfg),
        universe,
        n_days=N_DAYS,
        cfg=cfg,
        day_index=TradingDayIndex([20260601, 20260602, 20260603, 20260604]),
    )
    decision = harness.handle_score(
        20260602,
        seq,
        universe.iids,
        callback_scores(raw),
        weights,
        unlimited,
        nav=nav,
        prices=prices,
        valid=np.where(halted, 0, 1),
    )
    assert decision is not None and decision.tau == tau

    # 1) the context the strategy sees is identical field by field
    got, want = decision.snapshot, direct_ctx.snapshot
    assert got.tau == want.tau and got.nav == pytest.approx(want.nav)
    np.testing.assert_allclose(got.weights, want.weights, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        got.sellable_weights, want.sellable_weights, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        got.holdings_shares, want.holdings_shares, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(got.cash, want.cash, rtol=1e-12)

    ctx = harness.build_context(decision.frame, decision.snapshot, prices)
    np.testing.assert_array_equal(ctx.z_signals, direct_ctx.z_signals)
    np.testing.assert_array_equal(ctx.halted, direct_ctx.halted)
    np.testing.assert_array_equal(ctx.limit_up, direct_ctx.limit_up)
    np.testing.assert_array_equal(ctx.limit_down, direct_ctx.limit_down)
    assert (ctx.day, ctx.minute) == (direct_ctx.day, direct_ctx.minute)

    # 2) and therefore the decision is identical
    assert (decision.target_weights is None) == (direct_target is None)
    if direct_target is not None:
        np.testing.assert_allclose(
            decision.target_weights, direct_target, rtol=1e-12, atol=1e-12
        )


def test_adapter_alpha_reaches_the_term_structure_unchanged(
    panel, cfg, ts_model, universe
):
    """Round trip: raw signals -> callback -> frame -> z -> identical stage alphas."""
    from openpo.mpo.stages import build_stage_grid

    market, signals = panel
    seq = EPOCH_OFFSETS[1]
    tau = MINUTES_PER_DAY + seq
    raw = signals.signals[tau].copy()

    frame = build_score_frame(
        20260602, seq, callback_scores(raw), day=1,
        universe=universe, iids=universe.iids,
    )
    # The [N,3] matrix handed to the z-score layer IS the raw signal panel row.
    np.testing.assert_array_equal(frame.raw, raw)

    harness = ReplayHarness(
        make_strategy(ts_model, market, cfg), universe, n_days=N_DAYS, cfg=cfg
    )
    n = universe.n_assets
    snap = SnapshotAdapter(universe).build(
        tau, 1e8, np.full(n, 0.5 / n), np.full(n, 0.25 / n),
        market.prices[tau].astype(float),
    )
    z_adapter = harness.build_context(
        frame, snap, market.prices[tau].astype(float)
    ).z_signals
    z_direct = np.nan_to_num(Simulator(market, cfg)._zscore(raw))

    np.testing.assert_array_equal(z_adapter, z_direct)
    grid = build_stage_grid(tau, N_DAYS, cfg)
    np.testing.assert_allclose(
        ts_model.stage_alphas(z_adapter, grid),
        ts_model.stage_alphas(z_direct, grid),
        rtol=0,
        atol=0,
    )
