"""Tests for openpo.mpo.simulator + openpo.mpo.metrics (SPEC section 5).

Uses a hand-built deterministic MarketPanel (no dependency on synth.py) so the
suite stays fast and self-contained; sibling modules that the simulator
genuinely needs (execution, ledger) are guarded with importorskip because they
are implemented in parallel.

Calendar convention under test: tau = day*240 + minute, trading minutes are
plain tau differences, overnight = 0 trading minutes.

Execution convention under test (Iteration 43): the fill reference price of the
epoch bar ``(tau, tau_end]`` is the WINDOW MIDPOINT minute close,
``prices[(tau + tau_end) // 2]``, with ``*(1 +- slippage)`` on top; the
participation cap still consumes the full ``(tau, tau_end]`` volume pool. The
price pins that used to read ``prices[tau_end]`` were re-pinned deliberately --
see the block comment above ``_BAR_MINUTES``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("openpo.mpo.ledger")
execution = pytest.importorskip("openpo.mpo.execution")

from openpo.mpo import simulator as simulator_mod
from openpo.mpo.config import EPOCH_OFFSETS, MINUTES_PER_DAY, MpoConfig
from openpo.mpo.metrics import summarize
from openpo.mpo.simulator import Simulator
from openpo.mpo.types import (
    BacktestResult,
    MarketPanel,
    Order,
    SignalPanel,
    T1ViolationError,
)

N_ASSETS = 6
N_DAYS = 5
HALTED_ASSET = 5  # halted for the whole panel


# --------------------------------------------------------------------- fixtures
def build_panel() -> tuple[MarketPanel, SignalPanel]:
    """Tiny deterministic market: 6 assets x 5 days, ample volume, no limits."""
    t_total = N_DAYS * MINUTES_PER_DAY
    p0 = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    # Small deterministic per-minute drift, mixed signs; halted asset frozen.
    growth = (np.arange(N_ASSETS) - 2.5) * 2e-6
    growth[HALTED_ASSET] = 0.0
    taus = np.arange(t_total, dtype=np.float64)
    prices = p0[None, :] * (1.0 + growth[None, :]) ** taus[:, None]

    minute_amount = np.full((t_total, N_ASSETS), 5e6)  # CNY per minute
    prev_close = np.empty((N_DAYS, N_ASSETS))
    prev_close[0] = p0
    for d in range(1, N_DAYS):
        prev_close[d] = prices[d * MINUTES_PER_DAY - 1]

    halted = np.zeros((t_total, N_ASSETS), dtype=bool)
    halted[:, HALTED_ASSET] = True

    market = MarketPanel(
        n_assets=N_ASSETS,
        n_days=N_DAYS,
        prices=prices,
        minute_amount=minute_amount,
        prev_close=prev_close,
        limit_up=np.zeros((t_total, N_ASSETS), dtype=bool),
        limit_down=np.zeros((t_total, N_ASSETS), dtype=bool),
        halted=halted,
        adv_cny=np.full(N_ASSETS, 240 * 5e6),
        beta=np.ones(N_ASSETS),
        factor_exposures=np.ones((N_ASSETS, 1)),
        factor_cov_daily=np.array([[1e-4]]),
        specific_var_daily=np.full(N_ASSETS, 1e-4),
        meta={"seed": 7},
    )

    idx = np.arange(t_total)[:, None, None]
    assets = np.arange(N_ASSETS)[None, :, None]
    ks = np.arange(3)[None, None, :]
    signals = np.sin(0.013 * idx + 0.7 * assets + ks)
    signals[EPOCH_OFFSETS[0], 0, 0] = np.nan  # nan-safety probe at first epoch
    return market, SignalPanel(signals=signals, meta={"seed": 7})


class ScriptedStrategy:
    """Fixed target-weight schedule keyed on (day, minute); records contexts."""

    name = "scripted"

    def __init__(self, schedule):
        # schedule: {(day, minute): target array}; standing target persists.
        self.schedule = dict(schedule)
        self.current = np.zeros(N_ASSETS)
        self.contexts = []
        self.last_solve_time = None
        self.last_status = None

    def decide(self, ctx):
        self.contexts.append(ctx)
        key = (ctx.day, ctx.minute)
        if key in self.schedule:
            self.current = np.asarray(self.schedule[key], dtype=float)
        self.last_solve_time = 1e-3
        self.last_status = "optimal"
        return self.current.copy()


def target(weights_by_asset: dict) -> np.ndarray:
    w = np.zeros(N_ASSETS)
    for i, v in weights_by_asset.items():
        w[i] = v
    return w


# ------------------------------------------------------------------ main flows
def test_scripted_five_day_run_identities_and_t1():
    market, signals = build_panel()
    cfg = MpoConfig()
    w_day0 = target({0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, HALTED_ASSET: 0.1})
    w_day2 = target({0: 0.05, 1: 0.05, 2: 0.1, 3: 0.1, 4: 0.1})
    strat = ScriptedStrategy({(0, 5): w_day0, (2, 5): w_day2})

    res = Simulator(market, cfg).run(strat, signals, (0, N_DAYS))

    # Shape and hard invariants (daily identity asserted inside the simulator).
    assert res.daily_nav.shape == (N_DAYS + 1,)
    assert np.all(np.isfinite(res.daily_nav))
    assert res.t1_violations == 0
    assert res.rejected_orders == 0
    assert res.truncated_buys == 0
    assert res.meta["final_cash"] >= 0.0

    # External re-check of the close mark: NAV = cash + holdings @ close px.
    close_px = market.prices[N_DAYS * MINUTES_PER_DAY - 1]
    nav_rebuilt = res.meta["final_cash"] + res.meta["final_holdings_shares"] @ close_px
    assert abs(res.daily_nav[-1] - nav_rebuilt) <= 1e-6 * res.daily_nav[-1]

    # Trading happened on the buy day and the rebalance (sell) day only.
    assert res.daily_turnover[0] > 0.05
    assert res.daily_turnover[2] > 0.0
    assert res.daily_turnover[1] == 0.0 and res.daily_turnover[3] == 0.0

    # Sells occurred (day 2) => stamp tax collected; buys => commission.
    assert res.costs["stamp_tax"] > 0.0
    assert res.costs["commission"] > 0.0
    assert res.costs["impact_est"] > 0.0

    # Halted asset: target 0.1 requested but trading frozen -> never held,
    # and its z-score is zeroed in every DecisionContext.
    assert res.meta["final_holdings_shares"][HALTED_ASSET] == 0.0
    for ctx in strat.contexts:
        assert np.all(ctx.z_signals[HALTED_ASSET] == 0.0)
        assert np.all(np.isfinite(ctx.z_signals))  # nan probe neutralized

    # QP telemetry harvested once per decide (8 epochs x 5 days).
    assert len(res.qp_solve_times) == 8 * N_DAYS
    assert res.qp_status_counts == {"optimal": 8 * N_DAYS}


def test_same_day_sell_blocked_then_allowed_next_day():
    """T+1: shares bought at epoch 1 are not sellable until the next day."""
    market, signals = build_panel()
    cfg = MpoConfig()
    strat = ScriptedStrategy(
        {(0, 5): target({0: 0.1, 1: 0.1}), (0, 35): target({})}
    )
    sim = Simulator(market, cfg)

    res1 = sim.run(strat, signals, (0, 1))
    # Same-day liquidation is silently infeasible: sellable=0, planner emits
    # nothing, no violation raised, no stamp tax, position survives the day.
    assert res1.t1_violations == 0
    assert res1.costs["stamp_tax"] == 0.0
    assert np.all(res1.meta["final_holdings_shares"][[0, 1]] > 0.0)

    # Fresh run over two days: day-close roll unlocks, day-1 sells succeed.
    strat2 = ScriptedStrategy(
        {(0, 5): target({0: 0.1, 1: 0.1}), (0, 35): target({})}
    )
    res2 = sim.run(strat2, signals, (0, 2))
    assert res2.costs["stamp_tax"] > 0.0
    assert np.all(res2.meta["final_holdings_shares"] == 0.0)


def test_oversell_bypassing_planner_raises(monkeypatch):
    """A sell injected past plan_orders' truncation must hit the ledger wall."""
    market, signals = build_panel()
    cfg = MpoConfig()
    calls = {"n": 0}

    def evil_plan(target_w, snapshot, prices, cfg, marginal_score=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [Order(asset=0, shares=400.0)]  # buy 400 shares
        return [Order(asset=0, shares=-400.0)]  # oversell same-day inventory

    monkeypatch.setattr(execution, "plan_orders", evil_plan)
    strat = ScriptedStrategy({(0, 5): target({0: 0.05})})
    with pytest.raises(T1ViolationError):
        Simulator(market, cfg).run(strat, signals, (0, 1))
    assert calls["n"] >= 2


def _starved_sell_panel():
    """Panel where day-1 selling of asset 0 is volume-starved but buying isn't.

    Reproduces the smoke crash shape: the planner funds a large buy out of the
    EXPECTED proceeds of a sell that the participation cap then guts.
    """
    market, signals = build_panel()
    amt = market.minute_amount.copy()
    day1 = slice(MINUTES_PER_DAY, 2 * MINUTES_PER_DAY)
    amt[day1, 0] = 1e3  # 30-min bar ~3e4 CNY -> 10% cap ~300 shares
    return dataclasses.replace(market, minute_amount=amt), signals


def _drain_then_rotate_strategy():
    # Day 0: spend nearly all cash. Day 1: rotate asset 0 -> asset 3, a buy that
    # is only affordable if the (starved) sell really pays out.
    return ScriptedStrategy(
        {
            (0, 5): target({0: 0.3, 1: 0.3, 2: 0.3}),
            (1, 5): target({0: 0.0, 1: 0.3, 2: 0.3, 3: 0.3}),
        }
    )


def test_starved_sell_never_overdraws_cash():
    """Regression (smoke seed 20260821, tau=4685): settlement cash stayed >= 0.

    Planning may size buys against expected same-epoch proceeds; the matcher
    must cap the buy fills by the cash actually banked, count the shortfall as a
    truncated buy, and keep every hard identity intact.
    """
    market, signals = _starved_sell_panel()
    res = Simulator(market, MpoConfig()).run(
        _drain_then_rotate_strategy(), signals, (0, 2)
    )

    assert res.meta["final_cash"] >= 0.0
    assert res.truncated_buys > 0  # buys cut by volume and/or the cash floor
    assert res.t1_violations == 0
    assert np.all(np.isfinite(res.daily_nav))
    assert np.all(res.daily_nav > 0.0)


def test_starved_sell_overdraws_without_the_cash_cap(monkeypatch):
    """The scenario above genuinely bites: drop the cap and cash goes negative.

    Guards the regression from rotting into a no-op if the cap is ever bypassed.
    """
    market, signals = _starved_sell_panel()
    real_match = execution.match_orders

    def match_without_cap(*args, **kwargs):
        kwargs.pop("cash_available", None)  # pre-fix behaviour
        return real_match(*args, **kwargs)

    monkeypatch.setattr(execution, "match_orders", match_without_cap)
    with pytest.raises(AssertionError, match="cash went negative"):
        Simulator(market, MpoConfig()).run(
            _drain_then_rotate_strategy(), signals, (0, 2)
        )


def test_buy_and_hold_zero_cost_leaks_no_money():
    """With cost_mult=0, NAV must exactly track marked holdings + cash."""
    market, signals = build_panel()
    cfg = dataclasses.replace(MpoConfig(), cost_mult=0.0)

    class BuyOnce:
        name = "buy_and_hold"

        def __init__(self):
            self.done = False

        def decide(self, ctx):
            if self.done:
                return None
            self.done = True
            return target({i: 0.12 for i in range(5)})

    res = Simulator(market, cfg).run(BuyOnce(), signals, (0, N_DAYS))

    assert np.all(res.daily_turnover[1:] == 0.0)
    for key in ("commission", "stamp_tax", "slippage"):
        assert res.costs[key] == 0.0

    # After day 0 the book is frozen: NAV_k = cash + h @ close_px(day k-1),
    # exact to float noise -- any leak in the accounting shows up here.
    cash = res.meta["final_cash"]
    h = res.meta["final_holdings_shares"]
    for k in range(1, N_DAYS + 1):
        close_px = market.prices[k * MINUTES_PER_DAY - 1]
        expected = cash + h @ close_px
        assert abs(res.daily_nav[k] - expected) <= 1e-9 * expected

    # Zero-cost buy at reference price: day-0 NAV is preserved too. The
    # reference price is the MIDPOINT of the first epoch's bar (tau=5,
    # tau_end=35 -> minute 20), not its end (was minute 35 before Iteration 43;
    # see _bar_mark_tau).
    assert abs(res.daily_nav[1] - (res.daily_nav[0] + h @ (
        market.prices[MINUTES_PER_DAY - 1] - market.prices[20]
    ))) <= 1e-6 * res.daily_nav[0]


@pytest.mark.parametrize("as_range", [False, True])
def test_multi_day_range_runs_every_day(as_range):
    """Regression: a ``range`` day_range must not collapse to a single day.

    ``backtest.run_seed`` passes ``range(start, end)``; indexing that as a pair
    yields ``(start, start+1)`` and silently simulated ONE day. Pin the shapes
    and that NAV actually moves day over day.
    """
    market, signals = build_panel()
    cfg = MpoConfig()
    start, end = 1, N_DAYS
    day_range = range(start, end) if as_range else (start, end)
    strat = ScriptedStrategy({(1, 5): target({i: 0.1 for i in range(5)})})

    res = Simulator(market, cfg).run(strat, signals, day_range)

    n_eval = end - start
    assert res.daily_nav.shape == (n_eval + 1,)
    assert res.daily_turnover.shape == (n_eval,)
    assert res.meta["day_range"] == (start, end)
    # decide() called on every epoch of every evaluated day.
    assert len(strat.contexts) == len(EPOCH_OFFSETS) * n_eval
    assert sorted({c.day for c in strat.contexts}) == list(range(start, end))
    assert summarize(res)["n_days"] == n_eval

    # Trading happened on the first evaluated day and NAV then keeps moving as
    # the held book is marked to each day's close (prices drift by design).
    assert res.daily_turnover[0] > 0.05
    assert np.all(np.isfinite(res.daily_nav))
    daily_change = np.diff(res.daily_nav)
    assert np.all(daily_change != 0.0)


def test_day_range_rejects_non_consecutive_and_bad_step():
    market, signals = build_panel()
    cfg = MpoConfig()
    strat = ScriptedStrategy({})
    with pytest.raises(ValueError):
        Simulator(market, cfg).run(strat, signals, range(0, N_DAYS, 2))
    with pytest.raises(ValueError):
        Simulator(market, cfg).run(strat, signals, [0, 1, 3])


# --------------------------------------------------------------------- metrics
def test_summarize_on_real_run():
    market, signals = build_panel()
    cfg = MpoConfig()
    strat = ScriptedStrategy(
        {(0, 5): target({i: 0.1 for i in range(5)}), (2, 5): target({})}
    )
    res = Simulator(market, cfg).run(strat, signals, (0, N_DAYS))
    m = summarize(res)

    assert m["strategy"] == "scripted"
    assert m["n_days"] == N_DAYS
    assert m["t1_violations"] == 0
    assert m["avg_daily_turnover"] > 0.0
    assert np.isfinite(m["net_sharpe_annual"])
    assert np.isfinite(m["ann_return"]) and m["ann_vol"] >= 0.0
    assert 0.0 <= m["max_drawdown"] < 1.0
    # Slippage is exactly cfg.slippage of traded value -> 5.0 bp by design.
    assert m["cost_bp_breakdown"]["slippage"] == pytest.approx(5.0, abs=1e-6)
    # Commission ~2bp (charged on slippage-adjusted notional).
    assert m["cost_bp_breakdown"]["commission"] == pytest.approx(2.0, abs=0.05)
    # Stamp tax on the sell leg only: strictly between 0 and 5bp of total.
    assert 0.0 < m["cost_bp_breakdown"]["stamp_tax"] < 5.0
    assert m["n_qp_solves"] == 8 * N_DAYS
    assert m["qp_time_p50_s"] == pytest.approx(1e-3)
    assert m["qp_time_p95_s"] == pytest.approx(1e-3)


def test_summarize_synthetic_result_drawdown_and_no_trades():
    nav = np.array([1.0, 1.1, 0.99, 1.05])
    res = BacktestResult(
        strategy="s",
        seed=0,
        daily_nav=nav,
        daily_turnover=np.zeros(3),
        costs={"commission": 0.0, "stamp_tax": 0.0, "slippage": 0.0, "impact_est": 0.0},
    )
    m = summarize(res)
    assert m["max_drawdown"] == pytest.approx(1.0 - 0.99 / 1.1)
    assert m["avg_daily_turnover"] == 0.0
    assert all(v == 0.0 for v in m["cost_bp_breakdown"].values())
    assert m["qp_time_p50_s"] == 0.0 and m["n_qp_solves"] == 0
    assert m["ann_return"] == pytest.approx(1.05 ** (240 / 3) - 1.0)


# ------------------------------------------------- raw QP status telemetry
# Regression guard for a convicted diagnosability defect: the simulator used to
# derive the QP status from ``fail_soft_count`` deltas, collapsing every
# non-optimal outcome into one "fail_soft" bucket. A 100%-reproducible OSQP
# ``primal infeasible`` (T+1 x w_max box conflict) therefore read as generic
# solver flakiness for a whole 6310s benchmark run. The raw status string and
# the backend's own note must both reach BacktestResult.

N_EPOCHS = 8 * N_DAYS  # decide() calls per full-panel run


class RawStatusStrategy:
    """Duck-typed strategy publishing the RAW solver verdict per epoch.

    Alternates optimal / failed-with-note starting at optimal, so a full run
    splits the epochs exactly in half.
    """

    name = "raw-status"

    def __init__(self, n_assets=N_ASSETS, note="primal infeasible"):
        self.n_assets = n_assets
        self.note = note
        self.calls = 0
        self.last_solve_time = None
        self.last_status = None
        self.last_note = None

    def decide(self, ctx):
        self.calls += 1
        self.last_solve_time = 1e-3
        if self.calls % 2 == 1:
            # An optimal solve may still carry a note; it must NOT be tallied
            # as a failure reason.
            self.last_status = "optimal"
            self.last_note = "osqp status=solved"
            return np.zeros(self.n_assets)
        self.last_status = "failed"
        self.last_note = self.note
        return None


class LegacyFailSoftStrategy:
    """Pre-telemetry strategy: fail_soft_count only, no last_status/last_note."""

    name = "legacy"

    def __init__(self, n_assets=N_ASSETS):
        self.n_assets = n_assets
        self.calls = 0
        self.fail_soft_count = 0
        self.last_solve_time = 0.0

    def decide(self, ctx):
        self.calls += 1
        self.last_solve_time = 1e-3
        if self.calls % 2 == 0:
            self.fail_soft_count += 1
            return None
        return np.zeros(self.n_assets)


def test_raw_qp_status_and_failure_notes_are_tallied_verbatim():
    market, signals = build_panel()
    strat = RawStatusStrategy()

    res = Simulator(market, MpoConfig()).run(strat, signals, (0, N_DAYS))

    half = N_EPOCHS // 2
    assert strat.calls == N_EPOCHS
    # Raw status strings, not a single collapsed bucket.
    assert res.qp_status_counts == {"optimal": half, "failed": half}
    # The backend's own reason survives, counted per distinct string. The note
    # attached to the *optimal* solves is deliberately absent.
    assert res.meta["qp_failure_notes"] == {"primal infeasible": half}


def test_failure_notes_distinguish_multiple_reasons():
    """Two distinct reasons must not merge into one count."""
    market, signals = build_panel()

    class TwoReasonStrategy(RawStatusStrategy):
        def decide(self, ctx):
            self.calls += 1
            self.last_solve_time = 1e-3
            if self.calls % 3 == 0:
                self.last_status = "failed"
                self.last_note = "osqp status=primal infeasible"
                return None
            if self.calls % 3 == 1:
                self.last_status = "inaccurate"
                self.last_note = "osqp status=solved inaccurate"
                return np.zeros(self.n_assets)
            self.last_status = "optimal"
            self.last_note = None
            return np.zeros(self.n_assets)

    strat = TwoReasonStrategy()
    res = Simulator(market, MpoConfig()).run(strat, signals, (0, N_DAYS))

    n_failed = sum(1 for c in range(1, N_EPOCHS + 1) if c % 3 == 0)
    n_inacc = sum(1 for c in range(1, N_EPOCHS + 1) if c % 3 == 1)
    n_opt = N_EPOCHS - n_failed - n_inacc
    assert res.qp_status_counts == {
        "optimal": n_opt,
        "inaccurate": n_inacc,
        "failed": n_failed,
    }
    # 'inaccurate' is non-optimal too, so its note is recorded as well.
    assert res.meta["qp_failure_notes"] == {
        "osqp status=primal infeasible": n_failed,
        "osqp status=solved inaccurate": n_inacc,
    }


def test_legacy_strategy_without_last_status_keeps_fail_soft_bucket():
    market, signals = build_panel()
    strat = LegacyFailSoftStrategy()
    assert not hasattr(strat, "last_status")  # fallback precondition

    res = Simulator(market, MpoConfig()).run(strat, signals, (0, N_DAYS))

    half = N_EPOCHS // 2
    assert res.qp_status_counts == {"optimal": half, "fail_soft": half}
    # No note source -> the key exists but stays empty (never absent).
    assert res.meta["qp_failure_notes"] == {}


def test_failure_notes_key_always_present_and_json_serializable():
    """Artifact contract: the key exists on clean runs and survives json."""
    import json

    market, signals = build_panel()
    clean = Simulator(market, MpoConfig()).run(
        ScriptedStrategy({}), signals, (0, N_DAYS)
    )
    assert clean.meta["qp_failure_notes"] == {}

    res = Simulator(market, MpoConfig()).run(RawStatusStrategy(), signals, (0, N_DAYS))
    payload = json.dumps(
        {
            "qp_status_counts": res.qp_status_counts,
            "qp_failure_notes": res.meta["qp_failure_notes"],
        }
    )
    assert json.loads(payload) == {
        "qp_status_counts": {"optimal": N_EPOCHS // 2, "failed": N_EPOCHS // 2},
        "qp_failure_notes": {"primal infeasible": N_EPOCHS // 2},
    }


def test_qp_strategy_records_raw_solver_status_and_note():
    """End of the wire: _QpStrategy._solve_qp must publish sol.status/sol.note."""
    strategies = pytest.importorskip("openpo.mpo.strategies")
    pytest.importorskip("openpo.mpo.qp")

    class FakeSolution:
        def __init__(self, status, weights, solve_time, note):
            self.status = status
            self.weights = weights
            self.solve_time = solve_time
            self.note = note

    class AlternatingQpStrategy(strategies._QpStrategy):
        """Real QP build path, stubbed solver alternating optimal/failed."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self._name = "alternating"
            self.calls = 0

        def _call_solver(self, solve_mod, qp):
            self.calls += 1
            if self.calls % 2 == 1:
                return FakeSolution(
                    "optimal", [np.zeros(self.n_assets)], 1e-3, "osqp status=solved"
                )
            return FakeSolution("failed", None, 2e-3, "osqp status=primal infeasible")

        def decide(self, ctx):
            return self._solve_single_stage(self._z_of(ctx)[:, 0] * 1e-4, ctx)

    market, signals = build_panel()
    cfg = MpoConfig()
    strat = AlternatingQpStrategy(cfg=cfg, market=market)

    # Base-class defaults before any solve.
    assert strat.last_status is None and strat.last_note is None

    res = Simulator(market, cfg).run(strat, signals, (0, N_DAYS))

    half = N_EPOCHS // 2
    assert strat.calls == N_EPOCHS
    # Raw solver strings straight through, plus unchanged fail-soft accounting.
    assert res.qp_status_counts == {"optimal": half, "failed": half}
    assert res.meta["qp_failure_notes"] == {"osqp status=primal infeasible": half}
    assert strat.fail_soft_count == half
    # Last epoch (even call) failed -> properties reflect the latest solve.
    assert strat.last_status == "failed"
    assert strat.last_note == "osqp status=primal infeasible"


# ------------------------------------- matching bar vs. cfg.exec_minutes
# Regression guard for a convicted parameter conflation: the simulator used
# ``cfg.exec_minutes`` as the order-matching bar length, while SPEC section 4
# fixes that bar at the decision-epoch pitch (30 min, "决策后 30 分钟 VWAP 近似",
# ``participation_rate * bar 成交量`` 30min aggregate). ``cfg.exec_minutes`` is
# the S1 alpha window of stages.py only. Sweeping exec=60 therefore silently
# mutated the execution environment (overlapping bars -> double-counted
# participation volume, longer drift before the fill mark), which drove most of
# the Iteration 38 exec=60 Sharpe collapse.
#
# ITERATION 43 SEMANTIC CHANGE (deliberate re-pin, NOT test-gaming): the price
# pins in this block used to assert ``bar_px == prices[tau_end]`` (the bar-END
# close). A 24-seed publication-scale shadow re-mark convicted that mark as a
# fidelity artifact -- 12.95 bp/unit of implementation-delay cost vs 7.19 bp/unit
# under the true amount-weighted VWAP (44.5% overstated, ~1.18%/yr NAV, more
# than the explicit 5 bp slippage it is charged on top of). The fill mark is now
# the window MIDPOINT ``(tau + tau_end) // 2``, within 0.146 bp of the true
# VWAP. The pins below therefore moved from ``tau_end`` to the midpoint ON
# PURPOSE; the VOLUME pins are untouched (still the full ``(tau+1 .. tau_end]``
# pool), as are the participation non-overlap, cash-floor, T+1 and
# exec_minutes bit-identity guards.

_BAR_MINUTES = EPOCH_OFFSETS[1] - EPOCH_OFFSETS[0]  # 30, the frozen pitch


class BuyOnceStrategy:
    """Emits one target on the first epoch, then ``None`` forever.

    Guarantees exactly one ``plan_orders``/``match_orders`` call per run, so the
    captured bar bounds are unambiguous.
    """

    name = "buy-once"

    def __init__(self, weights):
        self._w = np.asarray(weights, dtype=np.float64)
        self.calls = 0

    def decide(self, ctx):
        self.calls += 1
        return self._w.copy() if self.calls == 1 else None


def _ramped_volume_panel():
    """Panel whose minute volume strictly increases with tau.

    Every distinct window then has a distinct sum, so the asserted bar amount
    pins the exact slice bounds (not merely the window length).
    """
    market, signals = build_panel()
    t_total = market.minute_amount.shape[0]
    ramp = 5e6 + 1e3 * np.arange(t_total, dtype=np.float64)
    amt = np.repeat(ramp[:, None], N_ASSETS, axis=1)
    return dataclasses.replace(market, minute_amount=amt), signals


def _scarce_volume_panel():
    """Asset 0 is volume-starved: a 30-min bar funds ~3000 shares, a 60-min ~6000."""
    market, signals = build_panel()
    amt = market.minute_amount.copy()
    amt[:, 0] = 1e4  # CNY/minute
    return dataclasses.replace(market, minute_amount=amt), signals


def _spy_on_match_orders(monkeypatch):
    """Record the (bar_amount, bar_px) each ``match_orders`` call is handed."""
    real = execution.match_orders
    seen = []

    def spy(orders, bar_amount, prices, *args, **kwargs):
        seen.append(
            {
                "bar_amount": np.array(bar_amount, dtype=np.float64),
                "bar_px": np.array(prices, dtype=np.float64),
            }
        )
        return real(orders, bar_amount, prices, *args, **kwargs)

    monkeypatch.setattr(execution, "match_orders", spy)
    return seen


def test_epoch_bar_minutes_derives_from_the_grid_and_rejects_non_uniform():
    """The bar length is the epoch pitch, validated at import time."""
    assert simulator_mod._EPOCH_BAR_MINUTES == _BAR_MINUTES == 30
    assert simulator_mod._epoch_bar_minutes(EPOCH_OFFSETS) == 30
    assert simulator_mod._epoch_bar_minutes((0, 15, 30, 45)) == 15
    for bad in [(5,), (), (5, 35, 66, 95), (5, 5), (35, 5)]:
        with pytest.raises(ValueError):
            simulator_mod._epoch_bar_minutes(bad)


def test_bar_mark_tau_is_the_window_midpoint(monkeypatch):
    """Iteration 43: the fill mark is the window MIDPOINT minute, floor-rounded.

    Pins the exact index formula ``(tau + tau_end) // 2`` on literal cases --
    including a close-truncated window and an odd-length one, where the floor
    picks the EARLIER central minute -- and then checks the simulator really
    hands ``prices[midpoint]`` to ``match_orders`` for the day's LAST epoch,
    whose bar is truncated at the close (215 -> (215, 239], mark 227).
    """
    # (tau, tau_end) -> mark. Full 30-min bars, then the truncated last epoch
    # of day 0 (24 min) and day 1 (24 min), then odd-length windows.
    cases = {
        (5, 35): 20,
        (35, 65): 50,
        (185, 215): 200,
        (215, 239): 227,  # close-truncated: 24-minute window
        (455, 479): 467,  # same, day 1
        (5, 34): 19,  # odd length (29 min): floor -> earlier central minute
        (5, 32): 18,  # odd length (27 min)
    }
    for (tau, tau_end), mark in cases.items():
        assert simulator_mod._bar_mark_tau(tau, tau_end) == mark
        assert tau < mark < tau_end  # inside the traded window, strictly

    # Every epoch of the frozen grid: mark strictly inside its own bar, and the
    # bars of consecutive epochs keep their marks in increasing order.
    marks = []
    for offset in EPOCH_OFFSETS:
        tau = offset
        tau_end = min(tau + _BAR_MINUTES, MINUTES_PER_DAY - 1)
        mark = simulator_mod._bar_mark_tau(tau, tau_end)
        assert tau < mark < tau_end
        marks.append(mark)
    assert marks == sorted(marks) and len(set(marks)) == len(marks)
    assert marks[0] == 20 and marks[-1] == 227

    # End to end on the truncated last epoch: trade only at offset 215.
    market, signals = _ramped_volume_panel()
    seen = _spy_on_match_orders(monkeypatch)
    Simulator(market, MpoConfig()).run(
        ScriptedStrategy({(0, 215): target({0: 0.1, 1: 0.1})}), signals, (0, 1)
    )

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0]["bar_px"], market.prices[227])
    # Volume pool unchanged by Iteration 43: the whole (215, 239] window.
    np.testing.assert_array_equal(
        seen[0]["bar_amount"], market.minute_amount[216:240].sum(axis=0)
    )
    assert not np.allclose(seen[0]["bar_px"], market.prices[239])  # not bar-end


@pytest.mark.parametrize("exec_minutes", [30, 60, 120])
def test_matching_bar_is_the_epoch_pitch_not_exec_minutes(monkeypatch, exec_minutes):
    """Fills always reference the 30-min bar aggregates, whatever exec_minutes is.

    Volume = the full ``(tau, tau+30]`` pool; price = that window's MIDPOINT
    close (Iteration 43; the pin was ``prices[tau_end]`` before).
    """
    market, signals = _ramped_volume_panel()
    cfg = dataclasses.replace(MpoConfig(), exec_minutes=exec_minutes)
    seen = _spy_on_match_orders(monkeypatch)

    Simulator(market, cfg).run(
        BuyOnceStrategy(target({0: 0.1, 1: 0.1})), signals, (0, 1)
    )

    assert len(seen) == 1
    tau = EPOCH_OFFSETS[0]  # day 0, first epoch
    tau_end = tau + _BAR_MINUTES
    expected_amt = market.minute_amount[tau + 1 : tau_end + 1].sum(axis=0)
    np.testing.assert_array_equal(seen[0]["bar_amount"], expected_amt)
    np.testing.assert_array_equal(seen[0]["bar_px"], market.prices[(tau + tau_end) // 2])
    # ... and NOT the pre-Iteration-43 bar-end close (prices strictly drift, so
    # these are genuinely different vectors).
    assert not np.allclose(seen[0]["bar_px"], market.prices[tau_end])

    if exec_minutes != _BAR_MINUTES:  # the conflated window was visibly different
        conflated = market.minute_amount[tau + 1 : tau + exec_minutes + 1].sum(axis=0)
        assert not np.allclose(seen[0]["bar_amount"], conflated)
        assert not np.allclose(seen[0]["bar_px"], market.prices[tau + exec_minutes])


@pytest.mark.parametrize("exec_minutes", [30, 60])
def test_participation_cap_uses_non_overlapping_30min_volume(exec_minutes):
    """exec_minutes=60 must not double-count the next epoch's volume pool."""
    market, signals = _scarce_volume_panel()
    cfg = dataclasses.replace(MpoConfig(), exec_minutes=exec_minutes)

    res = Simulator(market, cfg).run(
        BuyOnceStrategy(target({0: 0.02})), signals, (0, 1)
    )

    tau = EPOCH_OFFSETS[0]
    tau_end = tau + _BAR_MINUTES
    # Cap = participation_rate * (full 30-min pool) / (MIDPOINT mark price).
    # The pool is unchanged by Iteration 43; only the price converting CNY to
    # shares moved from prices[tau_end] to the window midpoint.
    px = float(market.prices[(tau + tau_end) // 2, 0])
    pool_30 = float(market.minute_amount[tau + 1 : tau_end + 1, 0].sum())
    cap_30 = (
        np.floor(cfg.participation_rate * pool_30 / px / cfg.lot_size) * cfg.lot_size
    )
    assert cap_30 > 0.0
    assert res.meta["final_holdings_shares"][0] == pytest.approx(cap_30)
    assert res.truncated_buys == 1  # the cap really bit (order >> cap)

    # The bug would have filled off the overlapping 60-min pool: strictly more.
    pool_60 = float(market.minute_amount[tau + 1 : tau + 61, 0].sum())
    cap_60 = (
        np.floor(cfg.participation_rate * pool_60 / px / cfg.lot_size) * cfg.lot_size
    )
    assert cap_60 > cap_30


@pytest.mark.parametrize("exec_minutes", [60, 120])
def test_run_is_bit_identical_across_exec_minutes(exec_minutes):
    """The simulator no longer reads cfg.exec_minutes anywhere.

    With a strategy that ignores cfg, a run at exec_minutes=60/120 must be
    bit-for-bit the exec_minutes=30 run -- NAV, turnover, every cost bucket and
    both truncation counters. Uses the starved-sell panel so participation caps
    and the match-time cash floor are genuinely exercised.
    """
    market, signals = _starved_sell_panel()
    base = Simulator(market, MpoConfig()).run(
        _drain_then_rotate_strategy(), signals, (0, 2)
    )
    other = Simulator(
        market, dataclasses.replace(MpoConfig(), exec_minutes=exec_minutes)
    ).run(_drain_then_rotate_strategy(), signals, (0, 2))

    assert base.truncated_buys > 0  # the interesting paths were hit
    np.testing.assert_array_equal(base.daily_nav, other.daily_nav)
    np.testing.assert_array_equal(base.daily_turnover, other.daily_turnover)
    assert base.costs == other.costs
    assert base.truncated_buys == other.truncated_buys
    assert base.rejected_orders == other.rejected_orders
    np.testing.assert_array_equal(
        base.meta["final_holdings_shares"], other.meta["final_holdings_shares"]
    )
    assert base.meta["final_cash"] == other.meta["final_cash"]


def test_last_status_and_last_note_are_read_only():
    strategies = pytest.importorskip("openpo.mpo.strategies")
    strat = strategies.Strategy()
    assert strat.last_status is None and strat.last_note is None
    with pytest.raises(AttributeError):
        strat.last_status = "optimal"
    with pytest.raises(AttributeError):
        strat.last_note = "primal infeasible"
