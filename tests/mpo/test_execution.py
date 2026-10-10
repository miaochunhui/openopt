"""Execution-layer tests for openpo.mpo.execution (SPEC section 4)."""

from __future__ import annotations

import numpy as np
import pytest

from openpo.mpo.config import MpoConfig
from openpo.mpo.execution import match_orders, plan_orders
from openpo.mpo.ledger import T1Ledger
from openpo.mpo.types import Order, PortfolioSnapshot, T1ViolationError

CFG = MpoConfig()  # lot 100, participation 0.10, slippage 5e-4, band 2bp


def make_snapshot(holdings, sellable, prices, nav=1e6, cash=None):
    holdings = np.asarray(holdings, dtype=np.float64)
    sellable = np.asarray(sellable, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)
    weights = holdings * prices / nav
    if cash is None:
        cash = nav - float(holdings @ prices)
    return PortfolioSnapshot(
        tau=5,
        nav=nav,
        cash=float(cash),
        holdings_shares=holdings,
        weights=weights,
        sellable_shares=sellable,
        sellable_weights=sellable * prices / nav,
    )


def no_flags(n):
    z = np.zeros(n, dtype=bool)
    return z.copy(), z.copy(), z.copy()  # limit_up, limit_down, halted


class TestPlanOrders:
    def test_buy_rounds_down_to_lots(self):
        snap = make_snapshot([0.0], [0.0], [10.0])
        # desired = 0.01234 * 1e6 / 10 = 1234 shares -> 1200 (12 lots)
        orders = plan_orders(np.array([0.01234]), snap, [10.0], CFG)
        assert orders == [Order(asset=0, shares=1200.0)]

    def test_no_trade_band_drops_sub_lot_buy(self):
        snap = make_snapshot([0.0], [0.0], [10.0])
        # desired = 90 shares < 1 lot -> dropped
        assert plan_orders(np.array([0.0009]), snap, [10.0], CFG) == []

    def test_no_trade_band_nav_fraction(self):
        # price 1 CNY: 1 lot = 100 CNY but 2bp of NAV = 200 CNY dominates.
        snap = make_snapshot([0.0], [0.0], [1.0])
        assert plan_orders(np.array([0.00015]), snap, [1.0], CFG) == []  # 150 CNY
        orders = plan_orders(np.array([0.0003]), snap, [1.0], CFG)  # 300 CNY
        assert orders == [Order(asset=0, shares=300.0)]

    def test_sell_capped_at_sellable_and_lot_rounded(self):
        # want to shed 600 shares but only 350 sellable -> 300 (lot floor).
        snap = make_snapshot([1000.0], [350.0], [10.0])
        orders = plan_orders(np.array([0.004]), snap, [10.0], CFG)
        assert orders == [Order(asset=0, shares=-300.0)]

    def test_full_clear_sells_odd_lot_tail(self):
        snap = make_snapshot([150.0], [150.0], [10.0])
        orders = plan_orders(np.array([0.0]), snap, [10.0], CFG)
        assert orders == [Order(asset=0, shares=-150.0)]

    def test_full_clear_exempt_from_band(self):
        # 50 shares @ 1 CNY = 50 CNY < band 200 CNY, but clearing is allowed.
        snap = make_snapshot([50.0], [50.0], [1.0])
        orders = plan_orders(np.array([0.0]), snap, [1.0], CFG)
        assert orders == [Order(asset=0, shares=-50.0)]

    def test_partial_sell_below_band_dropped(self):
        # partial (non-clearing) sell of 100 CNY < 200 CNY band -> dropped.
        snap = make_snapshot([1000.0], [1000.0], [1.0])
        assert plan_orders(np.array([0.0009]), snap, [1.0], CFG) == []

    def test_cash_truncation_default_priority(self):
        snap = make_snapshot([0.0, 0.0], [0.0, 0.0], [10.0, 10.0], cash=15000.0)
        orders = plan_orders(np.array([0.02, 0.01]), snap, [10.0, 10.0], CFG)
        # asset 0 has the larger |target deviation| -> funded first, shrunk to
        # the largest affordable lot count; asset 1 cannot afford one lot.
        assert len(orders) == 1
        assert orders[0].asset == 0
        assert orders[0].shares == 1400.0  # floor(15000/(1000*(1+7bp))) lots
        spend = orders[0].shares * 10.0 * (1.0 + CFG.buy_cost())
        assert spend <= snap.cash

    def test_cash_truncation_marginal_score_override(self):
        snap = make_snapshot([0.0, 0.0], [0.0, 0.0], [10.0, 10.0], cash=15000.0)
        orders = plan_orders(
            np.array([0.02, 0.01]), snap, [10.0, 10.0], CFG,
            marginal_score=np.array([0.1, 0.9]),
        )
        got = {o.asset: o.shares for o in orders}
        assert got[1] == 1000.0  # highest score funded in full
        assert got[0] == 400.0  # remainder buys 4 lots
        spend = sum(o.shares * 10.0 for o in orders) * (1.0 + CFG.buy_cost())
        assert spend <= snap.cash

    def test_same_epoch_sell_proceeds_fund_buys(self):
        # cash 0: the buy is only affordable using same-epoch sell proceeds.
        snap = make_snapshot([1000.0, 0.0], [1000.0, 0.0], [10.0, 10.0], cash=0.0)
        orders = plan_orders(np.array([0.0, 0.009]), snap, [10.0, 10.0], CFG)
        got = {o.asset: o.shares for o in orders}
        assert got[0] == -1000.0
        assert got[1] == 900.0
        proceeds = 1000.0 * 10.0 * (1.0 - CFG.sell_cost())
        assert 900.0 * 10.0 * (1.0 + CFG.buy_cost()) <= proceeds
        assert orders[0].shares < 0  # sells listed before buys

    def test_bad_price_skipped(self):
        snap = make_snapshot([0.0, 0.0], [0.0, 0.0], [10.0, 10.0])
        orders = plan_orders(
            np.array([0.01, 0.01]), snap, [np.nan, 10.0], CFG
        )
        assert [o.asset for o in orders] == [1]


class TestMatchOrders:
    def test_limit_up_rejects_buy(self):
        led = T1Ledger(np.zeros(1))
        lup, ldn, hlt = no_flags(1)
        lup[0] = True
        fills, rejected = match_orders(
            [Order(0, 500.0)], [1e6], [10.0], lup, ldn, hlt, led, CFG
        )
        assert fills == [] and rejected == 1
        assert led.bought_today[0] == 0.0

    def test_limit_down_rejects_sell_but_not_buy(self):
        led = T1Ledger(np.array([1000.0]))
        lup, ldn, hlt = no_flags(1)
        ldn[0] = True
        fills, rejected = match_orders(
            [Order(0, -500.0)], [1e6], [10.0], lup, ldn, hlt, led, CFG
        )
        assert fills == [] and rejected == 1
        fills, rejected = match_orders(
            [Order(0, 500.0)], [1e6], [10.0], lup, ldn, hlt, led, CFG
        )
        assert rejected == 0 and fills[0].shares == 500.0

    def test_halted_rejects_both_sides(self):
        led = T1Ledger(np.array([1000.0]))
        lup, ldn, hlt = no_flags(1)
        hlt[0] = True
        fills, rejected = match_orders(
            [Order(0, 500.0), Order(0, -500.0)],
            [1e6], [10.0], lup, ldn, hlt, led, CFG,
        )
        assert fills == [] and rejected == 2

    def test_participation_partial_fill_buy_lot_aligned(self):
        led = T1Ledger(np.zeros(1))
        # cap = 0.10 * 50_000 / 10 = 500 shares; residual 500 cancelled.
        fills, rejected = match_orders(
            [Order(0, 1000.0)], [50_000.0], [10.0], *no_flags(1), led, CFG
        )
        assert rejected == 0
        assert len(fills) == 1 and fills[0].shares == 500.0
        assert led.bought_today[0] == 500.0

    def test_participation_partial_fill_sell_allows_odd(self):
        led = T1Ledger(np.array([2000.0]))
        # cap = 0.10 * 53_000 / 10 = 530 shares (odd fill fine for sells).
        fills, _ = match_orders(
            [Order(0, -1000.0)], [53_000.0], [10.0], *no_flags(1), led, CFG
        )
        assert fills[0].shares == -530.0
        assert led.sold_today[0] == 530.0
        assert led.sellable()[0] == 1470.0  # residual cancelled, NOT locked

    def test_volume_starved_buy_is_not_a_rejection(self):
        led = T1Ledger(np.zeros(1))
        # cap = 0.10 * 9_000 / 10 = 90 shares < 1 lot -> zero fill, no reject.
        fills, rejected = match_orders(
            [Order(0, 1000.0)], [9_000.0], [10.0], *no_flags(1), led, CFG
        )
        assert fills == [] and rejected == 0

    def test_buy_costs_sign_and_magnitude(self):
        led = T1Ledger(np.zeros(1))
        fills, _ = match_orders(
            [Order(0, 500.0)], [1e8], [10.0], *no_flags(1), led, CFG
        )
        f = fills[0]
        assert f.price == pytest.approx(10.0 * (1.0 + 5e-4))  # buys pay up
        assert f.commission == pytest.approx(500.0 * f.price * 2e-4)
        assert f.stamp_tax == 0.0  # no stamp tax on buys
        assert f.slippage_cost == pytest.approx(500.0 * 10.0 * 5e-4)

    def test_sell_costs_sign_and_magnitude(self):
        led = T1Ledger(np.array([1000.0]))
        fills, _ = match_orders(
            [Order(0, -500.0)], [1e8], [10.0], *no_flags(1), led, CFG
        )
        f = fills[0]
        assert f.price == pytest.approx(10.0 * (1.0 - 5e-4))  # sells receive less
        assert f.price < 10.0 < 10.0 * (1.0 + 5e-4)
        assert f.commission == pytest.approx(500.0 * f.price * 2e-4)
        assert f.stamp_tax == pytest.approx(500.0 * f.price * 5e-4)
        assert f.slippage_cost == pytest.approx(500.0 * 10.0 * 5e-4)
        assert f.commission > 0 and f.slippage_cost > 0

    def test_cost_mult_scales_all_costs(self):
        cfg2 = MpoConfig(cost_mult=2.0)
        led = T1Ledger(np.zeros(1))
        fills, _ = match_orders(
            [Order(0, 500.0)], [1e8], [10.0], *no_flags(1), led, cfg2
        )
        f = fills[0]
        assert f.price == pytest.approx(10.0 * (1.0 + 2 * 5e-4))
        assert f.slippage_cost == pytest.approx(500.0 * 10.0 * 2 * 5e-4)
        assert f.commission == pytest.approx(500.0 * f.price * 2 * 2e-4)

    def test_ledger_hard_check_at_fill_time(self):
        # A rogue over-sized sell must be stopped by the ledger, not silently
        # clipped: the ledger is the independent hard T+1 check.
        led = T1Ledger(np.array([300.0]))
        with pytest.raises(T1ViolationError):
            match_orders(
                [Order(0, -500.0)], [1e8], [10.0], *no_flags(1), led, CFG
            )

    def test_sequential_sells_debit_at_fill_time(self):
        led = T1Ledger(np.array([500.0]))
        fills, _ = match_orders(
            [Order(0, -300.0)], [1e8], [10.0], *no_flags(1), led, CFG
        )
        assert led.sellable()[0] == 200.0
        with pytest.raises(T1ViolationError):
            match_orders(
                [Order(0, -300.0)], [1e8], [10.0], *no_flags(1), led, CFG
            )


def settle(fills, cash_open):
    """Replay the simulator's cash settlement over ``fills`` (SPEC section 5)."""
    cash = float(cash_open)
    for f in fills:
        if f.shares > 0:
            cash -= f.shares * f.price + f.commission
        else:
            cash += (-f.shares) * f.price - f.commission - f.stamp_tax
    return cash


def count_truncated_buys(orders, fills):
    """Replay the simulator's truncated-buy tally (filled < ordered)."""
    filled = {}
    for f in fills:
        filled[f.asset] = filled.get(f.asset, 0.0) + f.shares
    return sum(
        1 for o in orders if o.shares > 0 and filled.get(o.asset, 0.0) < o.shares - 1e-9
    )


class TestCashSettlementFloor:
    """Buys may never settle for more cash than the epoch actually has.

    ``plan_orders`` funds buys from ``cash + EXPECTED same-epoch sell proceeds``
    priced at the decision minute. The bar can invalidate that plan (sells cut
    by the participation cap / limit-down / halt, prices drifting between the
    decision minute and the fill mark), so ``match_orders`` settles sells first
    and truncates buys to the cash on hand.
    """

    # cash 0, sell 1000 sh @10 (planned proceeds ~9988) funding a 900 sh buy,
    # but the sell's bar only supports 0.10 * 30_000 / 10 = 300 shares.
    ORDERS = [Order(asset=0, shares=-1000.0), Order(asset=1, shares=900.0)]
    BAR = [30_000.0, 1e9]  # asset 0 volume-starved, asset 1 unconstrained
    PX = [10.0, 10.0]

    def test_naive_settlement_would_overdraw(self):
        """Baseline (cap disabled): this scenario really does go cash-negative."""
        led = T1Ledger(np.array([1000.0, 0.0]))
        fills, rejected = match_orders(
            self.ORDERS, self.BAR, self.PX, *no_flags(2), led, CFG
        )
        assert rejected == 0
        assert {f.asset: f.shares for f in fills} == {0: -300.0, 1: 900.0}
        assert settle(fills, 0.0) < -6000.0  # ~ -6009.9 CNY

    def test_cash_cap_truncates_buy_and_keeps_cash_non_negative(self):
        led = T1Ledger(np.array([1000.0, 0.0]))
        fills, rejected = match_orders(
            self.ORDERS, self.BAR, self.PX, *no_flags(2), led, CFG,
            cash_available=0.0,
        )
        assert rejected == 0  # cash starvation is a cancel, never a rejection
        assert fills[0].shares == -300.0  # sell settles first
        buy = fills[1]
        assert buy.asset == 1
        # 300 sh sold nets ~2996.4 CNY -> 2 lots affordable at 10.005 + 2bp comm.
        assert buy.shares == 200.0
        assert buy.shares % CFG.lot_size == 0
        assert settle(fills, 0.0) >= 0.0
        assert count_truncated_buys(self.ORDERS, fills) == 1

    def test_zero_cash_drops_buy_entirely(self):
        """No proceeds at all -> the buy is cancelled, not partially settled."""
        led = T1Ledger(np.zeros(1))
        fills, rejected = match_orders(
            [Order(asset=0, shares=500.0)], [1e9], [10.0], *no_flags(1), led, CFG,
            cash_available=0.0,
        )
        assert fills == [] and rejected == 0
        assert led.bought_today[0] == 0.0
        assert count_truncated_buys([Order(asset=0, shares=500.0)], fills) == 1

    def test_cap_spends_in_order_given(self):
        """Buys are served in planning (alpha priority) order; the tail is cut."""
        led = T1Ledger(np.zeros(3))
        orders = [Order(0, 500.0), Order(1, 500.0), Order(2, 500.0)]
        cash = 3.0 * 100.0 * 10.0 * (1.0 + 5e-4) * (1.0 + 2e-4) + 1.0
        fills, _ = match_orders(
            orders, [1e9] * 3, [10.0] * 3, *no_flags(3), led, CFG,
            cash_available=cash,
        )
        got = {f.asset: f.shares for f in fills}
        assert got == {0: 300.0}  # 3 affordable lots all go to the first buy
        assert settle(fills, cash) >= 0.0
        assert count_truncated_buys(orders, fills) == 3

    def test_ample_cash_leaves_fills_untouched(self):
        """The cap is a floor, not a tax: a funded plan fills exactly as before."""
        led_a, led_b = T1Ledger(np.zeros(1)), T1Ledger(np.zeros(1))
        args = ([Order(0, 500.0)], [1e9], [10.0], *no_flags(1))
        base, _ = match_orders(*args, led_a, CFG)
        capped, _ = match_orders(*args, led_b, CFG, cash_available=1e6)
        assert base == capped

    def test_sells_settle_before_buys_regardless_of_order_list(self):
        led = T1Ledger(np.array([0.0, 1000.0]))
        orders = [Order(asset=0, shares=900.0), Order(asset=1, shares=-1000.0)]
        fills, _ = match_orders(
            orders, [1e9, 1e9], [10.0, 10.0], *no_flags(2), led, CFG,
            cash_available=0.0,
        )
        assert [f.asset for f in fills] == [1, 0]  # sell first even though listed 2nd
        assert fills[0].shares == -1000.0
        assert fills[1].shares > 0.0  # the buy is funded by those proceeds
        assert settle(fills, 0.0) >= 0.0


class TestPlanMatchRoundTrip:
    def test_planned_orders_always_match_ledger_limits(self):
        rng = np.random.default_rng(7)
        n = 20
        prices = rng.uniform(5.0, 50.0, n)
        holdings = rng.integers(0, 50, n).astype(float) * 100.0
        sellable = np.floor(holdings * rng.uniform(0.0, 1.0, n))
        nav = 1e6
        snap = make_snapshot(holdings, sellable, prices, nav=nav)
        target = np.abs(rng.normal(0.0, 0.01, n))
        target /= max(target.sum(), 1.0)  # keep budget sane
        orders = plan_orders(target, snap, prices, CFG)
        led = T1Ledger(sellable)  # ledger sellable == snapshot sellable
        bar_amount = rng.uniform(1e4, 1e7, n)
        fills, rejected = match_orders(
            orders, bar_amount, prices, *no_flags(n), led, CFG
        )
        assert rejected == 0  # no limits/halts in this scenario
        for f in fills:
            if f.shares > 0:
                assert f.shares % CFG.lot_size == 0
        # every sell passed the T+1 hard check (no raise) and totals reconcile
        assert np.all(led.sold_today <= sellable + 1e-9)
