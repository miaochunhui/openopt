"""Ledger boundary tests for openpo.mpo.ledger (SPEC section 4)."""

from __future__ import annotations

import numpy as np
import pytest

from openpo.mpo.ledger import T1Ledger
from openpo.mpo.types import T1ViolationError


def make_ledger(prev=(100.0, 0.0, 150.0)) -> T1Ledger:
    return T1Ledger(np.asarray(prev, dtype=np.float64))


class TestInitAndState:
    def test_sellable_equals_prev_close_at_open(self):
        led = make_ledger()
        np.testing.assert_allclose(led.sellable(), [100.0, 0.0, 150.0])
        np.testing.assert_allclose(led.sold_today, 0.0)
        np.testing.assert_allclose(led.bought_today, 0.0)

    def test_init_copies_input(self):
        prev = np.array([100.0, 0.0, 150.0])
        led = T1Ledger(prev)
        prev[0] = 999.0
        assert led.sellable()[0] == 100.0

    def test_init_rejects_negative_and_nonfinite(self):
        with pytest.raises(ValueError):
            T1Ledger(np.array([-1.0, 5.0]))
        with pytest.raises(ValueError):
            T1Ledger(np.array([np.nan, 5.0]))
        with pytest.raises(ValueError):
            T1Ledger(np.array([]))


class TestT1Blocking:
    def test_buy_today_cannot_be_sold_today(self):
        """昨买今卖拦截: shares bought today stay locked until day close."""
        led = make_ledger()
        led.on_buy_fill(1, 200.0)
        assert led.bought_today[1] == 200.0
        assert led.sellable()[1] == 0.0  # buy did not unlock anything
        with pytest.raises(T1ViolationError):
            led.on_sell_fill(1, 100.0)

    def test_oversell_beyond_prev_close_raises(self):
        led = make_ledger()
        with pytest.raises(T1ViolationError):
            led.on_sell_fill(0, 100.0 + 1e-6)

    def test_tiny_float_tolerance_accepted(self):
        led = make_ledger()
        led.on_sell_fill(0, 100.0 + 5e-10)  # within 1e-9 tolerance
        assert led.sellable()[0] == pytest.approx(0.0, abs=1e-9)

    def test_sellable_never_negative(self):
        led = make_ledger()
        led.on_sell_fill(0, 100.0 + 5e-10)
        assert np.all(led.sellable() >= 0.0)


class TestFillAccounting:
    def test_sells_accumulate_and_shrink_sellable(self):
        led = make_ledger()
        led.on_sell_fill(0, 40.0)
        led.on_sell_fill(0, 25.0)
        assert led.sold_today[0] == 65.0
        assert led.sellable()[0] == 35.0

    def test_cancelled_residual_does_not_lock_inventory(self):
        """撤单不锁库存: only fills debit; a cancelled remainder stays sellable."""
        led = make_ledger()
        # Epoch 1: order for 100 fills only 30, residual 70 cancelled.
        led.on_sell_fill(0, 30.0)
        assert led.sellable()[0] == 70.0
        # Epoch 2: the remaining 70 are still sellable and fill fine.
        led.on_sell_fill(0, 70.0)
        assert led.sellable()[0] == 0.0
        with pytest.raises(T1ViolationError):
            led.on_sell_fill(0, 1.0)

    def test_full_clear_with_odd_lot_tail(self):
        """清仓尾零股: selling the exact 150-share position (odd 50 tail) is legal."""
        led = make_ledger()
        led.on_sell_fill(2, 150.0)
        assert led.sellable()[2] == 0.0
        with pytest.raises(T1ViolationError):
            led.on_sell_fill(2, 1.0)


class TestDayClose:
    def test_next_day_unlocks_buys_and_resets_counters(self):
        """次日解锁: day-close roll makes today's buys sellable tomorrow."""
        led = make_ledger()
        led.on_buy_fill(1, 200.0)
        led.on_sell_fill(0, 100.0)
        holdings = np.array([0.0, 200.0, 150.0])
        led.on_day_close(holdings)
        np.testing.assert_allclose(led.sellable(), holdings)
        np.testing.assert_allclose(led.sold_today, 0.0)
        np.testing.assert_allclose(led.bought_today, 0.0)
        led.on_sell_fill(1, 200.0)  # yesterday's buy now sellable

    def test_day_close_validates_input(self):
        led = make_ledger()
        with pytest.raises(ValueError):
            led.on_day_close(np.array([1.0, 2.0]))  # wrong size
        with pytest.raises(ValueError):
            led.on_day_close(np.array([1.0, -5.0, 0.0]))
        with pytest.raises(ValueError):
            led.on_day_close(np.array([1.0, np.inf, 0.0]))


class TestInterfaceHygiene:
    def test_snapshot_is_detached_copy(self):
        led = make_ledger()
        snap = led.snapshot()
        np.testing.assert_allclose(snap["sellable"], [100.0, 0.0, 150.0])
        snap["prev_close_shares"][0] = 0.0
        snap["sold_today"][0] = 99.0
        assert led.sellable()[0] == 100.0  # ledger unaffected

    def test_property_copies_are_detached(self):
        led = make_ledger()
        led.sold_today[0] = 99.0  # mutating the returned copy
        assert led.sold_today[0] == 0.0

    def test_invalid_fill_arguments(self):
        led = make_ledger()
        with pytest.raises(ValueError):
            led.on_sell_fill(0, -10.0)
        with pytest.raises(ValueError):
            led.on_buy_fill(0, 0.0)
        with pytest.raises(ValueError):
            led.on_buy_fill(0, np.nan)
        with pytest.raises(IndexError):
            led.on_buy_fill(3, 100.0)
