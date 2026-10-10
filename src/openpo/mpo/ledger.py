"""T+1 share ledger: independent hard enforcement layer (SPEC section 4).

The ledger tracks share inventory in SHARE units, deliberately decoupled from
the QP's weight-space T+1 constraints, so any violation upstream (planner bug,
rounding drift, race between orders) is caught at fill time.

Key conventions (project-wide, see docs/mpo-t1-SPEC.md):

- Trading minutes are plain tau differences; overnight counts as ZERO minutes
  (``trading_minutes_between(a, b) == b - a`` on the synthetic no-holiday
  calendar). The ledger itself is day-granular: state rolls only at
  :meth:`T1Ledger.on_day_close`.
- A-share T+1: shares bought today are locked until the next day-close roll.
  ``sellable = max(prev_close_shares - sold_today, 0)``.
- Orders never lock inventory -- only *fills* do. A cancelled remainder
  (partial fill + residual cancel) therefore leaves the unfilled shares
  sellable in later epochs of the same day.
"""

from __future__ import annotations

import numpy as np

from .types import T1ViolationError

# Absolute share tolerance for sell-fill validation (float round-trip noise).
_SELL_TOL = 1e-9


class T1Ledger:
    """Per-asset T+1 inventory ledger in share units.

    Parameters
    ----------
    prev_close_shares:
        [N] holdings at the previous day close. These are the only shares
        sellable today (net of what was already sold today).
    """

    def __init__(self, prev_close_shares: np.ndarray) -> None:
        prev = np.array(prev_close_shares, dtype=np.float64, copy=True).ravel()
        if prev.size == 0:
            raise ValueError("prev_close_shares must be non-empty")
        if not np.all(np.isfinite(prev)):
            raise ValueError("prev_close_shares must be finite")
        if np.any(prev < 0):
            raise ValueError("prev_close_shares must be non-negative")
        self._prev_close_shares = prev
        self._sold_today = np.zeros_like(prev)
        self._bought_today = np.zeros_like(prev)

    # ------------------------------------------------------------------ state
    @property
    def n_assets(self) -> int:
        return int(self._prev_close_shares.size)

    @property
    def prev_close_shares(self) -> np.ndarray:
        """[N] copy of previous-day close holdings."""
        return self._prev_close_shares.copy()

    @property
    def sold_today(self) -> np.ndarray:
        """[N] copy of shares sold (filled) so far today."""
        return self._sold_today.copy()

    @property
    def bought_today(self) -> np.ndarray:
        """[N] copy of shares bought (filled) so far today; locked until roll."""
        return self._bought_today.copy()

    def sellable(self) -> np.ndarray:
        """[N] shares sellable right now: max(prev_close - sold_today, 0)."""
        return np.maximum(self._prev_close_shares - self._sold_today, 0.0)

    def snapshot(self) -> dict:
        """Detached copy of full ledger state (safe to mutate/serialize)."""
        return {
            "prev_close_shares": self._prev_close_shares.copy(),
            "sold_today": self._sold_today.copy(),
            "bought_today": self._bought_today.copy(),
            "sellable": self.sellable(),
        }

    # ------------------------------------------------------------------ fills
    def _check_asset(self, asset: int) -> int:
        i = int(asset)
        if not 0 <= i < self.n_assets:
            raise IndexError(f"asset {asset} out of range [0, {self.n_assets})")
        return i

    @staticmethod
    def _check_shares(shares: float) -> float:
        q = float(shares)
        if not np.isfinite(q) or q <= 0.0:
            raise ValueError(f"fill shares must be a positive finite number, got {shares!r}")
        return q

    def on_sell_fill(self, asset: int, shares: float) -> None:
        """Record a sell fill; raises T1ViolationError if it exceeds sellable.

        Called at FILL time (not order time), so cancelled residuals never
        consume inventory.
        """
        i = self._check_asset(asset)
        q = self._check_shares(shares)
        avail = max(self._prev_close_shares[i] - self._sold_today[i], 0.0)
        if q > avail + _SELL_TOL:
            raise T1ViolationError(
                f"T+1 violation: sell {q:.6f} shares of asset {i} but only "
                f"{avail:.6f} sellable (prev_close={self._prev_close_shares[i]:.6f}, "
                f"sold_today={self._sold_today[i]:.6f})"
            )
        self._sold_today[i] += q

    def on_buy_fill(self, asset: int, shares: float) -> None:
        """Record a buy fill; the shares are locked until the next day close."""
        i = self._check_asset(asset)
        q = self._check_shares(shares)
        self._bought_today[i] += q

    # ------------------------------------------------------------------ roll
    def on_day_close(self, holdings_shares: np.ndarray) -> None:
        """Roll to the next day: unlock today's buys, reset daily counters.

        Parameters
        ----------
        holdings_shares:
            [N] end-of-day holdings from the accounting layer (authoritative;
            the ledger does not reconstruct them from fills on purpose, so a
            reconciliation mismatch upstream surfaces immediately).
        """
        h = np.array(holdings_shares, dtype=np.float64, copy=True).ravel()
        if h.size != self.n_assets:
            raise ValueError(f"holdings size {h.size} != n_assets {self.n_assets}")
        if not np.all(np.isfinite(h)):
            raise ValueError("holdings_shares must be finite")
        if np.any(h < -_SELL_TOL):
            raise ValueError("holdings_shares must be non-negative")
        self._prev_close_shares = np.maximum(h, 0.0)
        self._sold_today = np.zeros_like(self._prev_close_shares)
        self._bought_today = np.zeros_like(self._prev_close_shares)
