"""Synthetic trading calendar for MPO-T1 (see docs/mpo-t1-SPEC.md section 0).

Conventions (frozen, shared by all modules):

- A trading day has ``MINUTES_PER_DAY = 240`` minute bars, minute index in
  ``[0, 240)``. Global minute index ``tau = day * 240 + minute``.
- The calendar is synthetic: 5 days a week, **no holidays**, so ``tau`` itself
  is a trading-minute counter and the number of trading minutes between two
  taus is simply their difference: ``trading_minutes_between(a, b) = b - a``.
  Overnight gaps therefore count as **0 trading minutes** by construction
  (the last bar of day ``d`` is ``d*240 + 239`` and the first bar of day
  ``d+1`` is ``(d+1)*240``, one trading minute later).
- The "day close" tau of day ``d`` is the tau of its **last minute bar**,
  ``d*240 + 239``. Half-open stage intervals ``(start, end]`` that end at the
  close include that last bar.
- Decision epochs: 8 per day at minute offsets ``config.EPOCH_OFFSETS``
  ``(5, 35, 65, 95, 125, 155, 185, 215)``.

All functions are pure and integer-exact; array outputs are int64.
"""

from __future__ import annotations

import numpy as np

from openpo.mpo.config import EPOCH_OFFSETS, MINUTES_PER_DAY

# Fast membership test for is_epoch; EPOCH_OFFSETS is frozen in config.
_EPOCH_OFFSET_SET = frozenset(EPOCH_OFFSETS)


def day_of(tau: int) -> int:
    """Trading-day index containing global minute ``tau`` (floor division)."""
    return int(tau) // MINUTES_PER_DAY


def minute_of(tau: int) -> int:
    """Minute-of-day in ``[0, 240)`` for global minute ``tau``."""
    return int(tau) % MINUTES_PER_DAY


def day_close_tau(day: int) -> int:
    """Tau of the last minute bar (the close) of trading day ``day``.

    Equals ``day*240 + 239``. A stage ``(start, end]`` with
    ``end == day_close_tau(d)`` ends exactly at, and includes, the close bar
    of day ``d``; the immediately following stage starts on the next day.
    """
    return int(day) * MINUTES_PER_DAY + (MINUTES_PER_DAY - 1)


def trading_minutes_between(tau_a: int, tau_b: int) -> int:
    """Trading minutes elapsed from ``tau_a`` to ``tau_b``.

    With the synthetic no-holiday calendar this is exactly ``tau_b - tau_a``:
    overnight contributes 0 trading minutes because consecutive taus across a
    day boundary differ by 1. Kept as a named function so the convention is
    explicit at call sites (horizons/decay are all in trading minutes).
    """
    return int(tau_b) - int(tau_a)


def epoch_taus(n_days: int) -> np.ndarray:
    """All decision-epoch taus for a calendar of ``n_days`` days.

    Returns an int64 array of shape ``[n_days * len(EPOCH_OFFSETS)]``, sorted
    ascending: ``day*240 + offset`` for every day and every offset in
    ``config.EPOCH_OFFSETS``.
    """
    if n_days < 0:
        raise ValueError(f"n_days must be >= 0, got {n_days}")
    days = np.arange(n_days, dtype=np.int64) * MINUTES_PER_DAY
    offsets = np.asarray(EPOCH_OFFSETS, dtype=np.int64)
    return (days[:, None] + offsets[None, :]).ravel()


def is_epoch(tau: int) -> bool:
    """True iff ``tau`` falls on a decision epoch (minute offset in EPOCH_OFFSETS)."""
    return minute_of(tau) in _EPOCH_OFFSET_SET
