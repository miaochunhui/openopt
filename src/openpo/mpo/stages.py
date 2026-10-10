"""Stage grid construction for MPO-T1 (docs/mpo-t1-SPEC.md section 1).

Builds the (at most) 4-stage look-ahead grid of half-open trading-minute
intervals ``(start, end]`` used by the multi-period QP:

- S1 execution stage: ``(tau_now, min(tau_now + cfg.exec_minutes, today's close)]``
- S2 rest of today:   ``(S1.end, today's close]`` -- dropped when empty
- S3 next full day:   240 minutes
- S4 aggregate tail:  up to ``tau_now + cfg.horizon_minutes`` (trading
  minutes), truncated at the calendar end; dropped when empty.

Time conventions (see :mod:`openpo.mpo.calendar`): tau is a trading-minute
counter on a synthetic no-holiday calendar, so trading minutes between taus
equal their difference and overnight counts as 0 minutes. "Today's close" is
the tau of the last minute bar of the decision day (``day*240 + 239``); a
stage ending there includes the close bar and the next stage starts on the
following day.

Ablation (``cfg.max_stages < 4``): the full grid (with empty stages already
dropped) is truncated to its first ``max_stages`` stages. In particular
``max_stages == 1`` keeps only the execution stage, and ``max_stages == 2``
keeps S1+S2 -- or S1 plus the next full day when S2 is empty.
"""

from __future__ import annotations

from openpo.mpo.calendar import day_close_tau, day_of
from openpo.mpo.config import MINUTES_PER_DAY, MpoConfig
from openpo.mpo.types import Stage, StageGrid


def build_stage_grid(tau_now: int, n_days: int, cfg: MpoConfig) -> StageGrid:
    """Build the stage grid for a decision at ``tau_now`` on an ``n_days`` calendar.

    Parameters
    ----------
    tau_now : int
        Decision time (global trading minute). Must lie strictly before the
        close of its day (an execution stage of positive length must exist)
        and within the calendar ``[0, n_days*240)``.
    n_days : int
        Number of days in the synthetic calendar; the last representable tau
        is ``day_close_tau(n_days - 1) == n_days*240 - 1``.
    cfg : MpoConfig
        Uses ``exec_minutes``, ``horizon_minutes`` and ``max_stages``.

    Returns
    -------
    StageGrid
        Stages are contiguous (each start equals the previous end), have
        positive length, are re-indexed ``0..J-1``, and satisfy
        ``sum(length) == last.end - tau_now <= cfg.horizon_minutes``.
        ``today_stage_indices`` lists the stages ending on the decision day.

    Raises
    ------
    ValueError
        If ``tau_now`` is outside the calendar or at/after its day's close,
        or if the config stage/horizon parameters are non-positive.
    """
    tau_now = int(tau_now)
    n_days = int(n_days)
    if n_days <= 0:
        raise ValueError(f"n_days must be positive, got {n_days}")
    if cfg.exec_minutes <= 0 or cfg.horizon_minutes <= 0:
        raise ValueError(
            f"exec_minutes/horizon_minutes must be positive, got "
            f"{cfg.exec_minutes}/{cfg.horizon_minutes}"
        )
    if cfg.max_stages < 1:
        raise ValueError(f"max_stages must be >= 1, got {cfg.max_stages}")

    cal_end = day_close_tau(n_days - 1)  # last tau of the calendar (a close)
    if not 0 <= tau_now < cal_end + 1:
        raise ValueError(f"tau_now={tau_now} outside calendar [0, {cal_end}]")

    today = day_of(tau_now)
    close_today = day_close_tau(today)
    if tau_now >= close_today:
        # No positive-length execution stage can exist at/after the close.
        # Decision epochs (max offset 215 < 239) never hit this branch.
        raise ValueError(
            f"tau_now={tau_now} is at/after the day close ({close_today}); "
            "no execution stage possible"
        )

    # Absolute horizon end, truncated at the calendar's last bar.
    horizon_end = min(tau_now + cfg.horizon_minutes, cal_end)

    # Candidate right boundaries of S1..S4 in order. Each is additionally
    # clipped to horizon_end; the sweep below drops empty stages (e.g. S2 when
    # S1 already reaches the close, or S3/S4 past the calendar end).
    candidate_ends = (
        min(tau_now + cfg.exec_minutes, close_today),  # S1: exec, cut at close
        close_today,                                   # S2: rest of today
        close_today + MINUTES_PER_DAY,                 # S3: next day's close
        tau_now + cfg.horizon_minutes,                 # S4: horizon tail
    )

    stages: list[Stage] = []
    cursor = tau_now
    for end in candidate_ends:
        end = min(end, horizon_end)
        if end > cursor:  # positive length only; empty stages are dropped
            stages.append(Stage(index=len(stages), start=cursor, end=end))
            cursor = end

    # Ablation: keep only the first max_stages stages of the (dense) grid.
    if len(stages) > cfg.max_stages:
        stages = stages[: cfg.max_stages]

    today_idx = tuple(s.index for s in stages if day_of(s.end) == today)
    return StageGrid(tau_now=tau_now, stages=tuple(stages), today_stage_indices=today_idx)
