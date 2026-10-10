"""Tests for openpo.mpo.calendar and openpo.mpo.stages (SPEC section 1 invariants)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openpo.mpo import calendar as cal
from openpo.mpo.config import EPOCH_OFFSETS, MINUTES_PER_DAY, MpoConfig
from openpo.mpo.stages import build_stage_grid
from openpo.mpo.types import StageGrid

CFG = MpoConfig()


def cfg_with(**kw) -> MpoConfig:
    return dataclasses.replace(CFG, **kw)


# ---------------------------------------------------------------------------
# calendar.py
# ---------------------------------------------------------------------------


def test_day_minute_roundtrip():
    for tau in [0, 1, 239, 240, 241, 479, 480, 5 * 240 + 215, 99 * 240 + 239]:
        d, m = cal.day_of(tau), cal.minute_of(tau)
        assert 0 <= m < MINUTES_PER_DAY
        assert d * MINUTES_PER_DAY + m == tau


def test_day_close_tau_is_last_minute_of_day():
    for day in [0, 1, 7, 99]:
        close = cal.day_close_tau(day)
        assert cal.day_of(close) == day
        assert cal.minute_of(close) == MINUTES_PER_DAY - 1
        assert cal.day_of(close + 1) == day + 1  # next tau is next day's open


def test_trading_minutes_between_is_tau_difference():
    # Overnight counts 0 minutes: last bar of day d to first bar of day d+1 is 1 min.
    assert cal.trading_minutes_between(239, 240) == 1
    assert cal.trading_minutes_between(5, 2405) == 2400
    assert cal.trading_minutes_between(100, 100) == 0
    assert cal.trading_minutes_between(240, 239) == -1


def test_epoch_taus_layout():
    n_days = 3
    taus = cal.epoch_taus(n_days)
    assert taus.dtype == np.int64
    assert taus.shape == (n_days * len(EPOCH_OFFSETS),)
    assert np.all(np.diff(taus) > 0)  # sorted, unique
    expected = np.array(
        [d * MINUTES_PER_DAY + off for d in range(n_days) for off in EPOCH_OFFSETS]
    )
    np.testing.assert_array_equal(taus, expected)
    assert cal.epoch_taus(0).shape == (0,)


def test_is_epoch():
    for tau in cal.epoch_taus(2):
        assert cal.is_epoch(int(tau))
    for tau in [0, 4, 6, 239, 240, 240 + 214, 240 + 216]:
        assert not cal.is_epoch(tau)


# ---------------------------------------------------------------------------
# stages.py: full-grid invariants over every epoch of several days
# ---------------------------------------------------------------------------


def assert_grid_invariants(grid: StageGrid, tau_now: int, n_days: int, cfg: MpoConfig):
    stages = grid.stages
    assert isinstance(grid, StageGrid)
    assert grid.tau_now == tau_now
    assert 1 <= grid.n_stages <= min(4, cfg.max_stages)
    # re-indexed 0..J-1
    assert [s.index for s in stages] == list(range(len(stages)))

    # positive lengths, contiguous (=> non-overlapping half-open intervals)
    assert stages[0].start == tau_now
    for s in stages:
        assert s.length > 0
    for a, b in zip(stages, stages[1:]):
        assert b.start == a.end

    # total length: contiguity makes it last.end - tau_now, capped by horizon
    total = sum(s.length for s in stages)
    assert total == stages[-1].end - tau_now
    assert total <= cfg.horizon_minutes

    # calendar containment
    cal_end = cal.day_close_tau(n_days - 1)
    assert stages[-1].end <= cal_end

    # day-boundary alignment: every internal boundary is either the exec-stage
    # cut (tau_now + exec_minutes) or sits exactly on a day close.
    for s in stages[:-1]:
        assert (
            s.end == tau_now + cfg.exec_minutes
            or cal.minute_of(s.end) == MINUTES_PER_DAY - 1
        )
    # ... and every stage after the intraday ones starts on a day open
    # (previous close + 1), i.e. no stage straddles a partial day at its start.
    today = cal.day_of(tau_now)
    for s in stages:
        if cal.day_of(s.start) != today:
            assert cal.minute_of(s.start) == MINUTES_PER_DAY - 1  # a close tau
    # intraday stages (S1/S2) never cross the day boundary
    for s in stages:
        if cal.day_of(s.end) == today:
            assert cal.day_of(s.start + 1) == today

    # today_stage_indices == stages ending on the decision day, in order
    expected_today = tuple(s.index for s in stages if cal.day_of(s.end) == today)
    assert grid.today_stage_indices == expected_today
    assert len(expected_today) >= 1  # S1 always ends today


@pytest.mark.parametrize("day", [0, 1, 4, 7])
@pytest.mark.parametrize("offset", EPOCH_OFFSETS)
def test_full_grid_all_epochs(day, offset):
    n_days = 20  # large enough that tau_now + horizon fits for all tested days
    tau_now = day * MINUTES_PER_DAY + offset
    grid = build_stage_grid(tau_now, n_days, CFG)
    assert_grid_invariants(grid, tau_now, n_days, CFG)
    # far from the calendar end the horizon is never truncated
    assert grid.stages[-1].end == tau_now + CFG.horizon_minutes
    assert sum(s.length for s in grid.stages) == CFG.horizon_minutes


def test_full_grid_epoch_5_shapes():
    # tau_now = 5 on day 0: S1 30min, S2 rest of day, S3 next day, S4 tail.
    grid = build_stage_grid(5, 20, CFG)
    ends = [(s.start, s.end) for s in grid.stages]
    close0 = cal.day_close_tau(0)
    assert ends == [(5, 35), (35, close0), (close0, close0 + 240), (close0 + 240, 2405)]
    assert grid.stages[2].length == 240  # S3 = one full day
    assert grid.today_stage_indices == (0, 1)


def test_full_grid_epoch_215_degenerate_s2():
    # Last epoch of the day: exec truncated to the close (24 min), S2 empty
    # and dropped, J degenerates to 3; S3 is the full next day.
    tau_now = 215
    grid = build_stage_grid(tau_now, 20, CFG)
    assert grid.n_stages == 3
    s1, s3, s4 = grid.stages
    assert (s1.start, s1.end) == (215, cal.day_close_tau(0))
    assert s1.length == 24
    assert s3.length == 240 and cal.day_of(s3.end) == 1
    assert s4.end == tau_now + CFG.horizon_minutes
    assert grid.today_stage_indices == (0,)
    assert sum(s.length for s in grid.stages) == CFG.horizon_minutes
    assert_grid_invariants(grid, tau_now, 20, CFG)


def test_full_grid_epoch_215_all_days():
    n_days = 15
    for day in range(n_days - 1):
        tau_now = day * MINUTES_PER_DAY + 215
        grid = build_stage_grid(tau_now, n_days, CFG)
        assert_grid_invariants(grid, tau_now, n_days, CFG)
        assert 215 + tau_now not in [s.end for s in grid.stages[:-1]]  # no S2 cut


# ---------------------------------------------------------------------------
# calendar-end truncation
# ---------------------------------------------------------------------------


def test_calendar_end_truncates_tail():
    # Decision on day 8 of a 10-day calendar: S4 is cut at the calendar close.
    n_days = 10
    tau_now = 8 * MINUTES_PER_DAY + 5
    grid = build_stage_grid(tau_now, n_days, CFG)
    assert_grid_invariants(grid, tau_now, n_days, CFG)
    assert grid.stages[-1].end == cal.day_close_tau(n_days - 1)
    assert sum(s.length for s in grid.stages) < CFG.horizon_minutes


def test_last_day_drops_s3_and_s4():
    # Decision on the final day: only intraday stages survive.
    n_days = 5
    tau_now = 4 * MINUTES_PER_DAY + 65
    grid = build_stage_grid(tau_now, n_days, CFG)
    assert_grid_invariants(grid, tau_now, n_days, CFG)
    assert grid.n_stages == 2  # S1 + S2, both today
    assert grid.stages[-1].end == cal.day_close_tau(4)
    assert grid.today_stage_indices == (0, 1)


def test_last_day_last_epoch_single_stage():
    # Final day, epoch 215: everything collapses to one 24-minute stage.
    n_days = 3
    tau_now = 2 * MINUTES_PER_DAY + 215
    grid = build_stage_grid(tau_now, n_days, CFG)
    assert_grid_invariants(grid, tau_now, n_days, CFG)
    assert grid.n_stages == 1
    assert grid.stages[0].length == 24
    assert grid.today_stage_indices == (0,)


def test_second_to_last_day_keeps_s3_drops_s4():
    n_days = 4
    tau_now = 2 * MINUTES_PER_DAY + 5
    grid = build_stage_grid(tau_now, n_days, CFG)
    assert_grid_invariants(grid, tau_now, n_days, CFG)
    assert grid.n_stages == 3  # S1, S2, S3(=last day); S4 empty and dropped
    assert grid.stages[-1].length == 240
    assert grid.stages[-1].end == cal.day_close_tau(3)


# ---------------------------------------------------------------------------
# max_stages ablation
# ---------------------------------------------------------------------------


def test_max_stages_1_exec_only():
    cfg1 = cfg_with(max_stages=1)
    grid = build_stage_grid(5, 20, cfg1)
    assert grid.n_stages == 1
    assert (grid.stages[0].start, grid.stages[0].end) == (5, 5 + cfg1.exec_minutes)
    assert grid.today_stage_indices == (0,)
    # epoch 215: still truncated to the close
    grid = build_stage_grid(215, 20, cfg1)
    assert grid.n_stages == 1
    assert (grid.stages[0].start, grid.stages[0].end) == (215, cal.day_close_tau(0))


def test_max_stages_2_keeps_s1_s2():
    grid = build_stage_grid(5, 20, cfg_with(max_stages=2))
    assert grid.n_stages == 2
    assert (grid.stages[0].start, grid.stages[0].end) == (5, 35)
    assert grid.stages[1].end == cal.day_close_tau(0)
    assert grid.today_stage_indices == (0, 1)


def test_max_stages_2_empty_s2_falls_back_to_next_day():
    # Epoch 215: S2 empty, so the second stage is the full next day.
    grid = build_stage_grid(215, 20, cfg_with(max_stages=2))
    assert grid.n_stages == 2
    assert grid.stages[0].end == cal.day_close_tau(0)
    assert grid.stages[1].length == 240
    assert cal.day_of(grid.stages[1].end) == 1
    assert grid.today_stage_indices == (0,)


def test_max_stages_3_drops_tail():
    grid = build_stage_grid(5, 20, cfg_with(max_stages=3))
    assert grid.n_stages == 3
    assert grid.stages[-1].end == cal.day_close_tau(1)  # S3, no S4


@pytest.mark.parametrize("max_stages", [1, 2, 3])
@pytest.mark.parametrize("offset", EPOCH_OFFSETS)
def test_ablation_grids_are_prefixes_of_full_grid(max_stages, offset):
    n_days = 20
    tau_now = 3 * MINUTES_PER_DAY + offset
    full = build_stage_grid(tau_now, n_days, CFG)
    cfg = cfg_with(max_stages=max_stages)
    trunc = build_stage_grid(tau_now, n_days, cfg)
    assert_grid_invariants(trunc, tau_now, n_days, cfg)
    k = trunc.n_stages
    assert k == min(max_stages, full.n_stages)
    assert trunc.stages == full.stages[:k]
    assert trunc.today_stage_indices == tuple(
        i for i in full.today_stage_indices if i < k
    )


def test_max_stages_larger_than_grid_is_noop():
    grid4 = build_stage_grid(5, 20, CFG)
    grid9 = build_stage_grid(5, 20, cfg_with(max_stages=9))
    assert grid9.stages == grid4.stages


# ---------------------------------------------------------------------------
# validation errors
# ---------------------------------------------------------------------------


def test_invalid_tau_now_raises():
    with pytest.raises(ValueError):
        build_stage_grid(-1, 10, CFG)
    with pytest.raises(ValueError):
        build_stage_grid(10 * MINUTES_PER_DAY, 10, CFG)  # past calendar end
    with pytest.raises(ValueError):
        build_stage_grid(cal.day_close_tau(0), 10, CFG)  # at the close: no S1


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        build_stage_grid(5, 10, cfg_with(max_stages=0))
    with pytest.raises(ValueError):
        build_stage_grid(5, 10, cfg_with(exec_minutes=0))
    with pytest.raises(ValueError):
        build_stage_grid(5, 0, CFG)
