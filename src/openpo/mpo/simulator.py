"""Event-driven backtest simulator for MPO-T1 (SPEC section 5).

Drives one strategy through the epoch grid of a :class:`MarketPanel`:

    per epoch:  DecisionContext -> strategy.decide -> plan_orders -> match_orders
    per day:    ledger roll (T+1 unlock), NAV mark at close, hard identity check

Frozen project conventions (docs/mpo-t1-SPEC.md):

- ``tau = day*240 + minute``; trading minutes are plain tau differences and
  **overnight counts as 0 trading minutes** (synthetic no-holiday calendar,
  ``trading_minutes_between(a, b) == b - a``).
- Execution window of the epoch at ``tau`` is the 30-minute bar
  ``(tau, min(tau+30, day_close)]``; the last daily epoch (offset 215) gets a
  truncated 24-minute bar. Fills are priced off the window MIDPOINT minute
  close (``(tau + tau_end) // 2``, see :func:`_bar_mark_tau`) -- the VWAP
  approximation SPEC section 4 actually calls for; the participation cap still
  consumes the FULL ``(tau, tau_end]`` volume pool.
  That 30 is the DECISION-EPOCH PITCH (``EPOCH_OFFSETS`` spacing, see
  ``_EPOCH_BAR_MINUTES``) and is deliberately **independent of**
  ``cfg.exec_minutes``, which is the S1 alpha window of ``stages.py``. The two
  used to share ``cfg.exec_minutes``, so sweeping the exec axis silently
  changed the execution environment too (overlapping participation pools =
  double-counted volume, longer price drift before the fill mark).
- Halted names are doubly frozen: their z-scores are zeroed in the
  :class:`DecisionContext` AND their target weight is pinned to the current
  weight before order planning (plus ``match_orders`` rejects them again).
- Accounting identity, asserted at EVERY day close (tolerance ``1e-6 * NAV``):
  ``NAV_t = NAV_{t-1} + position_pnl - commission - stamp_tax - slippage``,
  where position pnl marks holdings to the bar reference prices used for
  execution. Cash must satisfy ``cash >= -1e-9`` at all times. T+1 violations
  cannot pass silently: the share ledger raises :class:`T1ViolationError`.
- The cash floor is enforced structurally, not by tolerance: ``match_orders``
  receives the epoch's opening cash, settles sells before buys and truncates
  buy fills to what that cash can actually pay for (commission and slippage
  included), counting each truncation in ``truncated_buys``. Planning may only
  size buys against EXPECTED proceeds at decision-minute prices, which the bar
  is free to invalidate (price drift, participation-capped sells, rejections).

Strategy protocol (duck-typed, see SPEC section 7):

- ``decide(ctx: DecisionContext) -> np.ndarray | None`` -- target weights [N]
  or ``None`` for "no trade this epoch".
- optional ``last_solve_time`` (seconds, float) -- harvested after each decide
  into ``BacktestResult.qp_solve_times``.
- optional ``last_status`` (str) -- the RAW solver verdict of the last solve
  ('optimal'/'inaccurate'/'failed'), tallied verbatim into
  ``qp_status_counts``. Strategies without it fall back to a status derived
  from the ``fail_soft_count`` delta (a single coarse ``fail_soft`` bucket).
- optional ``last_note`` (str) -- the backend's own diagnostic; counted per
  distinct string in ``meta['qp_failure_notes']`` whenever the status is
  non-optimal, so a reason like ``osqp status=primal infeasible`` survives
  into artifacts instead of being flattened into "the solver was flaky".
- optional ``last_marginal_score`` ([N] float) -- forwarded to ``plan_orders``
  as the cash-truncation priority (alpha marginal ranking).
- optional ``name`` (str) -- reported strategy label.
"""

from __future__ import annotations

import numpy as np

from . import execution
from .calendar import day_close_tau
from .config import EPOCH_OFFSETS, MINUTES_PER_DAY, MpoConfig
from .ledger import T1Ledger
from .types import BacktestResult, DecisionContext, MarketPanel, PortfolioSnapshot, SignalPanel

# Cash may only dip below zero by float noise.
_CASH_TOL = 1e-9
# Relative NAV tolerance for the daily accounting identity.
_NAV_IDENT_RTOL = 1e-6


def _epoch_bar_minutes(offsets=EPOCH_OFFSETS) -> int:
    """Order-matching bar length in trading minutes = the decision-epoch pitch.

    SPEC section 4 fixes the matching bar at "the 30 minutes after the
    decision" -- i.e. exactly the spacing of :data:`EPOCH_OFFSETS`, so that
    consecutive epochs consume NON-OVERLAPPING volume. Derived from the epoch
    grid instead of a hardcoded 30 so the two cannot drift apart, and validated
    at import time (a non-uniform grid would make "the bar" ill-defined and
    would double-count participation volume).
    """
    if len(offsets) < 2:
        raise ValueError(f"EPOCH_OFFSETS needs >= 2 epochs to define a bar, got {offsets}")
    spacing = int(offsets[1]) - int(offsets[0])
    if spacing <= 0 or any(int(b) - int(a) != spacing for a, b in zip(offsets, offsets[1:])):
        raise ValueError(f"EPOCH_OFFSETS must be uniformly spaced, got {offsets}")
    return spacing


# Matching-bar length; 30 for the frozen grid. NOT ``cfg.exec_minutes``.
_EPOCH_BAR_MINUTES = _epoch_bar_minutes()


def _bar_mark_tau(tau: int, tau_end: int) -> int:
    """Minute whose close prices the fills of bar ``(tau, tau_end]`` are marked at.

    The window MIDPOINT, ``(tau + tau_end) // 2`` -- a one-price approximation
    of the amount-weighted VWAP that SPEC section 4 specifies as the fill price
    ("30 分钟 VWAP 近似").

    Iteration 43 moved this off the previous bar-END mark (``tau_end``, the last
    print of the window). A 24-seed publication-scale shadow re-mark (bit-exact
    against the published run) measured the implementation-delay cost at
    12.95 bp/unit under the bar-end mark vs 7.19 bp/unit under the true
    amount-weighted VWAP: a 44.5% artifact worth ~1.18%/yr of NAV, LARGER than
    the explicit 5 bp slippage itself. Cause: the bar-end print is already the
    worst-case price of the window for a trader who decided at ``tau``, so
    charging it AND then ``*(1 +- slippage)`` double-charges the same friction.
    The midpoint mark sits within 0.146 bp of the measured true VWAP (97.5% of
    the correction) and needs no minute-VWAP aggregation.

    ``*(1 +- cfg.slippage)`` is still applied on top of this price by
    ``execution.match_orders``, and the participation cap still consumes the
    whole ``(tau, tau_end]`` volume pool -- only the price mark moved.

    Floor division is deliberate on odd-length windows (a close-truncated last
    epoch, e.g. ``(5, 34) -> 19``): the earlier of the two central minutes, so
    the mark can never sit past the window's midpoint.
    """
    return (int(tau) + int(tau_end)) // 2

# Optional sibling import: cross-sectional z-score from the term-structure
# module if available, else the local fallback below (same contract).
try:  # pragma: no cover - exercised only when the sibling module exists
    from .term_structure import zscore_cross_section as _ts_zscore
except Exception:  # ImportError or a partially-written module
    _ts_zscore = None


def _resolve_day_range(day_range) -> tuple[int, int]:
    """Normalize the ``day_range`` argument to half-open bounds ``(start, end)``.

    Accepted forms (all equivalent for ``start=8, end=16``):

    - ``range`` with unit step -- ``range(8, 16)`` (what ``backtest.py`` passes;
      indexing it as a pair would silently yield ``(8, 9)``, i.e. one day).
    - a 2-element ``(start, end)`` pair -- ``(8, 16)``.
    - any other sequence of *consecutive* day indices -- ``[8, 9, ..., 15]``.
    """
    if isinstance(day_range, range):
        if day_range.step != 1:
            raise ValueError(f"day_range must have unit step, got {day_range}")
        return int(day_range.start), int(day_range.stop)
    if isinstance(day_range, np.ndarray):
        days = [int(v) for v in day_range.ravel()]
    else:
        try:
            days = [int(v) for v in day_range]
        except TypeError as exc:  # not iterable at all
            raise TypeError(
                f"day_range must be a range, (start, end) pair or day sequence, "
                f"got {type(day_range).__name__}"
            ) from exc
    if len(days) == 2:  # legacy/explicit half-open bounds
        return days[0], days[1]
    if not days:
        raise ValueError("day_range is empty")
    if any(b - a != 1 for a, b in zip(days, days[1:])):
        raise ValueError(f"day_range must be consecutive days, got {days}")
    return days[0], days[-1] + 1


def _zscore_local(raw: np.ndarray) -> np.ndarray:
    """NaN-safe per-column cross-sectional z-score of a [N, K] slice.

    Columns with fewer than 2 finite entries or ~zero dispersion map to all
    zeros; non-finite inputs map to 0 (neutral signal).
    """
    x = np.asarray(raw, dtype=np.float64)
    z = np.zeros_like(x)
    for k in range(x.shape[1]):
        col = x[:, k]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            continue
        mu = col[finite].mean()
        sd = col[finite].std()
        if not np.isfinite(sd) or sd < 1e-12:
            continue
        z[finite, k] = (col[finite] - mu) / sd
    return z


class Simulator:
    """Backtest engine binding one market panel to one cost/execution config.

    Parameters
    ----------
    market:
        Frozen synthetic market (prices already encode halt freezes).
    cfg:
        Frozen :class:`MpoConfig`; ``cfg.nav_cny`` is the starting cash.
    """

    def __init__(self, market: MarketPanel, cfg: MpoConfig) -> None:
        self.market = market
        self.cfg = cfg
        t_expected = market.n_days * MINUTES_PER_DAY
        if market.prices.shape[0] != t_expected:
            raise ValueError(
                f"prices has {market.prices.shape[0]} rows, expected "
                f"n_days*{MINUTES_PER_DAY}={t_expected}"
            )
        # Daily vol per asset for the quadratic-impact *estimate* (not cash).
        self._sigma_d = np.sqrt(np.maximum(np.diag(market.daily_cov()), 0.0))
        self._adv_cny = np.asarray(market.adv_cny, dtype=np.float64).ravel()
        # Once the sibling z-score errors once we stop retrying it (fail-soft).
        self._use_ts_zscore = _ts_zscore is not None

    # ------------------------------------------------------------------ z-score
    def _zscore(self, raw: np.ndarray) -> np.ndarray:
        if self._use_ts_zscore:
            try:
                z = np.asarray(_ts_zscore(raw), dtype=np.float64)
                if z.shape == raw.shape and np.all(np.isfinite(z)):
                    return z
            except Exception:
                pass
            self._use_ts_zscore = False  # fall back permanently this run
        return _zscore_local(raw)

    # ------------------------------------------------------------------ run
    def run(
        self,
        strategy,
        signals: SignalPanel,
        day_range: range | tuple[int, int],
    ) -> BacktestResult:
        """Backtest ``strategy`` over trading days ``[start, end)``.

        ``day_range`` may be a ``range(start, end)``, a ``(start, end)`` pair or
        a sequence of consecutive day indices (see :func:`_resolve_day_range`).

        Returns a :class:`BacktestResult` with ``daily_nav`` of length
        ``end - start + 1`` (start NAV included) and ``daily_turnover`` of
        length ``end - start``. Raises ``AssertionError`` on any
        accounting-identity or cash-floor breach and propagates
        ``T1ViolationError`` from the ledger unchanged.
        """
        market, cfg = self.market, self.cfg
        n = market.n_assets
        start, end = _resolve_day_range(day_range)
        if not 0 <= start < end <= market.n_days:
            raise ValueError(f"day_range {day_range} outside [0, {market.n_days}]")
        if signals.signals.shape[0] != market.prices.shape[0]:
            raise ValueError("signals and market panels disagree on T")
        n_eval = end - start

        cash = float(cfg.nav_cny)
        holdings = np.zeros(n, dtype=np.float64)  # shares
        ledger = T1Ledger(np.zeros(n))

        daily_nav = np.empty(n_eval + 1, dtype=np.float64)
        daily_nav[0] = cash  # flat start: NAV == cash
        daily_turnover = np.zeros(n_eval, dtype=np.float64)
        costs = {"commission": 0.0, "stamp_tax": 0.0, "slippage": 0.0, "impact_est": 0.0}
        rejected = 0
        truncated_buys = 0
        qp_solve_times: list[float] = []
        qp_status_counts: dict[str, int] = {}
        qp_failure_notes: dict[str, int] = {}

        # Strategies that expose no ``last_status`` but do expose the fail-soft
        # counter get their status derived from the counter delta per decide.
        prev_fail_soft = getattr(strategy, "fail_soft_count", None)
        prev_fail_soft = None if prev_fail_soft is None else int(prev_fail_soft)

        # Marks for incremental position-pnl accrual (identity check input).
        last_px = market.prices[start * MINUTES_PER_DAY].astype(np.float64).copy()

        for d in range(start, end):
            nav_open = daily_nav[d - start]
            day_pnl = 0.0
            day_cash_costs = 0.0  # commission + stamp + slippage (hit cash/NAV)
            day_traded = 0.0
            close_tau = day_close_tau(d)

            for offset in EPOCH_OFFSETS:
                tau = d * MINUTES_PER_DAY + offset
                px_now = market.prices[tau].astype(np.float64)
                halted_now = np.asarray(market.halted[tau], dtype=bool)

                nav_now = cash + float(holdings @ px_now)
                weights = holdings * px_now / nav_now
                sellable = ledger.sellable()
                snapshot = PortfolioSnapshot(
                    tau=tau,
                    nav=nav_now,
                    cash=cash,
                    holdings_shares=holdings.copy(),
                    weights=weights,
                    sellable_shares=sellable,
                    sellable_weights=sellable * px_now / nav_now,
                )
                z = self._zscore(signals.signals[tau])
                z[halted_now] = 0.0  # signal freeze for halted names
                ctx = DecisionContext(
                    snapshot=snapshot,
                    z_signals=z,
                    prices=px_now,
                    limit_up=np.asarray(market.limit_up[tau], dtype=bool),
                    limit_down=np.asarray(market.limit_down[tau], dtype=bool),
                    halted=halted_now,
                    day=d,
                    minute=offset,
                )

                target_w = strategy.decide(ctx)

                # Harvest optional QP telemetry right after the decide call.
                solve_time = getattr(strategy, "last_solve_time", None)
                if solve_time is not None:
                    qp_solve_times.append(float(solve_time))
                status = getattr(strategy, "last_status", None)
                if status is None and prev_fail_soft is not None:
                    # Derive: counter bumped -> the QP was abandoned fail-soft,
                    # else a nonzero solve time means a solve did complete.
                    now_fail_soft = int(getattr(strategy, "fail_soft_count", prev_fail_soft))
                    if now_fail_soft > prev_fail_soft:
                        status = "fail_soft"
                    elif solve_time is not None and float(solve_time) > 0.0:
                        status = "optimal"
                    prev_fail_soft = now_fail_soft
                if status is not None:
                    status = str(status)
                    qp_status_counts[status] = qp_status_counts.get(status, 0) + 1
                    # Non-optimal: keep the backend's own reason verbatim so the
                    # root cause (e.g. 'primal infeasible') is legible without
                    # re-running the benchmark to reproduce it.
                    if status != "optimal":
                        note = getattr(strategy, "last_note", None)
                        if note:
                            note = str(note)
                            qp_failure_notes[note] = qp_failure_notes.get(note, 0) + 1

                if target_w is None:
                    continue  # explicit no-trade epoch
                target_w = np.asarray(target_w, dtype=np.float64).ravel()
                if target_w.size != n or not np.all(np.isfinite(target_w)):
                    raise ValueError(
                        f"strategy target_w must be a finite [{n}] vector, got "
                        f"shape {target_w.shape}"
                    )
                # Trade freeze for halted names: pin target to current weight.
                if halted_now.any():
                    target_w = target_w.copy()
                    target_w[halted_now] = weights[halted_now]

                score = getattr(strategy, "last_marginal_score", None)
                if score is not None:
                    score = np.asarray(score, dtype=np.float64).ravel()
                    if score.size != n or not np.all(np.isfinite(score)):
                        score = None
                orders = execution.plan_orders(
                    target_w, snapshot, px_now, cfg, marginal_score=score
                )
                if not orders:
                    continue

                # Execution bar: (tau, min(tau + epoch pitch, close)] -- clipped
                # at close, so bars of successive epochs tile the day without
                # overlap. The length is the epoch pitch (30), NEVER
                # cfg.exec_minutes: that config knob is the S1 alpha window
                # (stages.py) and must not perturb the execution environment.
                tau_end = min(tau + _EPOCH_BAR_MINUTES, close_tau)
                bar_amount = market.minute_amount[tau + 1 : tau_end + 1].sum(axis=0)
                # Fill mark = window MIDPOINT close (VWAP proxy, within 0.146bp
                # of the true amount-weighted VWAP; see _bar_mark_tau). The
                # participation pool above deliberately stays the FULL
                # (tau, tau_end] window -- the whole bar's volume is tradable,
                # only the average price of it is not the last print. Limit/halt
                # gating stays on the END-of-bar state (unchanged, conservative).
                mark_tau = _bar_mark_tau(tau, tau_end)
                bar_px = market.prices[mark_tau].astype(np.float64)

                # Accrue position pnl up to the execution mark BEFORE fills so
                # trade costs and price moves stay cleanly separated.
                day_pnl += float(holdings @ (bar_px - last_px))
                last_px = bar_px.copy()

                fills, rej = execution.match_orders(
                    orders,
                    bar_amount,
                    bar_px,
                    np.asarray(market.limit_up[tau_end], dtype=bool),
                    np.asarray(market.limit_down[tau_end], dtype=bool),
                    np.asarray(market.halted[tau_end], dtype=bool),
                    ledger,
                    cfg,
                    # Hard settlement floor: planning sized buys on decision-minute
                    # prices plus EXPECTED same-epoch proceeds; by fill time prices
                    # have drifted and sells may have filled short, so the matcher
                    # settles sells first and caps buys by the cash truly on hand.
                    cash_available=cash,
                )
                rejected += int(rej)

                nav_pre = cash + float(holdings @ bar_px)
                filled_by_asset = np.zeros(n, dtype=np.float64)
                epoch_cost = 0.0
                for f in fills:
                    if f.shares > 0:
                        cash -= f.shares * f.price + f.commission
                    else:
                        cash += (-f.shares) * f.price - f.commission - f.stamp_tax
                    holdings[f.asset] += f.shares
                    filled_by_asset[f.asset] += f.shares
                    fill_cost = f.commission + f.stamp_tax + f.slippage_cost
                    epoch_cost += fill_cost
                    costs["commission"] += f.commission
                    costs["stamp_tax"] += f.stamp_tax
                    costs["slippage"] += f.slippage_cost
                    day_traded += abs(f.shares) * bar_px[f.asset]
                day_cash_costs += epoch_cost

                # Quadratic impact ESTIMATE (0.5*eta*dw'D dw, D from SPEC s3);
                # informational only -- never touches cash or the identity.
                dw = np.abs(filled_by_asset) * bar_px / nav_now
                traded_mask = dw > 0
                if traded_mask.any():
                    adv_frac = self._adv_cny[traded_mask] / nav_now
                    d_diag = cfg.impact_coef * self._sigma_d[traded_mask] / np.maximum(adv_frac, 1e-12)
                    costs["impact_est"] += 0.5 * nav_now * float(d_diag @ dw[traded_mask] ** 2)

                # Buy orders not fully filled (participation cap / limit / halt /
                # match-time cash cap).
                for o in orders:
                    if o.shares > 0 and filled_by_asset[o.asset] < o.shares - 1e-9:
                        truncated_buys += 1

                # Per-epoch mini-identity: fills may only cost fees + slippage.
                nav_post = cash + float(holdings @ bar_px)
                assert abs(nav_post - (nav_pre - epoch_cost)) <= _NAV_IDENT_RTOL * max(
                    nav_pre, 1.0
                ), (
                    f"epoch accounting drift at tau={tau}: nav_pre={nav_pre:.6f} "
                    f"nav_post={nav_post:.6f} epoch_cost={epoch_cost:.6f}"
                )
                assert cash >= -_CASH_TOL, f"cash went negative at tau={tau}: {cash:.6f}"

            # ------------------------------------------------------ day close
            close_px = market.prices[close_tau].astype(np.float64)
            day_pnl += float(holdings @ (close_px - last_px))
            last_px = close_px.copy()
            nav_close = cash + float(holdings @ close_px)

            # Hard daily identity: NAV_t = NAV_{t-1} + pnl - cash costs.
            expected = nav_open + day_pnl - day_cash_costs
            tol = _NAV_IDENT_RTOL * max(abs(nav_close), abs(nav_open), 1.0)
            assert abs(nav_close - expected) <= tol, (
                f"daily NAV identity broken on day {d}: nav_close={nav_close:.6f} "
                f"expected={expected:.6f} (pnl={day_pnl:.6f}, costs={day_cash_costs:.6f})"
            )
            assert cash >= -_CASH_TOL, f"cash negative at day {d} close: {cash:.6f}"

            idx = d - start
            daily_nav[idx + 1] = nav_close
            daily_turnover[idx] = day_traded / nav_open if nav_open > 0 else 0.0
            holdings = np.maximum(holdings, 0.0)  # scrub float dust
            ledger.on_day_close(holdings)  # T+1 unlock roll

        traded_total = float(np.sum(daily_turnover * daily_nav[:-1]))
        return BacktestResult(
            strategy=getattr(strategy, "name", type(strategy).__name__),
            seed=int(market.meta.get("seed", signals.meta.get("seed", -1))),
            daily_nav=daily_nav,
            daily_turnover=daily_turnover,
            costs=costs,
            rejected_orders=rejected,
            truncated_buys=truncated_buys,
            t1_violations=0,  # the ledger raises instead of counting
            qp_solve_times=qp_solve_times,
            qp_status_counts=qp_status_counts,
            meta={
                "day_range": (start, end),
                "traded_value_cny": traded_total,
                "final_cash": cash,
                "final_holdings_shares": holdings.copy(),
                "qp_failure_notes": qp_failure_notes,
            },
        )
