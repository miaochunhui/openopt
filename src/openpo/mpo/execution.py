"""Order planning and 30-minute bar matching (SPEC section 4).

Pipeline per decision epoch: ``plan_orders`` turns the QP's stage-1 target
weights into lot-feasible, T+1-feasible, cash-feasible orders; ``match_orders``
executes them against the next 30-minute bar with a participation cap,
signed slippage, per-fill commission/stamp tax, and ledger debiting at fill
time.

Conventions (project-wide, see docs/mpo-t1-SPEC.md):

- Trading minutes are tau differences; overnight = 0 minutes. Execution here
  is epoch-local and never carries residual orders across epochs (残量撤单).
- Buys are lot multiples (``cfg.lot_size``, A-share 100); sells are lot
  multiples except a full liquidation, whose odd-lot tail may be sold whole.
- The execution price approximates the 30-min VWAP by the bar reference price
  times ``(1 +/- slippage)`` -- adverse sign: buys pay up, sells receive less.
- Planning prices at the DECISION minute, matching prices at the bar END, so a
  plan is never a cash guarantee: within the bar prices drift, sells can fill
  short of plan (participation cap / limit-down / halt) and buys pay
  commission on the drifted price. ``match_orders`` therefore settles sells
  before buys and caps buy fills by the cash actually on hand -- the hard
  floor that keeps settlement cash >= 0 (see :func:`match_orders`).
"""

from __future__ import annotations

import numpy as np

from .config import MpoConfig
from .ledger import T1Ledger
from .types import Fill, Order, PortfolioSnapshot

# Share tolerance when deciding whether a sell intends a full liquidation.
_CLEAR_TOL = 1e-9


def _buy_cash_outflow(exec_qty: float, exec_price: float, comm_rate: float) -> float:
    """Cash a buy fill actually takes out: notional + commission.

    Written in exactly the same float operations the simulator uses when it
    debits cash (``shares * price + commission``), so the affordability test
    below and the settlement that follows it can never disagree by an ulp.
    """
    notional = exec_qty * exec_price
    return notional + notional * comm_rate


def _affordable_buy_qty(
    exec_qty: float, exec_price: float, comm_rate: float, cash_left: float, lot: int
) -> float:
    """Largest lot-aligned quantity <= ``exec_qty`` whose outflow fits ``cash_left``."""
    if exec_qty <= 0.0 or exec_price <= 0.0 or cash_left <= 0.0:
        return 0.0
    if _buy_cash_outflow(exec_qty, exec_price, comm_rate) <= cash_left:
        return exec_qty  # affordable as-is, no truncation
    unit = _buy_cash_outflow(float(lot), exec_price, comm_rate)
    if unit <= 0.0:
        return 0.0
    max_lots = min(int(np.floor(cash_left / unit)), int(round(exec_qty / lot)))
    # Guard the floor against float rounding: shrink until the outflow really fits.
    while max_lots > 0 and _buy_cash_outflow(max_lots * lot, exec_price, comm_rate) > cash_left:
        max_lots -= 1
    return float(max_lots * lot)


def _validate_vector(x, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size != n:
        raise ValueError(f"{name} size {a.size} != {n}")
    return a


def plan_orders(
    target_w: np.ndarray,
    snapshot: PortfolioSnapshot,
    prices: np.ndarray,
    cfg: MpoConfig,
    marginal_score: np.ndarray | None = None,
) -> list[Order]:
    """Convert target weights into feasible orders (sells first, then buys).

    Steps, in order:

    1. Desired share delta ``(target_w - w_now) * NAV / price``.
    2. Buys: round DOWN to whole lots. Sells: cap at T+1 sellable shares,
       round down to whole lots -- unless the trade intends a full liquidation
       (desired >= current holdings), in which case the entire sellable
       position including the odd-lot tail is sold (清仓尾零股).
    3. No-trade band: drop orders with notional below
       ``max(lot_size * price, cfg.min_order_nav_frac * NAV)``. Full-clear
       sells are exempt so dust positions cannot linger forever.
    4. Cash check: total buy cash need (incl. buy-side costs) must fit within
       ``cash + expected same-epoch sell proceeds`` (A-share same-day proceeds
       are reusable). If not, buys are truncated in ascending priority order:
       priority = ``marginal_score`` if given (higher = keep first), else
       ``|target_w - w_now|``. The marginal surviving buy is shrunk to the
       largest affordable lot count.

    Assets with non-finite or non-positive prices are skipped (frozen/bad
    quotes are handled upstream; planning against them is meaningless).
    """
    n = snapshot.holdings_shares.size
    tw = _validate_vector(target_w, n, "target_w")
    px = _validate_vector(prices, n, "prices")
    nav = float(snapshot.nav)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError(f"snapshot.nav must be positive, got {nav}")
    lot = int(cfg.lot_size)
    if marginal_score is not None:
        score = _validate_vector(marginal_score, n, "marginal_score")
    else:
        score = np.abs(tw - snapshot.weights)

    band_frac_cny = cfg.min_order_nav_frac * nav
    sells: list[Order] = []
    buy_cands: list[tuple[float, int, float]] = []  # (score, asset, shares)

    for i in range(n):
        if not np.isfinite(px[i]) or px[i] <= 0.0:
            continue
        desired = (tw[i] - snapshot.weights[i]) * nav / px[i]
        band_cny = max(lot * px[i], band_frac_cny)

        if desired > 0.0:  # ---------------------------------------- buy leg
            shares = np.floor(desired / lot) * lot
            if shares <= 0.0 or shares * px[i] < band_cny:
                continue
            buy_cands.append((float(score[i]), i, float(shares)))
        elif desired < 0.0:  # --------------------------------------- sell leg
            want = -desired
            holding = float(snapshot.holdings_shares[i])
            sellable = float(snapshot.sellable_shares[i])
            full_clear = want >= holding - _CLEAR_TOL
            if full_clear:
                shares = sellable  # whole position incl. odd-lot tail
            else:
                shares = np.floor(min(want, sellable) / lot) * lot
            if shares <= 0.0:
                continue
            if not full_clear and shares * px[i] < band_cny:
                continue
            sells.append(Order(asset=i, shares=-float(shares)))

    # ------------------------------------------------------ cash truncation
    sell_cost = cfg.sell_cost()
    buy_cost = cfg.buy_cost()
    proceeds = sum(-o.shares * px[o.asset] * (1.0 - sell_cost) for o in sells)
    budget = float(snapshot.cash) + proceeds

    buy_cands.sort(key=lambda t: t[0], reverse=True)  # best marginal alpha first
    buys: list[Order] = []
    for _, i, shares in buy_cands:
        need = shares * px[i] * (1.0 + buy_cost)
        if need <= budget:
            buys.append(Order(asset=i, shares=shares))
            budget -= need
            continue
        # Shrink the marginal buy to the largest affordable lot count.
        lot_cost = lot * px[i] * (1.0 + buy_cost)
        max_lots = int(np.floor(budget / lot_cost)) if lot_cost > 0 else 0
        shrunk = float(max_lots * lot)
        if shrunk <= 0.0 or shrunk * px[i] < max(lot * px[i], band_frac_cny):
            continue  # cannot afford a band-passing order; drop
        buys.append(Order(asset=i, shares=shrunk))
        budget -= shrunk * px[i] * (1.0 + buy_cost)

    return sells + buys


def match_orders(
    orders: list[Order],
    bar_amount_30min: np.ndarray,
    prices: np.ndarray,
    limit_up: np.ndarray,
    limit_down: np.ndarray,
    halted: np.ndarray,
    ledger: T1Ledger,
    cfg: MpoConfig,
    cash_available: float | None = None,
) -> tuple[list[Fill], int]:
    """Match orders against one 30-minute bar; returns (fills, rejected_count).

    - Sells are always settled BEFORE buys within the epoch (A-share same-day
      proceeds are reusable), independent of the order list's own ordering.
    - ``cash_available`` (CNY on hand at the decision minute) is the hard
      settlement floor: buy fills are capped so that
      ``notional + commission <= cash still on hand`` after the epoch's sells
      have been banked, walking buys in the order given (i.e. the planning
      alpha-priority order from :func:`plan_orders`). A buy shrunk or dropped
      this way is lot-aligned, is NOT a rejection, and shows up as a truncated
      buy to the caller (filled < ordered). ``None`` disables the cap (unit
      tests / callers that do their own cash management).
      Planning admits buys up to ``cash + expected same-epoch proceeds`` priced
      at the DECISION minute (SPEC section 4); by fill time prices have moved
      and sells may have filled short, so without this cap settlement cash can
      go negative.
    - Rejections (counted): buy at limit-up, sell at limit-down, any order on
      a halted asset. Rejected orders produce no fill and touch no state.
    - Fill quantity is capped by ``cfg.participation_rate * bar_amount_30min``
      (CNY, converted to shares at the bar reference price). Buys stay lot
      multiples after capping; sells may fill odd amounts. The unfilled
      remainder is CANCELLED, never carried to the next epoch, and a
      volume-starved zero fill is NOT a rejection.
    - Execution price = bar reference price * (1 + slippage) for buys,
      * (1 - slippage) for sells; the per-fill ``slippage_cost`` is the CNY
      cost vs. the reference price. Commission on both sides, stamp tax on
      sells only; all rates scaled by ``cfg.cost_mult``.
    - Sells debit the T+1 ledger AT FILL TIME via ``on_sell_fill`` (which
      raises T1ViolationError on any breach -- the hard second check).
    """
    n = ledger.n_assets
    amt = _validate_vector(bar_amount_30min, n, "bar_amount_30min")
    px = _validate_vector(prices, n, "prices")
    lup = np.asarray(limit_up, dtype=bool).ravel()
    ldn = np.asarray(limit_down, dtype=bool).ravel()
    hlt = np.asarray(halted, dtype=bool).ravel()
    for name, arr in (("limit_up", lup), ("limit_down", ldn), ("halted", hlt)):
        if arr.size != n:
            raise ValueError(f"{name} size {arr.size} != {n}")

    slip = cfg.slippage * cfg.cost_mult
    buy_comm_rate = cfg.buy_commission * cfg.cost_mult
    sell_comm_rate = cfg.sell_commission * cfg.cost_mult
    stamp_rate = cfg.stamp_tax * cfg.cost_mult
    lot = int(cfg.lot_size)

    if cash_available is None:
        cash_left = None
    else:
        cash_left = float(cash_available)
        if not np.isfinite(cash_left):
            raise ValueError(f"cash_available must be finite, got {cash_available}")

    # Sells first, buys second (stable within each side): the buy-side cash cap
    # may only spend proceeds that are already banked.
    sequenced = [o for o in orders if float(o.shares) < 0.0]
    sequenced += [o for o in orders if float(o.shares) > 0.0]

    fills: list[Fill] = []
    rejected = 0
    for order in sequenced:
        i = int(order.asset)
        qty = float(order.shares)
        is_buy = qty > 0.0
        if hlt[i] or (is_buy and lup[i]) or (not is_buy and ldn[i]):
            rejected += 1
            continue
        if not np.isfinite(px[i]) or px[i] <= 0.0:
            rejected += 1
            continue

        cap_shares = cfg.participation_rate * max(amt[i], 0.0) / px[i]
        exec_qty = min(abs(qty), cap_shares)
        if is_buy:
            exec_qty = np.floor(exec_qty / lot) * lot  # buys stay lot-aligned
        if exec_qty <= 0.0:
            continue  # volume-starved: residual cancelled, not a rejection

        if is_buy:
            exec_price = px[i] * (1.0 + slip)
            if cash_left is not None:
                # Hard settlement cap: never spend cash we do not have.
                exec_qty = _affordable_buy_qty(
                    exec_qty, exec_price, buy_comm_rate, cash_left, lot
                )
                if exec_qty <= 0.0:
                    continue  # cash-starved: residual cancelled, not a rejection
            notional = exec_qty * exec_price
            commission = notional * buy_comm_rate
            stamp = 0.0
            ledger.on_buy_fill(i, exec_qty)
            signed_qty = exec_qty
            if cash_left is not None:
                cash_left -= notional + commission
        else:
            exec_price = px[i] * (1.0 - slip)
            notional = exec_qty * exec_price
            commission = notional * sell_comm_rate
            stamp = notional * stamp_rate
            ledger.on_sell_fill(i, exec_qty)  # fill-time debit; may raise
            signed_qty = -exec_qty
            if cash_left is not None:
                cash_left += notional - commission - stamp
        slippage_cost = exec_qty * px[i] * slip  # adverse move vs. reference

        fills.append(
            Fill(
                asset=i,
                shares=signed_qty,
                price=float(exec_price),
                commission=float(commission),
                stamp_tax=float(stamp),
                slippage_cost=float(slippage_cost),
            )
        )
    return fills, rejected
