"""Production prediction-data adapter layer for MPO-T1 (Iteration 39, BACKLOG P1).

Scope discipline (read this first)
---------------------------------
This module is the **本地契约腿 only**: it encodes the production ``HandleScore``
minute callback contract into the frozen MPO-T1 internal types and is verified
by contract tests against synthetic panels on this box. The **集群 30 日回放实测
leg is PENDING** -- there is no real prediction data on this machine
(``/weka-jd``, ``/ceph-jd``, ``~/alphadata`` all absent, see
``docs/HANDOFF-mpo-t1-20260821.md`` section 5), so nothing here has ever seen a
real score file. What the cluster leg still needs is listed in
:data:`CLUSTER_LEG_TODO`.

Production contract facts (all evidence is in-repo, other people's assets --
read-only prior art)
--------------------------------------------------------------------------
================================================  =========================================================
fact                                              evidence
================================================  =========================================================
callback signature is 8-positional, not 6          ``deliverables/opt_bt4_30day_benchmark.py:696``
                                                   ``HandleScore(date, seq, cash_score, cash_weight, iids,``
                                                   ``scores, weights, unlimited_weights)``
``scores`` is a dict keyed by ``identify_str``      ``deliverables/opt_openpo.py:736`` + ``:419``
(``'ScoreMain'``), value = ``(buy, sell)`` tuple    (payload docstring ``deliverables/opt_openpo.py:723``)
or a list of objects with ``.buy_score/.sell_score``
production alpha == ``buy_scores`` (verbatim)       ``deliverables/opt_openpo.py:768`` ``score = buy_scores``
                                                   ``deliverables/opt_bt4_30day_benchmark.py:439``
``sell_scores`` are a **gate**, not an alpha        ``deliverables/opt_openpo.py:765`` ``sell>500 -> no sell``
                                                   ``:767`` ``buy<-500 -> no buy``
                                                   ``:750`` spread gate ``buy.max()-sell.min()<=0.6``
seq range 1..237 == 交易分钟 9:31..14:57            ``deliverables/opt_openpo.py:407``
                                                   ``valid_seqs = list(range(1, 238))``
``unlimited_weights`` = T+1 sellable, weight units  ``deliverables/opt_openpo.py:731`` (re-marked to
relative to ``TOTAL_ASSETS``                        ``next_avg_px`` then clamped at 0 / nan_to_num)
``weights`` are normalized to sum 1 (no cash)       ``deliverables/opt_openpo.py:733-734``
costs: buy 2bp, sell 9bp, slippage 50bp             ``deliverables/probe_opt_bt4_real_shapes.py:283-285``
                                                   (``buy_commission_rate/sell_commission_rate/``
                                                   ``spread_slippage_rate``)
tradability flag ``valid``                          ``deliverables/opt_bt4_30day_benchmark.py:425``
                                                   ``GetFeatureValue("valid", date, seq)``
three score families d/m/t_alpha                    ``deliverables/probe_opt_bt4_real_shapes.py:269-272``
================================================  =========================================================

**Discrepancy vs the handoff note (report-worthy).** The handoff records
``HandleScore(date, seq, iids, (buy,sell), weights, unlimited_weights)``; the
actual production signature carries two extra leading arguments
(``cash_score``, ``cash_weight``) and ``scores`` is a *dict* whose single
``'ScoreMain'`` entry holds an **already-blended** score: the three families are
merged upstream by the file score provider via
``score_weight='0.0,0.33,0.23'`` and per-family ``score_scale`` 1.0/0.35/0.35
(``deliverables/probe_opt_bt4_real_shapes.py:268-272``). MPO-T1 needs the three
horizons *separately*, so the cluster leg must configure three score providers
(three ``identify_str`` keys) or read the ``d/m/t_alpha`` directories directly.
This adapter therefore takes a **mapping horizon-key -> payload** and fails loudly
if a horizon is missing, instead of pretending a blended score is a term
structure.

Design decisions taken here (each guarded by a test in
``tests/mpo/test_adapters.py``)
-------------------------------------------------------------------------
1. **buy/sell merge = buy_scores as the alpha** (``alpha_mode='buy'``, default).
   This is *convicted*, not 疑似: production feeds ``score = buy_scores`` to the
   optimizer verbatim and uses ``sell_scores`` only for tradability gates.
   ``alpha_mode='net'`` (``buy - sell``) exists as an experiment axis and is
   explicitly **not** the production semantic.
2. **seq -> minute is the identity** (``minute = seq``). Both count elapsed
   trading minutes from the 9:30 open: seq 1 == 9:31 and seq 237 == 14:57 is
   exactly 120 (9:30->11:30) + 117 (13:00->14:57) trading minutes, matching the
   ``valid_seqs`` comment. Production trades in bar ``seq+1``
   (``next_amount[seq+1]``), and the MPO execution stage is ``(tau_now,
   tau_now+30]`` -- i.e. also starts at ``minute+1`` -- so the identity map keeps
   决策/执行 alignment. Consequence: production has no score at minutes 0, 238,
   239; none of the 8 epoch offsets (5..215) lands there, so the epoch grid is
   unaffected.
3. **Sentinel scores are not alphas.** ``|score| > 500`` is a production
   tradability sentinel; feeding it into a cross-sectional z-score would blow up
   the whole cross-section, so sentinel entries become NaN in the raw matrix
   (z-score neutral 0) and are re-expressed as ``limit_up`` (cannot buy) /
   ``limit_down`` (cannot sell) masks, which is exactly what the QP box does
   (``src/openpo/mpo/qp.py:349-350``).
4. **Cross-horizon iid alignment = union with NaN fill**, first-appearance order
   over :data:`HORIZON_KEYS`. A name missing from one family is neutral in that
   column only; a name with no score in any family is treated as untradable
   (``halted``) because we have no basis to trade it.
5. **sellable passthrough**: ``snapshot.sellable_weights`` is
   ``unlimited_weights`` verbatim -- no re-marking, no cap by ``weights`` beyond a
   loud validation tolerance. The T+1 constraint layer owns the semantics.
6. **The production zero-trade spread gate is OFF by default.** Its threshold
   (0.6) is in production score units; on return-scale alphas it would gate
   every epoch and silently shut the strategy down. Opt in with
   ``buy_sell_diff=PRODUCTION_BUY_SELL_DIFF`` once the cluster leg measures the
   real score scale.

Nothing in this module mutates the frozen defaults in
:mod:`openpo.mpo.config`; :func:`production_mpo_config` returns a
``dataclasses.replace`` copy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpo.mpo.config import (
    EPOCH_OFFSETS,
    MINUTES_PER_DAY,
    SIGNAL_HORIZONS,
    MpoConfig,
)
from openpo.mpo.term_structure import zscore_cross_section
from openpo.mpo.types import DecisionContext, PortfolioSnapshot

__all__ = [
    "CLUSTER_LEG_TODO",
    "HORIZON_KEYS",
    "PRODUCTION_ALGO_PARAMS",
    "PRODUCTION_BUY_SELL_DIFF",
    "PRODUCTION_COSTS",
    "PRODUCTION_SCORE_SENTINEL",
    "PRODUCTION_SEQ_MAX",
    "PRODUCTION_SEQ_MIN",
    "CallbackRecord",
    "HorizonScores",
    "IidUniverse",
    "ProductionCosts",
    "ReplayDecision",
    "ReplayHarness",
    "ScoreFrame",
    "SnapshotAdapter",
    "TradingDayIndex",
    "buy_sell_from_payload",
    "build_score_frame",
    "is_epoch_seq",
    "minute_of_seq",
    "production_mpo_config",
    "seq_of_minute",
    "tau_of_seq",
    "union_universe",
]


# ---------------------------------------------------------------------------
# production constants
# ---------------------------------------------------------------------------

#: Production score-family directory keys, ordered short -> mid -> long so that
#: column ``k`` of the raw matrix pairs with ``config.SIGNAL_HORIZONS[k]``
#: (30 / 240 / 2400 trading minutes).
#:
#: 疑似 -- needs cluster verification: the repo pins the three directory names
#: (``deliverables/probe_opt_bt4_real_shapes.py:269-272``) and the per-family
#: scales, but nowhere states which family is which horizon. ``d/m/t`` reading
#: as day/mid/tick is a guess; the handoff calls them short/mid/long in this
#: order. If the cluster probe contradicts it, this tuple (and only this tuple)
#: needs reordering -- every consumer goes through it.
HORIZON_KEYS: tuple[str, str, str] = ("d_alpha", "m_alpha", "t_alpha")

PRODUCTION_SEQ_MIN = 1
PRODUCTION_SEQ_MAX = 237

#: ``|score| > 500`` is a tradability sentinel, not a prediction
#: (``deliverables/opt_openpo.py:765-767``).
PRODUCTION_SCORE_SENTINEL = 500.0

#: ``buy_sell_diff`` production default (``probe_opt_bt4_real_shapes.py:267``):
#: when no Barra style control is configured, production returns a flat
#: zero-trade vector if ``buy.max() - sell.min() <= 0.6``.
#:
#: **Not enabled by default** (``buy_sell_diff=None``) anywhere in this module:
#: 0.6 is expressed in production *score* units, which are demonstrably not
#: return units (the MPO term structure regresses scores onto returns). Enabling
#: it on return-scale alphas gates every epoch to zero-trade -- a silent,
#: total-shutdown failure mode. Turn it on only after the cluster leg measures
#: the real score scale.
PRODUCTION_BUY_SELL_DIFF = 0.6

#: Other production algo params, recorded with evidence but **not** applied by
#: :func:`production_mpo_config` (they belong to the production optimizer, whose
#: objective differs from the MPO-T1 QP). Pass them as explicit overrides when
#: the cluster leg wants full production parity.
PRODUCTION_ALGO_PARAMS: dict[str, float] = {
    # probe_opt_bt4_real_shapes.py:273 total_assets = 500e8
    "nav_cny": 500e8,
    # probe_opt_bt4_real_shapes.py:281 max_stock_weight = 0.005
    "w_max": 0.005,
    # probe_opt_bt4_real_shapes.py:283 default_trade_speed = 0.08
    "participation_rate": 0.08,
}

#: What the (still PENDING) cluster 30-day replay leg has to establish. Kept in
#: code so the gap cannot rot out of a doc. Entries that later got closed keep
#: their slot prefixed ``RESOLVED (...)`` with the evidence pointer, so the list
#: stays a ledger rather than a shrinking wishlist.
CLUSTER_LEG_TODO: tuple[str, ...] = (
    "expose the three d/m/t_alpha families separately (three score providers or "
    "direct directory reads); the current production path blends them into one "
    "'ScoreMain' payload via score_weight=0.0,0.33,0.23 and cannot feed a term "
    "structure",
    "confirm HORIZON_KEYS -> SIGNAL_HORIZONS (30/240/2400min) assignment against "
    "real per-family forward-return ICs",
    "confirm seq->minute identity by checking a real score timestamp against seq "
    "(only the valid_seqs comment backs it today)",
    "source limit-up/limit-down masks: production expresses them through "
    "max_buy_dw/max_sell_dw caps and the +-500 sentinels, not boolean flags; "
    "verify the sentinel mapping covers 涨跌停 or add the caps as box bounds",
    "source NAV/prices per callback (TOTAL_ASSETS and next_avg_px[seq+1]) and "
    "decide whether unlimited_weights must be re-marked as production does",
    "re-fit the term structure on real panels (fit_term_structure currently "
    "needs a MarketPanel; real replay needs a prices/returns feed for calibration)",
    "RESOLVED (Iteration 45, probe 2026-08-23) per-epoch solve wall at real "
    "cluster scale: N=5000 costs 2.23s with the low-rank factor risk assembly "
    "plus Clarabel (27x inside the 60s epoch budget; the old dense path never "
    "finished) -- so run the cluster leg with risk_form='auto' (or 'lowrank') "
    "and cfg.qp_backend='clarabel' whenever N is above ~2500",
    "measure the real score SCALE: buy_sell_diff=0.6 and the +-500 sentinels are "
    "in production score units, not return units, so both thresholds are "
    "un-calibrated here (the spread gate is therefore off by default)",
)


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProductionCosts:
    """Production cost trio, decomposed onto the frozen :class:`MpoConfig` fields.

    Production states three numbers (``probe_opt_bt4_real_shapes.py:283-285``):
    ``buy_commission_rate=0.0002`` (2bp), ``sell_commission_rate=0.0009`` (9bp,
    stamp duty included) and ``spread_slippage_rate=0.005`` (50bp).
    :class:`MpoConfig` splits the sell leg into ``sell_commission + stamp_tax``,
    so the 9bp is decomposed keeping the repo's canonical ``STAMP_TAX = 5bp``
    (``src/openpo/mpo/config.py:54``): ``sell_commission = 9bp - 5bp = 4bp``.
    The split is cosmetic for the QP -- only ``sell_cost()`` enters it.

    Slippage delta vs the synthetic default is the headline: **50bp vs 5bp, a
    10x jump**. Round-trip cost goes 19bp (synthetic 7bp buy + 12bp sell) ->
    111bp (52bp + 59bp). At 111bp a round trip needs ~5.8x more alpha to break
    even, so the production-cost run is a genuinely different optimization
    problem, not a tweak: expect far lower turnover and much smaller stage-1
    trades. Any synthetic conclusion about turnover does NOT transfer.
    """

    buy_commission: float = 2e-4
    sell_commission: float = 4e-4
    stamp_tax: float = 5e-4
    slippage: float = 50e-4

    @property
    def buy_total(self) -> float:
        """Total buy cost fraction (commission + slippage) = 52bp."""
        return self.buy_commission + self.slippage

    @property
    def sell_total(self) -> float:
        """Total sell cost fraction (commission + stamp + slippage) = 59bp."""
        return self.sell_commission + self.stamp_tax + self.slippage

    @property
    def sell_commission_all_in(self) -> float:
        """The single number production configures: 9bp incl. stamp duty."""
        return self.sell_commission + self.stamp_tax


PRODUCTION_COSTS = ProductionCosts()


def production_mpo_config(
    base: MpoConfig | None = None,
    costs: ProductionCosts = PRODUCTION_COSTS,
    **overrides: Any,
) -> MpoConfig:
    """Frozen defaults with the production cost trio swapped in.

    Only the four cost fields are replaced; everything else (gamma, w_max,
    stages, solver) stays at the frozen default unless passed in ``overrides``.
    Production portfolio params live in :data:`PRODUCTION_ALGO_PARAMS` and are
    deliberately *not* applied automatically -- swapping NAV to 500e8 and w_max
    to 0.005 changes the experiment, not the cost contract.

    >>> cfg = production_mpo_config()
    >>> round(cfg.buy_cost() * 1e4, 6), round(cfg.sell_cost() * 1e4, 6)
    (52.0, 59.0)
    """
    cfg = MpoConfig() if base is None else base
    return dataclasses.replace(
        cfg,
        buy_commission=costs.buy_commission,
        sell_commission=costs.sell_commission,
        stamp_tax=costs.stamp_tax,
        slippage=costs.slippage,
        **overrides,
    )


# ---------------------------------------------------------------------------
# (date, seq) -> (day, minute) -> tau
# ---------------------------------------------------------------------------

def minute_of_seq(seq: int) -> int:
    """Minute-of-day in ``[0, 240)`` for a production ``seq`` (identity map).

    ``seq`` and the MPO minute index both count elapsed trading minutes from the
    9:30 open, so the map is the identity (see the module docstring, decision 2).
    Raises ``ValueError`` outside ``[1, 237]``: an out-of-range seq means the
    caller is reading a different calendar convention and must be told loudly.
    """
    s = int(seq)
    if not PRODUCTION_SEQ_MIN <= s <= PRODUCTION_SEQ_MAX:
        raise ValueError(
            f"seq must be in [{PRODUCTION_SEQ_MIN}, {PRODUCTION_SEQ_MAX}] "
            f"(production valid_seqs, 9:31-14:57), got {seq}"
        )
    return s


def seq_of_minute(minute: int) -> int:
    """Inverse of :func:`minute_of_seq`; raises for minutes production never scores."""
    m = int(minute)
    if not PRODUCTION_SEQ_MIN <= m <= PRODUCTION_SEQ_MAX:
        raise ValueError(
            f"minute {minute} has no production score seq (scored minutes are "
            f"{PRODUCTION_SEQ_MIN}..{PRODUCTION_SEQ_MAX}); minutes 0/238/239 are "
            "unscored and no epoch offset lands there"
        )
    return m


def is_epoch_seq(seq: int, epoch_minutes: Sequence[int] = EPOCH_OFFSETS) -> bool:
    """True iff this production ``seq`` is an MPO decision epoch."""
    return minute_of_seq(seq) in set(int(m) for m in epoch_minutes)


def tau_of_seq(day: int, seq: int) -> int:
    """Global trading minute ``tau = day * 240 + minute_of_seq(seq)``."""
    d = int(day)
    if d < 0:
        raise ValueError(f"day index must be >= 0, got {day}")
    return d * MINUTES_PER_DAY + minute_of_seq(seq)


class TradingDayIndex:
    """Maps production trade dates (``YYYYMMDD`` ints) to 0-based day indices.

    The MPO calendar is a synthetic no-holiday 240-minute calendar: day indices
    must be *consecutive positions in the replay window*, never derived from the
    date arithmetic. Construct once per replay window from the ordered date list.
    """

    def __init__(self, dates: Sequence[int]) -> None:
        seq = [int(d) for d in dates]
        if not seq:
            raise ValueError("dates must be non-empty")
        if len(set(seq)) != len(seq):
            raise ValueError(f"dates contain duplicates: {sorted(seq)}")
        if any(b <= a for a, b in zip(seq, seq[1:])):
            raise ValueError(f"dates must be strictly ascending, got {seq}")
        self.dates: tuple[int, ...] = tuple(seq)
        self._pos = {d: i for i, d in enumerate(self.dates)}

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def day_of(self, date: Any) -> int:
        d = _date_int(date)
        try:
            return self._pos[d]
        except KeyError:
            raise ValueError(
                f"date {d} is not in the replay window "
                f"[{self.dates[0]}, {self.dates[-1]}] ({len(self.dates)} days)"
            ) from None

    def tau(self, date: Any, seq: int) -> int:
        return tau_of_seq(self.day_of(date), seq)


def _date_int(date: Any) -> int:
    """Production dates arrive as ``alpha.Date`` objects or plain ints."""
    if hasattr(date, "ToInteger"):
        return int(date.ToInteger())
    return int(date)


# ---------------------------------------------------------------------------
# iid alignment
# ---------------------------------------------------------------------------

def _iid_str(iid: Any) -> str:
    """Production iids are ``alpha.InstrumentID``; mirror the repo's coercion
    (``deliverables/opt_bt4_30day_benchmark.py:427``)."""
    if hasattr(iid, "ToString"):
        return str(iid.ToString())
    return str(iid)


class IidUniverse:
    """Fixed asset universe with positional lookup, loud on unknown names.

    The strategy layer is bound to a fixed ``N`` (``Sigma_d`` is ``[N, N]``), so
    every callback has to be projected onto one universe. Unknown iids raise:
    silently dropping a name would silently drop a position.
    """

    def __init__(self, iids: Sequence[Any]) -> None:
        ids = tuple(_iid_str(x) for x in iids)
        if not ids:
            raise ValueError("universe must be non-empty")
        if len(set(ids)) != len(ids):
            seen: set[str] = set()
            dupes = sorted({x for x in ids if x in seen or seen.add(x)})
            raise ValueError(f"universe has duplicate iids: {dupes[:10]}")
        self.iids = ids
        self._pos = {iid: i for i, iid in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.iids)

    @property
    def n_assets(self) -> int:
        return len(self.iids)

    def positions(self, iids: Sequence[Any]) -> np.ndarray:
        """Row indices of ``iids`` in this universe (``ValueError`` if unknown)."""
        names = [_iid_str(x) for x in iids]
        missing = [x for x in names if x not in self._pos]
        if missing:
            raise ValueError(
                f"{len(missing)} iid(s) absent from the universe of "
                f"{self.n_assets}: {missing[:10]}"
            )
        return np.fromiter((self._pos[x] for x in names), dtype=np.int64, count=len(names))

    def align(
        self,
        values: Any,
        iids: Sequence[Any],
        fill: float = np.nan,
        name: str = "values",
    ) -> np.ndarray:
        """Scatter ``values`` (indexed by ``iids``) onto a ``[n_assets]`` vector."""
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size != len(iids):
            raise ValueError(
                f"{name} has {v.size} entries but {len(iids)} iids were given"
            )
        out = np.full(self.n_assets, float(fill), dtype=np.float64)
        out[self.positions(iids)] = v
        return out


def union_universe(iid_lists: Iterable[Sequence[Any]]) -> IidUniverse:
    """Union of several iid lists in first-appearance order."""
    seen: dict[str, None] = {}
    for lst in iid_lists:
        for iid in lst:
            seen.setdefault(_iid_str(iid), None)
    return IidUniverse(tuple(seen))


# ---------------------------------------------------------------------------
# score ingestion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HorizonScores:
    """One score family's payload at one ``(date, seq)``."""

    iids: tuple[str, ...]
    buy_scores: np.ndarray
    sell_scores: np.ndarray

    @classmethod
    def from_payload(cls, iids: Sequence[Any], payload: Any) -> "HorizonScores":
        buy, sell = buy_sell_from_payload(payload)
        names = tuple(_iid_str(x) for x in iids)
        if buy.size != len(names) or sell.size != len(names):
            raise ValueError(
                f"score payload has buy={buy.size} sell={sell.size} entries but "
                f"{len(names)} iids"
            )
        return cls(iids=names, buy_scores=buy, sell_scores=sell)


def buy_sell_from_payload(payload: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalize every production score payload form to ``(buy, sell)`` float64.

    Supported (all three observed in repo prior art,
    ``deliverables/opt_openpo.py:737-745``):

    - ``(buy_scores, sell_scores)`` tuple / 2-element list of arrays,
    - list of objects exposing ``.buy_score`` / ``.sell_score``,
    - mapping with ``'buy_scores'`` / ``'sell_scores'`` keys (convenience).

    A dict keyed by ``identify_str`` (``{'ScoreMain': ...}``) is rejected with a
    pointer to the real problem: a blended ScoreMain payload cannot feed a
    three-horizon term structure.
    """
    if isinstance(payload, Mapping):
        if "buy_scores" in payload and "sell_scores" in payload:
            return (
                np.asarray(payload["buy_scores"], dtype=np.float64).ravel(),
                np.asarray(payload["sell_scores"], dtype=np.float64).ravel(),
            )
        raise ValueError(
            "score payload mapping must carry 'buy_scores'/'sell_scores'; got keys "
            f"{sorted(map(str, payload.keys()))[:5]}. A production "
            "{'ScoreMain': ...} dict holds the BLENDED score (score_weight "
            "0.0,0.33,0.23) and cannot feed a three-horizon term structure -- "
            "expose d/m/t_alpha separately (see CLUSTER_LEG_TODO)"
        )
    pair_like = isinstance(payload, tuple) or (
        isinstance(payload, list)
        and len(payload) == 2
        and all(isinstance(x, (np.ndarray, list, tuple)) for x in payload)
    )
    if pair_like:
        if len(payload) != 2:
            raise ValueError(
                f"tuple score payload must be (buy_scores, sell_scores), got "
                f"{len(payload)} elements"
            )
        return (
            np.asarray(payload[0], dtype=np.float64).ravel(),
            np.asarray(payload[1], dtype=np.float64).ravel(),
        )
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        items = list(payload)
        if items and all(
            hasattr(x, "buy_score") and hasattr(x, "sell_score") for x in items
        ):
            return (
                np.asarray([x.buy_score for x in items], dtype=np.float64),
                np.asarray([x.sell_score for x in items], dtype=np.float64),
            )
    raise ValueError(
        f"unsupported score payload type {type(payload).__name__}; expected a "
        "(buy, sell) tuple, a list of objects with .buy_score/.sell_score, or a "
        "mapping with 'buy_scores'/'sell_scores'"
    )


@dataclass(frozen=True)
class ScoreFrame:
    """One decision epoch's aligned signal cross-section.

    ``raw`` is the ``[N, 3]`` matrix handed to the z-score/term-structure layer:
    column ``k`` is family ``HORIZON_KEYS[k]`` == ``SIGNAL_HORIZONS[k]`` minutes.
    Non-finite entries mean "no usable prediction" (name absent from that family,
    NaN in the feed, or a ``|score| > sentinel`` tradability sentinel) and are
    neutralized to 0 downstream -- never silently imputed.
    """

    date: int
    seq: int
    day: int
    minute: int
    tau: int
    iids: tuple[str, ...]
    raw: np.ndarray  # [N, 3]
    buy: np.ndarray  # [N, 3]
    sell: np.ndarray  # [N, 3]
    limit_up: np.ndarray  # [N] bool -- cannot buy
    limit_down: np.ndarray  # [N] bool -- cannot sell
    halted: np.ndarray  # [N] bool -- cannot trade at all
    spread: np.ndarray  # [3] per-family buy.max() - sell.min()
    gated: bool  # production zero-trade gate fired
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_assets(self) -> int:
        return self.raw.shape[0]


def build_score_frame(
    date: Any,
    seq: int,
    horizon_scores: Mapping[str, Any],
    *,
    day: int,
    universe: IidUniverse | None = None,
    iids: Sequence[Any] | None = None,
    alpha_mode: str = "buy",
    sentinel: float = PRODUCTION_SCORE_SENTINEL,
    buy_sell_diff: float | None = None,
    valid: Any = None,
    halt_unscored: bool = True,
    horizon_keys: Sequence[str] = HORIZON_KEYS,
) -> ScoreFrame:
    """Align one minute's three score families into a :class:`ScoreFrame`.

    Parameters
    ----------
    horizon_scores:
        ``{horizon_key: payload}``. A payload is anything
        :func:`buy_sell_from_payload` accepts, or a :class:`HorizonScores` (use
        that form when the families have *different* iid vectors). All keys in
        ``horizon_keys`` must be present -- a missing family is a configuration
        error, not a NaN column.
    universe:
        Target universe. When ``None`` the union of the families' iids
        (first-appearance order) is used.
    iids:
        Shared iid vector for payloads given as bare ``(buy, sell)`` tuples.
    alpha_mode:
        ``'buy'`` (production semantic, default) or ``'net'`` (``buy - sell``,
        experiment axis only).
    valid:
        Optional production ``valid`` flag array aligned with ``iids``; ``0``
        means untradable -> ``halted``.
    """
    if alpha_mode not in ("buy", "net"):
        raise ValueError(f"alpha_mode must be 'buy' or 'net', got {alpha_mode!r}")
    keys = tuple(horizon_keys)
    if len(keys) != len(SIGNAL_HORIZONS):
        raise ValueError(
            f"horizon_keys must have {len(SIGNAL_HORIZONS)} entries (one per "
            f"SIGNAL_HORIZONS), got {keys}"
        )
    missing = [k for k in keys if k not in horizon_scores]
    if missing:
        raise ValueError(
            f"score families missing from the callback: {missing}; expected all of "
            f"{list(keys)}. MPO-T1 needs the three horizons separately (see "
            "CLUSTER_LEG_TODO)"
        )

    per_horizon: list[HorizonScores] = []
    for k in keys:
        payload = horizon_scores[k]
        if isinstance(payload, HorizonScores):
            per_horizon.append(payload)
            continue
        if iids is None:
            raise ValueError(
                f"family {k!r} payload has no iids; pass iids=... for shared-iid "
                "payloads or wrap each family in HorizonScores"
            )
        per_horizon.append(HorizonScores.from_payload(iids, payload))

    uni = universe if universe is not None else union_universe(h.iids for h in per_horizon)
    n = uni.n_assets

    buy = np.full((n, len(keys)), np.nan, dtype=np.float64)
    sell = np.full((n, len(keys)), np.nan, dtype=np.float64)
    n_missing: dict[str, int] = {}
    spread = np.full(len(keys), -np.inf, dtype=np.float64)
    for k_idx, (key, hs) in enumerate(zip(keys, per_horizon)):
        buy[:, k_idx] = uni.align(hs.buy_scores, hs.iids, name=f"{key}.buy_scores")
        sell[:, k_idx] = uni.align(hs.sell_scores, hs.iids, name=f"{key}.sell_scores")
        n_missing[key] = int(np.count_nonzero(~np.isfinite(buy[:, k_idx])))
        fb, fs = np.isfinite(buy[:, k_idx]), np.isfinite(sell[:, k_idx])
        if fb.any() and fs.any():
            spread[k_idx] = float(buy[fb, k_idx].max() - sell[fs, k_idx].min())

    # Sentinels: tradability flags masquerading as scores.
    sentinel = float(sentinel)
    cannot_buy = np.isfinite(buy) & (buy < -sentinel)
    cannot_sell = np.isfinite(sell) & (sell > sentinel)
    limit_up = cannot_buy.any(axis=1)  # conservative: any family flags it
    limit_down = cannot_sell.any(axis=1)

    raw = buy.copy() if alpha_mode == "buy" else buy - sell
    raw[np.abs(buy) > sentinel] = np.nan
    if alpha_mode == "net":
        raw[np.abs(sell) > sentinel] = np.nan

    halted = np.zeros(n, dtype=bool)
    if halt_unscored:
        halted |= ~np.isfinite(raw).any(axis=1)
    if valid is not None:
        if iids is None and universe is None:
            raise ValueError("valid=... requires iids=... or universe=... for alignment")
        valid_iids = iids if iids is not None else uni.iids
        v = uni.align(valid, valid_iids, fill=0.0, name="valid")
        halted |= v == 0

    gated = bool(buy_sell_diff is not None and np.nanmax(spread) <= float(buy_sell_diff))

    return ScoreFrame(
        date=_date_int(date),
        seq=int(seq),
        day=int(day),
        minute=minute_of_seq(seq),
        tau=tau_of_seq(day, seq),
        iids=uni.iids,
        raw=raw,
        buy=buy,
        sell=sell,
        limit_up=limit_up,
        limit_down=limit_down,
        halted=halted,
        spread=spread,
        gated=gated,
        diagnostics={
            "alpha_mode": alpha_mode,
            "horizon_keys": list(keys),
            "n_assets": n,
            "n_missing_per_family": n_missing,
            "n_limit_up": int(limit_up.sum()),
            "n_limit_down": int(limit_down.sum()),
            "n_halted": int(halted.sum()),
            "n_unscored": int((~np.isfinite(raw).any(axis=1)).sum()),
        },
    )


# ---------------------------------------------------------------------------
# snapshot adapter
# ---------------------------------------------------------------------------

class SnapshotAdapter:
    """Build :class:`~openpo.mpo.types.PortfolioSnapshot` from a callback.

    Production hands weight-unit vectors relative to ``TOTAL_ASSETS``:
    ``weights`` (normalized to sum 1, i.e. fully invested) and
    ``unlimited_weights`` (the T+1 sellable inventory). Shares are recovered as
    ``w * nav / price``; ``sellable_weights`` is ``unlimited_weights`` verbatim.

    Every check is loud (``ValueError``): shape, finiteness, positive prices,
    long-only weights, ``sum(weights) <= 1``, and ``sellable <= holdings``. No
    silent absorption -- production's own ``nan_to_num``/``maximum(0, ...)``
    behaviour is available but must be requested explicitly via
    ``nan_policy='zero'`` / ``clip_negative=True`` so a data outage cannot look
    like a flat book.

    ``iids`` may reorder the callback vectors onto the universe, but it must
    *cover* the universe: a name without a price has no recoverable share count,
    so partial coverage is an error rather than a fabricated price.

    Parameters
    ----------
    universe:
        Fixed universe; callback iids are projected onto it.
    nan_policy:
        ``'raise'`` (default) or ``'zero'`` (production parity).
    normalize_weights:
        ``True`` reproduces production's ``weights /= weights.sum()``.
    sellable_tolerance:
        Weight-unit slack allowed on ``sellable <= weights`` before raising.
        Production re-marks ``unlimited_weights`` from last daily close to
        ``next_avg_px``, so a real feed can exceed ``weights`` by an intraday
        return; the cluster leg must decide whether to widen this or re-mark.
    """

    def __init__(
        self,
        universe: IidUniverse,
        *,
        nan_policy: str = "raise",
        clip_negative: bool = False,
        normalize_weights: bool = False,
        sellable_tolerance: float = 1e-6,
        weight_sum_tolerance: float = 1e-6,
    ) -> None:
        if nan_policy not in ("raise", "zero"):
            raise ValueError(f"nan_policy must be 'raise' or 'zero', got {nan_policy!r}")
        self.universe = universe
        self.nan_policy = nan_policy
        self.clip_negative = bool(clip_negative)
        self.normalize_weights = bool(normalize_weights)
        self.sellable_tolerance = float(sellable_tolerance)
        self.weight_sum_tolerance = float(weight_sum_tolerance)

    # -- helpers -----------------------------------------------------------
    def vector(
        self,
        values: Any,
        iids: Sequence[Any] | None = None,
        name: str = "values",
        fill: float = np.nan,
    ) -> np.ndarray:
        """Public alignment+validation helper (used for the price vector)."""
        return self._vector(values, iids, name, fill)

    def _vector(self, values: Any, iids: Sequence[Any] | None, name: str, fill: float) -> np.ndarray:
        n = self.universe.n_assets
        if iids is None:
            v = np.asarray(values, dtype=np.float64).ravel()
            if v.size != n:
                raise ValueError(
                    f"{name} has {v.size} entries but the universe has {n} assets "
                    "(pass iids=... to align a partial vector)"
                )
            v = v.copy()
        else:
            v = self.universe.align(values, iids, fill=fill, name=name)
        bad = ~np.isfinite(v)
        if bad.any():
            if self.nan_policy == "raise":
                idx = np.flatnonzero(bad)[:10]
                raise ValueError(
                    f"{name} has {int(bad.sum())} non-finite entries at positions "
                    f"{idx.tolist()} (iids {[self.universe.iids[i] for i in idx]}); "
                    "pass nan_policy='zero' to reproduce production's nan_to_num"
                )
            v[bad] = float(fill) if np.isfinite(fill) else 0.0
        return v

    # -- build -------------------------------------------------------------
    def build(
        self,
        tau: int,
        nav: float,
        weights: Any,
        unlimited_weights: Any,
        prices: Any,
        *,
        iids: Sequence[Any] | None = None,
        cash: float | None = None,
    ) -> PortfolioSnapshot:
        n = self.universe.n_assets
        nav = float(nav)
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"nav must be finite and positive, got {nav}")
        if iids is not None:
            covered = {_iid_str(x) for x in iids}
            gap = [x for x in self.universe.iids if x not in covered]
            if gap:
                raise ValueError(
                    f"callback covers {len(covered)} of {n} universe names; "
                    f"{len(gap)} missing (e.g. {gap[:10]}). A name without a price "
                    "has no recoverable share count -- shrink the universe or fix "
                    "the feed instead of fabricating one"
                )

        w = self._vector(weights, iids, "weights", 0.0)
        u = self._vector(unlimited_weights, iids, "unlimited_weights", 0.0)
        px = self._vector(prices, iids, "prices", np.nan)
        if not np.all(px > 0):
            idx = np.flatnonzero(~(px > 0))[:10]
            raise ValueError(
                f"prices must be strictly positive; {len(idx)}+ bad entries at "
                f"{idx.tolist()} (iids {[self.universe.iids[i] for i in idx]})"
            )

        if self.clip_negative:
            w = np.maximum(w, 0.0)
            u = np.maximum(u, 0.0)
        else:
            for name, vec in (("weights", w), ("unlimited_weights", u)):
                neg = vec < -self.sellable_tolerance
                if neg.any():
                    idx = np.flatnonzero(neg)[:10]
                    raise ValueError(
                        f"{name} must be non-negative (long-only book); "
                        f"{int(neg.sum())} negative entries at {idx.tolist()}, "
                        f"min={vec.min():.6g}; pass clip_negative=True to "
                        "reproduce production's maximum(0, ...)"
                    )

        w_sum = float(w.sum())
        if self.normalize_weights:
            if w_sum <= 0:
                raise ValueError(
                    f"cannot normalize weights with sum {w_sum:.6g} (production "
                    "asserts a positive weight sum)"
                )
            w = w / w_sum
            u = u / w_sum
            w_sum = float(w.sum())
        if w_sum > 1.0 + self.weight_sum_tolerance:
            raise ValueError(
                f"sum(weights) = {w_sum:.8f} > 1: weights must be NAV fractions "
                "(pass normalize_weights=True for production's weights/=sum)"
            )

        excess = u - w
        over = excess > self.sellable_tolerance
        if over.any():
            idx = np.flatnonzero(over)[:10]
            raise ValueError(
                f"unlimited_weights exceeds weights for {int(over.sum())} names at "
                f"{idx.tolist()} (max excess {float(excess.max()):.6g} > tol "
                f"{self.sellable_tolerance:.6g}): T+1 sellable cannot exceed the "
                "holding. Production re-marks unlimited_weights to next_avg_px -- "
                "widen sellable_tolerance only with cluster evidence"
            )
        sellable_w = u  # verbatim passthrough; anything beyond tolerance raised above

        holdings = w * nav / px
        sellable_shares = sellable_w * nav / px
        cash_value = nav * (1.0 - w_sum) if cash is None else float(cash)
        if not np.isfinite(cash_value):
            raise ValueError(f"cash must be finite, got {cash_value}")

        return PortfolioSnapshot(
            tau=int(tau),
            nav=nav,
            cash=cash_value,
            holdings_shares=holdings,
            weights=w,
            sellable_shares=sellable_shares,
            sellable_weights=sellable_w,
        )


# ---------------------------------------------------------------------------
# replay harness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CallbackRecord:
    """One production minute callback, already unpacked.

    ``horizon_scores`` is ``{horizon_key: payload}`` (see
    :func:`build_score_frame`). ``nav`` is production ``TOTAL_ASSETS``;
    ``prices`` should be the execution reference price
    (``next_avg_px[seq+1]`` in production).
    """

    date: int
    seq: int
    iids: tuple[str, ...]
    horizon_scores: Mapping[str, Any]
    weights: np.ndarray
    unlimited_weights: np.ndarray
    nav: float
    prices: np.ndarray
    valid: np.ndarray | None = None


@dataclass(frozen=True)
class ReplayDecision:
    """Outcome of one decision epoch."""

    date: int
    seq: int
    day: int
    minute: int
    tau: int
    target_weights: np.ndarray | None
    status: str
    note: str | None
    solve_time: float
    frame: ScoreFrame
    snapshot: PortfolioSnapshot
    diagnostics: dict = field(default_factory=dict)

    @property
    def traded(self) -> bool:
        return self.target_weights is not None


class ReplayHarness:
    """Drive a :class:`~openpo.mpo.strategies.Strategy` from production callbacks.

    This is the seam the (PENDING) cluster 30-day replay plugs into: production
    calls :meth:`handle_score` once per minute with the raw callback arguments;
    non-epoch minutes return ``None`` (no decision), epoch minutes return a
    :class:`ReplayDecision` whose ``target_weights`` is the execution-stage
    target ``w^1`` (or ``None`` when the solver failed / the production
    zero-trade gate fired).

    The :class:`~openpo.mpo.types.DecisionContext` is built exactly as
    :meth:`openpo.mpo.simulator.Simulator.run` builds it -- NaN-safe
    cross-sectional z-score of the raw ``[N, 3]`` matrix, then z zeroed on
    halted names -- so an adapter-fed decision is bit-identical to the
    simulator-path decision on identical inputs (asserted by the parity test).
    """

    def __init__(
        self,
        strategy: Any,
        universe: IidUniverse,
        *,
        n_days: int,
        cfg: MpoConfig | None = None,
        epoch_minutes: Sequence[int] = EPOCH_OFFSETS,
        alpha_mode: str = "buy",
        sentinel: float = PRODUCTION_SCORE_SENTINEL,
        buy_sell_diff: float | None = None,
        horizon_keys: Sequence[str] = HORIZON_KEYS,
        snapshot_adapter: SnapshotAdapter | None = None,
        day_index: TradingDayIndex | None = None,
        halt_unscored: bool = True,
    ) -> None:
        self.strategy = strategy
        self.universe = universe
        self.n_days = int(n_days)
        if self.n_days <= 0:
            raise ValueError(f"n_days must be positive, got {n_days}")
        self.cfg = cfg if cfg is not None else production_mpo_config()
        self.epoch_minutes = tuple(int(m) for m in epoch_minutes)
        self.alpha_mode = alpha_mode
        self.sentinel = float(sentinel)
        self.buy_sell_diff = buy_sell_diff
        self.horizon_keys = tuple(horizon_keys)
        self.adapter = snapshot_adapter or SnapshotAdapter(universe)
        self.day_index = day_index
        self.halt_unscored = bool(halt_unscored)

        n_strategy = getattr(strategy, "n_assets", None)
        if n_strategy is not None and int(n_strategy) != universe.n_assets:
            raise ValueError(
                f"strategy is bound to {int(n_strategy)} assets but the universe "
                f"has {universe.n_assets}; MPO risk inputs are per-universe"
            )

        self.counters: dict[str, int] = {
            "callbacks": 0,
            "epochs": 0,
            "skipped_non_epoch": 0,
            "gated": 0,
            "fail_soft": 0,
            "decided": 0,
        }

    # -- day resolution ----------------------------------------------------
    def _day_of(self, date: Any, day: int | None) -> int:
        if day is not None:
            return int(day)
        if self.day_index is None:
            raise ValueError(
                "day index unavailable: construct ReplayHarness with "
                "day_index=TradingDayIndex(dates) or pass day=... explicitly"
            )
        return self.day_index.day_of(date)

    # -- context -----------------------------------------------------------
    def build_context(
        self, frame: ScoreFrame, snapshot: PortfolioSnapshot, prices: np.ndarray
    ) -> DecisionContext:
        """Frame + snapshot + aligned prices -> ``DecisionContext``.

        The z-score path is deliberately identical to
        ``Simulator.run`` (``src/openpo/mpo/simulator.py:238-239``): NaN-safe
        cross-sectional z-score, non-finite -> 0 (neutral), then halted names
        zeroed. Keeping this line-for-line identical is what makes the parity
        test possible.
        """
        z = np.nan_to_num(zscore_cross_section(frame.raw), nan=0.0, posinf=0.0, neginf=0.0)
        if frame.halted.any():
            z[frame.halted] = 0.0
        return DecisionContext(
            snapshot=snapshot,
            z_signals=z,
            prices=np.asarray(prices, dtype=np.float64),
            limit_up=frame.limit_up,
            limit_down=frame.limit_down,
            halted=frame.halted,
            day=frame.day,
            minute=frame.minute,
        )

    # -- main seam ---------------------------------------------------------
    def handle_score(
        self,
        date: Any,
        seq: int,
        iids: Sequence[Any],
        scores: Mapping[str, Any],
        weights: Any,
        unlimited_weights: Any,
        *,
        nav: float,
        prices: Any,
        valid: Any = None,
        day: int | None = None,
    ) -> ReplayDecision | None:
        """Production-shaped entry point; ``None`` on non-epoch minutes.

        Argument order mirrors the production callback tail
        ``(date, seq, iids, scores, weights, unlimited_weights)``; the extras
        (``nav``, ``prices``, ``valid``) are the production state the callback
        reads off ``self`` (``TOTAL_ASSETS``, ``next_avg_px[seq+1]``,
        ``GetFeatureValue('valid', ...)``) and must be supplied explicitly here.
        """
        self.counters["callbacks"] += 1
        minute = minute_of_seq(seq)
        if minute not in self.epoch_minutes:
            self.counters["skipped_non_epoch"] += 1
            return None
        self.counters["epochs"] += 1

        d = self._day_of(date, day)
        if d >= self.n_days:
            raise ValueError(
                f"day index {d} outside the harness calendar of {self.n_days} days"
            )
        frame = build_score_frame(
            date,
            seq,
            scores,
            day=d,
            universe=self.universe,
            iids=iids,
            alpha_mode=self.alpha_mode,
            sentinel=self.sentinel,
            buy_sell_diff=self.buy_sell_diff,
            valid=valid,
            halt_unscored=self.halt_unscored,
            horizon_keys=self.horizon_keys,
        )
        snapshot = self.adapter.build(
            frame.tau, nav, weights, unlimited_weights, prices, iids=iids
        )
        px = self.adapter.vector(prices, iids, "prices")

        diagnostics = dict(frame.diagnostics)
        diagnostics.update(
            {
                "nav": float(nav),
                "weight_sum": float(np.sum(snapshot.weights)),
                "sellable_sum": float(np.sum(snapshot.sellable_weights)),
                "spread": frame.spread.tolist(),
            }
        )

        if frame.gated:
            self.counters["gated"] += 1
            return ReplayDecision(
                date=frame.date,
                seq=frame.seq,
                day=frame.day,
                minute=frame.minute,
                tau=frame.tau,
                target_weights=None,
                status="zero_trade_gate",
                note=(
                    f"buy.max()-sell.min() <= buy_sell_diff={self.buy_sell_diff} "
                    "for every family (production returns a flat zero-trade vector)"
                ),
                solve_time=0.0,
                frame=frame,
                snapshot=snapshot,
                diagnostics=diagnostics,
            )

        ctx = self.build_context(frame, snapshot, px)
        target = self.strategy.decide(ctx)
        solve_time = float(getattr(self.strategy, "last_solve_time", 0.0) or 0.0)
        status = getattr(self.strategy, "last_status", None)
        note = getattr(self.strategy, "last_note", None)

        if target is None:
            self.counters["fail_soft"] += 1
            status = str(status) if status is not None else "fail_soft"
        else:
            target = np.asarray(target, dtype=np.float64).ravel()
            if target.size != self.universe.n_assets or not np.all(np.isfinite(target)):
                raise ValueError(
                    f"strategy returned a bad target at date={frame.date} "
                    f"seq={frame.seq}: shape {target.shape}, "
                    f"{int(np.count_nonzero(~np.isfinite(target)))} non-finite entries"
                )
            self.counters["decided"] += 1
            status = str(status) if status is not None else "optimal"

        diagnostics["solve_time"] = solve_time
        return ReplayDecision(
            date=frame.date,
            seq=frame.seq,
            day=frame.day,
            minute=frame.minute,
            tau=frame.tau,
            target_weights=target,
            status=status,
            note=None if note is None else str(note),
            solve_time=solve_time,
            frame=frame,
            snapshot=snapshot,
            diagnostics=diagnostics,
        )

    def run_day(
        self, callbacks: Iterable[CallbackRecord], *, day: int | None = None
    ) -> list[ReplayDecision]:
        """Replay one day of callbacks; returns the epoch decisions in order."""
        out: list[ReplayDecision] = []
        for cb in callbacks:
            decision = self.handle_score(
                cb.date,
                cb.seq,
                cb.iids,
                cb.horizon_scores,
                cb.weights,
                cb.unlimited_weights,
                nav=cb.nav,
                prices=cb.prices,
                valid=cb.valid,
                day=day,
            )
            if decision is not None:
                out.append(decision)
        return out
