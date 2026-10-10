"""MPO-T1 frozen shared data containers (Iteration 37).

All cross-module dataclasses live here so parallel implementers share one
vocabulary. Arrays are numpy float64 unless noted. Time index convention:
tau = day * 240 + minute, minute in [0, 240).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class T1ViolationError(AssertionError):
    """Raised by the ledger when a sell exceeds sellable inventory."""


@dataclass(frozen=True)
class Stage:
    """Half-open interval (start, end] in absolute trading minutes."""

    index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def midpoint_offset(self, tau_now: int) -> float:
        return 0.5 * (self.start + self.end) - tau_now


@dataclass(frozen=True)
class StageGrid:
    tau_now: int
    stages: tuple[Stage, ...]  # ordered, contiguous, first stage is executed
    today_stage_indices: tuple[int, ...]  # stages ending on the decision day

    @property
    def n_stages(self) -> int:
        return len(self.stages)


@dataclass
class MarketPanel:
    """Synthetic market data. T = n_days * 240 minutes."""

    n_assets: int
    n_days: int
    prices: np.ndarray  # [T, N] minute close prices
    minute_amount: np.ndarray  # [T, N] traded amount (CNY) per minute
    prev_close: np.ndarray  # [D, N] previous-day close (limit reference)
    limit_up: np.ndarray  # [T, N] bool, price at upper limit
    limit_down: np.ndarray  # [T, N] bool
    halted: np.ndarray  # [T, N] bool
    adv_cny: np.ndarray  # [N] average daily traded amount
    beta: np.ndarray  # [N]
    factor_exposures: np.ndarray  # [N, F] (market + style)
    factor_cov_daily: np.ndarray  # [F, F]
    specific_var_daily: np.ndarray  # [N]
    true_seg_alpha: np.ndarray | None = None  # [T, N, 3] oracle segment alphas
    meta: dict = field(default_factory=dict)

    def daily_cov(self) -> np.ndarray:
        x = self.factor_exposures
        return x @ self.factor_cov_daily @ x.T + np.diag(self.specific_var_daily)


@dataclass
class SignalPanel:
    """Observed model predictions, cross-sectionally raw (not z-scored)."""

    signals: np.ndarray  # [T, N, 3] -- columns: return_1_30/1_240/1_2400 predictions
    meta: dict = field(default_factory=dict)


@dataclass
class PortfolioSnapshot:
    """Decision-time state handed to strategies (weight units unless noted)."""

    tau: int
    nav: float
    cash: float
    holdings_shares: np.ndarray  # [N]
    weights: np.ndarray  # [N] holdings value / nav
    sellable_shares: np.ndarray  # [N] T+1 sellable right now
    sellable_weights: np.ndarray  # [N]


@dataclass
class DecisionContext:
    snapshot: PortfolioSnapshot
    z_signals: np.ndarray  # [N, 3] cross-sectionally z-scored signals at tau
    prices: np.ndarray  # [N] last prices
    limit_up: np.ndarray  # [N] bool
    limit_down: np.ndarray  # [N] bool
    halted: np.ndarray  # [N] bool
    day: int
    minute: int


@dataclass(frozen=True)
class Order:
    asset: int
    shares: float  # >0 buy, <0 sell (lots already applied)


@dataclass(frozen=True)
class Fill:
    asset: int
    shares: float  # signed, actually executed
    price: float  # execution price incl. slippage sign
    commission: float
    stamp_tax: float
    slippage_cost: float


@dataclass
class MpoSolution:
    status: str  # 'optimal' | 'inaccurate' | 'failed'
    weights: np.ndarray | None  # [J, N] stage-end target weights
    buys: np.ndarray | None  # [J, N]
    sells: np.ndarray | None  # [J, N]
    objective: float | None
    solve_time: float
    iterations: int = 0


@dataclass
class BacktestResult:
    strategy: str
    seed: int
    daily_nav: np.ndarray  # [D_eval + 1] NAV path (start included)
    daily_turnover: np.ndarray  # [D_eval] traded value / NAV
    costs: dict  # {'commission','stamp_tax','slippage','impact_est'} totals CNY
    rejected_orders: int = 0
    truncated_buys: int = 0
    t1_violations: int = 0  # must stay 0; ledger raises otherwise
    qp_solve_times: list = field(default_factory=list)
    qp_status_counts: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def daily_returns(self) -> np.ndarray:
        nav = np.asarray(self.daily_nav, dtype=float)
        return nav[1:] / nav[:-1] - 1.0

    def net_sharpe_annual(self) -> float:
        r = self.daily_returns()
        sd = r.std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            return 0.0
        return float(r.mean() / sd * np.sqrt(240.0))


@dataclass
class TermStructureDiagnostics:
    seg_ic: np.ndarray | None = None  # [3] in-sample segment spearman IC
    coef_table: np.ndarray | None = None  # [3 seg, 3 sig] fitted B
    rho_epoch: np.ndarray | None = None  # [3] epoch-grid AR(1) autocorr
    n_samples: int = 0
    notes: dict = field(default_factory=dict)
