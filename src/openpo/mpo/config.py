"""MPO-T1 frozen configuration (Iteration 37).

Single source of truth for all default parameters. Implementation modules must
not redefine these values. All horizons/decays are in TRADING minutes
(overnight counts as zero minutes).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field


MINUTES_PER_DAY = 240
EPOCH_OFFSETS = (5, 35, 65, 95, 125, 155, 185, 215)  # decision minutes within a day
SEGMENT_BOUNDS = ((0, 30), (30, 240), (240, 2400))  # non-overlapping alpha segments
SIGNAL_HORIZONS = (30, 240, 2400)  # return_1_30 / return_1_240 / return_1_2400
TRADING_DAYS_PER_YEAR = 240


@dataclass(frozen=True)
class SynthConfig:
    """Synthetic A-share market generator parameters."""

    n_assets: int = 300
    n_days: int = 100
    minutes_per_day: int = MINUTES_PER_DAY
    # latent alpha AR(1) half-lives in trading minutes, one per signal horizon
    alpha_half_lives: tuple[float, float, float] = (30.0, 240.0, 2400.0)
    alpha_cross_corr: float = 0.3
    target_ics: tuple[float, float, float] = (0.05, 0.04, 0.03)
    ic_mult: float = 1.0  # sensitivity axis
    market_vol_annual: float = 0.20
    beta_range: tuple[float, float] = (0.8, 1.2)
    n_style_factors: int = 8
    style_vol_annual_range: tuple[float, float] = (0.04, 0.08)
    idio_daily_vol: float = 0.025
    intraday_u_shape_peak: float = 1.5
    price_init_mean: float = 20.0
    price_init_spread: float = 10.0
    adv_median_cny: float = 8e7
    adv_sigma_log: float = 0.8
    limit_pct: float = 0.10
    limit_hit_rate_daily: float = 0.001
    halt_rate_daily: float = 0.005


# Canonical cost values (fractions): buy 2bp commission + 5bp slippage = 7bp,
# sell adds 5bp stamp tax = 12bp.
BUY_COMMISSION = 2e-4
SELL_COMMISSION = 2e-4
STAMP_TAX = 5e-4
SLIPPAGE = 5e-4


@dataclass(frozen=True)
class MpoConfig:
    """Strategy + experiment configuration."""

    # --- portfolio / risk ---
    nav_cny: float = 1e8
    gamma: float = 10.0  # daily-unit risk aversion
    w_max: float = 0.02
    impact_coef: float = 0.1  # eta scale for diagonal quadratic impact
    # --- costs (fractions) ---
    buy_commission: float = BUY_COMMISSION
    sell_commission: float = SELL_COMMISSION
    stamp_tax: float = STAMP_TAX
    slippage: float = SLIPPAGE
    cost_mult: float = 1.0
    # --- stages / decay ---
    exec_minutes: int = 30
    horizon_minutes: int = 2400
    max_stages: int = 4
    t1_enabled: bool = True
    decay_mult: float = 1.0  # sensitivity: strategy assumes rho^(1/decay_mult)
    cost_mode: str = "production"  # 'production' | 'axioma' (closed-form parity)
    axioma_delta: float = 1.0  # delta for cost_mode='axioma'
    # --- calibration ---
    calib_days: int = 20
    ridge_lambda: float = 1e-3
    winsor_sigma: float = 3.0
    # --- execution ---
    lot_size: int = 100
    participation_rate: float = 0.10
    min_order_nav_frac: float = 2e-4
    slippage_tolerance: float = 3e-3  # limit-price protection band
    # --- solver ---
    qp_backend: str = "osqp"
    qp_eps: float = 1e-6
    qp_max_iter: int = 20000

    def buy_cost(self) -> float:
        return (self.buy_commission + self.slippage) * self.cost_mult

    def sell_cost(self) -> float:
        return (self.sell_commission + self.stamp_tax + self.slippage) * self.cost_mult


@dataclass(frozen=True)
class ExperimentConfig:
    synth: SynthConfig = field(default_factory=SynthConfig)
    mpo: MpoConfig = field(default_factory=MpoConfig)
    n_seeds: int = 24
    seed_base: int = 20260821
    tuning_seed_base: int = 987654
    n_tuning_seeds: int = 5
    eval_start_day: int = 20  # == calib_days; eval window is [calib_days, n_days)
    bootstrap_iters: int = 10000


def config_hash(cfg) -> str:
    """SHA256 of a (possibly nested) dataclass, stable across runs."""

    def encode(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [encode(x) for x in obj]
        return obj

    blob = json.dumps(encode(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
