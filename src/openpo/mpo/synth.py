"""Synthetic A-share market generator for MPO-T1 (SPEC section 6).

``generate_market(scfg, seed)`` returns ``(MarketPanel, SignalPanel)`` for a
``N x D x 240``-minute synthetic universe with a *known* alpha term structure,
so every downstream module (term-structure calibration, MPO, oracle strategy)
can be scored against ground truth.

Frozen conventions repeated here on purpose
-------------------------------------------
- ``tau = day * minutes_per_day + minute``; ``T = n_days * minutes_per_day``.
- Every horizon / decay is counted in **trading minutes**; overnight counts as
  **zero** minutes, i.e. the latent alpha clock is the continuous trading
  minute clock ``tau`` (no holiday gaps in the synthetic calendar).
- All parameters come from :class:`openpo.mpo.config.SynthConfig`; nothing is
  re-defined here except the two SPEC constants below that are prose-only.

Generative model
----------------
Per-minute simple return of asset ``i`` at ``tau``::

    r[tau, i] = sum_k c_k * m_k[tau, i]                      (alpha)
              + u[minute] * ( beta_i * f_mkt[tau]            (market factor)
                            + 0.3 * X_i . g[tau]             (style factors)
                            + idio_vol_i / sqrt(M) * e[tau, i] )

- ``m_k`` are the three latent alphas: AR(1) on the trading-minute clock with
  half lives ``scfg.alpha_half_lives`` (``rho_k = 0.5 ** (1 / half_life_k)``),
  unit stationary variance, driven by innovations whose correlation matrix has
  ``scfg.alpha_cross_corr`` off-diagonals.  NOTE: imposing the target
  correlation on *levels* of AR(1)s with such different memories is not
  positive-semidefinite-feasible (it would need innovation correlation > 1), so
  the correlation is imposed on the innovations; the resulting *level*
  correlations are attenuated for widely separated half lives and are reported
  in ``meta['level_corr_m']``.
- ``u`` is the intraday U-shape (first/last minute ``scfg.intraday_u_shape_peak``
  times mid-day), RMS-normalised so it redistributes but does not change total
  daily variance.  It is applied to the *noise* components only, which keeps the
  alpha conditional mean exactly analytic (see below).
- ``c_k`` is calibrated so the true conditional mean reaches a cross-sectional
  IC of ``2 x scfg.target_ics`` per horizon (the oracle upper bound).

Because the alpha part carries no seasonality, the conditional mean of the
cumulated return is an exact AR(1) geometric sum::

    E[ sum_{j=a+1..b} r[tau+j, i] | F_tau ] = sum_k c_k * m_k[tau, i]
                                              * (rho_k^(a+1) - rho_k^(b+1))
                                                / (1 - rho_k)

which is what ``MarketPanel.true_seg_alpha[tau, i, s]`` stores for the three
non-overlapping segments ``config.SEGMENT_BOUNDS`` (market/style contributions
are deliberately excluded -- it is an alpha oracle, not a return oracle).

Observed predictions (``SignalPanel.signals``, RAW, not z-scored) are the
*cumulative*-horizon conditional means (segments telescope onto
``config.SIGNAL_HORIZONS``) plus gaussian noise whose per-horizon amplitude is
found by bisection so that the measured cross-sectional Spearman IC lands on
``scfg.target_ics * scfg.ic_mult`` on a post-burn-in window.

Microstructure: prices are GBM from ``U(price_init_mean +- price_init_spread)``
compounded day by day off the previous close; ADV is lognormal with median
``scfg.adv_median_cny`` and spread ``scfg.adv_sigma_log``, spread over the day
with the same U-shape; ``+-limit_pct`` limits are forced on
``scfg.limit_hit_rate_daily`` of stock-days (the breaching direction freezes for
the rest of that day); ``scfg.halt_rate_daily`` of stock-days are halted (price
frozen at the previous close, zero volume, ``halted`` flag set).  Signals are
kept finite on halted names (the simulator freezes them via ``halted``).

Determinism: one ``np.random.default_rng(seed)`` root, split into fixed
per-component child streams via ``Generator.spawn``, so the same seed is
bit-identical and no component can shift another's draws.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import rankdata

from .config import (
    SEGMENT_BOUNDS,
    SIGNAL_HORIZONS,
    TRADING_DAYS_PER_YEAR,
    SynthConfig,
)
from .types import MarketPanel, SignalPanel

__all__ = [
    "ORACLE_IC_MULTIPLE",
    "STYLE_RETURN_WEIGHT",
    "cross_sectional_spearman_ic",
    "forward_return",
    "generate_market",
    "ic_eval_taus",
    "intraday_shape",
    "measure_segment_ics",
    "measure_signal_ics",
    "segment_return",
    "u_shape_amount",
    "u_shape_vol",
]

# SPEC section 6 prose-only constants (not tunable knobs, hence not in config).
STYLE_RETURN_WEIGHT = 0.3  # weight of the style-factor block in minute returns
ORACLE_IC_MULTIPLE = 2.0  # true conditional mean IC target = 2x observed target

# Number of taus sampled when measuring / calibrating a cross-sectional IC.
_IC_SAMPLE_TAUS = 512
# Non-selected stock-days are kept this far inside the price limit band.
_LIMIT_BAND_MARGIN = 1e-3
# AR(1) recursion block length cap (rho^-B must stay numerically tame).
_AR_BLOCK_CAP = 512


# ---------------------------------------------------------------------------
# intraday seasonality
# ---------------------------------------------------------------------------

def intraday_shape(minutes_per_day: int, peak: float) -> np.ndarray:
    """Raw U-shape over a trading day: ``peak`` at both ends, 1.0 at mid-day."""
    m = int(minutes_per_day)
    if m < 1:
        raise ValueError(f"minutes_per_day must be >= 1, got {m}")
    if m == 1:
        return np.ones(1, dtype=np.float64)
    x = np.arange(m, dtype=np.float64) / (m - 1)
    return 1.0 + (float(peak) - 1.0) * 0.5 * (1.0 + np.cos(2.0 * np.pi * x))


def u_shape_vol(minutes_per_day: int, peak: float) -> np.ndarray:
    """Volatility multiplier: RMS-normalised so daily variance is unchanged."""
    s = intraday_shape(minutes_per_day, peak)
    return s / np.sqrt(np.mean(s * s))


def u_shape_amount(minutes_per_day: int, peak: float) -> np.ndarray:
    """Volume weights: sum to 1 so the daily total equals ADV."""
    s = intraday_shape(minutes_per_day, peak)
    return s / s.sum()


# ---------------------------------------------------------------------------
# IC measurement helpers (shared by the generator's calibration and by tests)
# ---------------------------------------------------------------------------

def _row_corr_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over rows of the per-row Pearson correlation of ``a`` and ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"need two identically shaped 2-D arrays, got {a.shape} {b.shape}")
    if a.shape[0] == 0 or a.shape[1] < 2:
        return 0.0
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", ac, bc)
    den = np.sqrt(np.einsum("ij,ij->i", ac, ac) * np.einsum("ij,ij->i", bc, bc))
    ok = den > 0.0
    if not ok.any():
        return 0.0
    return float(np.mean(num[ok] / den[ok]))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Cross-sectional (per-row) average ranks; NaNs rank last, deterministically."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = np.where(np.isfinite(x), x, np.inf)
    return rankdata(x, axis=1)


def cross_sectional_spearman_ic(signal: np.ndarray, future: np.ndarray) -> float:
    """Mean per-tau cross-sectional Spearman IC of ``signal`` vs ``future``.

    Both inputs are ``[S, N]`` (S sampled taus, N assets). Rows with zero
    dispersion are skipped.
    """
    return _row_corr_mean(_ranks(signal), _ranks(future))


def ic_eval_taus(
    n_minutes: int,
    horizon: int,
    max_half_life: float,
    n_max: int = _IC_SAMPLE_TAUS,
    offset: float = 0.0,
) -> np.ndarray:
    """Evenly spaced post-burn-in taus with ``tau + horizon <= T - 1``.

    Burn-in defaults to the longest alpha half life (the AR(1) transient) but
    is shrunk automatically on short panels so tiny test configurations still
    get a usable window instead of an empty one.  ``offset`` shifts the grid by
    that fraction of one stride (used by tests to score on a *different*
    subsample than the one the generator calibrated on).
    """
    t = int(n_minutes)
    h = int(horizon)
    hi = t - h  # tau <= T-1-h  ->  exclusive upper bound
    if hi <= 1:
        return np.empty(0, dtype=np.int64)
    burn = int(min(int(max_half_life), t // 4, max(hi // 2, 0)))
    if hi - burn < 2:
        burn = max(0, hi // 2)
    count = int(min(int(n_max), hi - burn))
    taus = np.linspace(burn, hi - 1, count)
    if offset and count > 1:
        taus = taus + offset * (taus[1] - taus[0])
    taus = np.clip(np.rint(taus), burn, hi - 1).astype(np.int64)
    return np.unique(taus)


def forward_return(prices: np.ndarray, taus: np.ndarray, horizon: int) -> np.ndarray:
    """``P[tau+horizon] / P[tau] - 1`` for every sampled tau -> ``[S, N]``."""
    taus = np.asarray(taus, dtype=np.int64)
    return prices[taus + int(horizon)] / prices[taus] - 1.0


def segment_return(prices: np.ndarray, taus: np.ndarray, bounds: tuple) -> np.ndarray:
    """``P[tau+b] / P[tau+a] - 1`` for a half-open segment ``(a, b]``."""
    taus = np.asarray(taus, dtype=np.int64)
    a, b = int(bounds[0]), int(bounds[1])
    return prices[taus + b] / prices[taus + a] - 1.0


def _max_half_life(market: MarketPanel, scfg: SynthConfig | None) -> float:
    if scfg is not None:
        return float(max(scfg.alpha_half_lives))
    hl = market.meta.get("alpha_half_lives")
    if hl is None:
        return float(max(SIGNAL_HORIZONS))
    return float(max(hl))


def measure_signal_ics(
    market: MarketPanel,
    signals: SignalPanel,
    scfg: SynthConfig | None = None,
    offset: float = 0.0,
    n_max: int = _IC_SAMPLE_TAUS,
) -> np.ndarray:
    """Measured cross-sectional Spearman IC of the observed signals ``[3]``.

    Horizon ``q`` is scored against the realised cumulative return over
    ``SIGNAL_HORIZONS[q]`` trading minutes on the post-burn-in window.
    Horizons with no usable window return ``nan``.
    """
    prices = market.prices
    sig = signals.signals
    hl = _max_half_life(market, scfg)
    out = np.full(len(SIGNAL_HORIZONS), np.nan, dtype=np.float64)
    for q, h in enumerate(SIGNAL_HORIZONS):
        taus = ic_eval_taus(prices.shape[0], h, hl, n_max=n_max, offset=offset)
        if taus.size < 2:
            continue
        out[q] = cross_sectional_spearman_ic(sig[taus][:, :, q], forward_return(prices, taus, h))
    return out


def measure_segment_ics(
    market: MarketPanel,
    scfg: SynthConfig | None = None,
    offset: float = 0.0,
    n_max: int = _IC_SAMPLE_TAUS,
) -> np.ndarray:
    """Oracle IC of ``true_seg_alpha`` vs realised segment returns ``[3]``."""
    if market.true_seg_alpha is None:
        raise ValueError("market.true_seg_alpha is required to measure segment ICs")
    prices = market.prices
    tsa = market.true_seg_alpha
    hl = _max_half_life(market, scfg)
    out = np.full(len(SEGMENT_BOUNDS), np.nan, dtype=np.float64)
    for s, (a, b) in enumerate(SEGMENT_BOUNDS):
        taus = ic_eval_taus(prices.shape[0], int(b), hl, n_max=n_max, offset=offset)
        if taus.size < 2:
            continue
        out[s] = cross_sectional_spearman_ic(
            tsa[taus][:, :, s], segment_return(prices, taus, (a, b))
        )
    return out


# ---------------------------------------------------------------------------
# analytic AR(1) building blocks
# ---------------------------------------------------------------------------

def _geometric_sum(rho: np.ndarray, a: int, b: int) -> np.ndarray:
    """``sum_{j=a+1..b} rho^j`` (elementwise over ``rho``)."""
    rho = np.asarray(rho, dtype=np.float64)
    out = np.empty_like(rho)
    near_one = np.abs(1.0 - rho) < 1e-15
    out[near_one] = float(b - a)
    r = rho[~near_one]
    out[~near_one] = (r ** (a + 1) - r ** (b + 1)) / (1.0 - r)
    return out


def _level_corr(rho: np.ndarray, corr_eps: np.ndarray) -> np.ndarray:
    """Stationary level correlation of AR(1)s driven by correlated innovations."""
    s = np.sqrt(1.0 - rho * rho)
    denom = 1.0 - np.outer(rho, rho)
    r = corr_eps * np.outer(s, s) / denom
    np.fill_diagonal(r, 1.0)
    return r


def _solve_alpha_scales(
    g: np.ndarray, var_target: np.ndarray, corr_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for ``c`` so that ``Var(sum_k c_k g[q,k] m_k) == var_target[q]``.

    ``g[q, k]`` is the AR(1) geometric sum of signal ``k`` over horizon ``q``.
    Ignoring the cross-signal correlation the system is linear in ``c**2``,
    which gives an excellent starting point; the correlated system is then
    refined with a 3-parameter least-squares fit on ``log c`` (deterministic).
    """
    g2 = g * g
    try:
        x0 = np.linalg.solve(g2, var_target)
    except np.linalg.LinAlgError:  # pragma: no cover - g2 is well conditioned
        x0 = np.linalg.lstsq(g2, var_target, rcond=None)[0]
    floor = 1e-14 * max(float(np.max(var_target)), 1e-30)
    c0 = np.sqrt(np.maximum(x0, floor))

    def model_var(c: np.ndarray) -> np.ndarray:
        a = g * c[None, :]  # [q, k]
        return np.einsum("qj,jk,qk->q", a, corr_m, a)

    log_target = np.log(var_target)

    def resid(p: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(model_var(np.exp(p)), 1e-300)) - log_target

    sol = least_squares(resid, np.log(c0), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    c = np.exp(sol.x)
    if not np.all(np.isfinite(c)) or np.max(np.abs(resid(sol.x))) > np.max(
        np.abs(resid(np.log(c0)))
    ):
        c = c0  # fail-soft: keep the closed-form start if the refinement is worse
    return c, model_var(c)


# ---------------------------------------------------------------------------
# latent alpha simulation
# ---------------------------------------------------------------------------

def _ar1_inplace(x: np.ndarray, rho: float, state: np.ndarray, block: int) -> None:
    """In-place AR(1) filter along axis 0 of ``x`` (innovations -> levels).

    ``x[t] <- rho * x[t-1] + sqrt(1-rho^2) * x[t]`` with ``x[-1] = state``.
    Blocked closed form (``rho^-i`` cumulative sums) -- fully vectorised, no
    Python loop over minutes.  The block length is capped so ``rho**-(b-1)``
    stays numerically tame; results are block-length invariant.
    """
    t = x.shape[0]
    s = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
    if rho <= 0.0:  # memoryless
        x *= s
        return
    if rho >= 1.0:  # degenerate: infinite half life, the level never moves
        x[:] = np.asarray(state, dtype=np.float64)[None, :]
        return
    b = int(min(block, t, max(1, 1 + int(np.log(1e12) / -np.log(rho)))))
    k = np.arange(b, dtype=np.float64)
    decay = rho ** k
    inv = rho ** (-k)
    cur = np.asarray(state, dtype=np.float64).copy()
    for start in range(0, t, b):
        n = min(b, t - start)
        blk = x[start : start + n]
        blk *= (s * inv[:n])[:, None]
        np.cumsum(blk, axis=0, out=blk)
        blk += (rho * cur)[None, :]
        blk *= decay[:n, None]
        cur = blk[n - 1].copy()


def _latent_alphas(rng, n_minutes: int, n_assets: int, rho: np.ndarray,
                   corr_eps: np.ndarray, corr_m: np.ndarray, block: int) -> np.ndarray:
    """Latent alpha panel ``m[k, T, N]`` (unit stationary variance)."""
    k = rho.size
    m = rng.standard_normal((k, n_minutes, n_assets))
    # correlate innovations in place (lower-triangular chol, reverse order)
    lower = np.linalg.cholesky(corr_eps)
    for j in range(k - 1, -1, -1):
        acc = lower[j, j] * m[j]
        for i in range(j):
            acc += lower[j, i] * m[i]
        m[j] = acc
    # stationary initial state, correlated like the levels
    init = rng.standard_normal((n_assets, k)) @ np.linalg.cholesky(corr_m).T
    for j in range(k):
        _ar1_inplace(m[j], float(rho[j]), init[:, j], block)
    return m


# ---------------------------------------------------------------------------
# signal noise calibration
# ---------------------------------------------------------------------------

def _ic_with_noise(true_sub, eps_sub, sigma, fut_ranks) -> float:
    return _row_corr_mean(_ranks(true_sub + sigma * eps_sub), fut_ranks)


def _bisect_noise_sigma(true_sub, eps_sub, fut_ranks, target, n_iter=40):
    """Smallest-bracket bisection on the noise scale; returns ``(sigma, ic)``."""
    ic0 = _ic_with_noise(true_sub, eps_sub, 0.0, fut_ranks)
    if not np.isfinite(ic0) or ic0 <= target:
        return 0.0, ic0  # target unreachable: keep the oracle (fail-soft)
    scale = float(np.std(true_sub))
    hi = scale if scale > 0 else 1.0
    for _ in range(30):
        if _ic_with_noise(true_sub, eps_sub, hi, fut_ranks) <= target:
            break
        hi *= 2.0
    lo = 0.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if _ic_with_noise(true_sub, eps_sub, mid, fut_ranks) > target:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    return sigma, _ic_with_noise(true_sub, eps_sub, sigma, fut_ranks)


def _analytic_noise_sigma(var_alpha: float, var_noise: float, target: float) -> float:
    """Closed-form noise scale when no window is long enough to measure an IC.

    From ``corr(A + n, A + N) = Var(A) / sqrt((Var A + s^2)(Var A + Var N))``,
    with a small-rho Spearman->Pearson correction (``6/pi * asin(rho/2)``).
    """
    t = float(target) / 0.9549
    t = min(max(t, 1e-9), 0.999)
    denom = t * t * (var_alpha + var_noise)
    if denom <= 0.0:
        return 0.0
    s2 = var_alpha * var_alpha / denom - var_alpha
    return float(np.sqrt(max(s2, 0.0)))


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def generate_market(scfg: SynthConfig, seed: int) -> tuple[MarketPanel, SignalPanel]:
    """Generate one synthetic market path + observed signal panel.

    Parameters
    ----------
    scfg : frozen :class:`~openpo.mpo.config.SynthConfig` (sole parameter source).
    seed : integer seed; the same seed reproduces bit-identical panels.

    Returns
    -------
    (MarketPanel, SignalPanel) -- see :mod:`openpo.mpo.types`. Calibration
    diagnostics (alpha scales, noise scales, achieved ICs) land in
    ``MarketPanel.meta`` / ``SignalPanel.meta``.
    """
    n = int(scfg.n_assets)
    d = int(scfg.n_days)
    mpd = int(scfg.minutes_per_day)
    if n < 2:
        raise ValueError(f"n_assets must be >= 2 (cross-sectional IC needs a panel), got {n}")
    if d < 1:
        raise ValueError(f"n_days must be >= 1, got {d}")
    if mpd < 2:
        raise ValueError(f"minutes_per_day must be >= 2, got {mpd}")
    t = d * mpd
    n_sig = len(SIGNAL_HORIZONS)
    if len(scfg.alpha_half_lives) != n_sig or len(scfg.target_ics) != n_sig:
        raise ValueError("alpha_half_lives / target_ics must have one entry per signal horizon")
    # The memory-lean signal build telescopes segments onto the horizons.
    cum_bounds = [(0, h) for h in SIGNAL_HORIZONS]
    tiled = [(SEGMENT_BOUNDS[0][0], SEGMENT_BOUNDS[s][1]) for s in range(n_sig)]
    if [tuple(x) for x in tiled] != cum_bounds or any(
        SEGMENT_BOUNDS[s][0] != SEGMENT_BOUNDS[s - 1][1] for s in range(1, n_sig)
    ):
        raise ValueError(
            "SEGMENT_BOUNDS must tile (0, SIGNAL_HORIZONS[q]] contiguously; "
            f"got {SEGMENT_BOUNDS} vs {SIGNAL_HORIZONS}"
        )

    # ---- fixed per-component child streams (determinism) ------------------
    (
        rng_alpha,
        rng_factor,
        rng_idio,
        rng_char,
        rng_micro,
        rng_amount,
        rng_event,
        rng_noise,
    ) = np.random.default_rng(int(seed)).spawn(8)

    # ---- intraday seasonality --------------------------------------------
    uv = u_shape_vol(mpd, scfg.intraday_u_shape_peak)
    ua = u_shape_amount(mpd, scfg.intraday_u_shape_peak)
    u_t = np.tile(uv, d)  # [T]

    # ---- static asset characteristics ------------------------------------
    beta = rng_char.uniform(scfg.beta_range[0], scfg.beta_range[1], size=n)
    n_style = int(scfg.n_style_factors)
    style_load = rng_char.standard_normal((n, n_style))
    style_vol_ann = rng_char.uniform(
        scfg.style_vol_annual_range[0], scfg.style_vol_annual_range[1], size=n_style
    )
    idio_scale = np.exp(0.25 * rng_char.standard_normal(n))
    idio_scale /= np.sqrt(np.mean(idio_scale * idio_scale))  # RMS 1 -> mean var exact
    idio_vol_daily = float(scfg.idio_daily_vol) * idio_scale

    sqrt_days = np.sqrt(float(TRADING_DAYS_PER_YEAR))
    sqrt_min = np.sqrt(float(mpd))
    mkt_vol_daily = float(scfg.market_vol_annual) / sqrt_days
    style_vol_daily = style_vol_ann / sqrt_days

    # ---- latent alphas ----------------------------------------------------
    half_lives = np.asarray(scfg.alpha_half_lives, dtype=np.float64)
    if np.any(half_lives <= 0):
        raise ValueError(f"alpha_half_lives must be positive, got {scfg.alpha_half_lives}")
    rho = 0.5 ** (1.0 / half_lives)
    corr_eps = np.full((n_sig, n_sig), float(scfg.alpha_cross_corr), dtype=np.float64)
    np.fill_diagonal(corr_eps, 1.0)
    corr_m = _level_corr(rho, corr_eps)
    block = int(min(_AR_BLOCK_CAP, max(mpd, 1)))
    m = _latent_alphas(rng_alpha, t, n, rho, corr_eps, corr_m, block)  # [3, T, N]

    # ---- factor + idiosyncratic minute returns ---------------------------
    f_mkt = rng_factor.standard_normal(t) * (mkt_vol_daily / sqrt_min) * u_t
    g_style = rng_factor.standard_normal((t, n_style))
    g_style *= (style_vol_daily / sqrt_min)[None, :]
    g_style *= u_t[:, None]

    r = rng_idio.standard_normal((t, n))
    r *= (idio_vol_daily / sqrt_min)[None, :]
    r *= u_t[:, None]
    r += f_mkt[:, None] * beta[None, :]
    r += (STYLE_RETURN_WEIGHT * g_style) @ style_load.T

    # ---- cross-sectional noise scale (per sqrt(minute)) -------------------
    base_h = int(min(mpd, max(2, t // 4)))
    csum = np.cumsum(r, axis=0)
    base_taus = ic_eval_taus(t, base_h, 0.0, n_max=_IC_SAMPLE_TAUS)
    if base_taus.size < 2:
        base_taus = np.array([0], dtype=np.int64)
    base_noise = csum[base_taus + base_h] - csum[base_taus]
    del csum
    var_noise_base = float(np.mean(np.var(base_noise, axis=1, ddof=1)))
    var_noise_per_min = var_noise_base / base_h
    var_noise = var_noise_per_min * np.asarray(SIGNAL_HORIZONS, dtype=np.float64)

    # ---- alpha scale calibration (oracle IC = 2x target) -----------------
    target_ic = np.asarray(scfg.target_ics, dtype=np.float64)
    obs_target = np.clip(target_ic * float(scfg.ic_mult), 0.0, 0.95)
    oracle_target = np.clip(
        np.maximum(ORACLE_IC_MULTIPLE * target_ic, 1.2 * obs_target), 1e-6, 0.95
    )
    g_mat = np.empty((n_sig, n_sig), dtype=np.float64)
    for q, h in enumerate(SIGNAL_HORIZONS):
        g_mat[q] = _geometric_sum(rho, 0, int(h))
    var_alpha_target = var_noise * (oracle_target**2 / (1.0 - oracle_target**2))
    c_alpha, var_alpha = _solve_alpha_scales(g_mat, var_alpha_target, corr_m)

    # alpha contribution to minute returns (no seasonality -> exact geo sums)
    r += np.tensordot(c_alpha, m, axes=([0], [0]))

    # ---- prices, limits, halts (sequential over days, vectorised inside) --
    px0 = rng_micro.uniform(
        scfg.price_init_mean - scfg.price_init_spread,
        scfg.price_init_mean + scfg.price_init_spread,
        size=n,
    )
    if np.any(px0 <= 0):
        raise ValueError("price_init_mean - price_init_spread must stay positive")
    halt_day = rng_event.random((d, n)) < float(scfg.halt_rate_daily)
    limit_day = rng_event.random((d, n)) < float(scfg.limit_hit_rate_daily)
    limit_dir = np.where(rng_event.random((d, n)) < 0.5, -1.0, 1.0)
    limit_minute = rng_event.integers(1, mpd, size=(d, n)) if mpd > 1 else np.zeros((d, n), int)

    prices = np.empty((t, n), dtype=np.float64)
    prev_close = np.empty((d, n), dtype=np.float64)
    limit_up = np.zeros((t, n), dtype=bool)
    limit_down = np.zeros((t, n), dtype=bool)
    halted = np.zeros((t, n), dtype=bool)
    lp = float(scfg.limit_pct)
    minute_idx = np.arange(mpd, dtype=np.float64)[:, None]

    base = px0.copy()
    for day in range(d):
        prev_close[day] = base
        sl = slice(day * mpd, (day + 1) * mpd)
        growth = np.cumprod(1.0 + r[sl], axis=0)  # [M, N]
        px = base[None, :] * growth

        # force the configured limit-hit rate: ramp to the band by a random minute
        cols = np.flatnonzero(limit_day[day])
        if cols.size:
            m0 = limit_minute[day, cols].astype(np.float64)[None, :]
            ramp = np.minimum(1.0, (minute_idx + 1.0) / (m0 + 1.0))
            gcol = growth[:, cols]
            px[:, cols] = base[cols][None, :] * (
                1.0 + limit_dir[day, cols][None, :] * lp * ramp + (gcol - 1.0) * (1.0 - ramp)
            )

        cap = base * (1.0 + lp)
        flr = base * (1.0 - lp)
        # Non-selected stock-days are compressed strictly inside the band so the
        # REALISED limit-hit rate equals scfg.limit_hit_rate_daily. Without this
        # the natural breach rate (2.5% daily vol vs a 10% band, running-maximum
        # reflection) is ~3x the configured rate and the knob is meaningless.
        if cols.size < n:
            free = ~limit_day[day]
            px[:, free] = np.clip(
                px[:, free],
                (flr * (1.0 + _LIMIT_BAND_MARGIN))[free][None, :],
                (cap * (1.0 - _LIMIT_BAND_MARGIN))[free][None, :],
            )
        up = np.logical_or.accumulate(px >= cap[None, :] * (1.0 - 1e-12), axis=0)
        dn = np.logical_or.accumulate(px <= flr[None, :] * (1.0 + 1e-12), axis=0)
        dn &= ~up  # a frozen upper limit wins; the two flags stay exclusive
        if up.any():
            px = np.where(up, np.broadcast_to(cap[None, :], px.shape), px)
        if dn.any():
            px = np.where(dn, np.broadcast_to(flr[None, :], px.shape), px)

        hcols = np.flatnonzero(halt_day[day])
        if hcols.size:  # halted: price frozen at the previous close all day
            px[:, hcols] = base[hcols][None, :]
            up[:, hcols] = False
            dn[:, hcols] = False

        prices[sl] = px
        limit_up[sl] = up
        limit_down[sl] = dn
        halted[sl] = halt_day[day][None, :]
        base = px[-1].copy()

    del r
    if not np.all(prices > 0.0):  # pragma: no cover - impossible at minute vols
        raise FloatingPointError("synthetic price path went non-positive")

    # ---- traded amounts ---------------------------------------------------
    adv_draw = float(scfg.adv_median_cny) * np.exp(
        float(scfg.adv_sigma_log) * rng_amount.standard_normal(n)
    )
    minute_amount = rng_amount.standard_normal((t, n))
    minute_amount *= 0.4  # lognormal minute noise, sigma_log = 0.4
    minute_amount -= 0.08  # -sigma^2/2 -> unit mean
    np.exp(minute_amount, out=minute_amount)
    minute_amount *= adv_draw[None, :]
    minute_amount *= np.tile(ua, d)[:, None]
    minute_amount.reshape(d, mpd, n).swapaxes(1, 2)[halt_day] = 0.0  # halted: no volume
    adv_cny = minute_amount.reshape(d, mpd, n).sum(axis=1).mean(axis=0)

    # ---- oracle segment alphas -------------------------------------------
    seg_coef = np.empty((n_sig, n_sig), dtype=np.float64)  # [seg, signal]
    for s, (a, b) in enumerate(SEGMENT_BOUNDS):
        seg_coef[s] = c_alpha * _geometric_sum(rho, int(a), int(b))
    true_seg_alpha = np.empty((t, n, n_sig), dtype=np.float64)
    for s in range(n_sig):  # out= keeps the peak footprint at one [T,N,3] copy
        np.einsum("ktn,k->tn", m, seg_coef[s], out=true_seg_alpha[:, :, s])
    level_corr = np.corrcoef(m[:, ::max(1, t // 512), :].reshape(n_sig, -1))
    del m

    # ---- observed signals = cumulative conditional mean + gaussian noise --
    signals = rng_noise.standard_normal((t, n, n_sig))  # reused as the output buffer
    max_hl = float(np.max(half_lives))
    sigma = np.zeros(n_sig, dtype=np.float64)
    achieved = np.full(n_sig, np.nan, dtype=np.float64)
    oracle_ic = np.full(n_sig, np.nan, dtype=np.float64)
    n_calib = np.zeros(n_sig, dtype=np.int64)
    calib_mode = []
    for q, h in enumerate(SIGNAL_HORIZONS):
        taus = ic_eval_taus(t, int(h), max_hl)
        n_calib[q] = taus.size
        analytic = _analytic_noise_sigma(
            float(var_alpha[q]), float(var_noise[q]), float(obs_target[q])
        )
        if taus.size < 2:
            # No window is long enough to measure this horizon (very short
            # panel): fall back to the closed-form population noise scale.
            sigma[q] = analytic
            calib_mode.append("analytic_no_window")
            continue
        true_sub = np.cumsum(true_seg_alpha[taus], axis=2)[:, :, q]
        eps_sub = signals[taus][:, :, q]
        fut_ranks = _ranks(forward_return(prices, taus, int(h)))
        oracle_ic[q] = _row_corr_mean(_ranks(true_sub), fut_ranks)
        sigma[q], achieved[q] = _bisect_noise_sigma(
            true_sub, eps_sub, fut_ranks, float(obs_target[q])
        )
        if sigma[q] <= 0.0:
            # The window's *measured* oracle IC already sits below the observed
            # target -- on long horizons few independent blocks fit in the panel
            # so this is estimation noise, not a modelling error. Zero noise
            # would make the signal a pure oracle, so use the population
            # closed form instead (fail-soft, recorded in meta).
            sigma[q] = analytic
            calib_mode.append("analytic_oracle_below_target")
            achieved[q] = _ic_with_noise(true_sub, eps_sub, sigma[q], fut_ranks)
        else:
            calib_mode.append("bisect")
    for day in range(d):  # chunked so no full-size temporary is needed
        sl = slice(day * mpd, (day + 1) * mpd)
        signals[sl] *= sigma[None, None, :]
        signals[sl] += np.cumsum(true_seg_alpha[sl], axis=2)

    # ---- risk model (exact daily covariance of the generative process) ----
    factor_exposures = np.column_stack([beta, style_load])
    factor_cov_daily = np.diag(
        np.concatenate(
            [[mkt_vol_daily**2], (STYLE_RETURN_WEIGHT * style_vol_daily) ** 2]
        )
    )
    # alpha is cross-sectionally idiosyncratic -> folded into specific variance
    day_q = int(np.argmin(np.abs(np.asarray(SIGNAL_HORIZONS) - mpd)))
    alpha_var_daily = float(var_alpha[day_q]) * (mpd / float(SIGNAL_HORIZONS[day_q]))
    specific_var_daily = idio_vol_daily**2 + alpha_var_daily

    meta = {
        "seed": int(seed),
        "generator": "openpo.mpo.synth.generate_market",
        "n_minutes": int(t),
        "minutes_per_day": mpd,
        "alpha_half_lives": [float(x) for x in half_lives],
        "rho_minute": [float(x) for x in rho],
        "c_alpha": [float(x) for x in c_alpha],
        "segment_coef": seg_coef.tolist(),
        "level_corr_m": level_corr.tolist(),
        "corr_eps_offdiag": float(scfg.alpha_cross_corr),
        "target_ic": [float(x) for x in target_ic],
        "observed_ic_target": [float(x) for x in obs_target],
        "oracle_ic_target": [float(x) for x in oracle_target],
        "oracle_ic_measured": [float(x) for x in oracle_ic],
        "measured_ic": [float(x) for x in achieved],
        "noise_sigma": [float(x) for x in sigma],
        "sd_alpha_horizon": [float(x) for x in np.sqrt(var_alpha)],
        "sd_noise_horizon": [float(x) for x in np.sqrt(var_noise)],
        "ic_calib_taus": [int(x) for x in n_calib],
        "ic_calib_mode": calib_mode,
        "burn_in_minutes": int(min(max_hl, t // 4)),
        "style_return_weight": STYLE_RETURN_WEIGHT,
        "n_style_factors": n_style,
        "idio_daily_vol_rms": float(np.sqrt(np.mean(idio_vol_daily**2))),
        "alpha_var_daily": alpha_var_daily,
        "limit_up_stock_days": int(np.sum(limit_up.reshape(d, mpd, n).any(axis=1))),
        "limit_down_stock_days": int(np.sum(limit_down.reshape(d, mpd, n).any(axis=1))),
        "halted_stock_days": int(np.sum(halt_day)),
        "signals_nan_on_halt": False,
        "segment_bounds": [list(b) for b in SEGMENT_BOUNDS],
        "signal_horizons": list(SIGNAL_HORIZONS),
    }

    market = MarketPanel(
        n_assets=n,
        n_days=d,
        prices=prices,
        minute_amount=minute_amount,
        prev_close=prev_close,
        limit_up=limit_up,
        limit_down=limit_down,
        halted=halted,
        adv_cny=adv_cny,
        beta=beta,
        factor_exposures=factor_exposures,
        factor_cov_daily=factor_cov_daily,
        specific_var_daily=specific_var_daily,
        true_seg_alpha=true_seg_alpha,
        meta=meta,
    )
    panel = SignalPanel(
        signals=signals,
        meta={
            "seed": int(seed),
            "columns": ["return_1_30", "return_1_240", "return_1_2400"],
            "raw": True,
            "noise_sigma": [float(x) for x in sigma],
            "measured_ic": [float(x) for x in achieved],
            "observed_ic_target": [float(x) for x in obs_target],
        },
    )
    return market, panel
