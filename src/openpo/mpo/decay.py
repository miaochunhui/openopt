"""AR(1) alpha-decay estimation for MPO-T1 (docs/mpo-t1-SPEC.md section 2 step 4).

Estimates, per signal column, the persistence of the cross-sectionally
z-scored prediction as an AR(1) coefficient, and converts it to the
**per-trading-minute** decay used by the stage-alpha term structure.

Pipeline (frozen by the SPEC)::

    per stock i:  z_k[e+1] = c + rho * z_k[e] + eps      (OLS on the epoch grid)
    rho_epoch_k  = cross-sectional MEDIAN of the per-stock OLS slopes
    rho_epoch_k += (1 + 3*rho_epoch_k) / T_eff           (Kendall bias fix)
    rho_epoch_k  = clip(rho_epoch_k, RHO_FLOOR, RHO_CEIL)
    rho_minute_k = rho_epoch_k ** (1 / 30)               (30 min between epochs)

Time conventions (project-wide): ``tau = day*240 + minute``; the synthetic
calendar has NO holidays, so consecutive decision epochs are ALWAYS exactly
``EPOCH_MINUTES = 30`` trading minutes apart -- including across a day
boundary (last epoch of day d is ``d*240+215``, first of day d+1 is
``d*240+245``). Overnight therefore counts as 0 extra trading minutes and the
epoch series of one stock is a single evenly-spaced AR(1) sample path.

Two deliberate readings of the SPEC prose, documented because they are
numerically load-bearing:

1. **The Kendall correction is applied at the EPOCH scale**, i.e. to the
   estimator's own sampling frequency, before the ``**(1/30)`` conversion.
   The SPEC sentence lists the minute conversion first, but the OLS bias
   ``-(1 + 3*rho)/T`` is a property of the *epoch-grid* regression; applied to
   ``rho_minute`` (which is ``1 - O(1/30)`` away from 1) a correction of order
   ``4/T_eff ~ 0.025`` would push it past 1.0 and clipping would erase ALL
   decay. See ``tests/mpo/test_term_structure.py::test_ar1_half_life_recovery``.
2. **``cfg.decay_mult`` is NOT baked into the estimate.** ``rho`` stays the
   raw per-minute estimate; the sensitivity axis is applied where the decay is
   consumed (``TermStructureModel.stage_alphas``) as
   ``rho_eff = rho ** (1 / decay_mult)`` -- see :func:`effective_rho`. Larger
   ``decay_mult`` => rho closer to 1 => the strategy assumes SLOWER decay.

Pure numpy, float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "EPOCH_MINUTES",
    "RHO_CEIL",
    "RHO_FLOOR",
    "DecayEstimate",
    "ar1_ols_slopes",
    "effective_rho",
    "estimate_ar1_decay",
    "half_life_minutes",
    "kendall_bias_correction",
    "rho_from_half_life",
]

# Trading minutes between two consecutive decision epochs (frozen: the epoch
# offsets (5,35,...,215) are 30 minutes apart and so is the day-boundary pair).
EPOCH_MINUTES = 30

# Open-interval clip for every rho we hand out: strictly inside (0, 1) so that
# rho**delta neither explodes nor kills the whole term structure.
RHO_FLOOR = 1e-6
RHO_CEIL = 1.0 - 1e-9

# A per-stock OLS slope is only usable if its regressor has some dispersion.
_VAR_FLOOR = 1e-24


@dataclass(frozen=True)
class DecayEstimate:
    """Per-signal AR(1) decay estimate (all arrays shape [K], float64)."""

    rho_minute: np.ndarray  # per-TRADING-MINUTE AR(1) coefficient, in (0,1)
    rho_epoch: np.ndarray  # per-epoch (30 min) coefficient after Kendall fix
    rho_epoch_raw: np.ndarray  # median OLS slope, before the Kendall fix
    t_eff: int  # lag-1 pairs per stock series (Kendall denominator)
    n_stocks_used: np.ndarray  # int per signal: stocks with a usable slope
    notes: dict = field(default_factory=dict)

    @property
    def half_life_minutes(self) -> np.ndarray:
        """Trading-minute half-lives implied by ``rho_minute``."""
        return half_life_minutes(self.rho_minute)


def ar1_ols_slopes(series: np.ndarray) -> np.ndarray:
    """Per-column lag-1 OLS slopes of an evenly-spaced panel.

    Parameters
    ----------
    series : np.ndarray
        ``[E, N]`` samples: ``E`` epoch observations of ``N`` independent
        series (stocks). NaNs are tolerated: a column's slope uses only the
        lag pairs where BOTH ends are finite.

    Returns
    -------
    np.ndarray
        ``[N]`` float64 slopes of ``x[e+1] = c + rho*x[e]`` (intercept fitted,
        i.e. covariances are computed about the pair means). Columns with
        fewer than 3 usable pairs or a degenerate regressor variance get NaN.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"series must be 2-D [E, N], got shape {x.shape}")
    if x.shape[0] < 4:
        return np.full(x.shape[1], np.nan)

    lhs = x[1:, :]  # x[e+1]
    rhs = x[:-1, :]  # x[e]
    ok = np.isfinite(lhs) & np.isfinite(rhs)
    n_pairs = ok.sum(axis=0)

    lhs = np.where(ok, lhs, 0.0)
    rhs = np.where(ok, rhs, 0.0)
    cnt = np.maximum(n_pairs, 1)
    mean_l = lhs.sum(axis=0) / cnt
    mean_r = rhs.sum(axis=0) / cnt
    dl = np.where(ok, lhs - mean_l, 0.0)
    dr = np.where(ok, rhs - mean_r, 0.0)
    cov = (dl * dr).sum(axis=0)
    var = (dr * dr).sum(axis=0)

    slopes = np.full(x.shape[1], np.nan)
    good = (n_pairs >= 3) & (var > _VAR_FLOOR)
    slopes[good] = cov[good] / var[good]
    return slopes


def kendall_bias_correction(rho: np.ndarray | float, t_eff: int) -> np.ndarray:
    """Kendall's small-sample bias correction ``rho + (1 + 3*rho)/T_eff``.

    The OLS AR(1) slope is biased downward by about ``(1 + 3*rho)/T``; this
    adds it back. ``t_eff <= 0`` is a no-op (nothing to correct).
    """
    r = np.asarray(rho, dtype=np.float64)
    if t_eff <= 0:
        return r.copy()
    return r + (1.0 + 3.0 * r) / float(t_eff)


def estimate_ar1_decay(
    z_epoch: np.ndarray,
    epoch_minutes: int = EPOCH_MINUTES,
) -> DecayEstimate:
    """Estimate the per-minute AR(1) decay of every signal column.

    Parameters
    ----------
    z_epoch : np.ndarray
        ``[E, N, K]`` cross-sectionally z-scored signals sampled on the
        decision-epoch grid, in ascending tau order and CONTIGUOUS (constant
        ``epoch_minutes`` spacing). Callers must slice the calibration window
        themselves -- this function never looks at anything else.
    epoch_minutes : int
        Trading minutes between consecutive rows (default 30).

    Returns
    -------
    DecayEstimate
        ``rho_minute`` is what the term structure consumes; ``rho_epoch`` /
        ``rho_epoch_raw`` / ``t_eff`` are kept for diagnostics. Signals whose
        per-stock slopes are all unusable fall back to ``RHO_FLOOR`` (no
        persistence => the segment weight carries the whole term structure)
        and are listed in ``notes['degenerate_signals']``.
    """
    z = np.asarray(z_epoch, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError(f"z_epoch must be 3-D [E, N, K], got shape {z.shape}")
    if epoch_minutes <= 0:
        raise ValueError(f"epoch_minutes must be positive, got {epoch_minutes}")
    n_epochs, _, n_sig = z.shape

    # Lag-1 pairs available per stock series -- Kendall's T.
    t_eff = max(int(n_epochs) - 1, 0)

    raw = np.full(n_sig, np.nan)
    used = np.zeros(n_sig, dtype=np.int64)
    degenerate: list[int] = []
    for k in range(n_sig):
        slopes = ar1_ols_slopes(z[:, :, k])
        finite = np.isfinite(slopes)
        used[k] = int(finite.sum())
        if used[k] > 0:
            raw[k] = float(np.median(slopes[finite]))
        else:
            degenerate.append(k)

    rho_epoch = kendall_bias_correction(np.where(np.isfinite(raw), raw, 0.0), t_eff)
    rho_epoch = np.clip(rho_epoch, RHO_FLOOR, RHO_CEIL)
    for k in degenerate:
        rho_epoch[k] = RHO_FLOOR
    rho_minute = np.clip(rho_epoch ** (1.0 / float(epoch_minutes)), RHO_FLOOR, RHO_CEIL)

    return DecayEstimate(
        rho_minute=rho_minute,
        rho_epoch=rho_epoch,
        rho_epoch_raw=raw,
        t_eff=t_eff,
        n_stocks_used=used,
        notes={
            "epoch_minutes": int(epoch_minutes),
            "n_epochs": int(n_epochs),
            "kendall_scale": "epoch",  # correction applied before **(1/30)
            "degenerate_signals": degenerate,
        },
    )


def effective_rho(rho_minute: np.ndarray | float, decay_mult: float) -> np.ndarray:
    """Sensitivity-axis decay ``rho ** (1 / decay_mult)`` (SPEC ``cfg.decay_mult``).

    ``decay_mult == 1`` returns the estimate untouched; ``decay_mult > 1``
    moves rho toward 1 (the strategy assumes the alpha decays SLOWER than
    estimated), ``< 1`` toward 0 (faster). Result stays in ``(0, 1)``.
    """
    r = np.asarray(rho_minute, dtype=np.float64)
    if not np.isfinite(decay_mult) or decay_mult <= 0:
        raise ValueError(f"decay_mult must be finite and positive, got {decay_mult}")
    return np.clip(np.clip(r, RHO_FLOOR, RHO_CEIL) ** (1.0 / float(decay_mult)),
                   RHO_FLOOR, RHO_CEIL)


def half_life_minutes(rho_minute: np.ndarray | float) -> np.ndarray:
    """Trading-minute half-life of a per-minute AR(1) coefficient."""
    r = np.clip(np.asarray(rho_minute, dtype=np.float64), RHO_FLOOR, RHO_CEIL)
    return -np.log(2.0) / np.log(r)


def rho_from_half_life(half_life: np.ndarray | float) -> np.ndarray:
    """Per-minute AR(1) coefficient with the given trading-minute half-life."""
    h = np.asarray(half_life, dtype=np.float64)
    if np.any(h <= 0):
        raise ValueError("half_life must be positive (trading minutes)")
    return np.clip(0.5 ** (1.0 / h), RHO_FLOOR, RHO_CEIL)
