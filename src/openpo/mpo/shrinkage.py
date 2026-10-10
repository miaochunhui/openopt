"""Robust per-horizon alpha shrinkage for MPO-T1 (Axioma 003/007 direction).

Authoritative design: ``docs/mpo-t1-shrinkage-design.md``. This module is
LEG 1: estimator + wrapper + tests only. Nothing here is wired into the
strategy pool yet -- leg 2 adds the single config flag after design review, so
this file must stay import-safe and side-effect free.

What this layer is (and what it is NOT)
---------------------------------------
``term_structure.fit_term_structure`` already shrinks, but a different thing:
ridge pulls the ESTIMATE of ``B`` toward the telescoping prior ``B0`` (a
structural prior, not zero) with ``lam = ridge_lambda * n_samples``. This
module shrinks the DECISION-TIME alpha magnitude toward ZERO by the estimated
signal-to-noise ratio of each (segment, signal) cell::

    alpha = (kappa (*) B_ridge) @ z          # (*) = elementwise on [S, K]

Committed kappa (design doc section 1, derivation sketch there)::

    ic[s,k,e] = spearman(z_e[:,k], winsor(r_seg[s]_e))   # per decision epoch
    IC[s,k]   = mean_e ic[s,k,e]           E[s,k] = number of usable epochs
    m_s       = max(1, (b_s - a_s) / 30)   # overlapping-window deflation
    E_eff     = E / m_s
    v_emp     = var(ic[s,k,:], ddof=1)                    # 0 when E < 2
    v_ana     = (1 - IC**2)**2 / max(n_med - 3, 1)        # Fisher-z floor
    sigma2    = max(v_emp, v_ana) / E_eff                 # +inf when E == 0
    kappa     = IC**2 / (IC**2 + sigma2) = t**2 / (1 + t**2),  t = IC / sigma

Degenerate behaviour (the ONLY tolerated fallback, and it is recorded, never
silent): a cell with ``E == 0`` gets ``sigma2 = +inf`` hence ``kappa = 0``,
i.e. that (segment, signal) alpha contribution is switched off and the cell is
listed in ``IcStats.notes['degenerate_cells']``. Every other failure mode
(shape/nan/range violations) raises.

``sigma2 -> 0`` gives ``kappa -> 1`` and :class:`ShrunkTermStructureModel`
then reproduces the wrapped model BITWISE (it forwards ``stage_alphas`` to an
inner :class:`~openpo.mpo.term_structure.TermStructureModel` that differs only
in ``B``, so the arithmetic path is identical).

Conventions inherited unchanged from ``term_structure`` / ``calendar``:
``tau = day*240 + minute``, tau differences ARE trading minutes (overnight = 0),
decision epochs every ``decay.EPOCH_MINUTES = 30`` minutes, segments
``config.SEGMENT_BOUNDS``, and the no-lookahead guard ``tau + b_s < calib*240``.

Pure numpy/scipy, float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from openpo.mpo.calendar import epoch_taus
from openpo.mpo.config import MINUTES_PER_DAY, SEGMENT_BOUNDS, MpoConfig
from openpo.mpo.decay import EPOCH_MINUTES
from openpo.mpo.term_structure import (
    TermStructureModel,
    winsorize_cross_section,
    zscore_cross_section,
)
from openpo.mpo.types import TermStructureDiagnostics

__all__ = [
    "IcStats",
    "ShrunkTermStructureModel",
    "estimate_ic_stats",
    "ic_stats_from_epochs",
    "kappa_from_ic",
    "overlap_multipliers",
    "shrink_term_structure",
    "shrinkage_kappa",
]

# Minimum usable cross-section for one IC observation (mirrors
# term_structure._MIN_CROSS_SECTION on purpose: same sample admission rule).
_MIN_CROSS_SECTION = 3
# Dispersion floor below which a cross-section counts as degenerate.
_SD_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def overlap_multipliers(
    seg_bounds=SEGMENT_BOUNDS,
    epoch_spacing: int = EPOCH_MINUTES,
) -> np.ndarray:
    """Independent-observation deflation factor ``m_s`` per segment.

    Decision epochs are ``epoch_spacing`` trading minutes apart, but segment
    ``s = (a, b]`` measures a ``b - a`` minute realized return, so consecutive
    epochs share most of their return window. ``m_s = max(1, (b-a)/spacing)``
    is the number of epochs that a single non-overlapping window spans -- the
    effective epoch count is ``E / m_s`` (design doc section 1 step 3). With
    the frozen defaults: ``(0,30] -> 1``, ``(30,240] -> 7``, ``(240,2400] -> 72``.
    """
    spacing = float(epoch_spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"epoch_spacing must be finite and positive, got {epoch_spacing}")
    out = np.empty(len(seg_bounds), dtype=np.float64)
    for s, (a, b) in enumerate(seg_bounds):
        length = float(b) - float(a)
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"segment {s} has non-positive length: bounds {(a, b)}")
        out[s] = max(1.0, length / spacing)
    return out


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Cross-sectional Spearman correlation of two 1-D vectors.

    Returns NaN when fewer than ``_MIN_CROSS_SECTION`` finite pairs remain or
    either rank vector is degenerate. Deliberately mirrors the private helper
    in ``term_structure`` (duplicated rather than imported: that module is
    frozen for this leg and its helper is private).
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"IC inputs must have equal shape, got {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < _MIN_CROSS_SECTION:
        return np.nan
    ra = rankdata(a[ok])
    rb = rankdata(b[ok])
    sa = ra.std()
    sb = rb.std()
    if sa <= _SD_FLOOR or sb <= _SD_FLOOR:
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def kappa_from_ic(ic: np.ndarray, ic_var: np.ndarray) -> np.ndarray:
    """Committed shrinkage factor ``kappa = IC^2 / (IC^2 + Var(IC))``.

    Parameters
    ----------
    ic : np.ndarray
        Point estimate of the (segment, signal) IC, any shape. Must be finite
        (a cell with no usable sample is encoded as ``ic = 0`` +
        ``ic_var = inf`` by :func:`ic_stats_from_epochs`, not as NaN). Values
        outside ``[-1, 1]`` are clipped -- a correlation cannot exceed 1.
    ic_var : np.ndarray
        Sampling VARIANCE of that estimate (``sigma2``, not the SD), same
        shape. Must be ``>= 0``; ``+inf`` is allowed and means "no usable
        sample" (drives kappa to exactly 0).

    Returns
    -------
    np.ndarray
        ``kappa`` in ``[0, 1]``, float64, same shape. Equals ``t^2/(1+t^2)``
        with ``t = IC / sigma``: monotone increasing in ``|IC|``, monotone
        decreasing in the sampling variance, ``0`` when ``IC == 0`` or the
        variance is infinite, ``1`` when the variance is 0 (certain estimate).
        The pathological ``IC == 0`` AND ``ic_var == 0`` cell maps to 0 (no
        signal to bet on) instead of ``0/0``.
    """
    a = np.asarray(ic, dtype=np.float64)
    v = np.asarray(ic_var, dtype=np.float64)
    if a.shape != v.shape:
        raise ValueError(f"ic {a.shape} and ic_var {v.shape} must have the same shape")
    if not np.all(np.isfinite(a)):
        raise ValueError("ic must be finite (use ic=0 with ic_var=inf for empty cells)")
    if np.any(np.isnan(v)) or np.any(v < 0.0):
        raise ValueError("ic_var must be >= 0 and non-NaN (+inf allowed for empty cells)")
    ic2 = np.clip(a, -1.0, 1.0) ** 2
    den = ic2 + v
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(den > 0.0, ic2 / den, 0.0)
    kappa = np.where(np.isfinite(kappa), kappa, 0.0)  # ic2/inf underflow guard
    return np.clip(kappa, 0.0, 1.0)


# ---------------------------------------------------------------------------
# IC statistics
# ---------------------------------------------------------------------------

@dataclass
class IcStats:
    """Per (segment, signal) IC point estimate and its sampling uncertainty.

    Attributes
    ----------
    ic : np.ndarray
        ``[S, K]`` mean per-epoch cross-sectional Spearman IC. Empty cells
        (no usable epoch) hold 0.0, never NaN.
    ic_var : np.ndarray
        ``[S, K]`` sampling variance of ``ic`` (``sigma2`` in the design doc),
        already deflated by the overlapping-window factor. ``+inf`` for empty
        cells.
    n_epochs : np.ndarray
        ``[S, K]`` int64 count of epochs that produced a finite IC.
    n_eff : np.ndarray
        ``[S, K]`` ``n_epochs / m_s`` -- effective independent epoch count.
    n_names : np.ndarray
        ``[S, K]`` median usable cross-section size across those epochs.
    overlap_mult : np.ndarray
        ``[S]`` the ``m_s`` used (see :func:`overlap_multipliers`).
    notes : dict
        Free-form telemetry (empty cells, variance components, provenance).
    """

    ic: np.ndarray
    ic_var: np.ndarray
    n_epochs: np.ndarray
    n_eff: np.ndarray
    n_names: np.ndarray
    overlap_mult: np.ndarray
    notes: dict = field(default_factory=dict)

    def ic_se(self) -> np.ndarray:
        """``sqrt(ic_var)`` -- standard error of the IC estimate."""
        return np.sqrt(self.ic_var)

    def t_stat(self) -> np.ndarray:
        """``IC / SE`` (0 where the SE is infinite, +-inf where the SE is 0)."""
        se = self.ic_se()
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(np.isfinite(se), self.ic / se, 0.0)
        return np.where(np.isnan(t), 0.0, t)

    def kappa(self) -> np.ndarray:
        """Shrinkage table ``[S, K]`` -- see :func:`kappa_from_ic`."""
        return kappa_from_ic(self.ic, self.ic_var)


def ic_stats_from_epochs(
    z_epoch: np.ndarray,
    r_epoch: np.ndarray,
    winsor_sigma: float = 3.0,
    seg_bounds=SEGMENT_BOUNDS,
    epoch_spacing: int = EPOCH_MINUTES,
    overlap_mult: np.ndarray | None = None,
    min_names: int = _MIN_CROSS_SECTION,
    notes: dict | None = None,
) -> IcStats:
    """IC table + sampling variance from epoch-level panels (design doc 1.1-1.4).

    Parameters
    ----------
    z_epoch : np.ndarray
        ``[E, N, K]`` cross-sectionally z-scored signals at each decision
        epoch. Non-finite entries are treated as missing names.
    r_epoch : np.ndarray
        ``[E, N, S]`` realized segment returns anchored at the same epochs
        (``P[tau+b]/P[tau+a] - 1``). NaN marks a sample the no-lookahead guard
        or a bad price dropped -- those names/epochs are excluded, they are
        NOT imputed.
    winsor_sigma : float
        Cross-sectional winsorization of the returns before ranking, for exact
        parity with the ridge fit. Rank-neutral except for the ties it creates
        at the clip, kept anyway so both layers admit the same sample.
    seg_bounds, epoch_spacing : see :func:`overlap_multipliers`.
    overlap_mult : np.ndarray, optional
        Override the ``[S]`` deflation factors. Pass ``ones`` only when the
        epoch samples really are non-overlapping (synthetic tests); the panel
        estimator never overrides them.
    min_names : int
        Minimum usable cross-section for an epoch to contribute an IC.
    notes : dict, optional
        Extra provenance merged into ``IcStats.notes``.

    Returns
    -------
    IcStats
    """
    z = np.asarray(z_epoch, dtype=np.float64)
    r = np.asarray(r_epoch, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError(f"z_epoch must be [E, N, K], got shape {z.shape}")
    if r.ndim != 3:
        raise ValueError(f"r_epoch must be [E, N, S], got shape {r.shape}")
    if z.shape[0] != r.shape[0] or z.shape[1] != r.shape[1]:
        raise ValueError(f"z_epoch {z.shape} and r_epoch {r.shape} disagree on (E, N)")
    n_epoch, _, n_sig = z.shape
    n_seg = r.shape[2]
    if n_seg != len(seg_bounds):
        raise ValueError(
            f"r_epoch has {n_seg} segment columns but seg_bounds has {len(seg_bounds)}"
        )
    if n_epoch == 0:
        raise ValueError("z_epoch/r_epoch contain no epochs (E == 0): nothing to estimate")
    if int(min_names) < _MIN_CROSS_SECTION:
        raise ValueError(f"min_names must be >= {_MIN_CROSS_SECTION}, got {min_names}")

    if overlap_mult is None:
        m = overlap_multipliers(seg_bounds, epoch_spacing)
    else:
        m = np.asarray(overlap_mult, dtype=np.float64).ravel()
        if m.shape != (n_seg,):
            raise ValueError(f"overlap_mult must have shape ({n_seg},), got {m.shape}")
        if not np.all(np.isfinite(m)) or np.any(m <= 0):
            raise ValueError(f"overlap_mult entries must be finite and > 0, got {m}")

    # Winsorize each (epoch, segment) cross-section once, over its finite names.
    r_w = np.empty_like(r)
    for e in range(n_epoch):
        for s in range(n_seg):
            r_w[e, :, s] = winsorize_cross_section(r[e, :, s], winsor_sigma)

    ic = np.zeros((n_seg, n_sig), dtype=np.float64)
    ic_var = np.full((n_seg, n_sig), np.inf, dtype=np.float64)
    n_used = np.zeros((n_seg, n_sig), dtype=np.int64)
    n_eff = np.zeros((n_seg, n_sig), dtype=np.float64)
    n_names = np.zeros((n_seg, n_sig), dtype=np.float64)
    v_emp_t = np.zeros((n_seg, n_sig), dtype=np.float64)
    v_ana_t = np.zeros((n_seg, n_sig), dtype=np.float64)
    degenerate: list[tuple[int, int]] = []

    for s in range(n_seg):
        for k in range(n_sig):
            ics: list[float] = []
            counts: list[int] = []
            for e in range(n_epoch):
                zz = z[e, :, k]
                rr = r_w[e, :, s]
                ok = np.isfinite(zz) & np.isfinite(rr)
                if ok.sum() < int(min_names):
                    continue
                val = _spearman_ic(zz[ok], rr[ok])
                if np.isfinite(val):
                    ics.append(val)
                    counts.append(int(ok.sum()))
            n_used[s, k] = len(ics)
            if not ics:
                # ONLY tolerated fallback: no sample -> kappa == 0 (recorded).
                degenerate.append((s, k))
                continue
            arr = np.asarray(ics, dtype=np.float64)
            ic[s, k] = float(arr.mean())
            n_names[s, k] = float(np.median(np.asarray(counts, dtype=np.float64)))
            eff = float(len(ics)) / float(m[s])
            n_eff[s, k] = eff
            v_emp = float(arr.var(ddof=1)) if arr.size >= 2 else 0.0
            dof = max(n_names[s, k] - 3.0, 1.0)  # Fisher-z SE of a rank correlation
            v_ana = (1.0 - min(ic[s, k] ** 2, 1.0)) ** 2 / dof
            v_emp_t[s, k] = v_emp
            v_ana_t[s, k] = v_ana
            base = max(v_emp, v_ana)  # never claim less noise than theory allows
            ic_var[s, k] = base / eff if eff > 0.0 else np.inf

    out_notes = {
        "n_epochs_total": int(n_epoch),
        "n_assets": int(z.shape[1]),
        "overlap_mult": m.tolist(),
        "epoch_spacing_minutes": int(epoch_spacing),
        "winsor_sigma": float(winsor_sigma),
        "min_names": int(min_names),
        "v_emp": v_emp_t.tolist(),
        "v_ana": v_ana_t.tolist(),
        "degenerate_cells": [list(c) for c in degenerate],
        "kappa_formula": "IC^2 / (IC^2 + Var(IC))",
    }
    if notes:
        out_notes.update(notes)
    return IcStats(
        ic=ic,
        ic_var=ic_var,
        n_epochs=n_used,
        n_eff=n_eff,
        n_names=n_names,
        overlap_mult=m,
        notes=out_notes,
    )


def estimate_ic_stats(
    market,
    signals,
    cfg: MpoConfig,
    calib_days: int | None = None,
    seg_bounds=SEGMENT_BOUNDS,
) -> IcStats:
    """Estimate the IC table on the calibration window ``[0, calib_days)``.

    Same window, same no-lookahead guard (``tau + b_s < calib_days*240``) and
    same winsorization as ``term_structure.fit_term_structure``, so kappa and
    ``B`` see exactly the same sample -- an in-sample optimism that is
    documented as risk 2 in the design doc (it biases kappa UP, i.e. toward the
    no-op direction).

    Parameters
    ----------
    market : MarketPanel
        Only ``prices`` ``[T, N]`` and ``n_days`` are read.
    signals : SignalPanel
        ``signals.signals`` ``[T, N, K]`` RAW (not z-scored) predictions.
    cfg : MpoConfig
        Uses ``calib_days`` and ``winsor_sigma``.
    calib_days : int, optional
        Overrides ``cfg.calib_days``.

    Returns
    -------
    IcStats
        With ``notes['max_tau_touched']`` for lookahead auditing.
    """
    n_days = int(getattr(market, "n_days"))
    prices = np.asarray(market.prices, dtype=np.float64)
    raw = np.asarray(signals.signals, dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError(f"signals.signals must be [T, N, K], got {raw.shape}")
    if prices.ndim != 2:
        raise ValueError(f"market.prices must be [T, N], got {prices.shape}")
    if prices.shape[0] != raw.shape[0] or prices.shape[1] != raw.shape[1]:
        raise ValueError(f"prices {prices.shape} and signals {raw.shape} disagree on (T, N)")

    days = int(cfg.calib_days if calib_days is None else calib_days)
    if days < 1:
        raise ValueError(f"calib_days must be >= 1, got {days}")
    if days > n_days:
        raise ValueError(f"calib_days={days} exceeds market.n_days={n_days}")
    t_calib = days * MINUTES_PER_DAY  # first tau OUTSIDE the calibration window
    if prices.shape[0] < t_calib:
        raise ValueError(
            f"panel has {prices.shape[0]} minutes, need {t_calib} for calib_days={days}"
        )

    taus = epoch_taus(days)
    n_seg = len(seg_bounds)
    z_epoch = np.stack([zscore_cross_section(raw[int(t)]) for t in taus], axis=0)
    r_epoch = np.full((taus.size, prices.shape[1], n_seg), np.nan, dtype=np.float64)
    max_tau_touched = -1
    for e, tau in enumerate(taus):
        for s, (a, b) in enumerate(seg_bounds):
            far = int(tau) + int(b)
            if far >= t_calib:
                continue  # NO-LOOKAHEAD GUARD (identical to fit_term_structure)
            near = int(tau) + int(a)
            p0 = prices[near]
            p1 = prices[far]
            max_tau_touched = max(max_tau_touched, far)
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = p1 / p0 - 1.0
            bad = ~np.isfinite(ret) | ~np.isfinite(p0) | (p0 <= 0)
            ret = np.where(bad, np.nan, ret)
            r_epoch[e, :, s] = ret

    return ic_stats_from_epochs(
        z_epoch,
        r_epoch,
        winsor_sigma=float(cfg.winsor_sigma),
        seg_bounds=seg_bounds,
        notes={
            "calib_days": days,
            "t_calib": int(t_calib),
            "max_tau_touched": int(max_tau_touched),
            "source": "estimate_ic_stats",
        },
    )


def shrinkage_kappa(stats: IcStats) -> np.ndarray:
    """``[S, K]`` shrinkage table of an :class:`IcStats` (see :func:`kappa_from_ic`)."""
    if not isinstance(stats, IcStats):
        raise TypeError(f"stats must be an IcStats, got {type(stats).__name__}")
    return stats.kappa()


# ---------------------------------------------------------------------------
# wrapper (application point)
# ---------------------------------------------------------------------------

@dataclass
class ShrunkTermStructureModel:
    """``TermStructureModel`` façade whose alphas are kappa-shrunk.

    Composition, not inheritance: the wrapper owns a fitted model (read-only)
    plus a ``[S, K]`` kappa table, and exposes the same consumption surface
    (``.B`` / ``.rho`` / ``.seg_bounds`` / ``.decay_mult`` / ``.effective_rho()``
    / ``.stage_alphas(z, grid)`` / ``.diagnostics``). ``stage_alphas`` is
    FORWARDED to an inner :class:`TermStructureModel` that differs from the
    wrapped one only in ``B = kappa * B_base`` -- no stage-alpha math is
    reimplemented here, so ``kappa == 1`` reproduces the wrapped model bitwise.

    The wrapped model is never mutated (its ``diagnostics`` object is copied
    into a new one carrying the kappa telemetry).

    Parameters
    ----------
    base : TermStructureModel
        Any object exposing ``.B``, ``.rho`` and (optionally) ``.seg_bounds``,
        ``.decay_mult``, ``.decay``, ``.diagnostics``. Wrapping a wrapper is
        allowed (the kappas then compose multiplicatively).
    kappa : np.ndarray
        ``[S, K]`` factors in ``[0, 1]``, matching ``base.B.shape``. A scalar
        or ``[S, 1]`` / ``[1, K]`` value is NOT broadcast implicitly -- pass
        the full table so the telemetry is unambiguous.
    stats : IcStats, optional
        The statistics kappa came from (kept for the report/telemetry).
    """

    base: TermStructureModel
    kappa: np.ndarray
    stats: IcStats | None = None

    def __post_init__(self) -> None:
        base_B = np.asarray(getattr(self.base, "B"), dtype=np.float64)
        if base_B.ndim != 2:
            raise ValueError(f"base.B must be 2-D [S, K], got shape {base_B.shape}")
        k = np.asarray(self.kappa, dtype=np.float64)
        if k.shape != base_B.shape:
            raise ValueError(
                f"kappa shape {k.shape} must equal base.B shape {base_B.shape} "
                "(no implicit broadcasting)"
            )
        if not np.all(np.isfinite(k)):
            raise ValueError("kappa must be finite everywhere")
        if np.any(k < 0.0) or np.any(k > 1.0):
            raise ValueError(f"kappa must lie in [0, 1], got range [{k.min()}, {k.max()}]")
        self.kappa = k

        seg_bounds = getattr(self.base, "seg_bounds", SEGMENT_BOUNDS)
        if len(seg_bounds) != base_B.shape[0]:
            raise ValueError(
                f"base.seg_bounds has {len(seg_bounds)} segments but B has {base_B.shape[0]} rows"
            )
        base_diag = getattr(self.base, "diagnostics", None)
        # Inner model: identical to base except for the shrunk B, so the
        # stage-alpha arithmetic path is byte-for-byte the same.
        self._inner = TermStructureModel(
            B=base_B * k,
            rho=np.asarray(getattr(self.base, "rho"), dtype=np.float64),
            seg_bounds=seg_bounds,
            diagnostics=self._build_diagnostics(base_diag, base_B * k),
            decay_mult=float(getattr(self.base, "decay_mult", 1.0)),
            decay=getattr(self.base, "decay", None),
        )

    # ------------------------------------------------------------ telemetry
    def _build_diagnostics(self, base_diag, shrunk_B: np.ndarray) -> TermStructureDiagnostics:
        """New diagnostics = copy of the base's + the kappa audit trail."""
        notes: dict = {}
        seg_ic = None
        rho_epoch = None
        n_samples = 0
        if base_diag is not None:
            notes.update(dict(getattr(base_diag, "notes", {}) or {}))
            seg_ic_raw = getattr(base_diag, "seg_ic", None)
            if seg_ic_raw is not None:
                seg_ic = np.array(seg_ic_raw, dtype=np.float64, copy=True)
            rho_raw = getattr(base_diag, "rho_epoch", None)
            if rho_raw is not None:
                rho_epoch = np.array(rho_raw, dtype=np.float64, copy=True)
            n_samples = int(getattr(base_diag, "n_samples", 0) or 0)
        notes["shrinkage_kappa"] = self.kappa.tolist()
        notes["shrinkage_kappa_min"] = float(self.kappa.min())
        notes["shrinkage_kappa_mean"] = float(self.kappa.mean())
        notes["shrinkage_B_unshrunk"] = np.asarray(
            getattr(self.base, "B"), dtype=np.float64
        ).tolist()
        if self.stats is not None:
            notes["shrinkage_ic"] = self.stats.ic.tolist()
            notes["shrinkage_ic_se"] = self.stats.ic_se().tolist()
            notes["shrinkage_ic_t"] = self.stats.t_stat().tolist()
            notes["shrinkage_n_epochs"] = self.stats.n_epochs.tolist()
            notes["shrinkage_n_eff"] = self.stats.n_eff.tolist()
            notes["shrinkage_stats_notes"] = dict(self.stats.notes)
        return TermStructureDiagnostics(
            seg_ic=seg_ic,
            coef_table=np.array(shrunk_B, dtype=np.float64, copy=True),
            rho_epoch=rho_epoch,
            n_samples=n_samples,
            notes=notes,
        )

    # ------------------------------------------- TermStructureModel surface
    @property
    def B(self) -> np.ndarray:
        """Shrunk coefficients ``kappa * base.B`` ``[S, K]``."""
        return self._inner.B

    @property
    def rho(self) -> np.ndarray:
        return self._inner.rho

    @property
    def seg_bounds(self):
        return self._inner.seg_bounds

    @property
    def decay_mult(self) -> float:
        return self._inner.decay_mult

    @property
    def decay(self):
        return self._inner.decay

    @property
    def diagnostics(self) -> TermStructureDiagnostics:
        return self._inner.diagnostics

    def effective_rho(self) -> np.ndarray:
        return self._inner.effective_rho()

    def stage_alphas(self, z: np.ndarray, grid) -> np.ndarray:
        """Kappa-shrunk per-stage alphas -- forwarded, math unchanged."""
        return self._inner.stage_alphas(z, grid)

    def unshrunk(self) -> TermStructureModel:
        """The wrapped model (unchanged, for A/B pairing)."""
        return self.base


def shrink_term_structure(
    model: TermStructureModel,
    market=None,
    signals=None,
    cfg: MpoConfig | None = None,
    calib_days: int | None = None,
    kappa: np.ndarray | None = None,
) -> ShrunkTermStructureModel:
    """One-call entry point for leg 2: fitted model -> shrunk model.

    Either pass ``kappa`` explicitly, or pass ``market``/``signals``/``cfg`` to
    estimate it on the calibration window via :func:`estimate_ic_stats`.
    Raises if neither route is fully specified (no silent identity wrapper).
    """
    if kappa is None:
        if market is None or signals is None or cfg is None:
            raise ValueError(
                "shrink_term_structure needs either kappa=... or "
                "market+signals+cfg to estimate it"
            )
        stats = estimate_ic_stats(market, signals, cfg, calib_days=calib_days)
        return ShrunkTermStructureModel(base=model, kappa=stats.kappa(), stats=stats)
    if market is not None or signals is not None:
        raise ValueError("pass either kappa=... or market+signals+cfg, not both")
    return ShrunkTermStructureModel(base=model, kappa=np.asarray(kappa, dtype=np.float64))
