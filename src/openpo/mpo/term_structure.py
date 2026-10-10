"""Alpha term structure for MPO-T1 (docs/mpo-t1-SPEC.md section 2).

Turns the three raw signal columns (``return_1_30`` / ``return_1_240`` /
``return_1_2400``) into per-stage expected returns for the multi-period QP:

    calibration (days [0, calib_days), frozen afterwards)
        1. cross-sectional z-score of the raw signals at every decision epoch
        2. realized segment returns  r_seg[s](tau,i) = P[tau+b]/P[tau+a] - 1
           for the non-overlapping segments SEGMENT_BOUNDS = ((0,30],(30,240],(240,2400]]
        3. winsorize targets at +-cfg.winsor_sigma cross-sectional sigmas
        4. pooled panel ridge per segment row, shrunk to the telescoping prior
        5. per-signal AR(1) decay on the epoch grid (openpo.mpo.decay)

    usage
        TermStructureModel.stage_alphas(z, grid) -> alpha[J, N]
        alpha[j,i] = sum_s  overlap((tau+a_s, tau+b_s], (stage_j.start, stage_j.end])
                             / (b_s - a_s)
                     * sum_k B[s,k] * rho_eff_k**delta_j * z[i,k]

Frozen conventions repeated on purpose:

- ``tau = day*240 + minute``; the synthetic calendar has NO holidays, so
  trading minutes between two taus are exactly their difference and overnight
  counts as ZERO minutes. Segment/stage arithmetic is therefore plain integer
  interval arithmetic on tau.
- ``delta_j`` = trading minutes from ``tau_now`` to stage j's MIDPOINT
  (``Stage.midpoint_offset``), **except the execution stage (index 0) which
  always uses delta = 0** -- its alpha is realized immediately.
- ``.rho`` holds the RAW per-minute AR(1) estimate. ``cfg.decay_mult`` is
  applied at consumption time inside :meth:`stage_alphas`
  (``rho_eff = rho ** (1/decay_mult)``, see :func:`openpo.mpo.decay.effective_rho`),
  so the same fitted model can be reused across the sensitivity axis.

**No-lookahead guard (explicit).** Segment 3 spans 2400 trading minutes = 10
trading days, so a naive sample at the end of the calibration window would
read prices from the evaluation window. Calibration therefore only keeps
samples whose FAR end stays inside the calibration window:

    keep tau  <=>  tau + b_s < calib_days * 240      (per segment s)

The guard is applied per segment (so the short segments still use the whole
window) and no price row at index ``>= calib_days*240`` is ever touched --
asserted by poisoning those rows in
``tests/mpo/test_term_structure.py::test_no_lookahead_past_calib_window``.

Pure numpy/scipy, float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from openpo.mpo.calendar import epoch_taus
from openpo.mpo.config import MINUTES_PER_DAY, SEGMENT_BOUNDS, MpoConfig
from openpo.mpo.decay import DecayEstimate, effective_rho, estimate_ar1_decay
from openpo.mpo.types import StageGrid, TermStructureDiagnostics

__all__ = [
    "TELESCOPING_PRIOR",
    "TermStructureModel",
    "fit_term_structure",
    "segment_overlap_weights",
    "winsorize_cross_section",
    "zscore_cross_section",
]

# Telescoping prior B0 (SPEC section 2 step 3): row s expresses segment s as
# the difference of the two enclosing cumulative-horizon signals. Each row is
# scaled by that segment's realized return scale before shrinkage.
TELESCOPING_PRIOR = np.array(
    [[1.0, 0.0, 0.0],
     [-1.0, 1.0, 0.0],
     [0.0, -1.0, 1.0]],
    dtype=np.float64,
)

# Minimum finite cross-section size for a usable sample (z-score + winsor need
# dispersion; a Spearman IC on < 3 names is noise).
_MIN_CROSS_SECTION = 3
# Dispersion floor below which a cross-section counts as degenerate.
_SD_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# cross-sectional primitives
# ---------------------------------------------------------------------------

def zscore_cross_section(raw: np.ndarray) -> np.ndarray:
    """NaN-safe cross-sectional z-score.

    Parameters
    ----------
    raw : np.ndarray
        ``[N]`` or ``[N, K]``: one cross-section of raw signal values (rows =
        stocks, columns = signals).

    Returns
    -------
    np.ndarray
        Same shape, float64. Each column is standardized over its FINITE
        entries only (mean 0, population std 1, ``ddof=0``). Non-finite inputs
        (NaN/inf) stay NaN in the output -- callers that need a number use
        ``np.nan_to_num`` (``stage_alphas`` does, the fit drops those rows).
        A degenerate column (fewer than 2 finite entries, or std <= 1e-12)
        maps its finite entries to 0.0.
    """
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"raw must be 1-D [N] or 2-D [N, K], got shape {x.shape}")
    flat = x.reshape(x.shape[0], -1) if x.ndim == 2 else x.reshape(-1, 1)

    out = np.full(flat.shape, np.nan, dtype=np.float64)
    for k in range(flat.shape[1]):
        col = flat[:, k]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            out[finite, k] = 0.0  # nothing to standardize against
            continue
        vals = col[finite]
        sd = vals.std()  # ddof=0, matches the simulator's local fallback
        if not np.isfinite(sd) or sd <= _SD_FLOOR:
            out[finite, k] = 0.0  # degenerate dispersion -> neutral
            continue
        out[finite, k] = (vals - vals.mean()) / sd
    return out.reshape(x.shape)


def winsorize_cross_section(values: np.ndarray, sigma: float) -> np.ndarray:
    """Clip one cross-section to ``mean +- sigma * std`` (population std).

    Non-finite entries and degenerate dispersion pass through unchanged;
    ``sigma <= 0`` disables winsorization.
    """
    v = np.asarray(values, dtype=np.float64).copy()
    if not np.isfinite(sigma) or sigma <= 0:
        return v
    finite = np.isfinite(v)
    if finite.sum() < 2:
        return v
    vals = v[finite]
    mu = vals.mean()
    sd = vals.std()
    if not np.isfinite(sd) or sd <= _SD_FLOOR:
        return v
    v[finite] = np.clip(vals, mu - sigma * sd, mu + sigma * sd)
    return v


def segment_overlap_weights(
    tau_now: int,
    stage_start: int,
    stage_end: int,
    seg_bounds=SEGMENT_BOUNDS,
) -> np.ndarray:
    """Fractional overlap of each alpha segment with an absolute stage.

    Segment ``s = (a, b]`` (trading minutes relative to ``tau_now``) is shifted
    to ``(tau_now + a, tau_now + b]``; the weight is
    ``overlap_minutes(shifted, (stage_start, stage_end]) / (b - a)`` -- in
    ``[0, 1]``, and 0 for a segment fully outside the stage. Mirrors
    ``openpo.mpo.strategies.segment_stage_weights`` (duplicated on purpose to
    keep the module import graph acyclic during parallel development).
    """
    w = np.zeros(len(seg_bounds), dtype=np.float64)
    for s, (a, b) in enumerate(seg_bounds):
        lo = max(int(stage_start), int(tau_now) + int(a))
        hi = min(int(stage_end), int(tau_now) + int(b))
        w[s] = max(0, hi - lo) / float(int(b) - int(a))
    return w


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two 1-D vectors (NaN if degenerate)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
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


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@dataclass
class TermStructureModel:
    """Fitted alpha term structure consumed by the strategy zoo.

    Attributes
    ----------
    B : np.ndarray
        ``[3 segments, 3 signals]`` ridge coefficients mapping z-scores to the
        realized return of each segment (units: fraction of price).
    rho : np.ndarray
        ``[3]`` per-TRADING-MINUTE AR(1) coefficients of the signals, in
        ``(0, 1)``. ``decay_mult`` is NOT baked in -- :meth:`stage_alphas`
        applies ``rho ** (1/decay_mult)``.
    seg_bounds : tuple
        The segment bounds the fit used (``config.SEGMENT_BOUNDS``).
    diagnostics : TermStructureDiagnostics
        In-sample segment ICs, coefficient table, epoch-grid rho, sample count.
    decay_mult : float
        Sensitivity multiplier taken from ``cfg.decay_mult`` at fit time.
    """

    B: np.ndarray
    rho: np.ndarray
    seg_bounds: tuple = SEGMENT_BOUNDS
    diagnostics: TermStructureDiagnostics = field(default_factory=TermStructureDiagnostics)
    decay_mult: float = 1.0
    decay: DecayEstimate | None = None

    def __post_init__(self) -> None:
        self.B = np.asarray(self.B, dtype=np.float64).reshape(len(self.seg_bounds), -1)
        self.rho = np.asarray(self.rho, dtype=np.float64).ravel()
        if self.rho.size != self.B.shape[1]:
            raise ValueError(
                f"rho has {self.rho.size} entries but B has {self.B.shape[1]} signal columns"
            )

    # -------------------------------------------------------------- decay
    def effective_rho(self) -> np.ndarray:
        """``rho ** (1 / decay_mult)`` -- the decay actually applied."""
        return effective_rho(self.rho, self.decay_mult)

    # -------------------------------------------------------------- alphas
    def stage_alphas(self, z: np.ndarray, grid: StageGrid) -> np.ndarray:
        """Per-stage expected returns for one decision (SPEC section 2).

        Parameters
        ----------
        z : np.ndarray
            ``[N, 3]`` cross-sectional z-scores at ``grid.tau_now`` (non-finite
            entries are neutralized to 0).
        grid : StageGrid
            Stage grid from ``openpo.mpo.stages.build_stage_grid``; only
            ``grid.tau_now`` and the stage intervals are used.

        Returns
        -------
        np.ndarray
            ``[J, N]`` float64, ``J = grid.n_stages``. Stage 0 (execution) is
            evaluated with ``delta = 0``; every later stage decays each signal
            by ``rho_eff_k ** delta_j`` with ``delta_j`` the trading minutes
            from ``tau_now`` to that stage's midpoint.
        """
        n_sig = self.B.shape[1]
        zz = np.nan_to_num(np.asarray(z, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if zz.ndim != 2 or zz.shape[1] != n_sig:
            raise ValueError(f"z must have shape [N, {n_sig}], got {zz.shape}")
        tau_now = int(grid.tau_now)
        rho_eff = self.effective_rho()

        alpha = np.empty((grid.n_stages, zz.shape[0]), dtype=np.float64)
        for j, stage in enumerate(grid.stages):
            seg_w = segment_overlap_weights(tau_now, stage.start, stage.end, self.seg_bounds)
            # Execution stage: alpha is captured now, no decay discount.
            delta = 0.0 if j == 0 else float(stage.midpoint_offset(tau_now))
            decay = np.ones(n_sig) if delta == 0.0 else rho_eff ** delta
            alpha[j] = zz @ ((seg_w @ self.B) * decay)
        return alpha


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def _ridge_row(
    Z: np.ndarray,
    r: np.ndarray,
    b0: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    """Ridge shrunk to a prior: ``min ||r - Z b||^2 + lam ||b - b0||^2``.

    ``lam = ridge_lambda * n_samples`` (SPEC section 2 step 3), so the penalty
    scales with the pooled panel size and the effective shrinkage stays
    sample-size independent.
    """
    n = Z.shape[0]
    k = Z.shape[1]
    if n == 0:
        return b0.copy()
    lam = float(ridge_lambda) * float(n)
    gram = Z.T @ Z + lam * np.eye(k)
    rhs = Z.T @ r + lam * b0
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - lam>0 keeps gram SPD
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def fit_term_structure(
    market,
    signals,
    cfg: MpoConfig,
    calib_days: int | None = None,
) -> TermStructureModel:
    """Calibrate the alpha term structure on days ``[0, calib_days)``.

    Parameters
    ----------
    market : MarketPanel
        Only ``prices`` ``[T, N]`` and ``n_days`` are read.
    signals : SignalPanel
        ``signals.signals`` ``[T, N, 3]`` RAW (not z-scored) predictions.
    cfg : MpoConfig
        Uses ``calib_days``, ``ridge_lambda``, ``winsor_sigma``, ``decay_mult``.
    calib_days : int, optional
        Overrides ``cfg.calib_days``. Must be >= 1 and <= ``market.n_days``.

    Returns
    -------
    TermStructureModel
        With ``B`` ``[3,3]``, per-minute ``rho`` ``[3]``, ``seg_bounds`` and
        ``diagnostics``. Never reads prices or signals at
        ``tau >= calib_days*240`` (no-lookahead guard, see module docstring).
    """
    n_days = int(getattr(market, "n_days"))
    prices = np.asarray(market.prices, dtype=np.float64)
    raw = np.asarray(signals.signals, dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError(f"signals.signals must be [T, N, K], got {raw.shape}")
    if prices.shape[0] != raw.shape[0] or prices.shape[1] != raw.shape[1]:
        raise ValueError(
            f"prices {prices.shape} and signals {raw.shape} disagree on (T, N)"
        )
    n_seg = len(SEGMENT_BOUNDS)
    n_sig = raw.shape[2]

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

    taus = epoch_taus(days)  # every decision epoch inside the calib window
    # ---- 1. cross-sectional z-scores at every calibration epoch -----------
    z_epoch = np.stack([zscore_cross_section(raw[t]) for t in taus], axis=0)  # [E,N,K]

    # ---- 2..3. per-segment samples (guarded), winsorized targets ----------
    B = np.zeros((n_seg, n_sig), dtype=np.float64)
    seg_ic = np.full(n_seg, np.nan)
    seg_scale = np.zeros(n_seg, dtype=np.float64)
    seg_n = np.zeros(n_seg, dtype=np.int64)
    seg_taus_used = np.zeros(n_seg, dtype=np.int64)
    max_tau_touched = -1

    for s, (a, b) in enumerate(SEGMENT_BOUNDS):
        chunks_z: list[np.ndarray] = []
        chunks_r: list[np.ndarray] = []
        for e, tau in enumerate(taus):
            far = int(tau) + int(b)
            # NO-LOOKAHEAD GUARD: the far price must stay inside the window.
            if far >= t_calib:
                continue
            near = int(tau) + int(a)
            p0 = prices[near]
            p1 = prices[far]
            max_tau_touched = max(max_tau_touched, far)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = p1 / p0 - 1.0
            zz = z_epoch[e]
            ok = np.isfinite(r) & np.isfinite(p0) & (p0 > 0) & np.all(np.isfinite(zz), axis=1)
            if ok.sum() < _MIN_CROSS_SECTION:
                continue
            chunks_z.append(zz[ok])
            chunks_r.append(winsorize_cross_section(r[ok], cfg.winsor_sigma))
        if not chunks_z:
            continue

        Z = np.concatenate(chunks_z, axis=0)
        rr = np.concatenate(chunks_r, axis=0)
        seg_n[s] = Z.shape[0]
        seg_taus_used[s] = len(chunks_z)
        # Realized return scale of this segment -> prior magnitude.
        seg_scale[s] = float(rr.std())
        b0 = TELESCOPING_PRIOR[s, :n_sig] * seg_scale[s]
        B[s] = _ridge_row(Z, rr, b0, cfg.ridge_lambda)

        # ---- in-sample cross-sectional Spearman IC (per epoch, averaged) --
        ics = []
        for zc, rc in zip(chunks_z, chunks_r):
            ic = _spearman(zc @ B[s], rc)
            if np.isfinite(ic):
                ics.append(ic)
        if ics:
            seg_ic[s] = float(np.mean(ics))

    # ---- 4. AR(1) decay on the epoch grid --------------------------------
    decay = estimate_ar1_decay(z_epoch)

    diagnostics = TermStructureDiagnostics(
        seg_ic=seg_ic,
        coef_table=B.copy(),
        rho_epoch=decay.rho_epoch.copy(),
        n_samples=int(seg_n.sum()),
        notes={
            "calib_days": days,
            "t_calib": int(t_calib),
            "n_epochs": int(taus.size),
            "n_samples_per_segment": seg_n.tolist(),
            "n_epochs_per_segment": seg_taus_used.tolist(),
            "seg_return_scale": seg_scale.tolist(),
            "prior_scaled": (TELESCOPING_PRIOR[:, :n_sig] * seg_scale[:, None]).tolist(),
            "rho_minute": decay.rho_minute.tolist(),
            "rho_epoch_raw": decay.rho_epoch_raw.tolist(),
            "half_life_minutes": decay.half_life_minutes.tolist(),
            "kendall_t_eff": int(decay.t_eff),
            "decay_mult": float(cfg.decay_mult),
            "max_tau_touched": int(max_tau_touched),
            "ridge_lambda": float(cfg.ridge_lambda),
            "winsor_sigma": float(cfg.winsor_sigma),
        },
    )
    return TermStructureModel(
        B=B,
        rho=decay.rho_minute,
        seg_bounds=SEGMENT_BOUNDS,
        diagnostics=diagnostics,
        decay_mult=float(cfg.decay_mult),
        decay=decay,
    )
