"""MPO-T1 strategy zoo (docs/mpo-t1-SPEC.md section 7).

Uniform protocol: every strategy exposes ``decide(ctx: DecisionContext) ->
np.ndarray | None`` (target weights [N] for the execution stage, or None for
"do not trade"), a ``name`` property (canonical roster name used by
``openpo.mpo.backtest``) and a ``last_solve_time`` property (seconds spent in
the most recent QP solve, 0.0 when no solve happened).

Roster (SPEC section 7):

1. :class:`MpoStrategy`            -- the multi-period optimizer (J <= 4).
2. :class:`SingleSignalStrategy`   -- one signal, single period (J = 1).
3. :class:`FixedWeightStrategy`    -- fixed-lambda composite, single period.
4. :class:`InformedTradingStrategy`-- Israelov-Katz: mid/long-horizon target
   with a short-signal gate on individual trades.
5. :class:`OracleMpoStrategy`      -- MPO fed with the TRUE segment alphas
   (upper bound for gain decomposition; not a competitor).
6. :class:`BuyAndHoldStrategy`     -- equal-weight anchor (leak detector).

Every optimizing strategy goes through the SAME QP layer
(``openpo.mpo.qp.build_mpo_qp`` + ``openpo.mpo.solve.solve_mpo_qp``), so the
T+1 constraints apply uniformly: with J = 1 the QP still enforces
``s^1 <= sellable_now``. Non-``optimal`` solver status is fail-soft: decide
returns None (no trade) and increments ``fail_soft_count``.

Time conventions (project-wide, repeated on purpose):

- ``tau = day * 240 + minute``; the synthetic calendar has NO holidays, so
  trading minutes between two taus equal their difference exactly and
  overnight counts as ZERO trading minutes.
- Alpha segments ``SEGMENT_BOUNDS = ((0,30],(30,240],(240,2400]]`` are
  relative half-open trading-minute intervals anchored at the decision tau.

The QP siblings (``openpo.mpo.qp`` / ``openpo.mpo.solve``) are imported
lazily inside the solve path so this module can be imported and partially
tested before those modules land (parallel implementation).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import numpy as np

from openpo.mpo.calendar import day_close_tau
from openpo.mpo.config import (
    EPOCH_OFFSETS,
    MINUTES_PER_DAY,
    SEGMENT_BOUNDS,
    SIGNAL_HORIZONS,
    MpoConfig,
)
from openpo.mpo.delay_cost import DelayCostConfig, delay_cost_coefficients
from openpo.mpo.stages import build_stage_grid
from openpo.mpo.types import DecisionContext, StageGrid

__all__ = [
    "Strategy",
    "MpoStrategy",
    "SingleSignalStrategy",
    "FixedWeightStrategy",
    "InformedTradingStrategy",
    "OracleMpoStrategy",
    "BuyAndHoldStrategy",
    "informed_gate",
    "segment_stage_weights",
]

# Column index of the short-horizon signal (return_1_30) in the z panel.
_SHORT_SIGNAL = 0
# Israelov-Katz gate band on the short-horizon z-score (SPEC section 7 item 4).
_GATE_BAND = 0.1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lazy_qp_modules():
    """Import the QP siblings lazily (they may not exist during parallel dev)."""
    qp_mod = importlib.import_module("openpo.mpo.qp")
    solve_mod = importlib.import_module("openpo.mpo.solve")
    return qp_mod, solve_mod


def segment_stage_weights(tau_now: int, stage_start: int, stage_end: int) -> np.ndarray:
    """Fractional overlap of each alpha segment with an absolute stage.

    Segment ``s = (a, b]`` (relative trading minutes) is shifted to the
    absolute interval ``(tau_now + a, tau_now + b]``; the returned weight is
    ``overlap_minutes(shifted_seg, (stage_start, stage_end]) / (b - a)``.
    Because the synthetic calendar has no holidays, interval intersection on
    raw taus IS the trading-minute overlap (overnight = 0 extra minutes).
    """
    w = np.zeros(len(SEGMENT_BOUNDS), dtype=np.float64)
    for s, (a, b) in enumerate(SEGMENT_BOUNDS):
        lo = max(stage_start, tau_now + a)
        hi = min(stage_end, tau_now + b)
        w[s] = max(0, hi - lo) / float(b - a)
    return w


def informed_gate(
    w_star: np.ndarray,
    w_now: np.ndarray,
    z_short: np.ndarray,
    band: float = _GATE_BAND,
) -> np.ndarray:
    """Israelov-Katz short-signal trade gate (SPEC section 7 item 4).

    Exact frozen formula::

        trade_i   = w_star_i - w_now_i
        blocked_i = sign(trade_i) * z_short_i < -band        (band = 0.1)
        target_i  = w_now_i  if blocked_i else w_star_i

    i.e. a trade passes only when the short-horizon z-score does not point
    against its direction by more than the band (same sign, or |z_short|
    within the band). Blocked names keep their current weight (suspended
    order); ``sign(0) = 0`` so zero trades are never blocked.
    """
    w_star = np.asarray(w_star, dtype=np.float64)
    w_now = np.asarray(w_now, dtype=np.float64)
    z_short = np.asarray(z_short, dtype=np.float64)
    trade = w_star - w_now
    blocked = np.sign(trade) * z_short < -band
    return np.where(blocked, w_now, w_star)


def _as_vector(x, n: int, name: str) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    if v.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {v.shape}")
    return v


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------

class Strategy:
    """Uniform strategy protocol.

    Subclasses set ``self._name`` in ``__init__`` and implement
    :meth:`decide`. ``fail_soft_count`` counts decisions abandoned because the
    QP did not return an ``optimal`` status (the caller then does not trade).

    ``last_status``/``last_note`` carry the RAW solver verdict of the most
    recent solve so the simulator can tally it verbatim. Collapsing every
    non-optimal outcome into one "fail_soft" bucket hid a 100%-reproducible
    OSQP ``primal infeasible`` (T+1 x w_max box conflict) behind what looked
    like generic solver flakiness for a whole benchmark run -- the root cause
    must be readable at the first scene.
    """

    def __init__(self) -> None:
        self._name = self.__class__.__name__
        self._last_solve_time = 0.0
        self._last_status: str | None = None
        self._last_note: str | None = None
        self.fail_soft_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def last_solve_time(self) -> float:
        """Seconds spent in the most recent QP solve (0.0 if none yet)."""
        return self._last_solve_time

    @property
    def last_status(self) -> str | None:
        """Raw solver status of the most recent solve ('optimal', 'inaccurate',
        'failed'), or None when no solve has run yet."""
        return self._last_status

    @property
    def last_note(self) -> str | None:
        """Raw solver diagnostic of the most recent solve (e.g. the backend's
        own ``osqp status=primal infeasible``), or None when absent."""
        return self._last_note

    def decide(self, ctx: DecisionContext) -> np.ndarray | None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# QP-backed base
# ---------------------------------------------------------------------------

class _QpStrategy(Strategy):
    """Shared risk/impact inputs + the common build-QP/solve/fail-soft path.

    Inputs may be given explicitly or derived from a ``market`` panel
    (integration glue for :func:`openpo.mpo.backtest._instantiate`):

    - ``Sigma_d``  [N,N] daily covariance; default ``market.daily_cov()``.
      A 2-D array passed as ``sigma_d`` is treated as the covariance too
      (the backtest candidate pool aliases ``sigma_d`` to the dense cov).
    - ``sigma_d``  [N] daily vols; default ``sqrt(diag(Sigma_d))``.
    - ``adv_frac`` [N] ADV_i / NAV; default ``market.adv_cny / cfg.nav_cny``.
    - ``n_days``   calendar length; default ``market.n_days``.

    Factor risk model (``factor_exposures`` [N,K] / ``factor_cov`` [K,K] /
    ``specific_var`` [N]): optional, defaults to the ``MarketPanel``'s own
    ``factor_exposures`` / ``factor_cov_daily`` / ``specific_var_daily`` when a
    ``market`` is given. It is handed to ``build_mpo_qp`` together with
    ``risk_form`` (default ``'auto'``), which only switches to the sparse
    low-rank assembly above ``qp.LOWRANK_AUTO_MIN_ASSETS`` (500) names. Every
    N <= 300 configuration -- i.e. the whole benchmark roster -- therefore keeps
    the byte-identical dense QP it had before this plumbing existed; the factor
    triplet is what makes N ~ 5000 solvable at all (see the ``openpo.mpo.qp``
    module docstring for the measured walls).

    Every concrete strategy forwards these four through a ``**risk_kwargs``
    tail, so they are accepted (but never required) by the whole roster and stay
    invisible to ``backtest._instantiate``'s candidate-pool matching -- which is
    what keeps the default builds byte-identical.
    """

    def __init__(
        self,
        Sigma_d=None,
        adv_frac=None,
        sigma_d=None,
        cfg: MpoConfig | None = None,
        n_days: int | None = None,
        market=None,
        factor_exposures=None,
        factor_cov=None,
        specific_var=None,
        risk_form: str = "auto",
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else MpoConfig()

        # Pool alias tolerance: a 2-D "sigma_d" is really the covariance.
        if sigma_d is not None:
            sigma_d = np.asarray(sigma_d, dtype=np.float64)
            if sigma_d.ndim == 2:
                if Sigma_d is None:
                    Sigma_d = sigma_d
                sigma_d = None

        if Sigma_d is None and market is not None:
            Sigma_d = market.daily_cov()
        if Sigma_d is None:
            raise ValueError("Sigma_d is required (directly or via market)")
        self.Sigma_d = np.asarray(Sigma_d, dtype=np.float64)
        if self.Sigma_d.ndim != 2 or self.Sigma_d.shape[0] != self.Sigma_d.shape[1]:
            raise ValueError(f"Sigma_d must be square [N,N], got {self.Sigma_d.shape}")
        self.n_assets = self.Sigma_d.shape[0]

        if sigma_d is None:
            sigma_d = np.sqrt(np.clip(np.diag(self.Sigma_d), 0.0, None))
        self.sigma_d = _as_vector(sigma_d, self.n_assets, "sigma_d")

        if adv_frac is None:
            if market is None:
                raise ValueError("adv_frac is required (directly or via market)")
            adv_frac = np.asarray(market.adv_cny, dtype=np.float64) / self.cfg.nav_cny
        self.adv_frac = _as_vector(adv_frac, self.n_assets, "adv_frac")

        if n_days is None:
            if market is None:
                raise ValueError("n_days is required (directly or via market)")
            n_days = market.n_days
        self.n_days = int(n_days)
        if self.n_days <= 0:
            raise ValueError(f"n_days must be positive, got {self.n_days}")

        # --- factor risk model (low-rank QP assembly, opt-in by size) -------
        # Taken from the market panel when it carries one and nothing was passed
        # explicitly. Adopted only when its cross-section matches this
        # strategy's N: a market whose risk model describes a different universe
        # than ``Sigma_d`` is a wiring bug, and the QP would be built from two
        # inconsistent risk models.
        if factor_exposures is None and factor_cov is None and specific_var is None:
            x = getattr(market, "factor_exposures", None)
            f = getattr(market, "factor_cov_daily", None)
            spec = getattr(market, "specific_var_daily", None)
            if x is not None and f is not None and spec is not None:
                x = np.asarray(x, dtype=np.float64)
                if x.ndim != 2 or x.shape[0] != self.n_assets:
                    raise ValueError(
                        f"market.factor_exposures must be [{self.n_assets},K] to "
                        f"match Sigma_d, got shape {x.shape}"
                    )
                factor_exposures = x
                factor_cov = np.asarray(f, dtype=np.float64)
                specific_var = _as_vector(spec, self.n_assets, "specific_var_daily")
        self.factor_exposures = factor_exposures
        self.factor_cov = factor_cov
        self.specific_var = specific_var
        self.risk_form = str(risk_form)

    # -- shared decision plumbing ------------------------------------------

    @staticmethod
    def _tau_of(ctx: DecisionContext) -> int:
        return int(ctx.day) * MINUTES_PER_DAY + int(ctx.minute)

    def _z_of(self, ctx: DecisionContext) -> np.ndarray:
        """[N,3] z-scores; NaNs (halted names) mapped to 0 = neutral alpha."""
        z = np.asarray(ctx.z_signals, dtype=np.float64)
        if z.shape != (self.n_assets, 3):
            raise ValueError(f"z_signals must be [{self.n_assets},3], got {z.shape}")
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _qp_build_kwargs(self, alpha: np.ndarray, ctx: DecisionContext) -> dict:
        """Extra keyword arguments for ``build_mpo_qp``; empty by default.

        Sole extension point for objective tweaks that only add per-name
        CONSTANTS to ``q`` (the delay-cost surcharge of
        :class:`MpoStrategy`). Returning ``{}`` -- what every strategy but the
        opt-in ``mpo_delayaware`` arm does -- leaves the assembled QP bit
        identical to the frozen SPEC objective.
        """
        return {}

    def _solve_qp(self, alpha: np.ndarray, grid: StageGrid, ctx: DecisionContext):
        """Build + solve the multi-stage QP; return w^1 or None (fail-soft).

        Positional contract of ``build_mpo_qp`` (SPEC section 3):
        (alpha[J,N], grid, w0[N], sellable_w[N], Sigma_d[N,N], adv_frac[N],
        sigma_d[N], limit_up_mask, limit_down_mask, cfg), plus the factor risk
        model / ``risk_form`` keywords (see the class docstring; ``None`` +
        ``'auto'`` is the no-op default) and whatever :meth:`_qp_build_kwargs`
        adds.
        """
        qp_mod, solve_mod = _lazy_qp_modules()
        snap = ctx.snapshot
        w0 = _as_vector(snap.weights, self.n_assets, "snapshot.weights")
        sellable = _as_vector(
            snap.sellable_weights, self.n_assets, "snapshot.sellable_weights"
        )
        alpha = np.asarray(alpha, dtype=np.float64)
        if alpha.shape != (grid.n_stages, self.n_assets):
            raise ValueError(
                f"alpha must be [{grid.n_stages},{self.n_assets}], got {alpha.shape}"
            )
        qp = qp_mod.build_mpo_qp(
            alpha,
            grid,
            w0,
            sellable,
            self.Sigma_d,
            self.adv_frac,
            self.sigma_d,
            np.asarray(ctx.limit_up, dtype=bool),
            np.asarray(ctx.limit_down, dtype=bool),
            self.cfg,
            factor_exposures=self.factor_exposures,
            factor_cov=self.factor_cov,
            specific_var=self.specific_var,
            risk_form=self.risk_form,
            **self._qp_build_kwargs(alpha, ctx),
        )
        sol = self._call_solver(solve_mod, qp)
        self._last_solve_time = float(getattr(sol, "solve_time", 0.0) or 0.0)
        # Raw verdict first: record before any fail-soft branching so the
        # backend's own reason reaches the simulator untranslated.
        status = getattr(sol, "status", None)
        self._last_status = None if status is None else str(status)
        note = getattr(sol, "note", None)
        self._last_note = None if note is None else str(note)
        if sol.status != "optimal" or sol.weights is None:
            self.fail_soft_count += 1
            return None
        w1 = np.array(sol.weights[0], dtype=np.float64, copy=True)  # execution stage
        # Halted names: keep current holdings (simulator freezes them anyway;
        # avoid planning unfillable orders on frozen prices).
        halted = np.asarray(ctx.halted, dtype=bool)
        if halted.any():
            w1[halted] = w0[halted]
        return w1

    def _call_solver(self, solve_mod, qp):
        """Adapter for ``solve_mpo_qp(qp, backend=..., cfg=...)`` variants."""
        fn = solve_mod.solve_mpo_qp
        params = inspect.signature(fn).parameters
        kwargs = {}
        if "backend" in params:
            kwargs["backend"] = self.cfg.qp_backend
        if "cfg" in params:
            kwargs["cfg"] = self.cfg
        return fn(qp, **kwargs)

    def _solve_single_stage(self, alpha1: np.ndarray, ctx: DecisionContext):
        """Solve the J=1 (max_stages=1) problem with the given stage alpha.

        The single stage is the execution stage ``(tau, min(tau+30, close)]``;
        the T+1 constraint degenerates to ``s^1 <= sellable_now`` inside the
        same QP layer used by the multi-period strategies.
        """
        tau = self._tau_of(ctx)
        cfg1 = dataclasses.replace(self.cfg, max_stages=1)
        grid = build_stage_grid(tau, self.n_days, cfg1)
        alpha = np.asarray(alpha1, dtype=np.float64).reshape(1, self.n_assets)
        return self._solve_qp(alpha, grid, ctx)


# ---------------------------------------------------------------------------
# 1. MPO (the subject)
# ---------------------------------------------------------------------------

class MpoStrategy(_QpStrategy):
    """Multi-period optimizer: stage grid -> term-structure alphas -> QP.

    Per decision epoch: ``build_stage_grid`` (J <= cfg.max_stages) ->
    ``ts_model.stage_alphas(z, grid)`` (segment-overlap x per-signal AR(1)
    decay) -> ``build_mpo_qp``/``solve_mpo_qp`` -> return w^1. Any
    non-optimal solver status is fail-soft (None, counted).

    ``delay_cost_cfg`` (optional, default None) turns on the
    implementation-delay surcharge of ``openpo.mpo.delay_cost``
    (``docs/mpo-t1-delaycost-design.md``): the per-name coefficients derived
    from the STAGE-1 alpha are added to the stage-1 buy/sell linear costs of the
    very same QP. Every other input is untouched, so the A/B against plain
    ``mpo`` is paired by construction. The arm's roster name flips to
    ``mpo_delayaware`` when the config is supplied -- one object, one name, no
    way to end up with a surcharged strategy reporting as ``mpo``.
    """

    def __init__(
        self,
        ts_model,
        Sigma_d=None,
        adv_frac=None,
        sigma_d=None,
        cfg: MpoConfig | None = None,
        n_days: int | None = None,
        market=None,
        delay_cost_cfg: DelayCostConfig | None = None,
        **risk_kwargs,
    ) -> None:
        super().__init__(Sigma_d, adv_frac, sigma_d, cfg, n_days, market, **risk_kwargs)
        self.ts_model = ts_model
        if delay_cost_cfg is not None and not isinstance(delay_cost_cfg, DelayCostConfig):
            raise TypeError(
                "delay_cost_cfg must be a delay_cost.DelayCostConfig, got "
                f"{type(delay_cost_cfg).__name__} (MpoConfig stays in cfg)"
            )
        self.delay_cost_cfg = delay_cost_cfg
        # Coefficient ledger of the most recent build (design doc F2: a mean
        # under ~1 bp means the wiring is wrong, so it must be observable).
        self.last_delay_extra: dict | None = None
        self._name = "mpo" if delay_cost_cfg is None else "mpo_delayaware"

    def _qp_build_kwargs(self, alpha: np.ndarray, ctx: DecisionContext) -> dict:
        """Stage-1 delay surcharge from ``alpha[0]``, or ``{}`` when disabled.

        ``alpha[0]`` IS the S1 expected return over the execution stage -- stage
        0 carries no AR(1) decay by the SPEC stage-alpha convention -- and the
        fill window coincides with it (both are ``cfg.exec_minutes``), which is
        why ``fill_window_scale`` defaults to 1.0.
        """
        cfg = self.delay_cost_cfg
        if cfg is None:
            return {}
        c_buy_extra, c_sell_extra = delay_cost_coefficients(alpha[0], cfg)
        self.last_delay_extra = {
            "mode": cfg.mode,
            "scale": float(cfg.scale),
            "c_buy_extra_mean": float(np.mean(c_buy_extra)),
            "c_buy_extra_max": float(np.max(c_buy_extra)),
            "c_sell_extra_mean": float(np.mean(c_sell_extra)),
            "c_sell_extra_max": float(np.max(c_sell_extra)),
        }
        kwargs = {"c_buy_extra": c_buy_extra, "c_sell_extra": c_sell_extra}
        if cfg.mode == "exact":
            # 'exact' hands a rebate to trading against the short alpha; legal
            # (the two extras sum to zero, so the wash pair stays penalized) but
            # it must be asked for explicitly on the QP side too.
            kwargs["allow_negative_extra"] = True
        return kwargs

    def decide(self, ctx: DecisionContext) -> np.ndarray | None:
        tau = self._tau_of(ctx)
        grid = build_stage_grid(tau, self.n_days, self.cfg)
        z = self._z_of(ctx)
        alpha = np.asarray(self.ts_model.stage_alphas(z, grid), dtype=np.float64)
        return self._solve_qp(alpha, grid, ctx)


# ---------------------------------------------------------------------------
# 2. single-signal single-period baselines
# ---------------------------------------------------------------------------

class SingleSignalStrategy(_QpStrategy):
    """One signal, single period (J = 1).

    The stage alpha is signal k's aggregate segment contribution over the
    REST OF TODAY ``(tau, today's close]``: the fitted B is used along the
    ``B[:, k]`` path only (other signal columns zeroed) with segment-overlap
    apportioning and no decay (this aggregate acts as stage 1, delta = 0 per
    the SPEC stage-alpha convention)::

        alpha_i = sum_s overlap((tau+a_s, tau+b_s], (tau, close]) / (b_s - a_s)
                        * B[s, k] * z[i, k]
    """

    def __init__(
        self,
        k: int,
        ts_model,
        Sigma_d=None,
        adv_frac=None,
        sigma_d=None,
        cfg: MpoConfig | None = None,
        n_days: int | None = None,
        market=None,
        **risk_kwargs,
    ) -> None:
        super().__init__(Sigma_d, adv_frac, sigma_d, cfg, n_days, market, **risk_kwargs)
        self.k = int(k)
        if not 0 <= self.k < 3:
            raise ValueError(f"signal index k must be in {{0,1,2}}, got {k}")
        self.ts_model = ts_model
        self._name = f"single_{SIGNAL_HORIZONS[self.k]}"

    def decide(self, ctx: DecisionContext) -> np.ndarray | None:
        tau = self._tau_of(ctx)
        close = day_close_tau(int(ctx.day))
        seg_w = segment_stage_weights(tau, tau, close)  # (tau, close] coverage
        B = np.asarray(self.ts_model.B, dtype=np.float64)
        coef = float(seg_w @ B[:, self.k])  # scalar: B[:,k] path only
        z = self._z_of(ctx)
        alpha1 = coef * z[:, self.k]
        return self._solve_single_stage(alpha1, ctx)


# ---------------------------------------------------------------------------
# 3. fixed-weight composite single-period baseline
# ---------------------------------------------------------------------------

class FixedWeightStrategy(_QpStrategy):
    """Fixed-lambda composite of the three signals, single period (J = 1).

    ``alpha_mix_i = sum_k lambdas_k * z[i,k] / std_k`` with std_k the
    decision-time cross-sectional std of z[:,k] (re-normalization; degenerate
    std falls back to 1). The mix is scaled by ``||B[0,:]||_2`` (norm of the
    fitted B row for the (0,30] segment) so the alpha magnitude is
    commensurate with the return units the QP costs/risk expect. ``lambdas``
    are selected on dedicated tuning seeds (see backtest.tune_fixed_weights).
    """

    def __init__(
        self,
        lambdas,
        ts_model=None,
        Sigma_d=None,
        adv_frac=None,
        sigma_d=None,
        cfg: MpoConfig | None = None,
        n_days: int | None = None,
        market=None,
        **risk_kwargs,
    ) -> None:
        super().__init__(Sigma_d, adv_frac, sigma_d, cfg, n_days, market, **risk_kwargs)
        lam = np.asarray(lambdas, dtype=np.float64)
        if lam.shape != (3,):
            raise ValueError(f"lambdas must have shape (3,), got {lam.shape}")
        if lam.min() < 0 or lam.sum() <= 0:
            raise ValueError(f"lambdas must be non-negative with positive sum, got {lam}")
        self.lambdas = lam
        self.ts_model = ts_model
        self._name = "fixed_weight"

    def _alpha_scale(self) -> float:
        """||B[0,:]|| of the fitted term structure; 1.0 without a model."""
        if self.ts_model is None:
            return 1.0
        B = np.asarray(self.ts_model.B, dtype=np.float64)
        scale = float(np.linalg.norm(B[0, :]))
        return scale if scale > 0.0 else 1.0

    def decide(self, ctx: DecisionContext) -> np.ndarray | None:
        z = self._z_of(ctx)
        std = z.std(axis=0, ddof=0)
        std = np.where(std > 1e-12, std, 1.0)  # cross-sectional re-normalization
        alpha1 = self._alpha_scale() * ((z / std) @ self.lambdas)
        return self._solve_single_stage(alpha1, ctx)


# ---------------------------------------------------------------------------
# 4. Israelov-Katz informed trading
# ---------------------------------------------------------------------------

class InformedTradingStrategy(_QpStrategy):
    """Mid/long-horizon single-period target with a short-signal trade gate.

    Target w* is the J=1 solution for the rest-of-today alpha computed along
    the B path with the SHORT signal column removed (B[:, 0] := 0)::

        alpha_i = sum_s seg_w_s * sum_{k in {1,2}} B[s,k] * z[i,k]

    Gate (exact frozen formula, see :func:`informed_gate`): with
    ``trade_i = w*_i - w_i``, a name is suspended (target_i = current w_i)
    iff ``sign(trade_i) * z_short_i < -0.1``.
    """

    def __init__(
        self,
        ts_model,
        Sigma_d=None,
        adv_frac=None,
        sigma_d=None,
        cfg: MpoConfig | None = None,
        n_days: int | None = None,
        market=None,
        **risk_kwargs,
    ) -> None:
        super().__init__(Sigma_d, adv_frac, sigma_d, cfg, n_days, market, **risk_kwargs)
        self.ts_model = ts_model
        self._name = "informed"

    def decide(self, ctx: DecisionContext) -> np.ndarray | None:
        tau = self._tau_of(ctx)
        close = day_close_tau(int(ctx.day))
        seg_w = segment_stage_weights(tau, tau, close)
        B = np.array(self.ts_model.B, dtype=np.float64, copy=True)
        B[:, _SHORT_SIGNAL] = 0.0  # drop the short-signal path from the target
        coef = seg_w @ B  # [3 signals]; column 0 is exactly zero
        z = self._z_of(ctx)
        alpha1 = z @ coef
        w_star = self._solve_single_stage(alpha1, ctx)
        if w_star is None:
            return None
        w0 = _as_vector(ctx.snapshot.weights, self.n_assets, "snapshot.weights")
        # Suspended names keep target = current weight => no order is planned
        # for them; the passed-through targets stay inside the QP box/T+1 set.
        return informed_gate(w_star, w0, z[:, _SHORT_SIGNAL], band=_GATE_BAND)


# ---------------------------------------------------------------------------
# 5. oracle MPO (gain-decomposition upper bound)
# ---------------------------------------------------------------------------

class OracleMpoStrategy(_QpStrategy):
    """MPO fed with the TRUE segment alphas from the market generator.

    Identical QP layer/constraints as :class:`MpoStrategy`, but the stage
    alphas bypass calibration: the true per-segment alphas are apportioned to
    stages by minute overlap WITHOUT decay -- the oracle values are already
    conditional expectations over each segment, so applying rho decay again
    would double-count::

        alpha[j,i] = sum_s overlap(SEG_s + tau, stage_j) / len(SEG_s)
                           * true_seg_alpha[t, i, s]

    ``true_seg_alpha`` may be sampled on the minute grid ([n_days*240, N, 3])
    or on the epoch grid ([n_days*8, N, 3]); both layouts are accepted.
    """

    def __init__(
        self,
        true_seg_alpha=None,
        Sigma_d=None,
        adv_frac=None,
        sigma_d=None,
        cfg: MpoConfig | None = None,
        n_days: int | None = None,
        market=None,
        **risk_kwargs,
    ) -> None:
        super().__init__(Sigma_d, adv_frac, sigma_d, cfg, n_days, market, **risk_kwargs)
        if true_seg_alpha is None and market is not None:
            true_seg_alpha = market.true_seg_alpha
        if true_seg_alpha is None:
            raise ValueError("true_seg_alpha is required (directly or via market)")
        tsa = np.asarray(true_seg_alpha, dtype=np.float64)
        if tsa.ndim != 3 or tsa.shape[1] != self.n_assets or tsa.shape[2] != 3:
            raise ValueError(
                f"true_seg_alpha must be [T,{self.n_assets},3], got {tsa.shape}"
            )
        t_minutes = self.n_days * MINUTES_PER_DAY
        t_epochs = self.n_days * len(EPOCH_OFFSETS)
        if tsa.shape[0] not in (t_minutes, t_epochs):
            raise ValueError(
                f"true_seg_alpha time axis {tsa.shape[0]} matches neither the "
                f"minute grid ({t_minutes}) nor the epoch grid ({t_epochs})"
            )
        self.true_seg_alpha = tsa
        self._name = "oracle_mpo"

    def _oracle_row(self, day: int, minute: int, tau: int) -> np.ndarray:
        """[N,3] true segment alphas at the decision time."""
        if self.true_seg_alpha.shape[0] == self.n_days * MINUTES_PER_DAY:
            return self.true_seg_alpha[tau]
        if minute not in EPOCH_OFFSETS:
            raise ValueError(
                f"minute {minute} is not on the epoch grid {EPOCH_OFFSETS}; "
                "epoch-grid true_seg_alpha cannot be indexed"
            )
        idx = day * len(EPOCH_OFFSETS) + EPOCH_OFFSETS.index(minute)
        return self.true_seg_alpha[idx]

    def decide(self, ctx: DecisionContext) -> np.ndarray | None:
        tau = self._tau_of(ctx)
        grid = build_stage_grid(tau, self.n_days, self.cfg)
        seg = np.nan_to_num(
            self._oracle_row(int(ctx.day), int(ctx.minute), tau),
            nan=0.0, posinf=0.0, neginf=0.0,
        )  # [N, 3]
        alpha = np.empty((grid.n_stages, self.n_assets), dtype=np.float64)
        for j, stage in enumerate(grid.stages):
            alpha[j] = seg @ segment_stage_weights(tau, stage.start, stage.end)
        return self._solve_qp(alpha, grid, ctx)


# ---------------------------------------------------------------------------
# 6. buy-and-hold anchor
# ---------------------------------------------------------------------------

class BuyAndHoldStrategy(Strategy):
    """Equal-weight buy at the first epoch, then hold forever.

    First decide() returns ``min(1/N, w_max)`` for every name (budget <= 1 by
    construction); every later call returns None (no trade). Serves as the
    simulator leak detector -- with zero alpha content its NAV must track the
    held portfolio exactly. No QP is involved, so ``last_solve_time`` is 0.
    """

    def __init__(self, cfg: MpoConfig | None = None, n_assets: int | None = None, market=None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else MpoConfig()
        if n_assets is None and market is not None:
            n_assets = market.n_assets
        self.n_assets = None if n_assets is None else int(n_assets)
        self._done = False
        self._name = "buy_hold"

    def decide(self, ctx: DecisionContext) -> np.ndarray | None:
        if self._done:
            return None
        self._done = True
        n = len(np.asarray(ctx.snapshot.weights))
        if self.n_assets is not None and n != self.n_assets:
            raise ValueError(f"snapshot has {n} assets, expected {self.n_assets}")
        return np.full(n, min(1.0 / n, self.cfg.w_max), dtype=np.float64)
