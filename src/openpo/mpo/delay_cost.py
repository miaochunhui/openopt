"""Implementation-delay cost coefficients for MPO-T1 (Iteration 42 leg 1).

Authoritative design: ``docs/mpo-t1-delaycost-design.md``. This module is
LEG 1: coefficient math + its own frozen config + tests only. Nothing here is
wired into the QP or the strategy pool yet -- leg 2 adds the objective term
after design review, so this file must stay import-safe and side-effect free.

The problem being priced
------------------------
``execution.match_orders`` fills at the BAR END reference price (SPEC section 4:
"planning prices at the DECISION minute, matching prices at the bar END"). A
newly opened position therefore forfeits the whole ``(0, exec_minutes]`` price
move -- you buy AFTER the move you predicted. But the QP objective credits the
stage-1 alpha to the full stage-end weight, ``-alpha[0] . w^1`` with
``w^1 = w^0 + dw^1``, so it over-credits by exactly ``alpha_S1 . (b^1 - s^1)``.
The pre-existing ``w^0`` part is real (you do hold through the window); only the
NET NEW trade part is fiction.

This is not zero-mean noise: you trade WITH your own alpha, so the forfeited
move has a fixed sign. Iteration 41 measured it at 13.2 bp per unit traded at
``bar = 30`` versus the 9.3 bp of explicit cost the QP does model.

Committed remedy (design doc section 2, primary = ``mode='signed'``)
--------------------------------------------------------------------
Undo the over-credit as a LINEAR per-name surcharge on the stage-1 buy/sell
split -- which is exactly where a per-unit-traded cost belongs::

    mode='signed'  (DEFAULT)   c_buy_extra  = max(+alpha_S1, 0)
                               c_sell_extra = max(-alpha_S1, 0)
    mode='exact'               c_buy_extra  = +alpha_S1
                               c_sell_extra = -alpha_S1
    mode='flat'                c_buy_extra  = c_sell_extra = flat_rate

``'exact'`` is the algebraically exact restoration (the new position's S1
contribution becomes exactly zero) but it hands a NEGATIVE cost -- a rebate --
to buying a name predicted to fall. That rebate is a bet on the shortest-horizon
alpha's sign being right, so the committed default clips it away: delay is a
cost, never a rebate, and ``c_base + extra >= c_base > 0`` always.

Sign asymmetry is the whole point and is deliberate: a positive 30-minute alpha
penalizes BUYS (you chase the move) and leaves sells alone; a negative alpha
penalizes SELLS.

Convexity / boundedness (design doc risk 2)
-------------------------------------------
The coefficients are pure functions of the alpha vector -- of the DATA ``z``,
never of the decision variables -- so they only add per-name constants to the
OSQP ``q`` vector. ``P`` is untouched and the problem stays the same convex QP.
The only unbounded direction of a linear-cost QP is the wash pair
(``b_i, s_i -> inf`` at fixed ``w``); its cost is
``(c_buy + extra_buy) + (c_sell + extra_sell)``, which is ``>= c_buy + c_sell``
in ``'signed'`` mode and EXACTLY ``c_buy + c_sell`` in ``'exact'`` mode (the two
extras sum to zero there), so wash trading stays penalized in both modes.

Units: everything is a return fraction (``1e-4 == 1 bp``), matching
``MpoConfig.buy_cost()`` / ``sell_cost()`` and ``TermStructureModel.stage_alphas``.
Pure numpy, float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DELAY_COST_MODES",
    "DelayCostConfig",
    "delay_cost_coefficients",
    "delay_cost_telemetry",
    "effective_trade_costs",
]

#: Supported coefficient modes; see the module docstring / design doc section 2.
DELAY_COST_MODES = ("signed", "exact", "flat")


@dataclass(frozen=True)
class DelayCostConfig:
    """Knobs for the delay-cost surcharge -- deliberately NOT part of ``MpoConfig``.

    ``MpoConfig`` is frozen by the SPEC and shared by every strategy; this layer
    is leg-1 research surface, so it carries its own config and leg 2 wires it
    in as a single optional argument. Passing an ``MpoConfig`` where a
    ``DelayCostConfig`` is expected raises (see :func:`delay_cost_coefficients`).

    Attributes
    ----------
    mode : str
        One of :data:`DELAY_COST_MODES`. ``'signed'`` (default) is the committed
        primary remedy, ``'exact'`` the un-clipped variant, ``'flat'`` the
        fallback of design doc section 2(b).
    scale : float
        Multiplier on the whole coefficient, ``>= 0``. ``0.0`` switches the
        surcharge off (returns exact zeros) without changing any other knob.
        ``0.5`` is the PRE-DECLARED retreat rung of the leg-2 A/B (falsification
        F3): bar-end fills are the pessimistic end of a 30-minute VWAP, so the
        measured 13.2 bp is an upper bound on the true delay.
    cap : float
        Per-name absolute cap on the coefficient, ``> 0``, in return fraction
        (default ``5e-3`` = 50 bp). A hard breaker against one blown-up alpha
        making a name untradeable; how often it fires is reported by
        :func:`delay_cost_telemetry`.
    flat_rate : float or None
        ``'flat'`` mode only, ``>= 0``, in return fraction. ``None`` derives it
        from the cross-section as ``mean(|alpha|)`` -- which is NAME-weighted and
        therefore a LOWER BOUND (~3x too small): the quantity remedy (b) actually
        needs is the TRADE-weighted average delay cost (13.2 bp measured, design
        doc section 2). Leg 2 must pass it explicitly. Setting it in any other
        mode raises (typo guard).
    fill_window_scale : float
        ``fill window minutes / S1 alpha window minutes``, ``> 0``, default 1.0
        (they coincide by construction: both are ``cfg.exec_minutes``). The
        Iteration 41 arms C/D showed the delay tracks the BAR length, not the
        stage length, so when the two are decoupled the coefficient must be
        rescaled. Linear apportionment matches the segment-overlap convention of
        ``term_structure.segment_overlap_weights``. Ignored by an EXPLICIT
        ``flat_rate`` (that is already a measured rate, not an alpha).
    """

    mode: str = "signed"
    scale: float = 1.0
    cap: float = 5e-3
    flat_rate: float | None = None
    fill_window_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in DELAY_COST_MODES:
            raise ValueError(
                f"mode must be one of {DELAY_COST_MODES}, got {self.mode!r}"
            )
        for name in ("scale", "cap", "fill_window_scale"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a real number, got {type(val).__name__}")
            if not np.isfinite(float(val)):
                raise ValueError(f"{name} must be finite, got {val}")
        if float(self.scale) < 0.0:
            raise ValueError(f"scale must be >= 0 (0 = off), got {self.scale}")
        if float(self.cap) <= 0.0:
            raise ValueError(f"cap must be > 0, got {self.cap}")
        if float(self.fill_window_scale) <= 0.0:
            raise ValueError(f"fill_window_scale must be > 0, got {self.fill_window_scale}")
        if self.flat_rate is not None:
            if isinstance(self.flat_rate, bool) or not isinstance(self.flat_rate, (int, float)):
                raise TypeError(
                    f"flat_rate must be a real number or None, got {type(self.flat_rate).__name__}"
                )
            if self.mode != "flat":
                raise ValueError(
                    f"flat_rate is only meaningful for mode='flat', got mode={self.mode!r} "
                    "(pass mode='flat' or drop flat_rate)"
                )
            if not np.isfinite(float(self.flat_rate)):
                raise ValueError(f"flat_rate must be finite, got {self.flat_rate}")
            if float(self.flat_rate) < 0.0:
                raise ValueError(
                    f"flat_rate must be >= 0 (a delay cost is a cost), got {self.flat_rate}"
                )


def _check_alpha(alpha_s1) -> np.ndarray:
    """Validate + copy the stage-1 alpha vector. Loud on every degenerate input."""
    a = np.asarray(alpha_s1, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"alpha_s1 must be 1-D [N], got shape {a.shape}")
    if a.size == 0:
        raise ValueError("alpha_s1 is empty (N == 0): nothing to price")
    bad = ~np.isfinite(a)
    if bad.any():
        idx = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"alpha_s1 must be finite everywhere: {int(bad.sum())} bad entries, "
            f"first at index {idx} (value {a[idx]!r}). Halted names must arrive "
            "as alpha 0.0, not NaN -- see term_structure.stage_alphas."
        )
    return a.copy()  # never alias the caller's buffer


def _check_cfg(cfg) -> DelayCostConfig:
    """Resolve/validate the config, with a loud guard against the MpoConfig swap."""
    if cfg is None:
        return DelayCostConfig()
    if isinstance(cfg, DelayCostConfig):
        return cfg
    if hasattr(cfg, "buy_cost") and hasattr(cfg, "sell_cost"):
        raise TypeError(
            "cfg must be a DelayCostConfig, got what looks like an MpoConfig. "
            "This layer has its OWN config on purpose (MpoConfig is frozen by the "
            "SPEC); pass DelayCostConfig(...) and keep MpoConfig for the base costs."
        )
    raise TypeError(f"cfg must be a DelayCostConfig or None, got {type(cfg).__name__}")


def delay_cost_coefficients(
    alpha_s1,
    cfg: DelayCostConfig | None = None,
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-name delay surcharge to add to the stage-1 buy / sell linear costs.

    Parameters
    ----------
    alpha_s1 : array_like
        ``[N]`` expected return over the S1 / fill window, i.e. ``alpha[0]`` out
        of ``TermStructureModel.stage_alphas`` (return fraction, undecayed --
        stage 0 carries ``delta = 0``). Must be finite; a halted name arrives as
        exactly 0.0 (its z-scores are neutralized upstream), never NaN.
    cfg : DelayCostConfig, optional
        Knobs; defaults to ``DelayCostConfig()`` (mode ``'signed'``, scale 1,
        cap 50 bp). An ``MpoConfig`` here raises :class:`TypeError`.
    mode : str, optional
        Overrides ``cfg.mode`` for one call (handy for A/B arms sharing a cfg).
        Must be in :data:`DELAY_COST_MODES`. An explicit ``cfg.flat_rate`` is
        only honoured when the EFFECTIVE mode is ``'flat'``; overriding away
        from ``'flat'`` raises rather than silently dropping it.

    Returns
    -------
    (c_buy_extra, c_sell_extra) : tuple of np.ndarray
        Two fresh ``[N]`` float64 arrays, return fractions, to be ADDED to
        ``cfg.buy_cost()`` / ``cfg.sell_cost()`` in the ``q`` slots of ``b^1`` /
        ``s^1``. Non-negative in ``'signed'`` / ``'flat'`` mode; possibly
        negative (a rebate) in ``'exact'`` mode. Clipped to ``[0, cap]``
        (``[-cap, cap]`` for ``'exact'``).

    Notes
    -----
    Pure function of the alpha vector and the config: no portfolio state, no
    decision variables, no RNG, no I/O, no mutation of the input. That is what
    keeps the leg-2 QP objective convex (design doc risk 2) and it is asserted
    mechanically in ``tests/mpo/test_delay_cost.py::TestPurity``.

    Examples
    --------
    Buying a name whose predicted 30-minute move is +15 bp costs you those
    15 bp of delay on top of the 7 bp explicit buy cost::

        >>> cb, cs = delay_cost_coefficients(np.array([15e-4, -15e-4, 0.0]))
        >>> (cb * 1e4).round(2)
        array([15.,  0.,  0.])
        >>> (cs * 1e4).round(2)
        array([ 0., 15.,  0.])
    """
    a = _check_alpha(alpha_s1)
    conf = _check_cfg(cfg)
    eff_mode = conf.mode if mode is None else str(mode)
    if eff_mode not in DELAY_COST_MODES:
        raise ValueError(f"mode must be one of {DELAY_COST_MODES}, got {eff_mode!r}")
    if conf.flat_rate is not None and eff_mode != "flat":
        raise ValueError(
            f"cfg.flat_rate={conf.flat_rate!r} is set but the effective mode is "
            f"{eff_mode!r}; refusing to silently ignore it"
        )

    a_eff = float(conf.fill_window_scale) * a

    if eff_mode == "signed":
        c_buy = np.maximum(a_eff, 0.0)
        c_sell = np.maximum(-a_eff, 0.0)
    elif eff_mode == "exact":
        c_buy = a_eff.copy()
        c_sell = -a_eff
    else:  # 'flat'
        if conf.flat_rate is None:
            # NAME-weighted, hence a documented LOWER bound (see DelayCostConfig).
            rate = float(np.mean(np.abs(a_eff)))
        else:
            rate = float(conf.flat_rate)  # already a rate: fill_window_scale N/A
        c_buy = np.full(a.shape, rate, dtype=np.float64)
        c_sell = np.full(a.shape, rate, dtype=np.float64)

    c_buy = c_buy * float(conf.scale)
    c_sell = c_sell * float(conf.scale)

    lo = -float(conf.cap) if eff_mode == "exact" else 0.0
    hi = float(conf.cap)
    return np.clip(c_buy, lo, hi), np.clip(c_sell, lo, hi)


def effective_trade_costs(
    alpha_s1,
    mpo_cfg,
    delay_cfg: DelayCostConfig | None = None,
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(c_buy[N], c_sell[N])`` = base explicit cost + delay surcharge.

    The leg-2 wiring surface: today ``qp.build_mpo_qp`` writes the SCALARS
    ``cfg.buy_cost()`` / ``cfg.sell_cost()`` into the ``b`` / ``s`` blocks of
    ``q``; leg 2 replaces the stage-1 slice with these per-name vectors and
    leaves stages 2..J on the scalars (design doc risk 4).

    Parameters
    ----------
    alpha_s1 : array_like
        ``[N]`` see :func:`delay_cost_coefficients`.
    mpo_cfg : MpoConfig
        Read-only; only ``buy_cost()`` and ``sell_cost()`` are called (duck
        typed so a test double works). A ``DelayCostConfig`` here raises.
    delay_cfg, mode : see :func:`delay_cost_coefficients`.

    Returns
    -------
    (c_buy, c_sell) : tuple of np.ndarray
        ``[N]`` float64 return fractions. With the frozen defaults and a +15 bp
        alpha name: ``c_buy = 7 bp + 15 bp = 22 bp``, ``c_sell = 12 bp``.
        In ``'exact'`` mode an entry CAN go negative; that is legal (the QP stays
        bounded, see the module docstring) and is counted in
        :func:`delay_cost_telemetry` under ``n_negative_effective``.
    """
    if isinstance(mpo_cfg, DelayCostConfig):
        raise TypeError(
            "mpo_cfg must be an MpoConfig (the base costs), got a DelayCostConfig "
            "-- arguments look swapped"
        )
    if not (hasattr(mpo_cfg, "buy_cost") and hasattr(mpo_cfg, "sell_cost")):
        raise TypeError(
            f"mpo_cfg must expose buy_cost()/sell_cost(), got {type(mpo_cfg).__name__}"
        )
    base_buy = float(mpo_cfg.buy_cost())
    base_sell = float(mpo_cfg.sell_cost())
    if not np.isfinite(base_buy) or not np.isfinite(base_sell):
        raise ValueError(
            f"base costs must be finite, got buy={base_buy}, sell={base_sell}"
        )
    extra_buy, extra_sell = delay_cost_coefficients(alpha_s1, delay_cfg, mode=mode)
    return base_buy + extra_buy, base_sell + extra_sell


def delay_cost_telemetry(
    alpha_s1,
    cfg: DelayCostConfig | None = None,
    mode: str | None = None,
    mpo_cfg=None,
) -> dict:
    """Audit trail for one decision epoch's coefficients (leg-2 artifact fodder).

    Falsification F2 of the design doc keys off ``c_buy_extra_mean``: if the
    surcharge lands under 1 bp the wiring is wrong and the A/B is void, so the
    number must reach the artifact rather than staying inside the solver.

    Returns
    -------
    dict
        JSON-safe scalars: the resolved knobs, alpha magnitudes, per-side extra
        mean/max, ``n_capped`` (entries the cap bound), and -- when ``mpo_cfg``
        is given -- effective cost means plus ``n_negative_effective``.
    """
    a = _check_alpha(alpha_s1)
    conf = _check_cfg(cfg)
    eff_mode = conf.mode if mode is None else str(mode)
    extra_buy, extra_sell = delay_cost_coefficients(a, conf, mode=mode)
    cap = float(conf.cap)
    n_capped = int(
        np.count_nonzero(np.isclose(np.abs(extra_buy), cap))
        + np.count_nonzero(np.isclose(np.abs(extra_sell), cap))
    )
    out = {
        "mode": eff_mode,
        "scale": float(conf.scale),
        "cap": cap,
        "fill_window_scale": float(conf.fill_window_scale),
        "flat_rate": None if conf.flat_rate is None else float(conf.flat_rate),
        "n_assets": int(a.size),
        "alpha_abs_mean": float(np.mean(np.abs(a))),
        "alpha_abs_max": float(np.max(np.abs(a))),
        "c_buy_extra_mean": float(np.mean(extra_buy)),
        "c_buy_extra_max": float(np.max(extra_buy)),
        "c_sell_extra_mean": float(np.mean(extra_sell)),
        "c_sell_extra_max": float(np.max(extra_sell)),
        "n_capped": n_capped,
    }
    if mpo_cfg is not None:
        c_buy, c_sell = effective_trade_costs(a, mpo_cfg, conf, mode=mode)
        out["base_buy_cost"] = float(mpo_cfg.buy_cost())
        out["base_sell_cost"] = float(mpo_cfg.sell_cost())
        out["c_buy_mean"] = float(np.mean(c_buy))
        out["c_sell_mean"] = float(np.mean(c_sell))
        out["n_negative_effective"] = int(
            np.count_nonzero(c_buy < 0.0) + np.count_nonzero(c_sell < 0.0)
        )
    return out
