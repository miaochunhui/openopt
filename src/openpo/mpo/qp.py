"""Multi-stage QP assembly for MPO-T1 (docs/mpo-t1-SPEC.md section 3).

Builds the OSQP-form problem

    min  0.5 x' P x + q' x + objective_offset
    s.t. l <= A x <= u

with the decision vector laid out stage-major inside three blocks::

    x = [ w^1 .. w^J | b^1 .. b^J | s^1 .. s^J ]  in R^{3JN}

``w^j`` are stage-end portfolio weights (fraction of NAV), ``b^j >= 0`` /
``s^j >= 0`` the buy / sell split of the stage-``j`` trade. Index helpers
:meth:`MpoQp.slice_w` / :meth:`MpoQp.slice_b` / :meth:`MpoQp.slice_s` and
:meth:`MpoQp.unpack` are the only sanctioned way to address the vector. The
low-rank risk form appends a fourth block of factor-exposure variables after
these three (see "Low-rank risk assembly" below); the three slices above are
unaffected by design.

Objective, production mode (``cfg.cost_mode == 'production'``)::

    sum_j [ -alpha[j] . w^j
            + (gamma/2) * (L_j/240) * w^j' Sigma_d w^j        # stage risk
            + c_buy * 1'b^j + c_sell * 1's^j                  # linear costs
            + (1/2) (b^j + s^j)' D (b^j + s^j) ]              # quadratic impact

with ``c_buy = cfg.buy_cost()``, ``c_sell = cfg.sell_cost()`` and
``D = diag(cfg.impact_coef * sigma_d / adv_frac)`` (the eta scale lives inside
``D`` via ``cfg.impact_coef``). ``Sigma_d`` is used densely by default, which is
exact and cheap while N stays small; see "Low-rank risk assembly" below for the
factor form that makes N ~ 5000 tractable.

Low-rank risk assembly (SPEC section 3, keyword-only ``factor_exposures`` /
``factor_cov`` / ``specific_var`` / ``risk_form``)
--------------------------------------------------------------------------
A dense ``Sigma_d`` puts an [N, N] block per stage into ``P``, so the KKT
factorisation cost grows like ``N^3``: measured on the synthetic risk model, the
dense assembly solves N = 300 in 0.13 s but is unusable by N = 5000: OSQP does
not get past the KKT factorisation inside a 900 s cap (zero iterations, 7.6 GB
resident, 50.1M nonzeros in ``P``), and Clarabel, which does finish it, needs
448 s -- still 7.5x over the 60 s epoch budget. With a
factor risk model ``Sigma_d = X F X' + diag(spec)`` (``X`` = [N, K] exposures,
``F`` = [K, K] factor covariance, ``spec`` = [N] specific variances, K = 9 in the
synthetic market) the same problem is assembled sparsely by adding the per-stage
factor-exposure variables ``y^j = X' w^j`` to the decision vector::

    x = [ w^1 .. w^J | b^1 .. b^J | s^1 .. s^J | y^1 .. y^J ]  in R^{3JN + JK}

with an extra equality row block ``factor_def`` (``J*K`` rows)::

    X' w^j - y^j = 0                                       for every stage j

and the risk term rewritten with no [N, N] block anywhere::

    (gamma/2) * (L_j/240) * ( y^j' F y^j + w^j' diag(spec) w^j )

which is algebraically IDENTICAL to ``(gamma/2)(L_j/240) w^j' Sigma_d w^j``.

Measured on this code (J = 4, K = 9, ``qp_eps = 1e-6``, synthetic risk model of
``tests/mpo/test_qp_lowrank.factor_case``), solve wall per problem::

    N       dense/OSQP   lowrank/OSQP   lowrank/Clarabel
    800       14.2 s        0.53 s          0.18 s
    1000      46.5 s        1.01 s            --
    5000    (capped out)   28.3  s          2.06 s

An independent probe on its own case generator (``/tmp/mpo_scale_probe``, same
J / K / eps) reproduces the picture with its own numbers -- N = 5000: dense/OSQP
capped out, dense/Clarabel 448 s, lowrank/OSQP 37.4 s, lowrank/Clarabel 2.23 s --
so the two levers below are the finding, not these particular seconds.

Two things follow. The reformulation is what makes N ~ 5000 feasible at all, and
at that size the BACKEND matters as much as the assembly: Clarabel's interior
point method eats the sparse equality block (2 s) where OSQP's ADMM grinds
(28 s), so the cluster leg runs with ``cfg.qp_backend = 'clarabel'``. The default
stays ``'osqp'`` -- it is the faster of the two at the sizes the default roster
uses, and this leg changes no defaults.

Parity against the dense assembly on the same risk model: ``max|dw| = 3.5e-9``
and ``8.6e-9`` relative on the objective at N = 80 with ``qp_eps = 1e-9`` (i.e.
the two assemblies agree to solver accuracy, not merely to eyeball accuracy). At
the production ``qp_eps = 1e-6`` the gap is ADMM slack, ~2.5e-5 in ``w`` at
N = 1000 -- which is why the parity test tightens ``qp_eps`` instead of loosening
its gate.

The ``y`` blocks are appended AFTER ``[w, b, s]`` on purpose: every existing
column slice (:meth:`MpoQp.slice_w` / ``slice_b`` / ``slice_s``), every
constraint block and every caller of :meth:`MpoQp.unpack` is unchanged, and the
extra variables stay an internal implementation detail (``unpack`` still returns
``(w, b, s)`` only; ``y`` is reachable through :meth:`MpoQp.unpack_factors` /
:meth:`MpoQp.slice_y` for diagnostics).

``risk_form`` selects the assembly:

- ``'auto'`` (default): low-rank iff the factor triplet was supplied AND
  ``N > LOWRANK_AUTO_MIN_ASSETS`` (500), else dense. The threshold keeps every
  existing N <= 300 caller -- the whole benchmark roster -- bit-for-bit on the
  dense path, and lets the cluster-scale runs pick the sparse form with no
  config change.
- ``'lowrank'``: always low-rank; requires the factor triplet.
- ``'dense'``: always dense, even when a factor triplet is given.

Calling ``build_mpo_qp`` without any of the four new keywords reproduces the
pre-existing assembly byte for byte (pinned by
``tests/mpo/test_qp_lowrank.py::test_default_call_is_byte_identical``).

Two deliberate restrictions:

- ``cost_mode='axioma'`` is DENSE ONLY. That mode's transition penalty couples
  neighbouring stages through ``Sigma_d`` (``-delta * Sigma_d`` off-diagonal
  blocks), which the ``y^j = X'w^j`` substitution would have to expand into
  cross-stage factor terms; the mode exists purely to reproduce the closed form
  of ``openpo.mpo.closed_form`` at N = 10, where dense is free. Asking for
  ``risk_form='lowrank'`` there raises instead of silently changing the parity
  gate's problem; ``'auto'`` falls back to dense.
- ``Sigma_d`` may be ``None`` -- and only ``None`` -- when the factor triplet is
  supplied and the low-rank form is actually selected, so a cluster-scale caller
  never has to materialise the [N, N] matrix it would not use.

The stage-1 linear costs accept an optional PER-NAME surcharge through the
keyword-only ``c_buy_extra`` / ``c_sell_extra`` arguments (both ``[N]``)::

    ... + (c_buy + c_buy_extra)' b^1 + (c_sell + c_sell_extra)' s^1     # j == 1
    ... + c_buy * 1'b^j + c_sell * 1's^j                                # j >= 2

That is the wiring surface of the implementation-delay surcharge
(``openpo.mpo.delay_cost``, design ``docs/mpo-t1-delaycost-design.md`` section 1):
only stage 1 is charged because only the stage-1 trade is actually EXECUTED --
later stages' fills are hypothetical replans, so pricing their delay would
double-count (design risk 4 accepts the resulting mild "later is cheaper" tilt,
which points the same way as the correction itself). Both default to ``None``,
which reproduces the scalar-cost objective bit for bit.

Objective, Axioma parity mode (``cfg.cost_mode == 'axioma'``)::

    sum_j [ -alpha[j] . w^j
            + (gamma/2) * w^j' Sigma_d w^j                    # UNSCALED risk
            + (delta/2) * dw^j' Sigma_d dw^j ]                # dw^j = w^j - w^{j-1}

i.e. *all* b/s cost terms (linear + impact) are replaced by the quadratic
transition penalty with ``delta = cfg.axioma_delta``, and the stage risk term
drops the ``L_j/240`` time scaling. Both deviations are deliberate: they are
exactly the assumptions behind Axioma 051 eq29-30, so an unconstrained J=2
instance of this mode reproduces :func:`openpo.mpo.closed_form.
closed_form_two_stage` to solver accuracy. In this mode ``b^j``/``s^j`` keep
only their dynamics + non-negativity roles (they carry no cost), so the
optimal split is degenerate; a tiny ridge (``bs_reg``, see below) is added to
pin it down and keep OSQP's polish step non-singular.

Constraints (row blocks in this order, see ``MpoQp.blocks``):

1. ``dynamics`` (J*N rows): ``w^j - w^{j-1} - b^j + s^j = 0`` with ``w^0 = w0``
   folded into the bound of the first block.
2. ``budget`` (J rows): ``budget_lo <= 1'w^j <= budget_hi`` (default 0..1, the
   slack is uninvested cash which earns nothing).
3. ``w_box`` (J*N rows): ``w_min <= w^j <= w_max`` (default 0..cfg.w_max).
   The upper bound is raised, per stage and name, to the tightest weight that
   is legally *reachable* (``w0 - max_cumulative_sell``) whenever T+1 or a
   stage-1 limit-down makes ``w_max`` unattainable -- see the "w_box
   feasibility repair" comment in :func:`build_mpo_qp`. Without it a holding
   that merely drifted above ``w_max`` (price move, no trade) while unsellable
   makes the box row contradict the dynamics + T+1 rows and the QP is
   genuinely primal infeasible. ``meta['w_box_relaxed']`` /
   ``meta['w_box_relax_max']`` report how often and by how much this fired.
4. ``b_box`` (J*N rows): ``0 <= b^j <= b_max``; a stage-1 ``limit_up`` name has
   its upper bound forced to 0 (cannot buy at the upper price limit).
5. ``s_box`` (J*N rows): ``0 <= s^j <= s_max``; a stage-1 ``limit_down`` name
   has its upper bound forced to 0.
6. ``t1_today`` (N rows, only when ``cfg.t1_enabled`` and the grid has today
   stages): ``sum_{j in grid.today_stage_indices} s^j <= sellable_w``.
7. ``t1_carry_j`` (N rows per non-today stage j, only when ``cfg.t1_enabled``):
   ``s^j <= w^{j-1}`` -- shares bought on a previous day are sellable, today's
   purchases are not. For the first stage (``j-1`` = current holdings) the
   right-hand side is the constant ``w0``.

8. ``factor_def`` (J*K rows, low-rank risk form only): ``X' w^j - y^j = 0``,
   the equality that defines the factor-exposure variables (see "Low-rank risk
   assembly" below). Absent on the dense path.

``cfg.t1_enabled = False`` drops blocks 6-7 (the T+1 ablation axis).

Optional keyword overrides (``build_mpo_qp(..., cfg, **overrides)``), each a
scalar or a broadcastable ``[N]`` / ``[J, N]`` array unless noted:

- ``w_min`` (default 0.0), ``w_max`` (default ``cfg.w_max``)
- ``b_max`` / ``s_max`` (default ``+inf``; limit masks still clamp stage 1)
- ``budget_lo`` (default 0.0), ``budget_hi`` (default 1.0) -- scalar or ``[J]``
- ``bs_reg``: ridge coefficient on ``b``/``s`` (default 0 in production mode,
  ``1e-9 * gamma * mean(diag(Sigma_d))`` in axioma mode). Scaled to the risk
  term so its bias on ``w`` stays far below the 1e-6 parity gate.
- ``adv_frac_floor`` (default 1e-8): floor applied to ``adv_frac`` before
  forming ``D``, so a zero-ADV name becomes prohibitively expensive to trade
  instead of dividing by zero.

Unknown override keys raise ``ValueError`` (typo guard). ``c_buy_extra`` /
``c_sell_extra`` are real keyword arguments, not overrides, so a misspelling of
either lands in ``**overrides`` and trips that same guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from openpo.mpo.config import MINUTES_PER_DAY, MpoConfig
from openpo.mpo.types import StageGrid

__all__ = ["MpoQp", "build_mpo_qp", "LOWRANK_AUTO_MIN_ASSETS", "RISK_FORMS"]

# ``risk_form='auto'`` switches to the low-rank factor assembly above this many
# names. Measured crossover (see the module docstring): dense is fine and exact
# up to a few hundred names, and is the only form the N <= 300 benchmark roster
# ever sees, so the threshold sits comfortably above it.
LOWRANK_AUTO_MIN_ASSETS = 500

RISK_FORMS = ("auto", "dense", "lowrank")

_OVERRIDE_KEYS = frozenset(
    {
        "w_min",
        "w_max",
        "b_max",
        "s_max",
        "budget_lo",
        "budget_hi",
        "bs_reg",
        "adv_frac_floor",
    }
)


@dataclass
class MpoQp:
    """OSQP-form data for the multi-stage problem plus unpacking metadata."""

    P: sp.csc_matrix  # [n_vars, n_vars] upper-triangular (OSQP convention)
    q: np.ndarray  # [n_vars]
    A: sp.csc_matrix  # [m, n_vars]
    l: np.ndarray  # [m]
    u: np.ndarray  # [m]
    n_stages: int  # J
    n_assets: int  # N
    n_factors: int = 0  # K, low-rank risk form only (0 == dense Sigma_d)
    objective_offset: float = 0.0  # constant term dropped from P/q
    blocks: dict = field(default_factory=dict)  # row-block name -> (start, stop)
    meta: dict = field(default_factory=dict)

    # -- variable indexing --------------------------------------------------

    @property
    def n_vars(self) -> int:
        """3JN, plus the JK internal factor-exposure variables when low-rank."""
        return 3 * self.n_stages * self.n_assets + self.n_stages * self.n_factors

    def slice_w(self, j: int) -> slice:
        n = self.n_assets
        return slice(j * n, (j + 1) * n)

    def slice_b(self, j: int) -> slice:
        n, jn = self.n_assets, self.n_stages * self.n_assets
        return slice(jn + j * n, jn + (j + 1) * n)

    def slice_s(self, j: int) -> slice:
        n, jn = self.n_assets, self.n_stages * self.n_assets
        return slice(2 * jn + j * n, 2 * jn + (j + 1) * n)

    def slice_y(self, j: int) -> slice:
        """Columns of the stage-``j`` factor exposures ``y^j = X' w^j`` [K].

        Only defined for the low-rank assembly; the dense form has no ``y``.
        """
        if self.n_factors <= 0:
            raise ValueError(
                "this MpoQp uses the dense risk form and carries no y variables "
                "(build with factor_exposures/factor_cov/specific_var to get them)"
            )
        k, base = self.n_factors, 3 * self.n_stages * self.n_assets
        return slice(base + j * k, base + (j + 1) * k)

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a primal solution into ``(weights, buys, sells)``, each [J, N].

        The low-rank form's trailing ``y`` block is a pure implementation detail
        of the risk term, so it is dropped here (use :meth:`unpack_factors` for
        it) and every caller keeps the same three-array contract.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.n_vars:
            raise ValueError(f"x must have {self.n_vars} entries, got {x.size}")
        j, n = self.n_stages, self.n_assets
        w = x[: j * n].reshape(j, n).copy()
        b = x[j * n : 2 * j * n].reshape(j, n).copy()
        s = x[2 * j * n : 3 * j * n].reshape(j, n).copy()
        return w, b, s

    def unpack_factors(self, x: np.ndarray) -> np.ndarray:
        """The [J, K] solved factor exposures ``y^j`` (low-rank form only).

        Diagnostics surface: ``y`` should equal ``w @ X`` to solver accuracy, so
        comparing the two is a direct check that the ``factor_def`` rows held.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.n_vars:
            raise ValueError(f"x must have {self.n_vars} entries, got {x.size}")
        if self.n_factors <= 0:
            raise ValueError(
                "this MpoQp uses the dense risk form and carries no y variables"
            )
        base = 3 * self.n_stages * self.n_assets
        return x[base:].reshape(self.n_stages, self.n_factors).copy()

    def objective_value(self, x: np.ndarray) -> float:
        """0.5 x'Px + q'x + offset (P is upper-triangular, so mirror it)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        px = self.P @ x
        # upper triangular storage: x'Px_full = 2*x'P_triu*x - sum(diag)*x^2
        diag = self.P.diagonal()
        quad = 2.0 * float(x @ px) - float(np.sum(diag * x * x))
        return 0.5 * quad + float(self.q @ x) + float(self.objective_offset)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_vector(v, n: int, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).ravel()
    if arr.size != n:
        raise ValueError(f"{name} must have {n} entries, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _broadcast_stage_asset(v, n_stages: int, n_assets: int, name: str) -> np.ndarray:
    """Broadcast a scalar / [N] / [J,N] override to a [J, N] float64 array."""
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full((n_stages, n_assets), float(arr))
    elif arr.ndim == 1 and arr.size == n_assets:
        out = np.tile(arr, (n_stages, 1))
    elif arr.shape == (n_stages, n_assets):
        out = arr.astype(np.float64, copy=True)
    else:
        raise ValueError(
            f"{name} must be scalar, [{n_assets}] or [{n_stages},{n_assets}], "
            f"got shape {arr.shape}"
        )
    if np.any(np.isnan(out)):
        raise ValueError(f"{name} contains NaN")
    return out


def _check_cost_extra(v, n_assets: int, name: str, allow_negative: bool) -> np.ndarray:
    """Validate a stage-1 per-name linear-cost surcharge; returns a fresh [N] copy.

    Deliberately stricter than the ``overrides`` broadcasters: no scalar and no
    ``[J, N]`` form is accepted, because "which stage does this apply to" is the
    whole subtlety of the surcharge (stage 1 only -- see the module docstring)
    and a silently broadcast scalar would hide a caller mistake.
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1 or arr.size != n_assets:
        raise ValueError(
            f"{name} must be a 1-D [{n_assets}] vector (stage-1 only), "
            f"got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        bad = int(np.flatnonzero(~np.isfinite(arr))[0])
        raise ValueError(
            f"{name} must be finite everywhere, first bad entry at index {bad}"
        )
    if not allow_negative and np.any(arr < 0.0):
        bad = int(np.flatnonzero(arr < 0.0)[0])
        raise ValueError(
            f"{name} must be non-negative (a delay cost is a cost): index {bad} "
            f"is {float(arr[bad])!r}. Pass allow_negative_extra=True for the "
            "delay_cost mode='exact' rebate variant, which is bounded because the "
            "two extras sum to zero (design doc risk 2)."
        )
    return arr.astype(np.float64, copy=True)


def _extra_meta(c_buy_extra, c_sell_extra) -> dict | None:
    """JSON-safe summary of the stage-1 surcharge, or None when there is none."""
    if c_buy_extra is None and c_sell_extra is None:
        return None
    out: dict[str, float] = {}
    for side, arr in (("buy", c_buy_extra), ("sell", c_sell_extra)):
        if arr is None:
            out[f"{side}_mean"] = 0.0
            out[f"{side}_max"] = 0.0
        else:
            out[f"{side}_mean"] = float(np.mean(arr))
            out[f"{side}_max"] = float(np.max(arr))
    return out


def _broadcast_stage(v, n_stages: int, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_stages, float(arr))
    if arr.ndim == 1 and arr.size == n_stages:
        return arr.astype(np.float64, copy=True)
    raise ValueError(f"{name} must be scalar or [{n_stages}], got shape {arr.shape}")


def _factor_model(
    factor_exposures, factor_cov, specific_var, n_assets: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Validate the ``(X, F, spec)`` triplet; ``None`` when none was supplied.

    All three or nothing: a partially supplied risk model is a caller bug that
    would otherwise silently fall back to the dense path at cluster scale (i.e.
    "the run never finishes" instead of "you forgot factor_cov").
    """
    given = [
        name
        for name, val in (
            ("factor_exposures", factor_exposures),
            ("factor_cov", factor_cov),
            ("specific_var", specific_var),
        )
        if val is not None
    ]
    if not given:
        return None
    if len(given) != 3:
        raise ValueError(
            "the low-rank risk model needs all of factor_exposures, factor_cov "
            f"and specific_var; got only {given}"
        )

    x = np.asarray(factor_exposures, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != n_assets:
        raise ValueError(
            f"factor_exposures must be [{n_assets},K], got shape {x.shape}"
        )
    n_factors = int(x.shape[1])
    if n_factors < 1:
        raise ValueError("factor_exposures must have at least one factor column")
    if not np.all(np.isfinite(x)):
        raise ValueError("factor_exposures contains non-finite values")

    f = np.asarray(factor_cov, dtype=np.float64)
    if f.shape != (n_factors, n_factors):
        raise ValueError(
            f"factor_cov must be [{n_factors},{n_factors}] to match "
            f"factor_exposures, got shape {f.shape}"
        )
    if not np.all(np.isfinite(f)):
        raise ValueError("factor_cov contains non-finite values")
    f = 0.5 * (f + f.T)  # same symmetrisation the dense path applies to Sigma_d

    spec = _as_vector(specific_var, n_assets, "specific_var")
    if np.any(spec < 0.0):
        bad = int(np.flatnonzero(spec < 0.0)[0])
        raise ValueError(
            f"specific_var must be non-negative, index {bad} is {float(spec[bad])!r}"
        )
    return x.astype(np.float64, copy=False), f, spec


def _resolve_risk_form(
    risk_form, has_factors: bool, n_assets: int, cost_mode: str
) -> str:
    """Map the requested ``risk_form`` onto ``'dense'`` / ``'lowrank'``.

    See the module docstring for the 'auto' threshold and the axioma-mode
    restriction; both are decided here so ``meta['risk_form']`` always reports
    the assembly that was actually built.
    """
    form = str(risk_form).lower()
    if form not in RISK_FORMS:
        raise ValueError(f"risk_form must be one of {RISK_FORMS}, got {risk_form!r}")

    if form == "lowrank":
        if not has_factors:
            raise ValueError(
                "risk_form='lowrank' needs factor_exposures/factor_cov/"
                "specific_var; pass them or leave risk_form at 'auto'"
            )
        if cost_mode == "axioma":
            raise ValueError(
                "risk_form='lowrank' is not available in cost_mode='axioma': that "
                "mode's transition penalty couples neighbouring stages through "
                "Sigma_d, and it exists only to reproduce closed_form_two_stage at "
                "N=10 where dense costs nothing. Use risk_form='dense' (or 'auto', "
                "which falls back to dense here) for the parity path."
            )
        return "lowrank"
    if form == "dense":
        return "dense"
    # 'auto': the factor form is a pure scale optimisation, so only take it when
    # it actually buys something and the objective is the production one.
    if has_factors and cost_mode == "production" and n_assets > LOWRANK_AUTO_MIN_ASSETS:
        return "lowrank"
    return "dense"


def build_mpo_qp(
    alpha,
    grid: StageGrid,
    w0,
    sellable_w,
    Sigma_d,
    adv_frac,
    sigma_d,
    limit_up,
    limit_down,
    cfg: MpoConfig,
    *,
    c_buy_extra=None,
    c_sell_extra=None,
    allow_negative_extra: bool = False,
    factor_exposures=None,
    factor_cov=None,
    specific_var=None,
    risk_form: str = "auto",
    **overrides,
) -> MpoQp:
    """Assemble the multi-stage MPO QP.

    Positional order is the caller contract used by
    ``openpo.mpo.strategies._QpStrategy._solve_qp``; see the module docstring
    for the objective, the constraint blocks and the accepted ``overrides``.

    Parameters
    ----------
    alpha : array [J, N]
        Per-stage expected returns (already decayed / aggregated upstream).
    grid : StageGrid
        Stage grid; ``J = grid.n_stages``, stage lengths ``L_j`` in trading
        minutes, ``grid.today_stage_indices`` drives the T+1 rows.
    w0 : array [N]
        Current portfolio weights (``w^0``, a constant of the problem).
    sellable_w : array [N]
        T+1 sellable inventory right now, in weight units.
    Sigma_d : array [N, N] or None
        Daily covariance matrix (symmetrised internally). May be ``None`` only
        when the factor triplet below is supplied AND the low-rank form is
        selected, so a cluster-scale caller can skip the [N, N] allocation.
    adv_frac : array [N]
        ``ADV_i / NAV``, used for the quadratic impact diagonal.
    sigma_d : array [N]
        Daily volatilities, used for the quadratic impact diagonal.
    limit_up, limit_down : bool arrays [N]
        Price-limit masks; applied to stage-1 buys / sells only.
    cfg : MpoConfig
        Frozen config (gamma, w_max, impact_coef, costs, cost_mode,
        axioma_delta, t1_enabled).
    c_buy_extra, c_sell_extra : array [N], optional (keyword-only)
        Per-name surcharge added to the STAGE-1 buy / sell linear costs, in
        return fraction (``1e-4 == 1 bp``) -- typically the output of
        :func:`openpo.mpo.delay_cost.delay_cost_coefficients`. ``None`` (both
        defaults) leaves ``q`` bit-identical to the scalar-cost assembly.
        Stage 1 only, on purpose: see the module docstring. Rejected in
        ``cost_mode='axioma'``, which carries no linear cost terms at all --
        silently ignoring them there would make the delay A/B look wired when it
        is not (design doc falsification F2 is exactly that failure mode).
    allow_negative_extra : bool, default False
        Permit negative surcharge entries, i.e. a delay REBATE. Needed only by
        ``delay_cost`` ``mode='exact'`` (``+alpha_S1`` on buys, ``-alpha_S1`` on
        sells). Legal because the QP stays bounded: the only unbounded direction
        of a linear-cost QP is the wash pair (``b_i, s_i -> inf`` at fixed
        ``w``), whose cost is ``(c_buy + extra_buy) + (c_sell + extra_sell)``,
        and in ``'exact'`` mode the two extras sum to EXACTLY zero, leaving
        ``c_buy + c_sell > 0`` (design doc risk 2 / ``delay_cost`` module
        docstring). In ``'signed'`` / ``'flat'`` mode both extras are already
        ``>= 0``, so the default guard is the right one.
    factor_exposures, factor_cov, specific_var : arrays, optional (keyword-only)
        Factor risk model ``Sigma_d = X F X' + diag(spec)``: ``X`` [N, K],
        ``F`` [K, K] (symmetrised internally), ``spec`` [N] ``>= 0``. All three
        or none. Supplying them makes the low-rank assembly *available*;
        ``risk_form`` decides whether it is used.
    risk_form : {'auto', 'dense', 'lowrank'}, default 'auto'
        ``'auto'`` picks low-rank only with a factor triplet, production cost
        mode and ``N > LOWRANK_AUTO_MIN_ASSETS`` (500); ``'lowrank'`` forces it
        (and raises without a triplet, or in ``cost_mode='axioma'``);
        ``'dense'`` forces the dense ``Sigma_d`` assembly even with a triplet.
        The selected form is reported in ``meta['risk_form']``. See the module
        docstring for the measured motivation.

    Returns
    -------
    MpoQp
        Ready for :func:`openpo.mpo.solve.solve_mpo_qp`.
    """
    bad = set(overrides) - _OVERRIDE_KEYS
    if bad:
        raise ValueError(
            f"unknown build_mpo_qp overrides {sorted(bad)}; "
            f"supported: {sorted(_OVERRIDE_KEYS)}"
        )

    if not isinstance(grid, StageGrid):
        raise TypeError(f"grid must be a StageGrid, got {type(grid).__name__}")
    n_stages = grid.n_stages
    if n_stages < 1:
        raise ValueError("grid must contain at least one stage")

    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 2 or alpha.shape[0] != n_stages:
        raise ValueError(
            f"alpha must be [{n_stages},N], got shape {alpha.shape}"
        )
    n_assets = int(alpha.shape[1])
    if not np.all(np.isfinite(alpha)):
        raise ValueError("alpha contains non-finite values")

    # ``Sigma_d=None`` is legal only on the low-rank path (checked below, once
    # the risk form is resolved); everything else validates exactly as before.
    sigma_full = None
    if Sigma_d is not None:
        sigma_full = np.asarray(Sigma_d, dtype=np.float64)
        if sigma_full.shape != (n_assets, n_assets):
            raise ValueError(
                f"Sigma_d must be [{n_assets},{n_assets}], got {sigma_full.shape}"
            )
        if not np.all(np.isfinite(sigma_full)):
            raise ValueError("Sigma_d contains non-finite values")
        sigma_full = 0.5 * (sigma_full + sigma_full.T)

    w0 = _as_vector(w0, n_assets, "w0")
    sellable_w = _as_vector(sellable_w, n_assets, "sellable_w")
    adv_frac = _as_vector(adv_frac, n_assets, "adv_frac")
    sigma_vol = _as_vector(sigma_d, n_assets, "sigma_d")
    limit_up = np.asarray(limit_up, dtype=bool).ravel()
    limit_down = np.asarray(limit_down, dtype=bool).ravel()
    if limit_up.size != n_assets or limit_down.size != n_assets:
        raise ValueError(
            f"limit_up/limit_down must have {n_assets} entries, got "
            f"{limit_up.size}/{limit_down.size}"
        )
    if np.any(sigma_vol < 0):
        raise ValueError("sigma_d must be non-negative")
    if np.any(adv_frac < 0):
        raise ValueError("adv_frac must be non-negative")

    cost_mode = str(cfg.cost_mode)
    if cost_mode not in ("production", "axioma"):
        raise ValueError(
            f"cfg.cost_mode must be 'production' or 'axioma', got {cost_mode!r}"
        )
    # --- stage-1 per-name linear-cost surcharge (delay cost) ---------------
    if cost_mode == "axioma" and (c_buy_extra is not None or c_sell_extra is not None):
        raise ValueError(
            "c_buy_extra/c_sell_extra require cost_mode='production': the axioma "
            "parity mode replaces every linear b/s cost with the quadratic "
            "transition penalty, so a per-unit-traded surcharge has nowhere to "
            "land. Refusing to accept coefficients that would be silently "
            "dropped (that is exactly design-doc falsification F2)."
        )
    if c_buy_extra is not None:
        c_buy_extra = _check_cost_extra(
            c_buy_extra, n_assets, "c_buy_extra", allow_negative_extra
        )
    if c_sell_extra is not None:
        c_sell_extra = _check_cost_extra(
            c_sell_extra, n_assets, "c_sell_extra", allow_negative_extra
        )

    # --- risk assembly: dense Sigma_d vs low-rank factor expansion ---------
    factors = _factor_model(factor_exposures, factor_cov, specific_var, n_assets)
    form = _resolve_risk_form(risk_form, factors is not None, n_assets, cost_mode)
    lowrank = form == "lowrank"
    if not lowrank:
        if sigma_full is None:
            raise ValueError(
                f"Sigma_d is required for the dense risk form (risk_form={form!r}); "
                "it may only be None when the factor triplet is given AND the "
                "low-rank form is selected"
            )
        factor_x = factor_f = factor_spec = None
        n_factors = 0
    else:
        factor_x, factor_f, factor_spec = factors
        n_factors = int(factor_x.shape[1])

    gamma = float(cfg.gamma)
    delta = float(cfg.axioma_delta)
    lengths = np.array([float(s.length) for s in grid.stages], dtype=np.float64)
    if np.any(lengths <= 0):
        raise ValueError(f"all stages must have positive length, got {lengths}")

    # --- overrides ---------------------------------------------------------
    w_min = _broadcast_stage_asset(
        overrides.get("w_min", 0.0), n_stages, n_assets, "w_min"
    )
    w_max = _broadcast_stage_asset(
        overrides.get("w_max", cfg.w_max), n_stages, n_assets, "w_max"
    )
    b_max = _broadcast_stage_asset(
        overrides.get("b_max", np.inf), n_stages, n_assets, "b_max"
    )
    s_max = _broadcast_stage_asset(
        overrides.get("s_max", np.inf), n_stages, n_assets, "s_max"
    )
    budget_lo = _broadcast_stage(
        overrides.get("budget_lo", 0.0), n_stages, "budget_lo"
    )
    budget_hi = _broadcast_stage(
        overrides.get("budget_hi", 1.0), n_stages, "budget_hi"
    )
    adv_floor = float(overrides.get("adv_frac_floor", 1e-8))
    if adv_floor <= 0:
        raise ValueError("adv_frac_floor must be positive")

    # Only the axioma default ridge reads it, and axioma is dense-only, so this
    # never forces an [N, N] touch on the low-rank path.
    mean_var = 0.0 if sigma_full is None else float(np.mean(np.diag(sigma_full)))
    default_bs_reg = 0.0
    if cost_mode == "axioma":
        # Pins the (cost-free) b/s split and keeps OSQP polish non-singular.
        default_bs_reg = 1e-9 * max(gamma, 1e-12) * max(mean_var, 1e-12)
    bs_reg = float(overrides.get("bs_reg", default_bs_reg))
    if bs_reg < 0:
        raise ValueError("bs_reg must be non-negative")

    # Stage-1 price limits: cannot buy at the upper limit / sell at the lower.
    b_max[0] = np.where(limit_up, 0.0, b_max[0])
    s_max[0] = np.where(limit_down, 0.0, s_max[0])

    today = tuple(int(j) for j in grid.today_stage_indices)
    if any(j < 0 or j >= n_stages for j in today):
        raise ValueError(
            f"grid.today_stage_indices {today} out of range for J={n_stages}"
        )

    # --- w_box feasibility repair (T+1 / limit-down interaction) -----------
    # An existing holding can drift above ``w_max`` without any trade (the
    # weight is price-driven), and T+1 can make it impossible to trim: shares
    # bought today are not sellable, and a limit-down name cannot be sold at
    # all in stage 1. Demanding ``w^j <= w_max`` then CONTRADICTS the dynamics
    # + T+1 rows and the whole QP becomes primal infeasible -- OSQP returns
    # 'primal infeasible', solve_mpo_qp fail-softs and the strategy stops
    # trading for the rest of the day (measured: a single unsellable overweight
    # of 1e-5 in weight units killed 87% of the J=1 baselines' epochs).
    # Repair: raise the box upper bound to the TIGHTEST weight that is legally
    # reachable in each stage, i.e. ``w0 - max_cumulative_sell_j``. The limit
    # still bans every *avoidable* overweight (nothing is relaxed when the
    # position can be trimmed); an unavoidable one is tolerated instead of
    # destroying the problem, and the objective (risk + alpha) still drives the
    # trim as soon as the shares become sellable. ``meta['w_box_relaxed']``
    # counts the touched (stage, name) entries for first-scene visibility.
    cum_sell = np.cumsum(s_max, axis=0)  # [J, N] max cumulative sell thru stage j
    today_cap = sellable_w if cfg.t1_enabled else np.full(n_assets, np.inf)
    reachable_lo = np.empty_like(w_max)
    for j in range(n_stages):
        if j in today:
            # today's stages share the single ``t1_today`` sell budget
            avail = np.minimum(cum_sell[j], today_cap)
        else:
            # ``t1_carry`` is s^j <= w^{j-1}: a past-day holding can be sold in
            # full, so w^j = 0 stays reachable unless s_max blocks it.
            avail = cum_sell[j]
        reachable_lo[j] = w0 - avail
    w_box_relaxed = int(np.count_nonzero(reachable_lo > w_max))
    w_box_relax_max = float(np.max(reachable_lo - w_max)) if w_box_relaxed else 0.0
    w_max = np.maximum(w_max, reachable_lo)

    jn = n_stages * n_assets
    # jk == 0 on the dense path, so every "3jn + jk" below degenerates to the
    # pre-existing layout and the assembled data stays byte-identical.
    jk = n_stages * n_factors
    n_vars = 3 * jn + jk
    sigma_sp = None if sigma_full is None else sp.csc_matrix(sigma_full)
    eye_n = sp.identity(n_assets, format="csc", dtype=np.float64)

    # --- P ----------------------------------------------------------------
    if lowrank:
        # w block: specific variance only (diagonal); the factor part of the risk
        # rides on the y block below, which is [K, K] per stage instead of [N, N].
        risk_coef = gamma * (lengths / MINUTES_PER_DAY)  # [J], production mode
        p_w = sp.diags(
            np.concatenate([risk_coef[j] * factor_spec for j in range(n_stages)]),
            format="csc",
        )
    else:
        w_blocks: list[list] = [[None] * n_stages for _ in range(n_stages)]
        for j in range(n_stages):
            if cost_mode == "production":
                coef = gamma * (lengths[j] / MINUTES_PER_DAY)
            else:
                # unscaled risk + transition curvature from terms j and j+1
                coef = gamma + delta * (2.0 if j < n_stages - 1 else 1.0)
            w_blocks[j][j] = coef * sigma_sp
            if cost_mode == "axioma" and j > 0:
                w_blocks[j][j - 1] = (-delta) * sigma_sp
                w_blocks[j - 1][j] = (-delta) * sigma_sp
        p_w = sp.bmat(w_blocks, format="csc")

    bs_blocks: list[list] = [[None] * (2 * n_stages) for _ in range(2 * n_stages)]
    for j in range(n_stages):
        if cost_mode == "production":
            impact_diag = (
                cfg.impact_coef * sigma_vol / np.maximum(adv_frac, adv_floor)
            )
            d_j = sp.diags(impact_diag + bs_reg, format="csc")
            cross = sp.diags(impact_diag, format="csc")
        else:
            d_j = bs_reg * eye_n
            cross = None
        bs_blocks[j][j] = d_j  # b^j
        bs_blocks[n_stages + j][n_stages + j] = d_j  # s^j
        if cross is not None:
            bs_blocks[j][n_stages + j] = cross
            bs_blocks[n_stages + j][j] = cross
    p_bs = sp.bmat(bs_blocks, format="csc")

    if lowrank:
        # y block: coef_j * F, one small dense [K, K] per stage.
        p_y = sp.block_diag(
            [sp.csc_matrix(risk_coef[j] * factor_f) for j in range(n_stages)],
            format="csc",
        )
        P = sp.block_diag((p_w, p_bs, p_y), format="csc")
    else:
        P = sp.block_diag((p_w, p_bs), format="csc")
    P = sp.triu(P, format="csc")
    P.eliminate_zeros()
    P.sort_indices()

    # --- q ----------------------------------------------------------------
    # y carries no linear term, so its trailing jk entries stay zero.
    q = np.zeros(n_vars, dtype=np.float64)
    q[:jn] = -alpha.reshape(-1)
    objective_offset = 0.0
    if cost_mode == "production":
        q[jn : 2 * jn] += float(cfg.buy_cost())
        q[2 * jn : 3 * jn] += float(cfg.sell_cost())
        # Stage-1 slice only: b^1 starts at jn, s^1 at 2jn (see MpoQp.slice_b/_s).
        if c_buy_extra is not None:
            q[jn : jn + n_assets] += c_buy_extra
        if c_sell_extra is not None:
            q[2 * jn : 2 * jn + n_assets] += c_sell_extra
    else:
        # (delta/2)(w^1 - w0)'Sigma(w^1 - w0): cross term into q, constant out.
        sigma_w0 = sigma_full @ w0
        q[:n_assets] += -delta * sigma_w0
        objective_offset = 0.5 * delta * float(w0 @ sigma_w0)

    # --- A, l, u ----------------------------------------------------------
    rows: list[sp.spmatrix] = []
    lows: list[np.ndarray] = []
    ups: list[np.ndarray] = []
    blocks: dict[str, tuple[int, int]] = {}
    cursor = 0

    def add(block: sp.spmatrix, lo: np.ndarray, hi: np.ndarray, name: str) -> None:
        nonlocal cursor
        block = sp.csr_matrix(block)
        if jk and block.shape[1] == 3 * jn:
            # Low-rank form: every pre-existing block is zero on the y columns,
            # so pad it here instead of touching each block's construction.
            block = sp.hstack(
                [block, sp.csr_matrix((block.shape[0], jk))], format="csr"
            )
        rows.append(block)
        lows.append(np.asarray(lo, dtype=np.float64).ravel())
        ups.append(np.asarray(hi, dtype=np.float64).ravel())
        blocks[name] = (cursor, cursor + block.shape[0])
        cursor += block.shape[0]

    def row_block(n_rows: int, parts: dict) -> sp.csr_matrix:
        """Assemble an [n_rows, 3JN] block from {(part, stage): matrix} entries.

        ``part`` is 'w' / 'b' / 's'; each matrix must be [n_rows, N].
        """
        cols = {"w": 0, "b": jn, "s": 2 * jn}
        pieces = [[sp.csr_matrix((n_rows, n_assets))] * (3 * n_stages)]
        for (part, j), mat in parts.items():
            idx = (cols[part] + j * n_assets) // n_assets
            pieces[0][idx] = sp.csr_matrix(mat)
        return sp.bmat(pieces, format="csr")

    # 1. dynamics: w^j - w^{j-1} - b^j + s^j = 0 (w^0 = w0 in the bound)
    eye_jn = sp.identity(jn, format="csc", dtype=np.float64)
    lag_w = sp.eye(jn, jn, k=-n_assets, format="csc", dtype=np.float64)
    dyn = sp.hstack([eye_jn - lag_w, -eye_jn, eye_jn], format="csr")
    rhs = np.zeros(jn, dtype=np.float64)
    rhs[:n_assets] = w0
    add(dyn, rhs, rhs, "dynamics")

    # 2. budget: budget_lo <= 1'w^j <= budget_hi
    bud = sp.hstack(
        [
            sp.kron(
                sp.identity(n_stages, format="csc"),
                np.ones((1, n_assets)),
                format="csc",
            ),
            sp.csc_matrix((n_stages, 2 * jn)),
        ],
        format="csr",
    )
    add(bud, budget_lo, budget_hi, "budget")

    # 3-5. simple bounds on w / b / s
    ident_block = sp.hstack(
        [sp.identity(jn, format="csc", dtype=np.float64), sp.csc_matrix((jn, 2 * jn))],
        format="csr",
    )
    add(ident_block, w_min.reshape(-1), w_max.reshape(-1), "w_box")
    add(
        sp.hstack(
            [
                sp.csc_matrix((jn, jn)),
                sp.identity(jn, format="csc", dtype=np.float64),
                sp.csc_matrix((jn, jn)),
            ],
            format="csr",
        ),
        np.zeros(jn),
        b_max.reshape(-1),
        "b_box",
    )
    add(
        sp.hstack(
            [sp.csc_matrix((jn, 2 * jn)), sp.identity(jn, format="csc", dtype=np.float64)],
            format="csr",
        ),
        np.zeros(jn),
        s_max.reshape(-1),
        "s_box",
    )

    # 6-7. T+1
    if cfg.t1_enabled:
        if today:
            t1 = row_block(n_assets, {("s", j): eye_n for j in today})
            add(t1, np.full(n_assets, -np.inf), sellable_w, "t1_today")
        for j in range(n_stages):
            if j in today:
                continue
            # s^j <= w^{j-1}; for j == 0 the rhs is the constant w0.
            parts = {("s", j): eye_n}
            if j == 0:
                hi = w0.copy()
            else:
                parts[("w", j - 1)] = -eye_n
                hi = np.zeros(n_assets)
            add(row_block(n_assets, parts), np.full(n_assets, -np.inf), hi, f"t1_carry_{j}")

    # 8. factor definition (low-rank form only): X' w^j - y^j = 0, J*K rows.
    # These are the rows that let the risk term be written without any [N, N]
    # block; they are equalities, so Clarabel puts them straight into its zero
    # cone and OSQP treats them as l == u.
    if lowrank:
        xt = sp.csr_matrix(factor_x.T)  # [K, N]
        fac = sp.hstack(
            [
                sp.block_diag([xt] * n_stages, format="csr"),  # w columns
                sp.csr_matrix((jk, 2 * jn)),  # b, s columns
                -sp.identity(jk, format="csr", dtype=np.float64),  # y columns
            ],
            format="csr",
        )
        add(fac, np.zeros(jk), np.zeros(jk), "factor_def")

    A = sp.vstack(rows, format="csc")
    A.sort_indices()
    l = np.concatenate(lows)
    u = np.concatenate(ups)
    if np.any(l > u + 1e-12):
        raise ValueError("infeasible bounds: some l > u after assembly")

    meta = {
        "cost_mode": cost_mode,
        "gamma": gamma,
        "delta": delta if cost_mode == "axioma" else None,
        "stage_lengths": tuple(int(x) for x in lengths),
        "risk_scale": tuple(
            float(lengths[j] / MINUTES_PER_DAY) if cost_mode == "production" else 1.0
            for j in range(n_stages)
        ),
        "t1_enabled": bool(cfg.t1_enabled),
        "today_stage_indices": today,
        # box-repair telemetry: how many (stage, name) upper bounds had to be
        # raised above w_max because T+1 / limit-down made w_max unreachable.
        "w_box_relaxed": w_box_relaxed,
        "w_box_relax_max": w_box_relax_max,
        "bs_reg": bs_reg,
        "buy_cost": float(cfg.buy_cost()) if cost_mode == "production" else 0.0,
        "sell_cost": float(cfg.sell_cost()) if cost_mode == "production" else 0.0,
        # Stage-1 surcharge ledger: None when no extras were passed. Design doc
        # falsification F2 keys off the mean being >= ~1 bp, so it must be
        # readable from the assembled problem, not only from the caller.
        "stage1_cost_extra": _extra_meta(c_buy_extra, c_sell_extra),
        "tau_now": int(grid.tau_now),
        "n_vars": n_vars,
        "n_cons": int(A.shape[0]),
        "w0": w0.copy(),
        # Which risk assembly was actually built, and what was asked for -- the
        # 'auto' decision is invisible from the caller's arguments alone, and at
        # cluster scale "silently went dense" is the difference between 2 s and
        # never finishing.
        "risk_form": form,
        "risk_form_requested": str(risk_form).lower(),
        "n_factors": n_factors,
    }
    return MpoQp(
        P=P,
        q=q,
        A=A,
        l=l,
        u=u,
        n_stages=n_stages,
        n_assets=n_assets,
        n_factors=n_factors,
        objective_offset=float(objective_offset),
        blocks=blocks,
        meta=meta,
    )
