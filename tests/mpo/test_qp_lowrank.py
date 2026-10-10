"""Low-rank (factor) risk assembly of the MPO-T1 QP -- parity + scale gate.

The dense ``Sigma_d`` assembly puts an [N, N] block per stage into ``P``, so the
KKT factorisation never completes at cluster scale (measured: N = 5000 does not
finish; N = 300 takes 0.13 s). ``build_mpo_qp(..., factor_exposures=,
factor_cov=, specific_var=)`` assembles the algebraically identical problem with
per-stage factor variables ``y^j = X' w^j`` instead, which is sparse.

What this module pins:

1. **Default-call byte identity** -- calling without the new keywords, and
   calling with them under ``risk_form='auto'`` at N < 500, must produce
   bit-for-bit the same ``P/q/A/l/u``. Any drift here is a silent change to the
   whole existing benchmark roster.
2. **Dense vs low-rank parity** -- same weights (< 1e-5) and same objective
   (< 1e-5 relative) on a synthetic factor model.
3. **Constraint preservation** -- T+1 lock, w_box repair, price limits and the
   budget row behave in ``risk_form='lowrank'`` exactly as the dense
   ``test_qp_analytic`` cases assert they do.
4. **Axioma parity mode stays dense** -- explicitly asking for low-rank there
   raises instead of quietly rebuilding the closed-form gate's problem.
5. **Medium-N smoke** -- N = 800 solves ``optimal`` on both OSQP and Clarabel,
   the two agree on the objective to 1e-4 relative, and low-rank is far faster
   than dense. Cross-solver *weights* are deliberately NOT gated at 1e-6: the
   objective is flat around the optimum there, so Clarabel and OSQP legitimately
   differ by ~1e-4 in ``w`` while agreeing to ~1e-5 in objective value.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

from openpo.mpo.config import MINUTES_PER_DAY, MpoConfig
from openpo.mpo.qp import LOWRANK_AUTO_MIN_ASSETS, build_mpo_qp
from openpo.mpo.solve import solve_mpo_qp
from openpo.mpo.stages import build_stage_grid
from openpo.mpo.types import Stage, StageGrid

BIG = 1e6
RELAXED = dict(
    w_min=-BIG, w_max=BIG, budget_lo=-BIG, budget_hi=BIG, b_max=BIG, s_max=BIG
)


# ---------------------------------------------------------------------------
# synthetic factor risk model (same structure as synth.generate_market, but
# without the minute panel, so large N stays cheap to construct)
# ---------------------------------------------------------------------------


def factor_case(n: int, seed: int = 20260823, n_factors: int = 9, alpha_ic: float = 0.10):
    """A complete build_mpo_qp input set with a factor risk model.

    Returns a dict with both the factor triplet and the dense ``Sigma_d`` it
    implies, so the two assemblies can be fed the SAME risk model exactly.
    """
    rng = np.random.default_rng(seed)
    cfg = MpoConfig()

    x = np.column_stack(
        [rng.uniform(0.8, 1.2, size=n), rng.standard_normal((n, n_factors - 1))]
    )
    f_diag = np.concatenate(
        [[(0.20 / np.sqrt(240.0)) ** 2], rng.uniform(0.04, 0.08, size=n_factors - 1) ** 2 / 240.0]
    )
    factor_cov = np.diag(f_diag)
    specific_var = (0.025 * np.exp(0.25 * rng.standard_normal(n))) ** 2

    sigma_full = (x * f_diag[None, :]) @ x.T
    sigma_full[np.diag_indices_from(sigma_full)] += specific_var
    sigma_vol = np.sqrt(np.diag(sigma_full))

    adv_frac = 8e7 * np.exp(0.8 * rng.standard_normal(n)) / cfg.nav_cny

    grid = build_stage_grid(10 * MINUTES_PER_DAY + 5, 100, cfg)
    lengths = np.array([float(s.length) for s in grid.stages])

    # ~40% of names held, budget ~0.95, all inside the box; ~70% T+1 free.
    raw = rng.exponential(size=n) * (rng.random(n) < 0.40)
    w0 = np.minimum(0.95 * raw / raw.sum(), cfg.w_max)
    sellable_w = w0 * (rng.random(n) < 0.70)

    alpha = (
        alpha_ic
        * rng.standard_normal((grid.n_stages, n))
        * sigma_vol[None, :]
        * np.sqrt(lengths / MINUTES_PER_DAY)[:, None]
    )

    return dict(
        cfg=cfg,
        grid=grid,
        n=n,
        alpha=alpha,
        w0=w0,
        sellable_w=sellable_w,
        Sigma_d=sigma_full,
        adv_frac=adv_frac,
        sigma_d=sigma_vol,
        limit_up=rng.random(n) < 0.001,
        limit_down=rng.random(n) < 0.001,
        factor_exposures=x,
        factor_cov=factor_cov,
        specific_var=specific_var,
    )


def build(case: dict, **kwargs):
    """``build_mpo_qp`` with the case's positional contract."""
    return build_mpo_qp(
        case["alpha"],
        case["grid"],
        case["w0"],
        case["sellable_w"],
        kwargs.pop("Sigma_d", case["Sigma_d"]),
        case["adv_frac"],
        case["sigma_d"],
        case["limit_up"],
        case["limit_down"],
        kwargs.pop("cfg", case["cfg"]),
        **kwargs,
    )


def triplet(case: dict) -> dict:
    return dict(
        factor_exposures=case["factor_exposures"],
        factor_cov=case["factor_cov"],
        specific_var=case["specific_var"],
    )


def same_data(a, b) -> None:
    """Assert two MpoQp carry bit-identical OSQP data (P, q, A, l, u)."""
    assert a.n_vars == b.n_vars
    for name in ("q", "l", "u"):
        va, vb = getattr(a, name), getattr(b, name)
        assert np.array_equal(va, vb), f"{name} differs"
    for name in ("P", "A"):
        ma, mb = sp.csc_matrix(getattr(a, name)), sp.csc_matrix(getattr(b, name))
        assert ma.shape == mb.shape, f"{name} shape differs"
        assert np.array_equal(ma.indptr, mb.indptr), f"{name} indptr differs"
        assert np.array_equal(ma.indices, mb.indices), f"{name} indices differs"
        assert np.array_equal(ma.data, mb.data), f"{name} data differs"
    assert a.blocks == b.blocks


# ---------------------------------------------------------------------------
# 1. the default path may not move a single byte
# ---------------------------------------------------------------------------


def test_default_call_is_byte_identical():
    """No new kwargs, and auto-with-factors below the threshold, agree exactly.

    ``risk_form='auto'`` at N < 500 must resolve to the dense assembly, so a
    caller that starts passing the factor triplet (which ``strategies`` now
    always does) gets the byte-for-bit problem it got before. Both variants are
    assembled in-process here, so this compares against the live dense code path
    rather than a checked-in fixture.
    """
    case = factor_case(120, seed=11)
    assert case["n"] < LOWRANK_AUTO_MIN_ASSETS

    plain = build(case)
    with_factors = build(case, **triplet(case))
    explicit_auto = build(case, risk_form="auto", **triplet(case))
    forced_dense = build(case, risk_form="dense", **triplet(case))

    for other in (with_factors, explicit_auto, forced_dense):
        same_data(plain, other)
        assert other.n_factors == 0
        assert other.meta["risk_form"] == "dense"
        assert "factor_def" not in other.blocks

    assert plain.meta["risk_form"] == "dense"
    assert plain.meta["risk_form_requested"] == "auto"
    assert with_factors.meta["risk_form_requested"] == "auto"
    assert forced_dense.meta["risk_form_requested"] == "dense"


def test_auto_switches_above_the_threshold_only():
    """The 'auto' decision is a pure function of N, factors and cost mode."""
    small = factor_case(80, seed=12)
    assert build(small, **triplet(small)).meta["risk_form"] == "dense"
    assert build(small, risk_form="lowrank", **triplet(small)).meta["risk_form"] == "lowrank"

    big = factor_case(LOWRANK_AUTO_MIN_ASSETS + 1, seed=13)
    qp = build(big, **triplet(big))
    assert qp.meta["risk_form"] == "lowrank"
    assert qp.meta["risk_form_requested"] == "auto"
    # ...and without a factor model, 'auto' has nothing to switch to.
    assert build(big).meta["risk_form"] == "dense"


# ---------------------------------------------------------------------------
# 2. dense vs low-rank parity
# ---------------------------------------------------------------------------


def test_lowrank_matches_dense_weights_and_objective():
    """Same problem, two assemblies: weights < 1e-5, objective < 1e-5 relative."""
    case = factor_case(80, seed=20260823)
    cfg = MpoConfig(qp_eps=1e-9)

    dense = build(case, cfg=cfg)
    low = build(case, cfg=cfg, risk_form="lowrank", **triplet(case))

    # Structure: y variables appended AFTER [w, b, s], nothing else moved.
    jn = case["grid"].n_stages * case["n"]
    jk = case["grid"].n_stages * case["factor_exposures"].shape[1]
    assert dense.n_vars == 3 * jn
    assert low.n_vars == 3 * jn + jk
    assert low.n_factors == case["factor_exposures"].shape[1]
    assert low.blocks["factor_def"] == (dense.A.shape[0], dense.A.shape[0] + jk)
    for name, span in dense.blocks.items():
        assert low.blocks[name] == span  # every pre-existing row block is in place
    assert np.array_equal(low.q[: 3 * jn], dense.q)
    assert np.all(low.q[3 * jn :] == 0.0)  # y carries no linear term

    sol_d = solve_mpo_qp(dense, "osqp", cfg)
    sol_l = solve_mpo_qp(low, "osqp", cfg)
    assert sol_d.status == "optimal", getattr(sol_d, "note", sol_d.status)
    assert sol_l.status == "optimal", getattr(sol_l, "note", sol_l.status)

    dw = float(np.max(np.abs(sol_l.weights - sol_d.weights)))
    rel_obj = abs(sol_l.objective - sol_d.objective) / max(abs(sol_d.objective), 1e-12)
    print(f"[lowrank parity N=80] max|dw|={dw:.3e} rel_obj={rel_obj:.3e}")
    assert dw < 1e-5, f"dense/lowrank weight parity {dw:.3e}"
    assert rel_obj < 1e-5, f"dense/lowrank objective parity {rel_obj:.3e}"

    # unpack still returns exactly (w, b, s) -- the y block is internal -- and
    # the solved y must satisfy the factor_def rows it was introduced by.
    x = _primal(low, cfg)
    w, b, s = low.unpack(x)
    assert w.shape == b.shape == s.shape == (case["grid"].n_stages, case["n"])
    assert np.max(np.abs(w - sol_l.weights)) < 1e-9  # same vector solve() unpacked
    y = low.unpack_factors(x)
    assert y.shape == (case["grid"].n_stages, low.n_factors)
    assert np.max(np.abs(y - w @ case["factor_exposures"])) < 1e-7
    assert low.objective_value(x) == pytest.approx(sol_l.objective, rel=1e-12)


def _primal(qp, cfg) -> np.ndarray:
    """Re-solve and return the RAW primal vector including the y block.

    ``solve_mpo_qp`` deliberately hands back only the unpacked (w, b, s), so the
    y-block assertions need the untruncated vector.
    """
    import osqp

    model = osqp.OSQP()
    model.setup(
        P=sp.csc_matrix(qp.P),
        q=qp.q,
        A=sp.csc_matrix(qp.A),
        l=qp.l,
        u=qp.u,
        polishing=True,
        eps_abs=cfg.qp_eps,
        eps_rel=cfg.qp_eps,
        max_iter=int(cfg.qp_max_iter),
        verbose=False,
    )
    x = np.asarray(model.solve(raise_error=False).x, dtype=np.float64)
    assert x.size == qp.n_vars
    return x


def test_lowrank_objective_value_equals_the_dense_quadratic_form():
    """``objective_value`` on the extended x reproduces the dense expression.

    Direct algebra check, no solver involved: for any feasible-in-the-factor-rows
    point, ``0.5 x'Px + q'x`` of the low-rank problem must equal the dense
    problem's value at the same (w, b, s).
    """
    case = factor_case(40, seed=77)
    cfg = case["cfg"]
    rng = np.random.default_rng(5)
    j, n = case["grid"].n_stages, case["n"]

    dense = build(case)
    low = build(case, risk_form="lowrank", **triplet(case))

    w = rng.uniform(0.0, cfg.w_max, size=(j, n))
    b = rng.uniform(0.0, 1e-3, size=(j, n))
    s = rng.uniform(0.0, 1e-3, size=(j, n))
    y = w @ case["factor_exposures"]  # the factor_def rows, satisfied exactly

    x_dense = np.concatenate([w.reshape(-1), b.reshape(-1), s.reshape(-1)])
    x_low = np.concatenate([x_dense, y.reshape(-1)])
    got, ref = low.objective_value(x_low), dense.objective_value(x_dense)
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-18)


# ---------------------------------------------------------------------------
# 3. every constraint block survives the reformulation
# ---------------------------------------------------------------------------


def lowrank_problem(alpha, grid, w0, sellable, cfg, n_factors=4, seed=3, **kwargs):
    """Small hand-made low-rank instance mirroring test_qp_analytic.solve_problem."""
    alpha = np.asarray(alpha, dtype=np.float64)
    n = alpha.shape[1]
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.standard_normal((n, n_factors - 1))])
    f_diag = np.full(n_factors, 1e-4)
    spec = np.full(n, 6.25e-4)
    sigma_full = (x * f_diag[None, :]) @ x.T
    sigma_full[np.diag_indices_from(sigma_full)] += spec
    limit_up = kwargs.pop("limit_up", np.zeros(n, bool))
    limit_down = kwargs.pop("limit_down", np.zeros(n, bool))
    qp = build_mpo_qp(
        alpha,
        grid,
        w0,
        sellable,
        sigma_full,
        kwargs.pop("adv_frac", np.full(n, 0.8)),
        np.sqrt(np.diag(sigma_full)),
        limit_up,
        limit_down,
        cfg,
        risk_form="lowrank",
        factor_exposures=x,
        factor_cov=np.diag(f_diag),
        specific_var=spec,
        **kwargs,
    )
    return qp, solve_mpo_qp(qp, cfg.qp_backend, cfg)


def test_lowrank_preserves_t1_lock_and_ablation():
    """Mirror of test_t1_blocks_today_sells_until_inventory_is_sellable."""
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    today = list(grid.today_stage_indices)
    n = 8
    w0 = np.full(n, 0.01)
    alpha = np.full((grid.n_stages, n), -20e-4)

    qp_l, locked = lowrank_problem(alpha, grid, w0, np.zeros(n), cfg)
    _, free = lowrank_problem(alpha, grid, w0, w0.copy(), cfg)
    assert qp_l.meta["risk_form"] == "lowrank" and "factor_def" in qp_l.blocks
    assert locked.status == "optimal" and free.status == "optimal"
    assert float(locked.sells[today].sum()) < 1e-5, "T+1 must forbid today's sells"
    assert float(locked.weights[0].min()) > 0.01 - 1e-5
    assert float(free.sells[today].sum()) > 0.5 * w0.sum()

    qp_off, ablated = lowrank_problem(
        alpha, grid, w0, np.zeros(n), MpoConfig(t1_enabled=False)
    )
    assert "t1_today" not in qp_off.blocks and "factor_def" in qp_off.blocks
    assert ablated.status == "optimal"
    assert float(ablated.sells[today].sum()) > 0.5 * w0.sum()


def test_lowrank_preserves_t1_carry_rows():
    """Mirror of test_t1_carry_rows_limit_later_stage_sells_to_prior_holdings."""
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    n = 6
    qp, sol = lowrank_problem(
        np.full((grid.n_stages, n), -20e-4), grid, np.full(n, 0.01), np.zeros(n), cfg
    )
    assert {"t1_carry_2", "t1_carry_3"} <= set(qp.blocks)
    assert sol.status == "optimal"
    for j in (2, 3):
        assert np.all(sol.sells[j] <= sol.weights[j - 1] + 1e-6)


def test_lowrank_preserves_budget_and_box():
    """Mirror of test_budget_and_box_constraints_hold."""
    # qp_eps is tightened one notch below the 1e-6 assertion tolerance: at the
    # production default the ADMM residual on this deliberately budget-binding
    # instance lands ~1.5e-6 negative on a few b/s entries, which is solver slack
    # (proportional to eps_abs), not a constraint the reformulation lost.
    cfg = MpoConfig(qp_eps=1e-8)  # w_max = 0.02, budget <= 1
    grid = build_stage_grid(35, 100, cfg)
    n = 60  # 60 * 0.02 = 1.2 > 1, so the budget row must bind
    rng = np.random.default_rng(99)
    alpha = np.abs(rng.normal(scale=40e-4, size=(grid.n_stages, n))) + 20e-4
    _, sol = lowrank_problem(alpha, grid, np.zeros(n), np.zeros(n), cfg)
    assert sol.status == "optimal"
    tol = 1e-6
    assert sol.weights.max() <= cfg.w_max + tol
    assert sol.weights.min() >= -tol
    assert sol.buys.min() >= -tol and sol.sells.min() >= -tol
    assert np.all(sol.weights.sum(axis=1) <= 1.0 + tol)
    assert sol.weights.sum(axis=1).max() > 0.9  # binding, not vacuous
    prev = np.zeros(n)
    for j in range(grid.n_stages):
        assert np.allclose(sol.weights[j] - prev, sol.buys[j] - sol.sells[j], atol=1e-6)
        prev = sol.weights[j]


def test_lowrank_preserves_price_limit_masks():
    """Mirror of test_price_limit_masks_zero_stage1_buy_and_sell."""
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    n = 6
    w0 = np.full(n, 0.01)
    alpha = np.full((grid.n_stages, n), 20e-4)
    alpha[:, 1] = -20e-4
    limit_up = np.zeros(n, bool)
    limit_up[0] = True
    limit_down = np.zeros(n, bool)
    limit_down[1] = True

    _, control = lowrank_problem(alpha, grid, w0, w0.copy(), cfg)
    qp, masked = lowrank_problem(
        alpha, grid, w0, w0.copy(), cfg, limit_up=limit_up, limit_down=limit_down
    )
    assert control.status == "optimal" and masked.status == "optimal"
    assert control.buys[0, 0] > 1e-4 and control.sells[0, 1] > 1e-4
    assert abs(float(masked.buys[0, 0])) < 1e-8
    assert abs(float(masked.sells[0, 1])) < 1e-8
    b_lo, _ = qp.blocks["b_box"]
    assert qp.u[b_lo] == 0.0 and np.isinf(qp.u[b_lo + n])


def test_lowrank_preserves_w_box_repair():
    """Mirror of test_drifted_overweight_under_t1_stays_feasible (case a)."""
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    n = 6
    alpha = np.full((grid.n_stages, n), -20e-4)
    drift = 2e-4
    w0 = np.full(n, cfg.w_max)
    w0[0] += drift

    qp, sol = lowrank_problem(alpha, grid, w0, np.zeros(n), cfg)
    assert sol.status == "optimal", getattr(sol, "note", sol.status)
    assert qp.meta["w_box_relaxed"] == len(grid.today_stage_indices)
    assert qp.meta["w_box_relax_max"] == pytest.approx(drift, rel=1e-9)
    for j in grid.today_stage_indices:
        assert sol.weights[j][0] == pytest.approx(w0[0], abs=1e-6)
        assert sol.weights[j][1:].max() <= cfg.w_max + 1e-6
    assert sol.weights[-1].max() < cfg.w_max

    qp_ok, sol_ok = lowrank_problem(alpha, grid, w0, np.full(n, drift), cfg)
    assert qp_ok.meta["w_box_relaxed"] == 0
    assert sol_ok.status == "optimal"
    assert sol_ok.weights.max() <= cfg.w_max + 1e-6


def test_lowrank_preserves_stage_length_risk_scaling():
    """Analytic single-name optimum: the L_j/240 factor must survive the rewrite.

    With one factor of variance ``fv`` and unit exposure plus specific variance
    ``sv``, ``Sigma_d = fv + sv`` for the single name, so the dense first-order
    condition of test_production_risk_scales_with_stage_length applies verbatim.
    """
    cfg = MpoConfig(w_max=1.0)
    fv, sv = 2.25e-4, 4.0e-4
    sigma2 = fv + sv
    adv_frac = np.array([0.8])
    alpha = np.array([[12e-4]])
    impact = cfg.impact_coef * np.sqrt(sigma2) / adv_frac[0]

    held = {}
    for length in (30, 240):
        grid = StageGrid(
            tau_now=5,
            stages=(Stage(index=0, start=5, end=5 + length),),
            today_stage_indices=(0,),
        )
        qp = build_mpo_qp(
            alpha,
            grid,
            np.zeros(1),
            np.zeros(1),
            np.array([[sigma2]]),
            adv_frac,
            np.array([np.sqrt(sigma2)]),
            np.zeros(1, bool),
            np.zeros(1, bool),
            cfg,
            risk_form="lowrank",
            factor_exposures=np.ones((1, 1)),
            factor_cov=np.array([[fv]]),
            specific_var=np.array([sv]),
        )
        sol = solve_mpo_qp(qp, cfg.qp_backend, cfg)
        assert sol.status == "optimal"
        assert qp.meta["risk_scale"] == (length / 240.0,)
        analytic = (alpha[0, 0] - cfg.buy_cost()) / (
            cfg.gamma * (length / 240.0) * sigma2 + impact
        )
        assert abs(float(sol.weights[0, 0]) - analytic) < 1e-6, (length, analytic)
        held[length] = float(sol.weights[0, 0])
    assert held[30] > 1.5 * held[240], held


# ---------------------------------------------------------------------------
# 4. input validation + the axioma restriction
# ---------------------------------------------------------------------------


def test_axioma_parity_mode_is_dense_only():
    """The closed-form gate's problem must not be silently reformulated."""
    case = factor_case(30, seed=8)
    cfg_ax = MpoConfig(cost_mode="axioma", gamma=10.0, t1_enabled=False)
    with pytest.raises(ValueError, match="not available in cost_mode='axioma'"):
        build(case, cfg=cfg_ax, risk_form="lowrank", **triplet(case), **RELAXED)
    # 'auto' falls back to dense instead of raising, so a strategy that always
    # forwards its factor model still works in parity mode.
    qp = build(case, cfg=cfg_ax, **triplet(case), **RELAXED)
    assert qp.meta["risk_form"] == "dense" and qp.n_factors == 0
    same_data(qp, build(case, cfg=cfg_ax, **RELAXED))
    # ...and it stays dense even above the auto threshold.
    big = factor_case(LOWRANK_AUTO_MIN_ASSETS + 1, seed=9)
    assert build(big, cfg=cfg_ax, **triplet(big), **RELAXED).meta["risk_form"] == "dense"


def test_factor_input_validation():
    case = factor_case(20, seed=4)
    x, f, spec = case["factor_exposures"], case["factor_cov"], case["specific_var"]

    with pytest.raises(ValueError, match="needs all of factor_exposures"):
        build(case, factor_exposures=x)
    with pytest.raises(ValueError, match="needs all of factor_exposures"):
        build(case, factor_exposures=x, factor_cov=f)
    with pytest.raises(ValueError, match="risk_form must be one of"):
        build(case, risk_form="sparse")
    with pytest.raises(ValueError, match="needs factor_exposures"):
        build(case, risk_form="lowrank")
    with pytest.raises(ValueError, match="factor_exposures must be"):
        build(case, factor_exposures=x[:-1], factor_cov=f, specific_var=spec)
    with pytest.raises(ValueError, match="factor_cov must be"):
        build(case, factor_exposures=x, factor_cov=f[:-1, :-1], specific_var=spec)
    with pytest.raises(ValueError, match="specific_var must have"):
        build(case, factor_exposures=x, factor_cov=f, specific_var=spec[:-1])
    with pytest.raises(ValueError, match="specific_var must be non-negative"):
        build(case, factor_exposures=x, factor_cov=f, specific_var=-spec)
    with pytest.raises(ValueError, match="factor_cov contains non-finite"):
        build(case, factor_exposures=x, factor_cov=np.full_like(f, np.nan), specific_var=spec)


def test_sigma_d_may_be_none_only_on_the_lowrank_path():
    """A cluster-scale caller can skip the [N, N] allocation it would not use."""
    case = factor_case(80, seed=6)
    cfg = MpoConfig(qp_eps=1e-9)
    ref = build(case, cfg=cfg, risk_form="lowrank", **triplet(case))
    without = build(case, cfg=cfg, Sigma_d=None, risk_form="lowrank", **triplet(case))
    same_data(ref, without)
    assert without.meta["w0"].shape == (case["n"],)

    with pytest.raises(ValueError, match="Sigma_d is required for the dense risk form"):
        build(case, Sigma_d=None)
    with pytest.raises(ValueError, match="Sigma_d is required for the dense risk form"):
        build(case, Sigma_d=None, risk_form="dense", **triplet(case))


def test_slice_and_unpack_guards():
    case = factor_case(20, seed=10)
    dense = build(case)
    low = build(case, risk_form="lowrank", **triplet(case))
    j, n, k = case["grid"].n_stages, case["n"], low.n_factors

    assert low.slice_w(1) == dense.slice_w(1)
    assert low.slice_b(1) == dense.slice_b(1)
    assert low.slice_s(1) == dense.slice_s(1)
    assert low.slice_y(0) == slice(3 * j * n, 3 * j * n + k)
    assert low.slice_y(j - 1) == slice(3 * j * n + (j - 1) * k, 3 * j * n + j * k)

    with pytest.raises(ValueError, match="dense risk form"):
        dense.slice_y(0)
    with pytest.raises(ValueError, match="dense risk form"):
        dense.unpack_factors(np.zeros(dense.n_vars))
    with pytest.raises(ValueError, match="entries"):
        low.unpack(np.zeros(low.n_vars - 1))
    with pytest.raises(ValueError, match="entries"):
        low.unpack_factors(np.zeros(low.n_vars - 1))


# ---------------------------------------------------------------------------
# 5. medium-N scale smoke: the whole point of the reformulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [800])
def test_medium_n_lowrank_solves_on_both_backends_and_beats_dense(n):
    """N = 800: lowrank+osqp and lowrank+clarabel agree, and dense is slower.

    Cross-backend gate is on the OBJECTIVE (1e-4 relative), not on weights: the
    objective is flat around this optimum, so the two solvers legitimately land
    ~1e-4 apart in ``w`` (Clarabel is the more feasible of the two) while
    agreeing to ~1e-5 in value. A 1e-6 weight gate here would be a false alarm
    generator, not a correctness check.
    """
    case = factor_case(n, seed=20260823)
    cfg = case["cfg"]  # production defaults, qp_eps = 1e-6

    t0 = time.perf_counter()
    low = build(case, risk_form="lowrank", **triplet(case))
    t_build_low = time.perf_counter() - t0
    assert low.meta["risk_form"] == "lowrank"
    assert low.n_vars == 3 * case["grid"].n_stages * n + case["grid"].n_stages * low.n_factors

    sol_low = solve_mpo_qp(low, "osqp", cfg)
    assert sol_low.status == "optimal", getattr(sol_low, "note", sol_low.status)

    sol_cl = solve_mpo_qp(low, "clarabel", cfg)
    if sol_cl.status == "failed":  # pragma: no cover - optional dependency
        pytest.skip(f"clarabel unavailable: {getattr(sol_cl, 'note', '')}")
    assert sol_cl.status == "optimal", getattr(sol_cl, "note", sol_cl.status)

    rel_obj = abs(sol_cl.objective - sol_low.objective) / max(abs(sol_low.objective), 1e-12)
    dw = float(np.max(np.abs(sol_cl.weights - sol_low.weights)))

    t0 = time.perf_counter()
    dense = build(case)
    t_build_dense = time.perf_counter() - t0
    sol_dense = solve_mpo_qp(dense, "osqp", cfg)
    assert sol_dense.status == "optimal", getattr(sol_dense, "note", sol_dense.status)

    print(
        f"\n[N={n} J={case['grid'].n_stages}] "
        f"lowrank/osqp {sol_low.solve_time:.2f}s (build {t_build_low:.2f}s) | "
        f"lowrank/clarabel {sol_cl.solve_time:.2f}s | "
        f"dense/osqp {sol_dense.solve_time:.2f}s (build {t_build_dense:.2f}s) | "
        f"speedup {sol_dense.solve_time / max(sol_low.solve_time, 1e-9):.1f}x | "
        f"clarabel-vs-osqp rel_obj {rel_obj:.2e} max|dw| {dw:.2e}"
    )

    assert rel_obj < 1e-4, f"cross-backend objective {rel_obj:.3e}"
    # Same optimum as the dense assembly, at the solver tolerance of this run.
    rel_dense = abs(sol_low.objective - sol_dense.objective) / max(abs(sol_dense.objective), 1e-12)
    assert rel_dense < 1e-4, f"lowrank vs dense objective {rel_dense:.3e}"
    # Wall-time sanity (informational, generous factor: this is the reason the
    # low-rank form exists, but CI machines are noisy).
    assert sol_low.solve_time < sol_dense.solve_time, (
        sol_low.solve_time,
        sol_dense.solve_time,
    )


# ---------------------------------------------------------------------------
# 6. strategy plumbing: the factor model reaches the QP, defaults unchanged
# ---------------------------------------------------------------------------


class _FixedAlphaModel:
    """Minimal term-structure stand-in: constant per-stage alpha."""

    B = np.zeros((3, 3))

    def __init__(self, value: float, n: int) -> None:
        self.value = float(value)
        self.n = int(n)

    def stage_alphas(self, z, grid):
        return np.full((grid.n_stages, self.n), self.value)


class _FakeMarket:
    """MarketPanel-shaped duck type carrying a factor risk model."""

    def __init__(self, case: dict) -> None:
        self.factor_exposures = case["factor_exposures"]
        self.factor_cov_daily = case["factor_cov"]
        self.specific_var_daily = case["specific_var"]
        self._cov = case["Sigma_d"]
        self.adv_cny = case["adv_frac"] * case["cfg"].nav_cny
        self.n_days = 100
        self.n_assets = case["n"]

    def daily_cov(self):
        return self._cov


def _ctx(case: dict):
    from openpo.mpo.types import DecisionContext, PortfolioSnapshot

    n = case["n"]
    prices = np.full(n, 20.0)
    nav = case["cfg"].nav_cny
    w0 = case["w0"]
    return DecisionContext(
        snapshot=PortfolioSnapshot(
            tau=10 * MINUTES_PER_DAY + 5,
            nav=nav,
            cash=nav * (1.0 - float(w0.sum())),
            holdings_shares=w0 * nav / prices,
            weights=w0.copy(),
            sellable_shares=case["sellable_w"] * nav / prices,
            sellable_weights=case["sellable_w"].copy(),
        ),
        z_signals=np.zeros((n, 3)),
        prices=prices,
        limit_up=np.zeros(n, bool),
        limit_down=np.zeros(n, bool),
        halted=np.zeros(n, bool),
        day=10,
        minute=5,
    )


def test_strategy_adopts_the_market_factor_model_without_changing_defaults():
    """MpoStrategy carries the panel's factor model; N=300-scale stays dense."""
    from openpo.mpo.strategies import MpoStrategy

    case = factor_case(60, seed=14)
    market = _FakeMarket(case)
    strat = MpoStrategy(_FixedAlphaModel(20e-4, case["n"]), market=market, cfg=case["cfg"])

    assert strat.factor_exposures is case["factor_exposures"]
    assert strat.risk_form == "auto"
    built: list = []
    real_build = build_mpo_qp

    import openpo.mpo.qp as qp_mod

    def spy(*args, **kwargs):
        qp = real_build(*args, **kwargs)
        built.append(qp)
        return qp

    qp_mod.build_mpo_qp = spy
    try:
        w1 = strat.decide(_ctx(case))
    finally:
        qp_mod.build_mpo_qp = real_build

    assert w1 is not None and w1.shape == (case["n"],)
    assert len(built) == 1
    # N = 60 < 500: 'auto' keeps the dense assembly the benchmark has always used.
    assert built[0].meta["risk_form"] == "dense"
    assert built[0].meta["risk_form_requested"] == "auto"
    assert built[0].n_factors == 0

    # No market and nothing passed -> no factor model at all, same as before.
    bare = MpoStrategy(
        _FixedAlphaModel(20e-4, case["n"]),
        Sigma_d=case["Sigma_d"],
        adv_frac=case["adv_frac"],
        cfg=case["cfg"],
        n_days=100,
    )
    assert bare.factor_exposures is None
    assert bare.factor_cov is None and bare.specific_var is None


def test_strategy_forced_lowrank_uses_the_factor_assembly():
    """``risk_form='lowrank'`` reaches build_mpo_qp and still returns a w^1."""
    from openpo.mpo.strategies import MpoStrategy

    case = factor_case(60, seed=15)
    strat = MpoStrategy(
        _FixedAlphaModel(20e-4, case["n"]),
        market=_FakeMarket(case),
        cfg=case["cfg"],
        risk_form="lowrank",
    )
    import openpo.mpo.qp as qp_mod

    built: list = []
    real_build = build_mpo_qp

    def spy(*args, **kwargs):
        qp = real_build(*args, **kwargs)
        built.append(qp)
        return qp

    qp_mod.build_mpo_qp = spy
    try:
        w1 = strat.decide(_ctx(case))
    finally:
        qp_mod.build_mpo_qp = real_build

    assert len(built) == 1
    assert built[0].meta["risk_form"] == "lowrank"
    assert built[0].n_factors == case["factor_exposures"].shape[1]
    assert w1 is not None and np.all(np.isfinite(w1))


def test_strategy_rejects_a_market_risk_model_of_the_wrong_size():
    """A panel describing a different universe than Sigma_d is a wiring bug."""
    from openpo.mpo.strategies import MpoStrategy

    case = factor_case(30, seed=16)
    market = _FakeMarket(case)
    with pytest.raises(ValueError, match="market.factor_exposures must be"):
        MpoStrategy(
            _FixedAlphaModel(20e-4, 10),
            Sigma_d=np.eye(10) * 1e-4,
            adv_frac=np.full(10, 0.8),
            n_days=100,
            market=market,
            cfg=case["cfg"],
        )
