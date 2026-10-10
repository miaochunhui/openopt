"""Analytic + economic validation of the MPO-T1 multi-stage QP.

Acceptance gate (docs/mpo-t1-SPEC.md section 3 / section 10): the numerical QP
in ``cost_mode='axioma'`` with all box/budget/T+1 constraints relaxed must
reproduce the Axioma 051 eq29-30 closed form to ``max|diff| < 1e-6``. The rest
of the file pins the economics that the QP is supposed to produce *emergently*
(T+1 lock, round-trip costs, stage-length risk scaling, price limits) plus the
fail-soft contract.

All problems here are tiny (N <= 300 only in the parity-free timing sense);
the whole module runs in a couple of seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpo.mpo.closed_form import closed_form_two_stage
from openpo.mpo.config import MpoConfig
from openpo.mpo.qp import build_mpo_qp
from openpo.mpo.solve import solve_mpo_qp
from openpo.mpo.stages import build_stage_grid
from openpo.mpo.types import MpoSolution, Stage, StageGrid

# Bound used to switch box/budget constraints off (SPEC: "+-1e6").
BIG = 1e6
RELAXED = dict(
    w_min=-BIG, w_max=BIG, budget_lo=-BIG, budget_hi=BIG, b_max=BIG, s_max=BIG
)


def random_spd(n: int, seed: int, scale: float = 1e-4) -> np.ndarray:
    """Random SPD daily covariance: ~2.5% daily vol, full correlation matrix."""
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n, n))
    return scale * (b @ b.T / n + 0.5 * np.eye(n))


def solve_problem(
    alpha,
    grid: StageGrid,
    w0,
    sellable_w,
    cfg: MpoConfig,
    *,
    Sigma=None,
    adv_frac=None,
    limit_up=None,
    limit_down=None,
    **overrides,
) -> tuple[object, MpoSolution]:
    """Build + solve with the same positional contract as ``strategies``."""
    alpha = np.asarray(alpha, dtype=np.float64)
    n = alpha.shape[1]
    Sigma = random_spd(n, seed=5) if Sigma is None else np.asarray(Sigma, float)
    adv_frac = np.full(n, 0.8) if adv_frac is None else np.asarray(adv_frac, float)
    limit_up = np.zeros(n, bool) if limit_up is None else limit_up
    limit_down = np.zeros(n, bool) if limit_down is None else limit_down
    qp = build_mpo_qp(
        alpha,
        grid,
        w0,
        sellable_w,
        Sigma,
        adv_frac,
        np.sqrt(np.diag(Sigma)),
        limit_up,
        limit_down,
        cfg,
        **overrides,
    )
    return qp, solve_mpo_qp(qp, cfg.qp_backend, cfg)


def two_stage_grid() -> StageGrid:
    """J=2, two full days, no stage ends today (T+1 rows are irrelevant)."""
    return StageGrid(
        tau_now=0,
        stages=(Stage(index=0, start=0, end=240), Stage(index=1, start=240, end=480)),
        today_stage_indices=(),
    )


# ---------------------------------------------------------------------------
# 1. Closed-form parity -- the acceptance gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 101, 20260821])
def test_axioma_two_stage_matches_closed_form(seed):
    """Unconstrained J=2 axioma-mode QP == Axioma 051 eq29-30, < 1e-6.

    ``qp_eps`` is tightened to 1e-9 for this test: OSQP's polish step cannot
    factorise the (deliberately cost-free) b/s block, so the achieved primal
    accuracy is set by the ADMM tolerance alone. At the production default
    (1e-6) the residual is ~1e-5 on this deliberately unconstrained instance
    with O(1) weights; 1e-9 costs ~100 extra ADMM iterations on N=10.
    """
    n = 10
    rng = np.random.default_rng(seed)
    sigma = random_spd(n, seed=seed)
    alpha = rng.normal(scale=2e-3, size=(2, n))
    w0 = rng.uniform(0.0, 0.02, size=n)

    cfg = MpoConfig(
        cost_mode="axioma",
        axioma_delta=1.0,
        gamma=10.0,
        t1_enabled=False,
        qp_eps=1e-9,
    )
    grid = two_stage_grid()
    qp, sol = solve_problem(
        alpha, grid, w0, np.zeros(n), cfg, Sigma=sigma, **RELAXED
    )
    assert sol.status == "optimal", getattr(sol, "note", sol.status)
    assert qp.meta["cost_mode"] == "axioma"
    assert qp.meta["risk_scale"] == (1.0, 1.0)  # no L_j/240 in parity mode

    w1_ref = closed_form_two_stage(
        alpha[0], alpha[1], sigma, cfg.gamma, cfg.axioma_delta, w0
    )
    diff1 = float(np.max(np.abs(sol.weights[0] - w1_ref)))
    assert diff1 < 1e-6, f"stage-1 parity {diff1:.3e} (seed {seed})"

    # Second FOC: (gamma+delta) w2 = Sigma^-1 alpha2 + delta w1.
    w2_ref = (
        np.linalg.solve(sigma, alpha[1]) + cfg.axioma_delta * w1_ref
    ) / (cfg.gamma + cfg.axioma_delta)
    diff2 = float(np.max(np.abs(sol.weights[1] - w2_ref)))
    assert diff2 < 1e-6, f"stage-2 parity {diff2:.3e} (seed {seed})"

    # b/s stay a valid non-negative split of the stage trades.
    dw1 = sol.weights[0] - w0
    assert np.allclose(sol.buys[0] - sol.sells[0], dw1, atol=1e-7)
    assert sol.buys.min() > -1e-7 and sol.sells.min() > -1e-7


def test_closed_form_rejects_bad_input():
    n = 4
    sigma = random_spd(n, seed=3)
    a = np.zeros(n)
    with pytest.raises(ValueError):
        closed_form_two_stage(a, np.zeros(n + 1), sigma, 1.0, 1.0, a)
    with pytest.raises(ValueError):
        closed_form_two_stage(a, a, sigma, 0.0, 1.0, a)  # gamma must be > 0
    with pytest.raises(ValueError):
        closed_form_two_stage(a, a, -sigma, 1.0, 1.0, a)  # not PD


def test_parity_holds_for_clarabel_fallback():
    """The fallback backend solves the same problem (looser tolerance)."""
    n = 10
    rng = np.random.default_rng(4242)
    sigma = random_spd(n, seed=4242)
    alpha = rng.normal(scale=2e-3, size=(2, n))
    w0 = rng.uniform(0.0, 0.02, size=n)
    cfg = MpoConfig(cost_mode="axioma", gamma=10.0, t1_enabled=False, qp_eps=1e-9)
    qp = build_mpo_qp(
        alpha,
        two_stage_grid(),
        w0,
        np.zeros(n),
        sigma,
        np.full(n, 0.8),
        np.sqrt(np.diag(sigma)),
        np.zeros(n, bool),
        np.zeros(n, bool),
        cfg,
        **RELAXED,
    )
    sol = solve_mpo_qp(qp, backend="clarabel", cfg=cfg)
    if sol.status == "failed":  # pragma: no cover - optional dependency
        pytest.skip(f"clarabel unavailable: {getattr(sol, 'note', '')}")
    w1_ref = closed_form_two_stage(
        alpha[0], alpha[1], sigma, cfg.gamma, cfg.axioma_delta, w0
    )
    assert float(np.max(np.abs(sol.weights[0] - w1_ref))) < 1e-5


# ---------------------------------------------------------------------------
# 2. Emergent economics: alpha persistence decides whether we trade at all
# ---------------------------------------------------------------------------


def test_short_only_alpha_untradeable_persistent_alpha_traded():
    """Same stage-1 alpha, different persistence -> opposite decisions.

    Entering costs ``cfg.buy_cost()`` = 7bp (2bp commission + 5bp slippage) and,
    with T+1 plus zero sellable inventory, nothing bought in the execution
    stage can be sold before the next day (exit costs a further 12bp). A 5bp
    alpha that lives only in the 30-minute execution stage therefore cannot pay
    for the round trip and the optimizer buys nothing; the *same* 5bp alpha
    repeated over the whole 2400-minute horizon (20bp cumulative) is worth
    paying for.
    """
    cfg = MpoConfig()  # gamma=10, default costs, t1_enabled=True
    grid = build_stage_grid(5, 100, cfg)
    assert grid.n_stages == 4 and grid.today_stage_indices == (0, 1)
    n = 20
    w0 = np.zeros(n)
    sellable = np.zeros(n)

    short_only = np.zeros((4, n))
    short_only[0] = 5e-4  # 5bp, execution stage only
    persistent = np.full((4, n), 5e-4)  # same 5bp in every stage

    _, sol_short = solve_problem(short_only, grid, w0, sellable, cfg)
    _, sol_pers = solve_problem(persistent, grid, w0, sellable, cfg)

    assert sol_short.status == "optimal" and sol_pers.status == "optimal"
    assert float(sol_short.buys[0].max()) < 1e-6, "short-only alpha must not trade"
    assert float(np.abs(sol_short.weights).max()) < 1e-6
    assert float(sol_pers.buys[0].min()) > 1e-3, "persistent alpha must trade"


# ---------------------------------------------------------------------------
# 3. T+1 constraint actually binds
# ---------------------------------------------------------------------------


def test_t1_blocks_today_sells_until_inventory_is_sellable():
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    today = list(grid.today_stage_indices)
    n = 8
    w0 = np.full(n, 0.01)
    alpha = np.full((4, n), -20e-4)  # every stage wants the position gone

    _, locked = solve_problem(alpha, grid, w0, np.zeros(n), cfg)
    _, free = solve_problem(alpha, grid, w0, w0.copy(), cfg)

    assert locked.status == "optimal" and free.status == "optimal"
    assert float(locked.sells[today].sum()) < 1e-5, "T+1 must forbid today's sells"
    assert float(locked.weights[0].min()) > 0.01 - 1e-5  # position cannot shrink today
    assert float(free.sells[today].sum()) > 0.5 * w0.sum(), "sellable inventory must sell"
    assert float(free.weights[0].max()) < 0.01 - 1e-4

    # Ablation: with t1_enabled=False the same locked problem sells immediately.
    cfg_off = MpoConfig(t1_enabled=False)
    qp_off, ablated = solve_problem(alpha, grid, w0, np.zeros(n), cfg_off)
    assert "t1_today" not in qp_off.blocks
    assert ablated.status == "optimal"
    assert float(ablated.sells[today].sum()) > 0.5 * w0.sum()


def test_t1_carry_rows_limit_later_stage_sells_to_prior_holdings():
    """A non-today stage can only sell what the previous stage holds."""
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    n = 6
    qp, sol = solve_problem(
        np.full((4, n), -20e-4), grid, np.full(n, 0.01), np.zeros(n), cfg
    )
    assert {"t1_carry_2", "t1_carry_3"} <= set(qp.blocks)
    assert sol.status == "optimal"
    for j in (2, 3):
        assert np.all(sol.sells[j] <= sol.weights[j - 1] + 1e-6)

    # Grid with no stage ending today: the first stage's carry row is bounded by
    # the constant w0 (everything on the book is already sellable).
    w0 = np.full(n, 0.004)
    qp0, sol0 = solve_problem(
        np.full((2, n), -20e-4), two_stage_grid(), w0, np.zeros(n), cfg
    )
    assert "t1_today" not in qp0.blocks and "t1_carry_0" in qp0.blocks
    lo, hi = qp0.blocks["t1_carry_0"]
    assert np.allclose(qp0.u[lo:hi], w0)
    assert sol0.status == "optimal"
    assert np.all(sol0.sells[0] <= w0 + 1e-6)
    assert float(sol0.sells[0].sum()) > 0.5 * w0.sum()  # and it does sell


def test_drifted_overweight_under_t1_stays_feasible():
    """A price-drifted overweight that T+1 forbids trimming must not kill the QP.

    First-scene finding (2026-08-22, seed 20260821): weights are price-driven,
    so a holding parked at the ``w_max`` box drifts above it with NO trade at
    all. Shares bought today are not sellable, so ``w^j <= w_max`` contradicts
    the dynamics + ``t1_today`` rows and the problem is *genuinely* primal
    infeasible (OSQP 'primal infeasible'; an LP min-violation check confirms no
    feasible point exists). An unsellable excess of 9.4e-6 in weight units was
    enough to fail-soft 87% of the J=1 baselines' decision epochs. The box
    upper bound is therefore raised to the tightest weight each stage can
    legally reach; the limit still binds wherever it is attainable.
    """
    cfg = MpoConfig()  # w_max = 0.02
    grid = build_stage_grid(5, 100, cfg)  # J=4, today stages = (0, 1)
    n = 6
    alpha = np.full((grid.n_stages, n), -20e-4)  # every stage wants it gone
    drift = 2e-4  # weight units; measured drifts ran 1e-5 .. 1.5e-3
    w0 = np.full(n, cfg.w_max)
    w0[0] += drift  # only name 0 drifted above the limit

    # (a) nothing sellable today: the excess is unavoidable, so it is tolerated
    qp, sol = solve_problem(alpha, grid, w0, np.zeros(n), cfg)
    assert sol.status == "optimal", getattr(sol, "note", sol.status)
    # exactly the today stages of the one drifted name were relaxed
    assert qp.meta["w_box_relaxed"] == len(grid.today_stage_indices)
    assert qp.meta["w_box_relax_max"] == pytest.approx(drift, rel=1e-9)
    for j in grid.today_stage_indices:
        assert sol.weights[j][0] == pytest.approx(w0[0], abs=1e-6)  # cannot trim
        assert sol.weights[j][1:].max() <= cfg.w_max + 1e-6  # others still capped
    # next day the shares are sellable again and the position is trimmed
    assert sol.weights[-1].max() < cfg.w_max
    # SAFETY INVARIANT: the repaired bound is never above the *current* holding,
    # so no NEW overweight can ever be bought -- only an inherited one tolerated.
    lo, hi = qp.blocks["w_box"]
    assert np.all(
        qp.u[lo:hi] <= np.maximum(cfg.w_max, np.tile(w0, grid.n_stages)) + 1e-15
    )

    # (b) the excess IS sellable today: nothing is relaxed and the box binds
    qp_ok, sol_ok = solve_problem(alpha, grid, w0, np.full(n, drift), cfg)
    assert qp_ok.meta["w_box_relaxed"] == 0
    assert sol_ok.status == "optimal"
    assert sol_ok.weights.max() <= cfg.w_max + 1e-6

    # (c) only part of the excess is sellable: the bound is the reachable weight
    half = 0.4 * drift
    qp_h, sol_h = solve_problem(alpha, grid, w0, np.full(n, half), cfg)
    assert qp_h.meta["w_box_relaxed"] == len(grid.today_stage_indices)
    assert sol_h.status == "optimal"
    assert sol_h.weights[0][0] == pytest.approx(w0[0] - half, abs=1e-6)

    # (d) stage-1 limit-down blocks the sell even with sellable inventory: the
    # relaxation is confined to stage 1, the second today stage still caps.
    limit_down = np.zeros(n, bool)
    limit_down[0] = True
    qp_d, sol_d = solve_problem(
        alpha, grid, w0, w0.copy(), cfg, limit_down=limit_down
    )
    assert qp_d.meta["w_box_relaxed"] == 1  # stage 0 of name 0 only
    assert sol_d.status == "optimal"
    assert sol_d.weights[0][0] == pytest.approx(w0[0], abs=1e-6)
    assert sol_d.weights[1][0] <= cfg.w_max + 1e-6

    # (e) no drift anywhere -> the box is untouched (no silent relaxation)
    qp_n, _ = solve_problem(alpha, grid, np.full(n, 0.01), np.zeros(n), cfg)
    assert qp_n.meta["w_box_relaxed"] == 0
    lo, hi = qp_n.blocks["w_box"]
    assert np.all(qp_n.u[lo:hi] == cfg.w_max)


# ---------------------------------------------------------------------------
# 4. Production risk term must carry the L_j/240 scaling (hidden-bug guard)
# ---------------------------------------------------------------------------


def test_production_risk_scales_with_stage_length():
    """Single-name single-stage problems against their analytic optimum.

    Interior first-order condition with ``w0 = 0`` (so ``b = w``, ``s = 0``):

        alpha = c_buy + gamma*(L/240)*sigma^2*w + D*w

    A 30-minute stage is charged 1/8 of a day's risk, so it must hold a
    materially *larger* position than an otherwise identical 240-minute stage.
    Dropping the ``L_j/240`` factor (or applying it twice) breaks both the
    analytic value and the ordering below.
    """
    cfg = MpoConfig(w_max=1.0)
    sigma2 = 6.25e-4  # (2.5% daily vol)^2
    sigma = np.array([[sigma2]])
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
        qp, sol = solve_problem(
            alpha, grid, np.zeros(1), np.zeros(1), cfg, Sigma=sigma, adv_frac=adv_frac
        )
        assert sol.status == "optimal"
        assert qp.meta["risk_scale"] == (length / 240.0,)
        analytic = (alpha[0, 0] - cfg.buy_cost()) / (
            cfg.gamma * (length / 240.0) * sigma2 + impact
        )
        assert abs(float(sol.weights[0, 0]) - analytic) < 1e-6, (length, analytic)
        held[length] = float(sol.weights[0, 0])

    assert held[30] > 1.5 * held[240], held


# ---------------------------------------------------------------------------
# 5. Price limits
# ---------------------------------------------------------------------------


def test_price_limit_masks_zero_stage1_buy_and_sell():
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    n = 6
    w0 = np.full(n, 0.01)
    alpha = np.full((4, n), 20e-4)
    alpha[:, 1] = -20e-4  # name 1 wants out, everybody else wants in
    limit_up = np.zeros(n, bool)
    limit_up[0] = True
    limit_down = np.zeros(n, bool)
    limit_down[1] = True

    _, control = solve_problem(alpha, grid, w0, w0.copy(), cfg)
    qp, masked = solve_problem(
        alpha, grid, w0, w0.copy(), cfg, limit_up=limit_up, limit_down=limit_down
    )
    assert control.status == "optimal" and masked.status == "optimal"
    # Control: both trades happen when no limit is hit.
    assert control.buys[0, 0] > 1e-4
    assert control.sells[0, 1] > 1e-4
    # Masked: stage-1 buy of the limit-up name and sell of the limit-down name die.
    assert abs(float(masked.buys[0, 0])) < 1e-8
    assert abs(float(masked.sells[0, 1])) < 1e-8
    # Later stages are untouched by the masks (limits are a stage-1 state).
    b_lo, b_hi = qp.blocks["b_box"]
    assert qp.u[b_lo] == 0.0  # b^1 of name 0
    assert np.isinf(qp.u[b_lo + n])  # b^2 of name 0
    assert masked.buys[1:, 0].max() > 1e-4 or masked.weights[1:, 0].max() > 1e-4


# ---------------------------------------------------------------------------
# 6. Constraint compliance + fail-soft
# ---------------------------------------------------------------------------


def test_budget_and_box_constraints_hold():
    cfg = MpoConfig()  # w_max = 0.02, budget <= 1
    grid = build_stage_grid(35, 100, cfg)
    n = 60  # 60 * 0.02 = 1.2 > 1, so the budget row must bind somewhere
    rng = np.random.default_rng(99)
    alpha = np.abs(rng.normal(scale=40e-4, size=(grid.n_stages, n))) + 20e-4
    _, sol = solve_problem(alpha, grid, np.zeros(n), np.zeros(n), cfg)
    assert sol.status == "optimal"
    tol = 1e-6
    assert sol.weights.max() <= cfg.w_max + tol
    assert sol.weights.min() >= -tol
    assert sol.buys.min() >= -tol and sol.sells.min() >= -tol
    assert np.all(sol.weights.sum(axis=1) <= 1.0 + tol)
    assert sol.weights.sum(axis=1).max() > 0.9  # binding, not vacuous
    # dynamics identity per stage
    prev = np.zeros(n)
    for j in range(grid.n_stages):
        assert np.allclose(sol.weights[j] - prev, sol.buys[j] - sol.sells[j], atol=1e-6)
        prev = sol.weights[j]


def test_scipy_fallback_agrees_with_osqp():
    """Last-resort dense backend reproduces the primary backend's solution."""
    cfg = MpoConfig(qp_eps=1e-9)
    grid = build_stage_grid(5, 100, cfg)
    n = 5
    alpha = np.full((grid.n_stages, n), 20e-4)
    qp, ref = solve_problem(alpha, grid, np.zeros(n), np.zeros(n), cfg)
    assert ref.status == "optimal"
    sol = solve_mpo_qp(qp, backend="scipy", cfg=cfg)
    assert sol.status == "optimal", getattr(sol, "note", sol.status)
    assert float(np.max(np.abs(sol.weights - ref.weights))) < 1e-6


def test_fail_soft_on_infeasible_problem():
    """Infeasible bounds must return a non-optimal status, not raise."""
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    n = 5  # 5 * w_max = 0.10 < required 0.50 invested
    _, sol = solve_problem(
        np.zeros((4, n)), grid, np.zeros(n), np.zeros(n), cfg, budget_lo=0.5
    )
    assert sol.status != "optimal"
    assert sol.weights is None and sol.objective is None
    assert sol.solve_time >= 0.0


def test_fail_soft_on_bad_backend_and_bad_input():
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    n = 3
    qp, _ = solve_problem(np.zeros((4, n)), grid, np.zeros(n), np.zeros(n), cfg)
    assert solve_mpo_qp(qp, backend="not-a-solver", cfg=cfg).status == "failed"
    assert solve_mpo_qp("not a qp", cfg=cfg).status == "failed"  # type: ignore[arg-type]


def test_build_rejects_unknown_override_and_bad_shapes():
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    n = 4
    args = (
        np.zeros((4, n)),
        grid,
        np.zeros(n),
        np.zeros(n),
        random_spd(n, seed=1),
        np.full(n, 0.8),
        np.full(n, 0.025),
        np.zeros(n, bool),
        np.zeros(n, bool),
        cfg,
    )
    with pytest.raises(ValueError, match="unknown build_mpo_qp overrides"):
        build_mpo_qp(*args, w_maxx=0.05)
    with pytest.raises(ValueError, match="alpha must be"):
        build_mpo_qp(np.zeros((3, n)), *args[1:])
    with pytest.raises(ValueError, match="Sigma_d must be"):
        build_mpo_qp(args[0], grid, args[2], args[3], np.eye(n + 1), *args[5:])


def test_unpack_and_objective_are_consistent():
    """MpoQp.unpack / objective_value agree with the reported solution."""
    cfg = MpoConfig()
    grid = build_stage_grid(5, 100, cfg)
    n = 5
    alpha = np.full((grid.n_stages, n), 30e-4)
    qp, sol = solve_problem(alpha, grid, np.zeros(n), np.zeros(n), cfg)
    assert sol.status == "optimal"
    x = np.concatenate(
        [sol.weights.reshape(-1), sol.buys.reshape(-1), sol.sells.reshape(-1)]
    )
    w, b, s = qp.unpack(x)
    assert np.array_equal(w, sol.weights) and np.array_equal(b, sol.buys)
    assert np.array_equal(s, sol.sells)
    assert qp.objective_value(x) == pytest.approx(sol.objective, rel=1e-9, abs=1e-12)
    assert sol.objective < 0.0  # profitable alpha net of costs
    with pytest.raises(ValueError):
        qp.unpack(x[:-1])
