# Optional Reversible Dense QP Scaling

This research wrapper is opt-in via `openopt.scaling`. It calls the existing
dense QP solver, restores primal and signed dual variables, and independently
checks the candidate against an untouched original-coordinate CPU float64
snapshot. Neither `openopt.qp` nor the package facade/default API is changed.
Passing evidence is numerical and approximate, not an exact certificate,
commercial qualification, or performance result.

## API And Coordinates

```python
from openopt.qp import QPConfig
from openopt.scaling import ScalingConfig, equilibrate_qp, solve_scaled_qp

result = solve_scaled_qp(problem, QPConfig(), ScalingConfig(iterations=10))
assert result.status == "solved" and result.evidence.passed
x, y = result.x, result.y                 # original coordinates
scaled_status = result.scaled_result.status

prepared = equilibrate_qp(problem)
x_hat, y_hat = prepared.scaling.to_scaled(x, y)
x_again, y_again = prepared.scaling.restore(x_hat, y_hat)
identity = solve_scaled_qp(problem, scaling=ScalingConfig(method="none"))
```

For positive diagonal `D`, `E` and positive scalar objective factor `c`:

```text
P_hat = c D P D       q_hat = c D q
A_hat = E A D        l_hat = E l       u_hat = E u
x = D x_hat          y = E y_hat / c
x_hat = D^-1 x       y_hat = c E^-1 y
objective_hat = c objective
```

Indeed, the scaled stationarity equation is
`c D P D x_hat + c D q + D A.T E y_hat = 0`.
Substituting the restoration formulas gives
`c D (P x + q + A.T y) = 0`. Positive factors preserve finite-bound
ordering, infinite sides and multiplier signs: positive `y` denotes an upper
bound, negative `y` a lower bound, and equality multipliers are unrestricted.
Dropping `c` or using `E^-1` to restore `y` is incorrect.

`DiagonalScaling` also supports explicit user-supplied factors and `apply`.
Factors own their input copies. Tensor fields remain mutable, so every
application/restoration validates them again; the wrapper itself never
mutates caller inputs.

## Bounded Ruiz Procedure

Each of the requested iterations measures infinity norms of variable columns
and constraint rows of the symmetric KKT block `[P A.T; A 0]`. The variable
norm is `max(max_i |P_ij|, max_i |A_ij|)`; the constraint norm is
`max_j |A_ij|`. Each diagonal update uses the inverse square root of its
norm. Exact zero rows/columns receive an identity update, not a division by
zero. Accumulated factors are clipped to the caller's finite positive limits
in logarithmic arithmetic before computing the actual incremental step.

When enabled, objective normalization follows each diagonal update using
`max(max_ij |P_ij|, max_j |q_j|)` and its reciprocal. The accumulated `c`
has the same explicit bounds. This is a bounded infinity-norm Ruiz variant,
not a claim of bitwise equivalence to OSQP's preprocessing. Bounds and
objective data are transformed with the actual clipped increments, not
unclipped proposals. No convergence or speedup is guaranteed.
When a large constraint row competes with much smaller curvature in one
variable, balancing KKT entry magnitudes can leave strongly unequal curvature.
The retained original extreme-unit case is feasible with a known KKT witness
but reaches `max_iterations` with fixed rho. Explicit adaptive rho solves
that same problem; no automatic rho selection is introduced. This wrapper
does not resolve all ill-conditioned cases or substitute for preconditioning.

Defaults are ten iterations and factors in `[1e-6, 1e6]`.
`method="none"` performs no equilibration and calls the same dense path.
Objective scaling can be disabled independently. Solver tolerances are in
scaled coordinates; `check_atol/check_rtol` are separate original-coordinate
candidate tolerances, defaulting to `1e-7`. The independent check additionally
requires dual availability and a numerical objective gap, unlike dense ADMM
termination. Tolerance-level feasibility/stationarity is never an exact proof.

## Validation And Failure Semantics

Empty constraints, LP/PSD models, zero rows/columns, equality, one-sided and
fully free rows are supported. Empty decision vectors, malformed dimensions,
non-tensors, complex/sparse inputs, nonconvex/asymmetric data, nonfinite
matrices/candidates, NaN/reversed/invalid infinite bounds, nonpositive or
nonfinite factors, invalid iterations/tolerances and nonrepresentable
reciprocals raise.

Every scaling/restoration product checks overflow and loss of nonzero values
to zero. Finite bounds may not silently become infinite, and dtype conversion
may not erase nonzero model coefficients. Nonzero representable subnormals
are retained; the existing independent checker may conservatively reject
underflow during its own arithmetic. Intermediate overflow/zero-underflow is
rejected even when a different multiplication order could avoid it; this
wrapper is not arbitrary-precision arithmetic or a dynamic rescaling retry.

The raw dense result is retained separately. If its status is `solved` but
the independent original check fails, wrapper status is `candidate_rejected`.
An iteration limit is never promoted, even if a planted candidate happens
to pass. Invalid arithmetic raises instead of retrying unscaled or changing
device. The reported objective comes from original-coordinate evidence;
scaled residuals remain only in `scaled_result`, never relabeled as original
residuals.

Independent verification explicitly copies original data and restored
candidates to CPU float64 and recomputes feasibility, stationarity,
complementarity, bound signs, dual objective and gap. It consults no scaled
residual, status, factorization or regularization. An ill-conditioned
original PSD spectrum may leave the dual unavailable and reject an otherwise
useful candidate. Float32 solutions are checked against the original values,
not a rounded float32 model. Explicit CUDA requests retain the dense solver's
strict availability check; GPU execution of this wrapper is unverified here.

## Acceptance And Reproduction

Only the existing CPU environment is used, without installation:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/iteration_gate.py \
  smoke --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/scaling/smoke
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py \
  --target tests/test_qp_scaling.py --target tests/test_general_qp.py \
  --target tests/test_qp_certificates.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/scaling/acceptance
```

The suite checks explicit equations and both multiplier signs, known KKT
solutions, OSQP in original coordinates, a disparate-unit ill-conditioned
QP, identity-path parity, semidefinite/LP/empty/zero structure, bounds and
factor limits, bad dimensions/nonfinite/extreme negative controls,
input ownership, strict CUDA requests, original-versus-rounded checking,
scaled-success/original-failure rejection and no status promotion.
The machine-readable test writes external `scaling-acceptance.json`, records
its path in JUnit, and round-trips finite JSON with source hashes, original
candidates/evidence, OSQP evidence and wrong-primal/wrong-dual controls.
`pytest.xml` and `verification.json` provide actual counts and run outcomes.
The explicit CUDA test covers known KKT in float32/float64 and the
ill-conditioned case in float64, verifies all result/factor tensor placements,
restores and independently checks original coordinates against known KKT
witnesses and numerical duality gaps, rejects
wrong dual signs, and writes `cuda-scaling-acceptance.json` externally. Only
these three cases skip when CUDA is unavailable; a coordinator may run this
same file through the authorized 4500 K8s route. No GPU run is claimed here.
`test_cuda_original_coordinate_restoration_kkt_only` imports neither OSQP nor
SciPy. Both imports are local to the CPU `reference` helper, so a GPU image
without them can collect and run `tests/test_qp_scaling.py -k cuda` without
installation. CPU OSQP matched evidence remains separate and mandatory in
the complete CPU acceptance run.

The seat's own preflight was 79 passed and four CUDA skips. Initial targeted
v1 evidence is retained under `acceptance-v1/tests-jgkdyz1q` (197 passed,
seven failed, three CUDA skips): a redundant-active-constraint fixture
incorrectly asserted a unique dual multiplier, one negative control had
insufficient scale separation, and the extreme-unit QP did not converge.
V2 retained the two nonconvergence failures (202 passed, three CUDA skips)
under `acceptance-v2/tests-028wcccm`; the diagnostic reproduction is
`ill-diagnostic/tests-q_je4ztl`. V3 under `acceptance-v3/tests-kje0c92p`
(204 passed, three CUDA skips) changed a row unit and therefore did **not**
close the original failure. The final suite restores the original case
exactly, leaves original oracle tolerances at `atol=rtol=1e-7`, and retains
fixed-rho negative results. Historical failures and v3 are not substituted
for acceptance of the original problem.

## Original-Problem Rho Ablation

Four matched runs are retained under
`ablation-v1/tests-2w5p40__/pytest-tmp/test_original_ill_conditioned_{0,1,2,3}/rho-cost-ablation.json`
and linked by JUnit. All use the same problem, thirty Ruiz iterations,
solver `eps_abs=eps_rel=1e-10`, 20,000 maximum iterations, and unchanged
original-coordinate candidate tolerances. An ablation test passing means
its diagnostic expectations held, not that every solver run succeeded.

| Rho | Cost Scaling | Status | Iterations | Original Gap |
| --- | --- | --- | --- | --- |
| Fixed 1 | Off | max_iterations | 20000 | 8.0443929741 |
| Fixed 1 | On | max_iterations | 20000 | 8.0443929741 |
| Adaptive | Off | solved | 7625 | 9.5882342e-8 |
| Adaptive | On | solved | 7625 | 9.5882342e-8 |

Measured scaled `P` is approximately
`[[2e-6, 5e-4], [5e-4, 1]]`, with condition number `571428.86`;
the normalized constraint rows include `[1, 0]` and `[0.001, 1]`.
Balancing entries has not eliminated the slow curvature direction.
Fixed rho overweights the constraint penalty relative to that direction.
Cost scaling is not the observed cause: its final factor is approximately
one and switching it off leaves iterations and gap effectively unchanged.
Adaptive rho reduces the penalty, with nine updates and final
`rho=0.00165969037487`, and the same original-coordinate checker passes:
feasibility `9.5882e-8`, stationarity `1.9149e-7`, complementarity
`9.5882e-8`, gap `9.5882e-8`. Stationarity and feasibility acceptance is
componentwise with the unchanged absolute/relative thresholds, not the
global maximum printed in the report. OSQP independently passes this same
original model.

The test-only explicit configuration is `adaptive_rho=True`,
`adaptive_rho_interval=25`, `adaptive_rho_tolerance=2`, `rho_min=1e-10`,
with all other dense parameters unchanged. Known-KKT cases still exercise
fixed rho. Original ill-conditioned CPU/CUDA acceptance and machine-readable
evidence use the explicit adaptive configuration; the wrapper does not enable
it on callers' behalf. The remaining fixed-rho failure is a documented
limitation, not a claimed general convergence improvement or speed result.

## Final Evidence

The original-problem v4 targeted run is
`/hf3fs-jd/data/tmp/openopt-release-20261010/scaling/acceptance-v4/tests-eqzqv3zw/`:
JUnit contains 214 cases, 211 passed, three explicit CUDA skips, zero failures
and zero errors. Its external `scaling-acceptance.json` contains two original
positive candidates and four wrong-primal/wrong-dual negative controls, with
matching source hashes. The four `rho-cost-ablation.json` files preserve
fixed-rho failures alongside adaptive successes on the unchanged problem.
Final CPU smoke is
`/hf3fs-jd/data/tmp/openopt-release-20261010/scaling/final-smoke/smoke-w8sa4ak4/`:
83 cases, 79 passed, four CUDA skips, zero failures/errors. These runs overlap
and must not be summed as distinct coverage.

After adding this evidence entry, the same complete targeted command is
rerun under the external `acceptance-v5` directory to pin this document's
final hash as well. The unique run's `verification.json`, `pytest.xml` and
JUnit-linked candidate/ablation reports are the final artifact authority.
The full repository regression and actual CUDA execution remain unverified
by this seat; no speed, commercial or exact-certificate claim is made.

Scope is exactly `src/openopt/scaling.py`, `tests/test_qp_scaling.py`, and this
document. The explicit scoped assignment takes precedence over the generic
repository requirement to update design/report pages: those pages are outside
this seat's exclusive write set and must be integrated by the coordinator.
No main-tree, infrastructure, GPU submission, installation, credential,
nested-agent or remote-push operation is part of this work.
