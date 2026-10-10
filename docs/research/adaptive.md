# Optional Dense QP Adaptive Rho

## Contract

Only residual-balanced scalar rho is implemented. Diagonal scaling is deferred:
it would require a separately verified original-coordinate primal/dual mapping
and stopping contract. This change is not GPU or commercial qualification.
The main integration owner must update `docs/qp-design.md` and generated reports;
those files are outside the adaptive seat's exclusive write set.

All pre-existing `QPConfig` defaults are unchanged. New fields are appended:

| Field | Default | Meaning |
| --- | --- | --- |
| `adaptive_rho` | `False` | Explicit opt-in |
| `adaptive_rho_interval` | `100` | Positive integer update cadence |
| `adaptive_rho_tolerance` | `5.0` | Required multiplicative change, strictly > 1 |
| `rho_min` | `1e-6` | Finite positive lower bound |
| `rho_max` | `1e6` | Finite positive upper bound |

Initial rho must lie within the bounds when adaptation is enabled; disabled
legacy rho values remain valid. Results append `rho_updates` and `final_rho`.
Disabled solves retain exactly the frozen baseline's tensors, status, iteration
count, objective and residuals, checked in float32/float64 with both factor
application paths against Git object `fcefac1:src/openopt/qp.py`.
This equivalence applies to normal finite arithmetic, not known overflow bugs.

## Update And Failure Boundary

At each update interval, normalize `||Ax-z||inf` by
`max(||Ax||inf, ||z||inf, eps_abs)` and `||Px+q+A'y||inf` by
`max(||Px||inf, ||q||inf, ||A'y||inf, eps_abs)`.
Multiply rho by the square root of their ratio, clip to the explicit bounds,
then apply the multiplicative change threshold. Log arithmetic avoids overflow;
zero/zero retains rho, a sole zero selects the corresponding bound subject
to that same threshold. This follows OSQP's residual-balancing principle,
not its time-based scheduling, row-wise penalties or internal scaling.
Reference: OSQP official solver documentation, section "rho step-size";
`https://osqp.org/docs/solver/index.html` (official source
`osqp/osqp/docs/solver/index.rst`, read on 2026-10-10).
the installed official OSQP 1.0.4 `auxil.c::compute_rho_estimate/adapt_rho`
provides the normalization and threshold implementation reference.

Update and termination intervals are independent. Adaptation never enables
early termination between the original `check_interval` checks. Original
primal, stationarity and normal-cone stopping criteria are unchanged.
No update occurs after a solved check or the final iteration.
Unconstrained problems skip adaptation because rho cannot alter their system.

Each accepted update refactors `P + rho A'A + sigma I`; the finite system,
Cholesky info and finite factor are checked. Failure raises with iteration and
rho; backend exceptions retain their first cause. No old factor is silently
used with a new rho. The unscaled multiplier `y` is not rescaled.
Rho/factor updates are local to one solve; prepared solves restart from the
initial factor and rho. The adaptive workspace additionally owns cached `A'A`.
Explicit CUDA validation is unchanged; there is no fallback.

## Reproduction

Run from `/data/dev/openopt-wt-qp3-20261010-adaptive`:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python scripts/verify.py --target tests/test_qp_adaptive.py --artifact-dir /3fs/research3fs/data/tmp/openopt-qp3-20261010-adaptive
```

Verification creates unique external directories with complete logs, JUnit
cases/properties and terminal `verification.json`. CPU ill-conditioned
diagonal QPs span eight orders of quadratic curvature with upper-active,
lower-active and equality witnesses. Both linear solver paths must match
OSQP and independent original-coordinate NumPy KKT calculations. Fixed tiny
rho is a deliberately poor negative control, not a tuned performance baseline.
Other negative controls reject invalid settings, injected Cholesky info/error/
nonfinite factors, unavailable explicit CUDA, infeasible and unbounded false
certificates. No local GPU or cluster operations are permitted in this seat.

Timing speedup, general rho robustness, float32 adaptive accuracy, diagonal
scaling, actual CUDA execution and commercial comparisons are not qualified
by these CPU tests. GPU execution belongs to the main agent's <=3-card runner.

## First Failure And Repair

The first targeted run `tests-913txf71` under the artifact root above failed
with pytest rc=1: 30 passed, 7 failed. Full `pytest.log`, JUnit traceback and
terminal FAIL are retained. Six failures were the oracle's weak-curvature
solution error: OSQP reported solved with `eps_rel=1e-10` and polishing, but
its first coordinate differed from the analytic witness by about 0.018
(upper/lower) or 0.0074 (equality). Openopt's independent KKT and witness
checks already passed. The changed oracle removes relative tolerance and
polishing; the solution/objective acceptance thresholds are not relaxed.
The seventh failure found an unnecessary unconstrained rho refactor; skipping
adaptation for zero constraint rows removes it. This is a changed-code/oracle
rerun, not a blind retry or promotion of the failed evidence.

The second run `tests-dgie1qww` failed rc=1 with 35 passed, 2 failed:
only the equality oracle hit OSQP iteration-limit status 7 at absolute
tolerance `1e-12`, below the cancellation scale of its roughly 20,000-sized
stationarity terms. The final oracle uses absolute `1e-10`, relative zero,
still requires `<1e-4` solution agreement, and now explicitly verifies the
oracle itself against the analytic witness. Neither prior FAIL is overwritten.
Additional seeded coupled dense QPs exercise non-diagonal curvature and
rectangular constraints, with planted primal/dual witnesses and OSQP.

## Independent Overflow Repair

Main review identified unchecked residual inputs, tolerances and objective.
A real CPU reproduction, `P=I`, `q=[-1e200]`, zero constraint rows, returns
finite x/y and `status="solved"` from frozen `fcefac1` despite a nonfinite
objective. The pre-repair red test is retained as `tests-mnur6th6`, pytest rc=1.
This is an explicit default error-path correction, not an adaptive option
or an attempt to preserve unsafe baseline behavior. Finite normal solves remain
bitwise unchanged. Another real case uses `eps_rel=1e308, q=[-3]` and proves
the old stopping tolerance overflows while the old status is solved.

Before any solved decision, require finite Ax/z/Px/A'y, all three residuals,
and both stopping tolerances. Before any return, require a finite objective.
Failure raises with iteration/rho; no `inf <= inf` comparison can accept a
nonfinite residual/tolerance. A backend-output injection separately exercises
finite iterates with overflowing Px. These checks apply with adaptation both
disabled and enabled; they never change device or solver.
The first post-repair run `tests-apsyksty` retained rc=1 (43 passed, 1 failed):
the finite-x injection used bounded constraints and also overflowed y, reaching
the objective guard rather than the intended residual-input guard. Its
corrected fixture uses an unconstrained scalar P=10,000 so y stays empty/
finite and only Px overflows. Both real overflow controls already passed.

## Accepted CPU Evidence

All directories below are under
`/3fs/research3fs/data/tmp/openopt-qp3-20261010-adaptive/`.

| Run | Terminal Evidence |
| --- | --- |
| `tests-f4g2axog` | PASS, 44 tests, no skips; finished 2026-10-10T07:52:26Z |
| `smoke-9ewp5gh4` | PASS, 79 passed and 4 CUDA skips; finished 2026-10-10T07:52:48Z |

For both linear paths, active upper/lower cases need 150 iterations and
3 rho updates, final rho 1772.24735, primal residual 6.54e-10, stationarity
6.96e-7, and OSQP objective gap 6.54e-6. Equality needs 125 iterations and
4 updates, final rho 11288.5874, primal 1.94e-13, stationarity 3.65e-9 and
objective gap 1.27e-11. These residuals meet the unchanged original-coordinate
absolute/relative tolerances. The deliberately poor fixed-rho control reaches
5000 iterations without solving. No time-based or GPU speedup is inferred.
Raw counts, properties, first-cause failures and dependency/device metadata
remain in each run's `pytest.xml`, `pytest.log` and `verification.json`.
