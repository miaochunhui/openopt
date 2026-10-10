# Independent Dense QP Numerical Evidence

This module checks supplied candidates for
`min 0.5 x.T P x + q.T x` subject to `l <= A x <= u`.
It does not generate certificates, solve another optimization problem, alter
solver state, or turn `max_iterations` into infeasible/unbounded.
**A passing report is approximate numerical evidence, not an exact proof,
certified lower bound, or production qualification.**

## API And Boundary

Import directly from `openopt.certificates`; the package facade and solver are
unchanged:

```python
from openopt.certificates import (
    check_qp_candidate, check_primal_infeasibility, check_unbounded_direction,
)

candidate = check_qp_candidate(problem, x, y, atol=1e-8, rtol=1e-8)
farkas = check_primal_infeasibility(problem, v)
ray = check_unbounded_direction(problem, d, feasible_x=x0)
report = candidate.to_dict()
```

`problem` has the dense `P/q/A/l/u` attributes of `QPProblem`. NumPy numeric
arrays and CPU dense real tensors are copied into independent float64 arrays.
Non-CPU tensors are rejected, never silently transferred. No solver residual,
projection variable, cached factor, scaling, or regularization is consulted.
No GPU work, cluster access, optional solver installation, or credentials are
required. Empty constraint sets are supported; empty decision vectors are not.

Malformed dimensions, nonfinite matrices/candidates, NaN/reversed bounds,
lower `+inf`, upper `-inf`, complex/sparse data, and invalid tolerances raise.
`atol > 0` and `0 <= rtol < 1` must be finite. Arithmetic overflow/underflow raises
with the original arithmetic exception chained; it never returns a NaN report.
Well-formed but incorrect candidates return `passed=False` and named checks.
Zero directions fail. Each nonzero direction is normalized by its infinity
norm before testing, making positive rescaling immaterial.

Reports contain `checks`, `metrics`, `thresholds`, `unavailable`, and an explicit
qualification. Unavailable objective/gap values are `None` (JSON `null`);
all reported numbers are finite. Feasibility, stationarity and complementarity
are checked componentwise; the reported maximum threshold is descriptive,
not the acceptance rule. A huge unrelated bound cannot mask another row's
feasibility error. Negative gaps are preserved, not clipped.

## Candidate And Dual Objective

Use the signed multiplier convention `y > 0` for upper bounds, `y < 0` for
lower bounds, and unrestricted signs for equalities.

The independent checks are original-coordinate bound violations,
`P x + q + A.T y`, bound-sign admissibility, complementarity, dual availability,
and an absolute/relative signed-gap test. For each finite side, complementarity
uses `y_i * (bound_i - (Ax)_i)` for the side selected by the sign of `y_i`.
Nonzero multipliers of an absent infinite side fail **without tolerance**,
including tiny wrong-sign multipliers. Zero multipliers are masked before
any endpoint product, avoiding `0 * infinity`.

Write `b = q + A.T y` and
`support(y) = sum(u_i*y_i for y_i>0) + sum(l_i*y_i for y_i<0)`.
The Lagrangian infimum is

```text
g(y) = -0.5 b.T P^dagger b - support(y),  when b belongs to range(P).
g(y) = -infinity, otherwise (or when support is infinite).
gap  = primal_objective - g(y).
```

Derivation: diagonalize symmetric PSD `P = V diag(lambda) V.T`. In a positive
eigendirection, minimizing `0.5 lambda z^2 + (V.T b)_i z` contributes
`-0.5 (V.T b)_i^2 / lambda`. In a zero eigendirection a nonzero linear
coefficient makes the infimum negative infinity. This is the value-range
condition; simply computing a pseudoinverse value without it is incorrect.
For an LP (`P=0`), numerical dual availability requires `q + A.T y` to be zero
within the stated range tolerance.

The checker explicitly computes `||b - P P^dagger b||_inf` and compares it
with `atol + rtol ||b||_inf`. It never adds `sigma I`, inverts a regularized
matrix, or substitutes `-0.5 x.T P x - support(y)` when stationarity fails.
The finite dual value is a numerical estimate conditional on the reported
range check, **not a rigorously certified lower bound**: a nonzero nullspace
component below tolerance can still mean the mathematical infimum is
negative infinity. Exact/interval arithmetic is outside this API.

Spectral decisions are deliberately conservative. Eigenvalues below
`-64*eps64*n*max(abs(lambda))` reject nonconvex data. Only exact computed zeros
and positive eigenvalues above that threshold form a resolved spectrum.
Tiny positive or slightly negative eigenvalues at the threshold yield
`unavailable` rather than being silently treated as zero; no small positive
curvature is turned into an unbounded ray. This may reject usable numerical
witnesses on rotated singular or ill-conditioned data. Symmetry is checked
at the same roundoff scale; only the spectral analysis uses the symmetric
average, while original-coordinate residuals use the original `P`.

## Directions And Strict Sign Checks

A primal-infeasibility witness `v` must satisfy

```text
A.T v = 0
support(v) < 0
```

The numerical annihilation test uses `atol` on the normalized direction, and
the support must be less than
`-(atol + rtol * sum(abs(endpoint_products)))`. Zero and tiny separation
margins do not pass. Infinite-support signs fail strictly. This implements
a numerical separating direction, not an exact Farkas proof.

A descent/recession direction `d` must satisfy

```text
P d = 0,  q.T d < 0
(A d)_i = 0   if both bounds are finite
(A d)_i >= 0  if only the lower bound is finite
(A d)_i <= 0  if only the upper bound is finite
no condition  if both bounds are infinite
```

Two-sided row equalities use normalized `atol`; one-sided signs are strict.
Descent uses the same strict margin as separation. Curvature requires a
resolved spectrum, `||P d||_inf <= atol`, and positive-eigenspace coordinates
no greater than `64*eps64*n`. Thus even `P=[1e-12]` cannot pass a unit ray
just because `||P d||` is below an ordinary feasibility tolerance.

Primal unboundedness additionally requires a supplied `feasible_x`, checked
independently in original coordinates. Without that base, the direction
checks remain visible but `passed=False` and the base is unavailable.
An infeasible primal problem can also have a valid dual-infeasibility direction;
the direction alone must not be called primal unboundedness.
Tolerance-level equalities/base feasibility remain numerical approximations:
even a tiny nonzero equality residual can invalidate an infinite exact ray.
The API makes no exact, interval, or solver-termination claim.

## Primary Sources

Official sources read on 2026-10-10:

- OSQP, *The solver*, "Primal infeasibility" and "Dual infeasibility":
  `https://osqp.org/docs/solver/index.html`
  and the official source
  `https://raw.githubusercontent.com/osqp/osqp/master/docs/solver/index.rst`.
  These give the signed box-support, stationarity, and recession equations.
- MOSEK Optimizer API for Python 11.2.6, section 12.4.2, equation (12.24),
  and section 12.4.3:
  `https://docs.mosek.com/11.2/pythonapi/prob-def-quadratic.html`.
  Its stationarity-constrained quadratic dual and additional `Q^o x = 0`
  dual-infeasibility condition agree after translating multiplier signs.
  The range formula above is derived explicitly here, not attributed to a
  remembered pseudoinverse formula in either source.

The browser/search tool returned empty responses; independent read-only HTTP
retrieval succeeded for both official pages and OSQP's source. No mathematical
condition was inferred from a gateway success code.

## Acceptance

CPU-only, existing reference environment, no dependency installation:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python scripts/verify.py \
  --target tests/test_qp_certificates.py --target tests/test_general_qp.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-20261010-certificates/acceptance
```

The tests include a planted dense scaled-row witness compared with OSQP,
singular PSD range/non-range, LP, unconstrained/free rows, equality and one-sided
bounds, both multiplier signs, overflow, malformed/nonfinite inputs, missing
feasible bases, and tiny positive curvature. Named mutation controls reject
wrong primal/dual candidates, broken annihilation, wrong ray curvature/signs,
and missing bases. The final test writes `numerical-acceptance.json` into the
external pytest temporary directory, records its path in JUnit XML, and checks
the artifact round-trip, UTC timestamp, source hashes, nonempty cases, and
expected verdicts. Additional tests patch solver and inverse/Cholesky entries
to fail if consulted, cover rotated singular PSD, and reject even tiny
wrong-sign one-sided recession directions.
`verification.json` and `pytest.xml` give actual test counts, failures, skips
and run times. Skipped GPU tests are not GPU qualification.

Scope is exactly `src/openopt/certificates.py`,
`tests/test_qp_certificates.py`, and this document; `qp.py` and `__init__.py`
are read-only. No solver algorithm/parameter or benchmark methodology changes,
and no generated reports are edited. The infra rule router rejected this
worktree's edit target as outside its own root (exit 2); this is recorded as a
scope mismatch, not treated as permission. Applicable openopt rules and the
explicit three-file assignment govern these edits. An overbroad initial
artifact search encountered unrelated 3FS permission errors and was stopped
(exit 130); subsequent evidence collection is confined to this task directory.

The first targeted run (`acceptance-v1/tests-g9f4mwqh`) retained a failed
artifact: 103 tests passed, one failed because a NumPy boolean from the
curvature check was not JSON serializable. Report construction now explicitly
normalizes booleans and finite floats; acceptance asserts native field types
and round-trips positive, negative and unavailable reports. Original failure
logs are retained; changed-code verification uses a new artifact directory.
The changed-code v2 targeted run (`acceptance-v2/tests-gm8vu0of`) passed all
104 tests with no skips before the additional independence/sign controls.
V3 passed 108 tests. Final review added extreme-range normalization controls:
`[1e300, -1e-300]` must not erase a nonzero infinite-support component during
division, nor may a ray lose its nonzero component silently.
`underflow-red/tests-dh1du_ci` retained both expected regression failures
(`DID NOT RAISE`, exit 1). Arithmetic now explicitly raises on underflow too;
changed-code verification uses v4 rather than overwriting red evidence.
