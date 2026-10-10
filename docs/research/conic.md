# Explicit Conic Model And CPU Reference

This module adds a dense convex conic **contract**, independent numerical
candidate checks, and an explicitly selected third-party Clarabel CPU reference.
It is not an Openopt-native conic solver, GPU implementation, certified optimizer,
commercial comparison, or performance qualification. No native GPU SOC prototype
is included: correctness and honest scope take priority over simulated coverage.

## Model And API

Import directly from `openopt.conic`; this scoped change does not edit the package
facade, dependencies, QP algorithm, shared reports, or other seats' files.

```python
import numpy as np
from openopt.conic import ConicProblem, ConicConfig, Zero, Nonnegative, SOC, solve_conic

# min t, u=3, v=4, (t,u,v) in Lorentz cone; analytic optimum (5,3,4).
problem = ConicProblem(
    P=np.zeros((3, 3)), q=np.array([1., 0., 0.]),
    A=np.vstack([[[0., 1., 0.], [0., 0., 1.]], -np.eye(3)]),
    b=np.array([3., 4., 0., 0., 0.]),
    cones=(Zero(2), SOC(3)),
)
result = solve_conic(problem, ConicConfig(backend="clarabel_reference", device="cpu"))
```

The contract is

```text
minimize  0.5 x.T P x + q.T x
subject to A x + s = b, s in K = K0 x K1 x ...
```

`P` is a full symmetric PSD `(n,n)` matrix, `q` is `(n,)`, `A` is `(m,n)`,
and `b` is `(m,)`. Nonempty decision vectors are required. An unconstrained
problem uses explicit `(0,n)` A, `(0,)` b, and `cones=()`; a size-0 product cone
is valid, not a missing model. Dense real CPU Torch tensors, NumPy arrays, and
numeric sequences are copied to float64. Nonfinite/complex/sparse data and
non-CPU tensors are rejected; no implicit host transfer occurs.

Cone blocks describe **consecutive rows in the supplied order**; dimensions must
sum to `m`. Repeated or mixed blocks are allowed. Cone aliases and their full
names are `Zero`/`ZeroCone`, `Nonnegative`/`NonnegativeCone`, and
`SOC`/`SecondOrderCone`:

- `Zero(d)`, d >= 1: s=0, thus Ax=b. The dual is unrestricted R^d.
- `Nonnegative(d)`, d >= 1: s>=0, thus Ax<=b. Its dual is itself.
- `SOC(d)`, d >= 2: s=(t,u), t>=||u||_2. Its dual is itself.

SOC has its **axis first**, not last. To model x in SOC, use A=-I, b=0.
Rotated SOC, exponential/power/PSD cones, sparse operators, batching, autograd,
and arbitrary dtype/device dispatch are not supported or inferred.

## Independent Numerical Checks

`check_cone_feasibility(vector, cones, dual=False, atol=..., rtol=...)` requires no
Clarabel or SciPy. Zero residuals use absolute tolerance; Nonnegative signs use
componentwise limits; SOC uses `max(||u||_2-t,0)` with a block-local scale.
A huge unrelated coordinate/block cannot hide another constraint violation.
The SOC norm uses stable hypot reduction, not a sum of potentially overflowing
squares. Arithmetic overflow/underflow raises with its cause, not a NaN report.

`check_conic_candidate(problem, x, s, z, ...)` independently recomputes:

- Ax+s-b, componentwise with original-coordinate scales.
- Membership of both supplied s and the original slack b-Ax in K.
- z in K*: Zero dual rows are free; Nonnegative/SOC dual rows are constrained.
- P x + q + A.T z, componentwise.
- s_block.T z_block, separately for every cone block to avoid cancellation.
- A range-aware PSD dual objective and the signed primal-dual gap.

The multiplier convention is
`L(x,s,z)=0.5*x.T*P*x+q.T*x+z.T*(Ax+s-b)`.
Infimizing over s in K gives a finite value only for z in K*. Writing
`h=q+A.T*z`, the infimum over x is

```text
dual(z) = -0.5 h.T P^dagger h - b.T z, if h belongs to range(P).
gap     = primal(x) - dual(z).
```

For LPs, P=0 and numerical dual availability therefore requires h=0 within
the componentwise stated range tolerance. This code does not substitute
`-0.5*x.T*P*x-b.T*z` when stationarity fails, and does not regularize P to create
a fictitious dual value. It diagonalizes the symmetric PSD matrix independently.
Only computed exact zero eigenvalues and positive eigenvalues above
`64*eps64*n*max(abs(eigenvalues))` form a resolved spectrum; unresolved tiny
eigenvalues make the dual unavailable rather than being silently discarded.
Roundoff-level asymmetry is averaged; larger asymmetry or clearly negative
curvature raises. Clarabel receives the upper triangle of this same full P.

Reports use `NumericalEvidence`: native bool checks, finite metrics/thresholds,
unavailable values as `None` (JSON null), and an explicit numerical qualification.
Negative gaps are retained, never clipped. A passing report is approximate
numerical evidence, **not an exact proof or certified lower bound**. In particular,
tolerance-level nullspace/range and cone checks do not establish exact feasibility.
No backend residual, backend objective, projection, or cached solver factor
is used to decide these checks.

## Strict Reference And Status

`ConicConfig()` intentionally has **no default backend**. Only an explicit
`backend="clarabel_reference"` plus `device="cpu"` executes Clarabel. CUDA requests
raise before loading optional dependencies, even on an available CUDA machine.
There is no `"auto"`, no silent fallback, no implicit installation.
This conic module loads Clarabel and SciPy only when solving; missing or broken
imports raise with the original import error chained. Other backend failures
propagate. The existing `openopt` compatibility facade eagerly imports SciPy
through `openpo.exact_solver`: importing `openopt.conic` without SciPy installed
therefore still fails in that facade, before this module executes. Fixing that
pre-existing package-level limitation requires another seat's authorization.
Normal package imports are independently tested without Clarabel; this module's
SciPy-free import/checks are tested under a namespace-only parent pointing to the
real source files, not with stubbed SciPy or solver dependencies.

The result records `backend`, `backend_version`, `device`, original
`backend_status`, `iterations`, and raw x/s/z. `certified` is always False.
For `Solved`, success is `reference_solved` only if independent checks pass;
otherwise `reference_candidate_rejected`. `AlmostSolved` remains
`reference_almost_solved` even if numerical checks pass. Both expose recomputed
candidate objective/evidence.

`PrimalInfeasible` and `DualInfeasible` remain
`reference_primal_infeasible_unverified` and
`reference_dual_infeasible_unverified` (with distinct almost variants).
They have no optimality evidence or objective. **DualInfeasible is not relabeled
primal unboundedness**: that requires a separately established primal feasible
base plus a descent/recession witness, which this module does not verify.
Likewise no Clarabel infeasibility ray is promoted to an Openopt certificate.
Iteration/time/numerical/progress limits stay distinct; unknown statuses raise.

## Reproduction And Scope

Use the existing interpreter and hide CUDA; no installation, cluster submission,
credentials, production changes, nested agents, main-tree edits, or push:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py \
  --target tests/test_conic.py --target tests/test_general_qp.py \
  --target tests/test_qp_certificates.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/conic/acceptance
```

The tests execute real installed Clarabel for analytic SOC, constrained and
unconstrained QP, non-diagonal P, infeasible, unbounded example (raw dual-infeasible
status), and iteration limit. A quadratic SOC projection has analytic solution
`(2.5,1.5,2,-2)`, objective `-4.25`, and SOC/Nonnegative/Zero order; another planted
SPD case uses repeated SOC blocks and a supplied complementary primal/dual
witness. They also test wrong candidates/dual signs/axis,
signed negative gaps, per-block cancellation, unresolved PSD range, huge unrelated
scales, malformed/nonfinite data, explicit CUDA refusal with availability true
and false, missing optional dependencies, fresh-process lazy import, backend
exceptions, and spoofed zero residual/objective reports.
The external pytest temp folder receives `conic-acceptance.json` with source
SHA256, time, actual backend version/status, x/s/z, checks, and a wrong-x negative
control. Its path is recorded in JUnit; `verification.json` and `pytest.xml` carry
the actual counts. The direct Clarabel comparison validates adapter wiring, not
independent algorithm agreement; analytic primal/dual witnesses provide that
separate correctness basis.

Re-read the resulting run directory, JUnit counts, six real cases, negative
control and current three-file hashes with:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=src /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  tests/test_conic.py --audit-artifacts /path/to/external/tests-RUN_DIRECTORY
```

The audit raises on missing/mismatched evidence, empty tests, skipped cases,
invalid status promotion or changed source hashes. Its successful terminal
line is `CONIC_ARTIFACT_AUDIT verdict=PASS` with a current UTC timestamp.

Preflight ran on 2026-10-10 in
`conic/preflight/smoke-922sm9nf`: 79 passed, 4 explicit CUDA skips.
Implementation acceptance is not inferred from preflight.
First acceptance (`conic/acceptance-v1/tests-_u8xrd3z`) preserved two failures
and 218 passes: the fresh-package probe exposed the facade's eager SciPy import,
and an extreme-range hypot test incorrectly expected underflow when the stable
norm performs no underflowing operation. Tests now distinguish package Clarabel
laziness from scoped-module SciPy laziness and assert finite negative evidence
for that stable norm, with separate real overflow/underflow controls. No facade
edit or silent dependency workaround is used.
V2 (`conic/acceptance-v2/tests-9_zwbp0b`) passed 220 tests without skips on
Clarabel 0.11.1 at 2026-10-10T08:43:27 UTC. Final acceptance adds mixed/repeated
SOC quadratic cases, malformed backend return controls and source/JUnit audit;
its current evidence must be read from the final run rather than inferred
from this earlier count.
The first final run (`conic/final/tests-8oy2ipn6`) retains 228 passes and one
failure: the repeated-SOC planted case returned Clarabel `Solved` but independent
local complementarity correctly rejected a roughly `5e-8` residual. The reference
solver's global relative objective-gap stopping rule does not imply every local
cone check passes. The test now explicitly asserts that rejection, then solves
this case with `eps_abs=1e-9, eps_rel=0`; the checker is not loosened and the
default result is not promoted. Final smoke
`conic/final-smoke/smoke-cdiba8q4` passed 79 tests with 4 explicit CUDA skips.
All red-run logs/JUnit remain intact; conic is the task/evidence owner.
The infra-only edit router returned exit 2 for this outside-root worktree;
it was not treated as permission or bypassed by modifying infra. The explicit
user assignment and openopt AGENTS define the exact three-file write set:
`src/openopt/conic.py`, `tests/test_conic.py`, and this document.
Shared QP design/generated reports are outside this seat's authorization;
the lead must integrate this new scope and evidence into release documentation.
No GPU/commercial/performance claim is made, and native GPU SOC remains future work.

## Primary Sources

The Clarabel official Python getting-started documentation was read on 2026-10-10:
`https://clarabel.org/stable/python/getting_started_py/`.
It specifies the model, upper-triangular CSC P storage, ordered cone mapping,
x/s/z outputs, and distinct Solved/AlmostSolved/PrimalInfeasible/DualInfeasible
termination statuses. Browser search returned no usable text; direct read-only
HTTP retrieval of the official page succeeded. The PSD pseudoinverse/range dual
above is derived explicitly here from the Lagrangian, not a guessed backend
formula. No dependency or solver source is vendored.
