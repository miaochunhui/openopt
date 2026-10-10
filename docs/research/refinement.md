# Opt-In QP Refinement

This is a separate numerical candidate transformation, not a new default
solver path or a complete arbitrary-precision optimizer. Import
`openopt.refinement` explicitly. `qp.py` and the public facade are unchanged.

## Method And Acceptance

For the original convex dense QP `min 0.5*x.T*P*x + q.T*x`, `l <= A*x <= u`,
guess active constraints from the baseline candidate's original-coordinate
distance to finite bounds. All exact equalities are active. The explicitly
configured active tolerance is only a guess; it never relaxes acceptance.
Two nearby bounds on a non-equality are ambiguous and explicitly rejected.
There is no active-set search or claim that every correct optimum has a
nonsingular KKT matrix.

With active row matrix `B` and active bounds `b`, solve
`[P B.T; B 0] [x; y_active] = [-q; b]` on the requested PyTorch device.
Inactive multipliers are zero. An unregularized LU factorization is reused
for residual corrections `delta = LU_solve(rhs - K*solution)`. The residual
is calculated in explicitly selected `dtype`; `factor_dtype` may be the
same precision or explicitly float32 with float64 residuals. No CPU,
least-squares, pseudoinverse, external optimizer or regularization fallback
is permitted. Numerical rank ambiguity/singularity is `unknown`, not success.

Each proposed candidate is copied explicitly to CPU float64 for the existing
independent `check_qp_candidate`, against frozen pre-cast original model data.
This is a host verification step, not a host solve. Acceptance requires every
original feasibility, stationarity, multiplier-sign, complementarity and
duality-gap gate to pass. In addition the maximum original feasibility,
stationarity, sign, complementarity and absolute gap cannot exceed the
baseline values, without rounding allowance. Baseline unavailable gaps are
not treated as zero, but the new gap must be available and pass.
Objective decrease alone is not acceptance: an infeasible point can have a
better objective.

The best fully checked nonworsening candidate is retained. Stagnation,
step-limit and every attempted candidate's check are recorded. If none is
acceptable, return explicit `rejected` or `unknown`, the reason, baseline-value
clones and last attempted candidate/evidence. Malformed inputs and invalid
or unrepresentable arithmetic raise; errors are not converted into success.
Inputs are never mutated. Rejection retains the baseline device/dtype; no
implicit device switching is possible because x/y must already reside on the
explicit device.

## Usage And Recovery

```python
from openopt.refinement import RefinementConfig, refine_qp_candidate
import torch

polished = refine_qp_candidate(
    original_problem, baseline.x, baseline.y,
    RefinementConfig(device="cuda:0", dtype=torch.float64,
                     factor_dtype=torch.float32),
)
# Inspect status/reason/evidence; accepted is numerical, not an exact proof.
```

Omitting this explicit call is the baseline/recovery path. There are no
default-path edits to reverse. A rejected result contains separate tensors
with exactly the baseline values, rather than silently returning a new solve.
The same-problem accuracy A/B is a correctness measurement; no end-to-end
performance comparison or commercial superiority is claimed.

## References And License

Mathematical references only, with no third-party source copied:

- OSQP solver documentation, Polishing section:
  `https://osqp.org/docs/solver/index.html#polishing`.
- Tobias Weber, Sebastian Sager and Ambros Gleixner, *Solving Quadratic
  Programs to High Precision using Scaled Iterative Refinement*, 2019,
  `https://arxiv.org/pdf/1803.07178` (v2, 19 August 2019).

The second reference develops rational scaled QP-oracle refinement and
arbitrary precision under assumptions. This implementation does not implement
that algorithm, rational arithmetic or its convergence guarantees. It only
implements active-set polishing and finite-precision linear-system residual
correction independently under this repository's 0BSD license.

## Verification And CUDA Handoff

Run locally using the existing validation environment and no GPU:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_qp_refinement.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/refinement/unit
```

`test_original_coordinate_accuracy_ab_and_three_negative_controls`,
`test_actual_admm_candidate_same_problem_polishing` and
`test_residual_correction_float32_factor_float64_residual` save complete
original model, baseline/refined x/y, explicit config, independent checks,
source hashes and three failing mutations in external JSON. Bound infinities
are encoded as `"Infinity"` / `"-Infinity"` strings; JSON never contains
non-standard NaN/Infinity literals. Readback reconstructs the same QP and
recomputes its original checks.

The separate disk replay entry does not invoke a solver. Supply the producer
JUnit's sidecar path/hash and the integrated current source root. Output must
be a new external file:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_qp_refinement.py --check-sidecar SIDEcar.json --sha256 CALLER_PIN --source-root /data/dev/openopt --output /hf3fs-jd/data/tmp/openopt-release-20261010/refinement/replay-new.json
```

It verifies caller-pinned bytes, exact source membership/current hashes,
the loaded checker source, recomputes original model/baseline/refined/negative checks, and independently
reapplies acceptance and unchanged-baseline rejection. A changed threshold,
model, x, y, source or byte pin is rejected in negative controls. Do not
recompute a caller pin to bless a changed old artifact.

Only the parent coordinator submits cluster work. CUDA targets in
`tests/test_qp_refinement.py`, selectable with `-k cuda_refinement`, are:

- `test_cuda_refinement_frozen_model_accuracy_ab`: three explicit dtype/factor
  combinations, same-model A/B, device assertions, no input mutation and
  complete independently replayable JSON linked and hashed in JUnit.
- `test_cuda_refinement_actual_residual_correction`: a non-dyadic, non-diagonal
  KKT problem requiring correction after the float32 factor's initial candidate
  fails the original float64 gate; complete model/config/check sidecar.
- `test_cuda_refinement_negative_controls`: wrong active set, singular KKT
  and wrong multiplier sign, each explicitly retains the CUDA baseline and
  saves its model/config/vectors/independent-check sidecar. A passing negative
  test is not accepted candidate qualification.

Local CUDA skips are UNVERIFIED, never GPU acceptance. Parent integration must
update shared design/report entries, replay numerical sidecars and run the
final frozen-source CPU/wheel/GPU gates. This seat does not edit shared
documentation, publish, push, install dependencies or change infrastructure.

## Retained Initial Failure

The initial unit run recorded `1 failed, 42 passed, 6 CUDA skipped` under
the external `refinement/unit-v1/tests-iplwo4cd` directory. A wrong active
multiplier sign had been conservatively classified `unknown` merely because
infinite box support makes the dual objective unavailable. This is a known
failed bound-sign check, not unresolved arithmetic. The implementation now
classifies any failed feasibility/stationarity/sign/complementarity check as
`rejected` before considering unavailable dual evidence. Singular/unresolved
KKT and otherwise passing KKT with unavailable original gap remain `unknown`.
The old failed run is retained and not counted as acceptance.

Initial reference/worktree baseline is
`0e34eb37bc8be4b6561bd18e5c5f6405aceede24`. Only this document, the new module
and its tests are in the seat's write set.
The initial non-diagonal diagnosis used a 25-iteration ADMM check interval;
that baseline had already converged near machine precision before checking.
Those artifacts are retained, not evidence of a substantial precision gain.
The final correction A/B uses a one-iteration check interval in both CPU
and CUDA fixtures, retains the same problem and original strict gate, and
asserts that the original ADMM baseline and initial coarse KKT candidate both
fail the strict gate before correction passes it.
