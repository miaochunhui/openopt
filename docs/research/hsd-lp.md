# Experimental Native Dense HSD LP

## Scope And Algorithm

Only `src/openopt/hsd_lp.py` is a new opt-in solver module; it is not exported
through the default API and does not change existing solvers or checkers.
`HSDLPConfig(enabled=True)` is mandatory. `P` must be exactly zero, including
subnormal nonzero entries. Supported products are zero, nonnegative and
ordinary Lorentz cones. RSOC, sparse, PSD/QP, full generic conic programs,
presolve, scaling, acceleration and general commercial LP are not implemented.
The binary64 dense embedding dimension is bounded by 256 and iteration count
by 10000; invalid/empty models, nonfinite arithmetic and device mismatches fail
loudly before they can become accepted statuses.

Original convention: minimize `q.T x`, `A x + s = b`, `s in K`.

```text
Q = [[0, A.T, q], [-A, 0, b], [-q.T, -b.T, 0]]
C = R^n x K* x R+
t = solve(I + Q, w)
u = projection_C(2 t - w)
v = u - (2 t - w)
w = w + u - t
```

The initial state is always `w=(0,0,1)`. A prepared workspace reuses the LU but
resets state on every solve; it is not a warm-start interface. `I+Q` belongs
to the resolvent, not a regularized original model. No inverse, normal-equation
model, diagonal perturbation or third-party solver is used.

Primary source is the official SCS algorithm page (title: SCS 3.3.1):
`https://www.cvxgrp.org/scs/algorithm/index.html`, retrieved 2026-10-10 through
the authorized astrill URL entry at explicit `4m`, one download attempt.
The 17637-byte HTML is external under
`/hf3fs-jd/data/tmp/openopt-release-20261010/native-hsd-lp/primary-source/scs-algorithm.html`,
SHA256 `2b63d5780b8ef6883081840ea12bd8de5809cea8f93290cb17c6272bf74ca9e7`.
The log is `primary-source/logs/url-fetch-20261010-205332-4080030.log`.
It records exact HTTP/local/remote byte agreement, wget rc zero, download
progress approximately 31.6 KB/s, matching byte hashes and `ASTRiLL_FETCH ok=1`.
No package installation or credential change occurred. The source was
read in full; the implementation is new code, not an SCS package binding.

## Numerical Acceptance And Device Boundary

Six original-model candidate gates are primal equation, both supplied/original
slack cone membership, strict dual cone membership, componentwise stationarity,
componentwise nonnegative/zero or block SOC complementarity, and objective-pair
gap. Every equation/component threshold is saved, not replaced by a global
maximum. `eps_abs`, `eps_rel`, `eps_ray` cannot exceed `1e-7`. An approximate LP
stationarity multiplier is not claimed to define an exact finite dual bound.

Independent original-coordinate audits use bounded exact dyadic products and
accumulation (`Fraction`), analytic cone inequalities and represented binary64
metrics. They never call the iteration projection or a reference solver.
Nonrepresentable metrics fail loudly. Strict dual/ray nonnegative signs and
dyadic SOC membership prevent wrong-sign tiny multipliers being clipped or
accepted as having finite support. Primal feasibility is numerical.

Normalized Farkas rays require `b.T y ~ -1`, strict `y in K*`, and componentwise
`A.T y ~ 0` with an absolute ray threshold tightened by
`min(1,max(abs(A))/norm_inf(b))`. Recession rays require strict `-A d in K` and
`q.T d ~ -1`. Two consecutive original checks must pass before a ray status.
These are bounded numerical evidence, not exact mathematical certificates.

`solved_candidate` requires all six gates. `primal_infeasible_ray` records
numerical Farkas evidence. `dual_infeasible_ray` deliberately does NOT assert
primal unboundedness: a ray alone is insufficient if the primal is infeasible.
Only a separately supplied and original-model-checked `feasible_base` permits
`unbounded_candidate`. The solver never manufactures a base. Otherwise the
status remains `max_iterations`; rejected checks and final embedding vectors
are retained. No old result is overwritten or retrospectively qualified.

LU, resolvent solve, DR, projection and embedding residual computations use
the actual requested torch CPU/CUDA device and float64. Explicit CUDA failure
does not call a CPU factor or solver. Input upload, original-model host audits,
source hashing and scalar synchronization are disclosed and are not GPU-only
verification. Factor/system snapshots are compared on device. Source bytes,
original array bytes/cones, device tensors, typed settings and actual torch
runtime/thread configuration are bound to the prepared workspace; mutation
rejects before a new solve, with another integrity check after the solve.

## Validation Entry

Use the existing CPU environment with CUDA hidden and thread variables set to
one. All logs, JSON and JUnit are external. Run the project pre/post smoke and
the bounded new suite; the verification entry preserves source before/after,
actual dependency versions, test counts and exit codes.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_hsd_lp.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/native-hsd-lp/owner-unit
```

Owner-only CUDA target is `tests/test_hsd_lp.py::test_cuda_actual_residency_and_analytic_candidate`
with `OPENOPT_RUN_HSD_CUDA_TESTS=1` and an actual CUDA environment. Missing CUDA
under opt-in is FAIL, not skip or fallback. This seat does not submit GPU work.
Analytic LP equality/lower-bound/SOC, infeasible and recession examples use
actual native solves and independent checks, plus Clarabel on the unchanged
original model. External per-case JSON retains settings/runtime, arrays,
iterations, row thresholds, residuals, model/source hashes and is read back.
Negative controls cover wrong rays/signs, tiny feasible coefficients,
componentwise threshold masking, model/source/factor/runtime changes, wrong
device, nonfinite factors/iterates and wrong resolvent results.

Pre-smoke at base `842523fb4690232fbf93c8a88325990d066d4c44`:
`pre-smoke/smoke-c637nggo/verification.json`, 80 PASS, four explicit local CUDA
skips, 8.83 pytest seconds, source consistency PASS.
Development `unit-v3/tests-3fdea0c4/verification.json`: 56 PASS, one explicit
CUDA target skip, 7.51 pytest seconds, source consistency PASS. The original
`unit-v1/tests-st35fj5g` stays FAIL (48 PASS, one FAIL, one skip): the negative
control used `flatten()` on a noncontiguous LU and mutated a copy. The test now
directly indexes the actual tensor and asserts it changed before rejection.
`unit-v2/tests-g551tn2b` passed 49/one skip after that test-only repair.

The consolidated eight-case receipt is produced by
`test_actual_receipt_and_rehashed_tamper_controls` as `actual-cpu.json` in its
external pytest temporary directory. Every case includes original arrays and
cone/model hashes; the failure remains in the denominator. Actual observed
native solves: eight; actual observed LU solves: 201 (iterations
20/30/40/50/20/20/20/1). There is one LU factor per workspace. Intra-op was
actually one, inter-op actually 128; both are saved, not relabelled as one.
Three separate candidate tests also execute Clarabel references and the
unchanged original conic checker; Farkas/recession reference statuses are
retained as reference output, not an independent exact certificate.

The test file offers bounded no-solve replay, requiring a caller byte pin:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/dev/openopt-wt-release4-20261010-native-hsd-lp/src /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_hsd_lp.py --replay /absolute/external/actual-cpu.json --sha256 FULL_SHA256
```

Replay checks original model bytes, source bytes, exact predefined
case/status/setting denominator, original numerical checks and no fabricated
base. Five coordinated rehashed controls reject wrong model, wrong ray,
dropped failure, changed settings and fabricated base. This bounded local
numerical replay does not attest past native execution or all archived runtime
fields against an external trusted anchor. Actual observed call counts belong
to the controlled test run; they are not remote attestation.

Clean frozen receipts are external under the `frozen-unit` and `frozen-smoke`
roots, populated after commit without changing these three files. CPU results
do not establish CUDA acceptance, universal
robustness, public/industrial coverage, exact proofs or commercial qualification.
