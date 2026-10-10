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

## Loaded Binding Repair V1

The independent `9c6b69bda8e86ad8cd3b198ebb72afa155186bff` review of owner
`695df3b1f998ab4ede15c0172cd948cba1910173` remains historical FAIL:
43 passed, two failed, zero skipped, 8.40 seconds. Its receipt is
`/hf3fs-jd/data/tmp/openopt-release-20261010/hsd-independent/clean-full-v1/review-j3id7xaj/receipt.json`,
SHA256 `438aa68a33aa98d18b81b2c6e2ae7e4a32723ced2c1c1c54eb6c7f0dcabd55c8`.
Loading an altered checker and then restoring trusted disk bytes falsely
accepted an equality candidate at iteration four (`x=2.7473684210526317`
instead of two; four of six original gates failed) and a Farkas ray at
iteration two (`y=[1,1.5]`, original `A.T y=0.5`, threshold at most `1e-7`).
Neither restored disk hashes nor a captured projector identity bound those
actually loaded functions. No historical result is upgraded.

The repair preserves numerical operations, thresholds, defaults and opt-in
device policy. It adds a bounded local loaded-code contract for:

- Every HSD top-level function, helper, method, property and default; the
  original-model/cone dataclasses and dimension helper; the projection and
  its finite/norm/scalar/cone/rotated helpers in native conic.
- Source-compiled code objects, actual globals, closures, imported module
  identities/cache/origins, cone/type aliases, constants, exact configuration
  class and fields. Dataclass-generated assignment/comparison/frozen/repr
  templates and closures are compared to stdlib-generated structural
  templates, without executing project source.
- The actual `torch.no_grad` wrapper code and its actual `func` and context
  closure, not arbitrary traversal of `__wrapped__`. A wrapper with a trusted
  `__wrapped__` but different executable code or closure is rejected.
- A separately compiled guard entry before preparation, guard helper
  bootstrap checks, then retained source buffers and verified entry at solve
  entry and after iteration completion, before returning a result. Existing
  model, source, tensor and runtime integrity checks remain active.

The guard compiles trusted local source buffers without executing them.
At preparation, the intended source files and installed Python/stdlib/torch
runtime are trust anchors. A caller must separately pin the intended clean
source; self-consistent malicious source on disk is not authenticated by
this guard. This is not protection against arbitrary hostile interpreter,
stdlib/dependency replacement, concurrent mutation-and-restoration between
checks, or remote attestation. Runtime monkeypatches/wrappers are not silently
accepted just because they preserve a module name or `__wrapped__`.
There is extra CPU inspection/source IO overhead; no speedup is claimed.

All repair evidence is external under
`/hf3fs-jd/data/tmp/openopt-release-20261010/hsd-loaded-repair/`.
The author's same load/restore controls actually failed twice before repair:
`red-v1/tests-mo7_a372/verification.json`. Development
`full-binding-v4/tests-3f0epe72/verification.json` is 103 passed, one explicitly
disabled local CUDA target, zero failed (35.77 seconds), with unchanged
dirty-source consistency. It is not clean-source acceptance. Earlier failures
are retained: v1's old source-path fixture violated the origin contract; v2
exposed class `__getattribute__` interception before instance integrity
dispatch. The entry check now precedes that dispatch. The 42 prepare/solve
closure controls include guard helpers and entry replacement; another control
changes checker defaults during the actual LU solve and requires post-solve
rejection. Two load/restore JSON records retain original model, rejection
reason and altered/restored byte hashes.

The only skipped author test is
`test_cuda_actual_residency_and_analytic_candidate`: absent
`OPENOPT_RUN_HSD_CUDA_TESTS=1` explicitly skips local CUDA. Under explicit
opt-in, unavailable CUDA fails. No GPU submission, GPU proof, commercial
qualification or general solver acceptance occurs here.

After a clean commit, rerun `scripts/verify.py tests --target
tests/test_hsd_lp.py` under the CPU environment. The test-file CLI
`--audit-unit /absolute/verification.json --sha256 FULL_SHA256 --owner-commit
FULL_COMMIT` checks current clean complete source membership, actual exact
104-item collection/JUnit membership, both load/restore rejections, the forty
serialized closure controls (the two direct closure-cell controls are
asserted in pytest), and the one named CUDA skip. It writes
`author-receipt.json` beside verification; no solve occurs in this audit.
`--scan-tool /data/dev/infra/ops/dsh/scan-secrets-precommit.py --needle-file
/home/infra/.config/dsh-n600-token --artifact-dir /absolute/external/root`
scans only the three repair paths, keeps the needle solely in memory, and
records positive/cleared sensitivity controls without secret-tail rendering.

The frozen independent 45 tests and document are not edited. Author execution
of those unchanged bytes against a new clean owner is available with
`--unchanged-review /data/dev/openopt-wt-release4-20261010-hsd-independent
--reviewer-commit 9c6b69bda8e86ad8cd3b198ebb72afa155186bff --owner-commit
FULL_NEW_OWNER --artifact-dir /absolute/external/root`. It creates a
non-Git external source mirror, pins both complete source sets, compares
actual 45-item collection/JUnit membership and retains any FAIL. The old
test's internal historical owner label is disclosed, never used to
authenticate the new owner. Its projector counting fixture replaces the HSD
imported alias with an arbitrary observation wrapper; that conflicts with
this strict alias contract. Do not weaken the guard, edit the original
negative assertions, or claim independent closure to hide that incompatibility.
Fresh caller-pinned independent review remains required.
