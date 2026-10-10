# Explicit Public CPU Robustness Experiment

## V3 Executable Worker Binding

This new scope implements `public-qp-worker-v3` in
`scripts/qualify_public_qp.py`, based strictly on
`acc199306936922b6d75330884310a58cde84a9b`. It does not cherry-pick preview
`a668d72987c128c84083228bb30b0e419076deab` or other unreviewed algorithms.
The preview remains CONTRACT_ONLY_NOT_EXECUTION. The original qualifier v1
and robustness v2 entry points/contracts are unchanged; neither accepts v3
results. V2 still rejects the new refinement field rather than widening its
old field set. V3 is a separate executable/versioned path, not backfilled v1/v2.

Only bounded synthetic frozen models are authorized here: one to four cases,
1 <= n <= 8 and m <= 64 after original variable bounds become solver rows.
Public origin is explicitly rejected even with source acceptance. No public
real model was solved, downloaded or redistributed. No GPU, installation,
production interface change, numerical tolerance relaxation or license claim
is included. A completed execution is not qualification:
`EXECUTION_COMPLETE_NOT_QUALIFICATION`, public/industrial counts zero.
Numerical failure remains UNQUALIFIED and cannot become success via refinement.

The complete producer-owned contract specifies all 19 QPConfig, seven
ScalingConfig and nine RefinementConfig fields, including
`check_arithmetic=reproducible_binary64_v2`. Refinement and final independent
candidate replay use that bounded arithmetic explicitly; the unchanged QP
internal termination and scaling validation still use their original numpy
checker. There is no arithmetic fallback or claim that all internals use v2.
Six original-model gates remain atol=rtol=1e-7 with exact wrong-infinite-side
support/sign rejection, range/dual/gap availability and component thresholds.
Solve targets are 1e-10; Openopt/OSQP max iterations 2000, Clarabel 200.
The bounded reproducible arithmetic rejects unsupported dimensions,
nonrepresentable/subnormal arithmetic and unresolved spectra. It is not an
exact proof, a complete LP/QP interface or evidence of general robustness.

OSQP's full requested/effective field sets and Clarabel's complete 45-field
settings derive from the explicitly pinned, unchanged v2 contract source,
not saved result metadata. All fields/types/values must match independently
at requested, effective, before-solve and after-solve stages. Arbitrary
resealing of plan, rows, worker receipts and byte inventory cannot redefine
the local producer contract. Clarabel presolve=false/max_threads=1 and OSQP
builtin/direct/warm_starting=0 are mandatory. Missing/new fields fail loudly.
No caller setting/source/device overrides are accepted on v3.

Each case/backend gets one worker with exactly two labelled attempts:
cold creates inputs/equilibration/workspace and solves; hot reuses that same
workspace but starts a new zero/default-initialized solve. Openopt initializes
x=y=0 and z=projection(0), resetting rho/factor on each call; OSQP disables
warm starts; Clarabel uses DefaultSolver's default initialization, not a warm
start API. No previous candidate is passed to hot or to another backend.
Both modes include host output and Openopt's original-coordinate refinement.
Hot excludes setup/equilibration/initial factorization. This is not a timing
race, general cold/hot equivalence guarantee or speed claim.

Actual solver entry counts use an observer of the unwrapped Python bound
workspace method and temporary wrappers around the true pybind bound method,
always calling the original method and restoring it immediately. Every mode
requires workspace_solve=1, other_solver=warm_start=0; OSQP also observes
one native solve under its wrapper. Each Openopt mode records one refinement
proposal, not an extra solver attempt. A missing solve must fail rather than
incrementing a loop counter and claiming execution. Test replay traps Python
and native solve/refinement entries, including newly loaded frozen modules.

Source identity checks immutable base bytes via Git, all seven selected source
pins and copied bytes, and actual core module __file__/__spec__.origin before
and after the worker. The producer itself must match the local trusted
runner.py, not only a rehashed saved runner. Runtime is explicitly Python
3.10.20, numpy 2.2.6, scipy 1.15.3, torch 2.8.0+cu128 (CPU execution), osqp
1.0.4, clarabel 0.11.1 and highspy 1.15.1. Entry/native-module bytes, package
versions and selected loaded OpenBLAS/OpenMP library bytes are pinned.
Actual OpenBLAS/OpenMP thread queries and torch intra/inter-op must be one;
OMP/OPENBLAS/MKL=1, CUDA_VISIBLE_DEVICES=-1 and no-bytecode settings must match.
Missing native-thread observation fails; no new dependency is installed.
This selected loaded-byte closure is not every shared library/package,
attestation against malicious trusted code, or proof of hardware execution.

Result/plan/corpus pins, complete case x backend x cold/hot membership,
original-array hashes before/after, exact inventory, mode-specific sidecars,
raw vectors, transformations, status and all independent numerical gates
are replayed without solve/refinement. Ruiz factors are recomputed without
solving, restored x/y and signed Clarabel cone mappings are verified.
Rejected refinement retains baseline values; accepted proposals require all
original gates and nonworsening metrics. Exhausted status cannot be upgraded.
Failures stop new affected workers immediately, preserve firstcause/log/exit,
and retain the full predeclared denominator. Failed runs are not replay-qualified.
Outputs are exclusive-create and outside the worktree; old evidence is immutable.

### V3 Verification Entry

Default tests do not require the exact producer runtime or mounted public
evidence. Actual local-native execution, coordinated v3 artifact tampering and
runtime probes require `OPENOPT_RUN_PUBLIC_V3_WORKER_TESTS=1`. Enabled missing
or incompatible dependencies/runtime fail loudly, not skip. All actual
models are synthetic MPS generated in external pytest tmp. No fabricated
metadata is treated as worker execution.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_qualify_public_qp.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-v3-worker-binding/default-new
sudo -n -u infra env OPENOPT_RUN_PUBLIC_V3_WORKER_TESTS=1 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_qualify_public_qp.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-v3-worker-binding/optin-new
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/qualify_public_qp.py replay --version public-qp-worker-v3 --result EXTERNAL_RESULT --result-sha256 CALLER_PIN
```

Standalone `run --version public-qp-worker-v3 --corpus FROZEN_SYNTHETIC_DIR
--manifest-sha256 CALLER_CORPUS_PIN --artifact-dir NEW_3FS_DIR` executes
the frozen contract. `worker --version public-qp-worker-v3` additionally
requires the caller plan SHA and worker index; it is not an unpinned entry.
The numerical gate never maps completion exit 0 to public qualification PASS.
Main owns any later real public-model authorization and independent review.

### Retained V3 Development Reds

Evidence root:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-v3-worker-binding/`.
Pre-smoke `pre-smoke/smoke-5kw08h1_` is 80 passed/four CUDA skips/8.18s,
source consistency PASS. Cheap initial controls
`cheap-controls-v1/tests-_du1xcrp` passed 14/4.17s.
`actual-v1/tests-n_7eitre` retains one test failure/42.66s: all 12 real
synthetic solves finished, but replay's parent torch inter-op was not one.
V3 replay now explicitly initializes and observes the required threads.
`call-observed-v3/tests-8oyjsrn8` retains one setup error/11.30s: no_grad
decorator hid the original solve's self from the profiler. The observer now
unwraps the actual method. `call-observed-v4/tests-qm080024` retains one
setup error/17.91s: OSQP pybind does not emit profiler c_call events.
Actual native entry wrappers fixed observation without loosening counts.
`call-observed-v5/tests-97q1lfdn` passed actual 12-solve/replay test/46.03s.
These are development receipts, not clean-source final acceptance.
Final frozen receipt coordinates follow separately; no historical red is
overwritten or counted as a passing execution.

Final development full-file receipts (same final producer/test bytes, before
this documentation-only receipt addition):

- Default `default-final-v1/tests-0skr5_mk/verification.json`, SHA256
  `7b5a8cf2c35fa9b508dfe3644a69aa484475248abee69b929deafa6fd899f612`:
  50 passed, 23 explicit exact-runtime execution skips, 43.91s.
- Opt-in `optin-final-v1/tests-8u27gbzf/verification.json`, SHA256
  `17bb580d3dd99b6434d9945f95fafc240a786259d4a86598957372bda00efc84`:
  73 passed, zero failures/errors/skips, 130.73s. Source consistency PASS,
  294 files, finished 2026-10-10T12:43:45.966997+00:00.
- Actual synthetic execution is
  `optin-final-v1/tests-8u27gbzf/pytest-tmp/v3-actual0/execution/result.json`,
  SHA256 `dc46e4d8ae9a76326bc7d813f0cb69aba18e924c4fd046c32fcabddc7484e3dd`;
  plan SHA256 `c0295136c0d744b383cf2b4d53ba72f88e38705e7e7aaee5bdc162eaff46045f`.
  Actual counts: 12 workspace solves, four nested OSQP native calls, four
  Openopt refinement proposals, zero other-solver/warm-start calls.
  Twelve of twelve original candidates pass the unchanged six gates;
  numerical PASS is synthetic only, public/industrial counts zero.
- External `optin-final-v1/tests-8u27gbzf/pytest-tmp/v3-actual0/replay.json`
  round-trips JSON, replaying 12/12 with solve_calls=0. It was checked at
  2026-10-10T12:42:53.431856+00:00, not asserted as permanent acceptance.

Opt-in negatives include six fully coordinated settings reseals (OSQP
max_iter/warm_starting, Clarabel max_iter/max_threads/presolve_enable,
Openopt check_arithmetic), eleven identity/denominator/summary variants,
all four settings stages' field/type controls, five installed-version
differences, invalid worker plan pin, observed missing-solve failure,
actual worker firstcause/stop-without-retry and accepted-refinement with
max_iterations remaining UNQUALIFIED. The original v1 CPU format/function
regressions also run; tests/receipts are not distinct industrial cases.
Clean frozen post-smoke/verification receipts are handed off externally,
without editing these immutable development artifacts or historical results.

This is a three-file, opt-in experiment, not a solver/checker/runner/importer
repair or a qualification report. Baseline is the caller-pinned negative
two-LP run; its six attempts are copied byte-for-byte and replayed, not rerun.
Each case receives exactly one new attempt per predeclared arm. Failed and
budget-exhausted rows remain in the denominator. No automatic retry, tiny-dual
clipping, tolerance widening, GPU, download or dependency installation occurs.
The original corpus arrays, variable bounds and objective offsets are unchanged.
Source review does not become legal permission: license_verified and public
redistribution remain false. No speed/commercial/industrial claim is made.

## V2 Settings Repair

Current producer is `public-robustness-v2`, based on scoped commit
`efc156774e16768c4ee385c0fb879f9e30626684`. The v1 producer had a P2
integrity gap: it checked only OSQP scaling/eps and Clarabel equilibration/
tol_feas, not complete requested/effective settings. Coordinating edits of
result rows, worker rows, progress snapshots and inventory hashes could change
max_iter/warm_starting/max_threads/presolve while old replay still passed.
Historical numeric vectors/verdicts below are retained, not full-settings
acceptance. Main's original frozen efc1567 branch remains the v1 replay entry.
V2 rejects v1 by version and producer identity; it does not relabel, backfill
runtime metadata into, or requalify the original sealed experiment.

V2 has a producer-owned exact contract: dictionary field membership, recursive
types and values must match, without tolerances or bool/int coercion.
Openopt solver/scaling/refinement settings are checked in full, including an
explicit default-field contract rather than accepting arbitrary frozen defaults.
OSQP 1.0.4's 28 effective fields and Clarabel 0.11.1's 45 effective fields are
enumerated from actual settings objects; unrecognized/new/missing fields fail.
OSQP uses explicitly selected builtin/direct algebra with every writable field
requested (enum controls use the wrapper's solver_type/cg_preconditioner aliases).
Clarabel explicitly requests all 45 fields, including the infinity time-limit
encoded as the JSON string `"Infinity"`. Actual backend settings are saved and
checked before and after solve, as well as the full requested/settings records.
The plan, result and every successful worker must independently match this
contract; coordinating a plan/worker/result reseal cannot redefine it.
Solve targets, original numerical 1e-7 component gates and strict sign rules are
unchanged. This repair does not change a solver/checker/importer or numerical
qualification decision.

Runtime is pinned to Python 3.10.20, numpy 2.2.6, scipy 1.15.3,
torch 2.8.0+cu128 (CPU here), osqp 1.0.4 and clarabel 0.11.1. Metadata and
loaded-module versions must agree. The plan/result/worker also bind SHA256
of loaded package entry modules, OSQP's interface and builtin extension and
Clarabel's extension to the replaying environment. This is selected loaded
module identity, not a complete shared-library/package closure or proof of
hardware execution. A different version needs a separately reviewed versioned
contract, not acceptance of a nearby version.
Worker environment is explicit OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=
MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=-1 and PYTHONDONTWRITEBYTECODE=1;
observed torch intra/inter-op counts must both be one. These configured counts
and environment are saved and pinned, not inferred from process presence.

No real v2 experiment has been run. New real attempts are held for main review.
Setup-only native backend observation and saved-candidate replay tests verify
the integrity repair, not new solver robustness or execution of v2 on real LPs.
The external `test-only-metadata-copy-not-a-run` fixtures deliberately add v2
metadata to copies of old vectors solely to exercise replay gates. They are
not worker receipts or a legal/numerical qualification artifact.

## P2 Evidence

### Portable Test Boundary

The default suite requires the declared test dependencies, not the producer's
exact local Python/package/native-module pins or a mounted real-v1 corpus.
Saved synthetic candidates and an explicitly labelled synthetic runtime double
exercise serialization, all complete settings gates, coordinated reseals,
denominators and numerical replay. The runtime double is confined to the test
fixture; the producer contract is unchanged. It is not a runtime observation,
worker execution receipt, real-model experiment or qualification.

Only `OPENOPT_RUN_PUBLIC_EVIDENCE_TESTS=1` enables the real-v1 metadata-copy
controls and native settings/runtime/thread setup probes. Without it those
tests are explicitly skipped, not reported as passed. An enabled run with
missing, unsafe or incorrectly pinned files fails loudly before testing;
unsupported local runtime fails its producer contract. The optional
`OPENOPT_PUBLIC_ROBUSTNESS_V1_RESULT` supplies a relocated result path, never a
different pin. Full copied-member validation still occurs on replay.
Copies carry `test_fixture_kind=test-only-metadata-copy-not-a-run` and
`worker_execution_verified=false`; v2 metadata fabricated for a negative
fixture does not establish execution of v2. Original v1 bytes remain unchanged.

The direct external unit entry now works in both modes. Default verification
does not read the local real-v1 path and records null real/baseline pins plus
`public_evidence_status=NOT_REQUESTED`. Opt-in records pinned bytes and
`METADATA_ONLY_NOT_EXECUTION`. Both retain actual pass/skip/failure counts, a
zero-solve trap, source-before/after hashes and JSON readback. No full-repository
portable-clone acceptance or native backend execution is inferred from this
scoped suite.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_experiment_public_robustness.py --unit-artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/portable-default-new
sudo -n -u infra env OPENOPT_RUN_PUBLIC_EVIDENCE_TESTS=1 CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_experiment_public_robustness.py --unit-artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/portable-optin-new
```

Portable verification, external root
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/`:

- Default `portable-default-v2/p2-unit-m1tldfv2/verification.json`, SHA256
  `8b9bc472b7253fb7e649a63432fd676e6b6aa7bba1cd05fda34fed34dc993579`,
  finished `2026-10-10T10:48:55.932879+00:00`: 68 passed, nine explicit local
  evidence/probe skips, zero failures/errors, 244.69 seconds.
  `OPENOPT_PUBLIC_ROBUSTNESS_V1_RESULT=/no-such-mounted-corpus/result.json`
  was deliberately set while evidence opt-in was zero; no local real files
  or exact local runtime were required. Real/baseline pins are null.
- Opt-in `portable-optin-v2/p2-unit-ga31eyj5/verification.json`, SHA256
  `dab0c7346c123122e8e16b3edf5bfd1edffe0e94758afc9697f607301ccf3fcb`,
  finished `2026-10-10T10:49:33.770906+00:00`: 77 passed, zero failures/
  errors/skips, 282.51 seconds. Real/baseline bytes match the original pins.
  This includes five metadata-only real coordinated settings mutations,
  two native settings setup probes, actual thread/runtime observation and
  refusal to promote the original v1 result, never new backend solves.
- Enabled missing-path preflight
  `portable-missing-v2/p2-unit-14dl958l/verification.json`, SHA256
  `8211e6709d8fafbb58d3f435265aef8156df5cf60a0bd4863e57270b4b500374`,
  is FAIL with ValueError and process exit 1; testing never started.
  This is intentional negative evidence, not a skipped/successful suite.

Both complete suites record solver_calls=0, before/after source equality and
JSON readback, with actual summaries in `pytest.log` and counts in `pytest.xml`.
Default tests retain all five coordinated settings mutations, 21 complete
membership/type/stage controls, four synthetic runtime/thread reseals, five
controlled installed-package version mismatch controls, a Python version
control, worker-thread mismatch and three disabled/missing/wrong-byte evidence
controls. Metadata version controls supply all other versions as test doubles;
they do not require the host to match the producer's supported environment.

Retained portability reds are `portable-default-v1/p2-unit-bbv6nvt1/`
(68 passed, nine skipped, one error) and
`portable-optin-v1/p2-unit-qjfo4d_6/` (77 passed, one error).
The new Python-version negative initially replaced the whole `sys` object;
the no-solve guard then could not restore its profiler at fixture teardown.
Only `sys.version` is now replaced. No producer code or contract was changed.
The final tested producer/test hashes are
`91f8786f81bff6f6a32a5f0f2763c47e2e0b5846c0699a8eea503e37ee6ac88f` /
`71d8216508ff91430884987ba6942c3aca67b1661a1d1a5b5e6004567d197e6f`.
This evidence paragraph is a subsequent documentation-only update, not a
replacement for either receipt's document hash. This is scoped portability
evidence in the existing environment, not a full clone/version-matrix test.

All evidence is under
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/`.
Repair-before red is `p2-red-v1/tests-gpzft2hx/`: 5 failed, 1 passed, exit 1.
Each of OSQP max_iter=9999/warm_starting=1 and Clarabel max_iter=199/
max_threads=2/presolve_enable=true produced `DID NOT RAISE` after a fully
coordinated reseal of an external original-v1 copy. The original result remains
SHA256 `aa584b62f97349f37afad438dfbd87320ace3408df4d7e574a0989370970e8e7`.
The actual supported-version defaults were read into external JSON without solve.

The first settings repair verification, `p2-green-v2/tests-1zlxr1u2/`,
passed 32 tests, including native setup-only observations, five coordinated
contract+row mutations, 21 membership/type/per-stage controls, four runtime/
thread mutations and refusal to upgrade v1. Its generic verifier also ran an
unrelated legacy CPU example after these tests; that receipt is NOT a zero-solve
verification. No real public model was solved. The final entry below deliberately
does not invoke that generic verifier or its example.

Retained final-entry red is `p2-no-solve-v3/p2-unit-2w84jv7u/`: 56 passed,
5 failed, zero errors/skips, solver_calls=0. All five settings mutations had
already been rejected correctly; a misplaced old-v1 assertion then referenced
an absent monkeypatch fixture and raised NameError. The test block was moved
back to its own test before the changed-source final verification.
Neither failed receipt nor its fixture copies overwrite a real result.

The historical direct unit entry ran the complete then-current scoped suite, used external
pytest temporary data/logs/JUnit, traps Python solver entries, forbids backend
solve calls during replay, verifies original real-result pins, and round-trips
the final JSON. Existing unit workers now use planted saved candidates instead
of real backend solves; they do not establish reference execution coverage.
Setup-only probes construct workspaces without calling solve. Final-entry
verification reported actual counts and zero solver_calls rather than treating
exit zero as qualification:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_experiment_public_robustness.py --unit-artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/p2-final-new
```

Historical pre-portability changed-code receipt is
`p2-final-v4/p2-unit-xs_u24tu/verification.json`, finished
`2026-10-10T10:30:45.156300+00:00`, SHA256
`68a562fd15eed185973dc04c890e51bcfd4e728870827c591546470af743f9f5`.
Actual JUnit: 63 passed, zero failures/errors/skips; solver_calls=0; JSON
readback and before/after source hashes match. The five required mutations
each fail both with coordinated plan/contract/row edits and with coordinated
row-only edits. Installed-version and actual worker-thread negative controls
also fail before execution. Original real-v1 and baseline SHA256 remain
`aa584b62f97349f37afad438dfbd87320ace3408df4d7e574a0989370970e8e7` /
`b10d49a32a13a9f34b8bf86ea6bd8a126dc779bf72a47db3f069420f653456ef`.
Verified producer/test SHA256 are
`91f8786f81bff6f6a32a5f0f2763c47e2e0b5846c0699a8eea503e37ee6ac88f` /
`a010bd8eba75a035dffdd6d886f9646611f8390fa637c58551a5745815b72892`.
This receipt paragraph is a later documentation-only update; it does not
change the tested producer/test bytes or replace that receipt's document pin.

No smoke/full-regression/GPU/wheel or new real solve is claimed for this repair.
The sole pending real execution/qualification owner is the main coordinator.

## Arms And Gates

All arms use CPU float64. Independent full candidate/status/reported-objective
checks remain eps_abs=eps_rel=1e-7, directions 1e-8. Solve targets are separate:

- Openopt: bounded Ruiz (10 iterations, factors [1e-6,1e6], objective scaling
  enabled), fresh adaptive-rho dense solve at 1e-10 with 10000 maximum
  iterations, then explicit original-coordinate active-set refinement
  (three corrections, active guess 1e-5/1e-7, independent acceptance 1e-7).
  Refinement cannot turn max_iterations into strict solved. Rejected/unknown
  proposals retain the restored candidate; last proposal/checks are saved too.
- OSQP: fresh workspace, library default scaling, strict 1e-10 solve target,
  10000 maximum iterations, no warm start. Effective settings are saved.
- Clarabel: library default equilibration, strict 1e-10 solve target,
  200 maximum iterations, one thread, presolve disabled to retain transparent
  finite-bound mapping. Original signed y and cone z/mapping are saved.

The Openopt arm changes both scaling and postprocessing and tightens the solve
target; reference arms change scaling and targets. This is a combined robustness
experiment, not an isolated causal scaling ablation or a fair timing race.
Original strict sign/range/gap rules are never replaced by backend status.

## Evidence And Replay

`run --baseline PATH --baseline-sha256 PIN --artifact-dir NEW_3FS_DIRECTORY`
copies pinned baseline/source files and saves one worker result/vector/
transformation sidecar per case/arm. Parent execution has an explicit at-most
180-second wall budget including input preflight and worker startup; exhaustion
is BUDGET_EXHAUSTED, never success. Logs retain worker errors and original
causes. Each worker runs one solver call; refinement is a candidate
transformation rather than a second solver attempt. Replay never solves.

`replay --result PATH --result-sha256 PIN` verifies raw byte hashes, exact
sidecar inventory, original baseline replay, case/arm/source membership,
unchanged model hashes, original complete independent checks and all failed
denominators. It checks scaling restoration and refinement nonworsening/
unchanged-rejection semantics and reconstructs Clarabel's signed dual mapping.
Frozen producer source must match the replaying producer. This is diagnostic
integrity, not an exact mathematical certificate or public qualification PASS.

Source baseline is `e566b0b8163f099a29a0d554fa3b87ad15e4fb67`.
The infra edit router rejects cross-repository targets as outside its root;
applicable openopt AGENTS and the explicit three-file user scope govern this
work. Shared design/report pages are outside scope, not edited here.
Only the main coordinator submits cluster work; this seat does not.

## Retained Failure And Real Results

The initial cheap unit run is retained at
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/unit-v1/tests-yfgpvmk0/`.
It recorded 7 passed, 3 failed and 9 errors. The first cause was producer misuse
of the importer's exclusive-create JSON writer for updating a worker row,
progress and result. No real model was run in this failed phase. Worker-original
and parent-final rows now have distinct files, progress files are append-only
numbered snapshots, and terminal result is written once. Unit-v2 passed 19;
unit-v3 passed 22 including status-upgrade and exact Ruiz-factor replay controls.
Old evidence is retained, not counted as acceptance or overwritten.

The one real batch finished `2026-10-10T09:56:26.886824+00:00` in
`25.376415964012267` seconds, below its explicit 180-second budget:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/real-v1/result.json`.
Result SHA256:
`aa584b62f97349f37afad438dfbd87320ace3408df4d7e574a0989370970e8e7`.
It replayed baseline 6/6 and new 6/6 at `2026-10-10T09:56:53.327460+00:00`.
New denominator is six, all attempted, zero FAIL/budget-exhausted, five
UNQUALIFIED and one numerical QUALIFIED. Terminal execution status is
`EXPERIMENT_COMPLETE_NOT_QUALIFICATION`; numerical and original verdicts
remain UNQUALIFIED. Execution exit zero means completed evidence, not all-pass.

| Arm | afiro | adlittle | New Qualified / Denominator |
| --- | --- | --- | --- |
| Openopt Ruiz + refinement | max_iterations 10000 | max_iterations 10000 | 0 / 2 |
| OSQP default scaling, strict | solved, strict sign fails | max_iterations 10000 | 0 / 2 |
| Clarabel default equilibration, strict | full numerical gates pass | component complementarity fails | 1 / 2 |

Both Openopt refinements are unknown with `singular_or_unresolved_kkt`; no
candidate proposal was accepted or baseline value replaced. Adlittle original
feasibility/stationarity/complementarity maxima are
`0.0006463185027314466`, `0.06076771075350962`, `0.6428541482602704`.
Afiro scaled primal/dual residuals are 1.7053025658242404e-13/
1.700029006457271e-15, already below 1e-10. Strict infinite-support/sign evidence
still blocks full gap termination (wrong-side multiplier 5.551115123125784e-16).
It is not a failure to reach the primal/dual residual target. The previously
passing original baseline candidate is not replaced by this failed arm.

Afiro OSQP has original feasibility `1.2660450465773465e-9`,
stationarity `2.977019359695254e-11` and complementarity maximum
`5.852098652344556e-9`, but wrong infinite-side multiplier
`2.8355450826891544e-17` still makes support/dual/gap unavailable.
Adlittle OSQP has feasibility `0.002561842997682362`, stationarity
`0.05176518931972396`, complementarity `4.516118764172443`, plus strict-sign
failure. These are failed numerical candidates, not hidden convergence wins.

Afiro Clarabel passes with original signed gap `2.090996531478595e-8`.
Adlittle Clarabel passes feasibility/stationarity/sign/range/gap, but its
componentwise complementarity fails (maximum `1.1127175773220218e-7`);
the unrelated maximum threshold `0.013005056723289846` cannot mask it.
Original dual range residual is `1.2423939210748358e-8` and signed gap
`4.293484380468726e-6`. No tolerance is widened to promote this candidate.
This combined-arm experiment does not isolate scaling causality and does
not establish general robustness, a speedup, or two-problem qualification.

Caller inputs remain:
baseline SHA256 `b10d49a32a13a9f34b8bf86ea6bd8a126dc779bf72a47db3f069420f653456ef`;
corpus SHA256 `3a434edfa8165167f56be25732840f03bbace1a5f5b7d1de7cdab989673d3f0f`;
source acceptance SHA256 `1f82c8df6e9258bd8598884dd2bfee77f9e97f2ce7ec9f46035c5604822519fa`.
All raw bytes, arrays/model hashes and offsets are verified in the copied
baseline. Model SHA256 is
`16cb1249ad14facc394b318d3da4b2270390ea3d1c5537388aa798ca779d7d96`
for afiro and
`fb4c43a1366a429976b185e4841f302da1357100a2c42f6ca0fd23e7614051f5`
for adlittle. Original checker SHA256 remains
`e7ece6f726bb7613d504ff6d84eedc20fb1a5879213a653680a26f8a64eb4c9c`.

The external v2 replay receipt entry accepts `--check-result V2_RESULT
--sha256 CALLER_V2_PIN --output NEW_EXTERNAL_JSON`. It writes outside the
sealed inventory and never solves. There is no real v2 result/pin yet.
Do not use it to update the historical v1 receipt; use main's retained efc1567
source for v1 replay, with the P2 limitation explicitly carried forward.

## Historical V1 Acceptance

Existing receipt:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/real-replay-v1.json`,
SHA256 `c17f1f02d0350aab55c4fd1cca94b42e776bf4d1668eb1bea6f3ff5cb9c06e5a`.
Its independent read-only replay at `2026-10-10T09:58:19.122842+00:00`
checks old 6/6 and new 6/6, retains all five new failures, and round-trips JSON
with qualification_upgraded=false.

Final targeted CPU run:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/final-tests-v1/tests-grgz29il/verification.json`,
finished `2026-10-10T09:59:43.572573+00:00`: 213 passed, ten explicit CUDA
skips, zero failures/errors. Source consistency PASS, 254 files. Targets are
this experiment (23 tests), existing scaling/refinement and qualification.
Negative controls include caller pin, vector bytes, resealed transformation,
tolerance, denominator, baseline verdict, clipping claim, extra inventory,
source membership, CPU budget and failed-worker accounting. The synthetic
accepted-refinement/max_iterations control explicitly keeps UNQUALIFIED.
Unit fixtures are not real public or industrial coverage.

Final CPU smoke:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/final-smoke-v1/smoke-sg8yaurb/verification.json`,
finished `2026-10-10T09:58:29.732412+00:00`: 80 passed, four CUDA skips,
zero failures/errors, source consistency PASS. Counts overlap, not additive.
No full-repository, wheel, GPU, commercial, speed or legal acceptance is claimed.

The verified producer/test SHA256 pins are
`8ceb0f2b0dd732fccfae1f1c13afdf585ee9aba0a59cbb8038fd4c4874d42e18` /
`3d336ab8b71a63be04259e2b80bdd888d99dbcb52574a8fe4c2eb196625d409c`.
These tests ran on the stated base commit plus exactly the three scoped
untracked files; this final evidence section is a documentation-only update.
The original negative result SHA256 is still the caller pin. Main source,
solver/checker/runner/importer and reports have no changes in this scope.
