# Cluster CPU Sharded Regression

`scripts/cluster_sharded_regression.py` transports the existing sharder to
zero-GPU Kubernetes CPU Jobs. The development seat only implements and tests
this entry with a fake Kubernetes transport. Actual image compatibility,
cluster placement and end-to-end full regression remain UNVERIFIED until the
authorized coordinator submits and collects real Jobs.

## Frozen Source And Runtime

First use `verify_sharded.py plan` on the final clean commit. Its complete
pytest nodeids, original source hashes and file-LPT assignment are preserved.
The cluster entry creates an external depth-one Git clone of the exact source
branch, checks out the frozen commit and removes the remote. It checks all
source files and commit/tree identity before and after; no credentials or
unrelated source history are copied. The cloned Git data is required by the
existing `verify.py` source gate and source-sensitive tests.

The cluster plan refers to that shared source rather than a host-only path.
The runtime probe executes the image's `python3` by default. It runs the
existing `versions()` minimum gates, imports every required dependency plus
`openopt`/`openpo` from the frozen source, records Python/executable/packages,
checks Git and rejects visible CUDA. Missing dependencies fail and are
preserved with partial dependency inventory and the original error. Optional
Gurobi/CPLEX/MOSEK/cuOpt discovery is recorded without importing or solving;
discovery is not licensed execution and commercial qualification remains
UNVERIFIED. Only an authorized existing runtime with the required dependencies
can proceed. There is no pip/apt installation or host-environment fallback.

All later workers require byte-pinned probe evidence and identical runtime
fingerprints. Kubernetes collection also checks a single successful Pod,
owner UID, zero restarts, observed resolved image digest, command and frozen
Job/Pod settings. A mutable image tag alone is not runtime identity.

## Admission And Failure

The coordinator is the only submitter. Each explicit `run --submit` admits
one ready stage; `collect` takes a bounded, immediate snapshot instead of
blocking or automatically submitting more work:

1. `PREPARED -> PROBE -> PROBE_PASSED`: one zero-GPU runtime/collection probe.
2. `PROBE_PASSED -> SMOKE -> SMOKE_PASSED`: the predicted most expensive whole
   shard, using exactly the existing `verify.py` explicit target entry.
3. `SMOKE_PASSED -> SHARDS -> SHARDS_PASSED`: remaining shards submitted
   independently, eligible to run in parallel. Actual overlap is not assumed.
4. `SHARDS_PASSED -> CHECK -> PASS`: same-runtime check Job calls the original
   sharder `check` on pinned combined report/JUnit and all per-shard sidecars.

No retries or object deletion/TTL cleanup are configured. Every Job has a
deadline at most 300 seconds; shard execution uses its existing plan deadline.
Timeout/failed Job/nonzero container/partial output/runtime drift blocks
further admission and retains the first cause, raw Job/Pod query evidence,
container logs when available, worker logs and each shard's JUnit/report.
Already submitted Jobs are not silently canceled or counted as unsubmitted.
`ADMITTING` is a deliberate fail-closed recovery state after interruption:
the coordinator must inspect receipts, never blindly resubmit it. `COLLECTING`
is likewise reserved while one coordinator reads evidence; a second collector
cannot update the same ledger. Previously accepted source/runtime/Job/Pod
evidence is re-read before new admission. Job membership is checked explicitly,
including exactly one probe, every shard once, and one independent check.

Kubernetes commands only use the caller's existing environment. This tool
does not read credentials, modify roles, install dependencies or use a GPU
runtime class. Pods disable service-account token mounting, request only CPU
and memory and set both CUDA and NVIDIA visibility off.

## Coordinator CLI

Only the main coordinator may execute `run --submit`. The development seat
must not run any real Kubernetes create operation.

Regenerate the sharder plan only after final integration and a clean commit.
Append the new measured cost input after the older JUnit histories:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify_sharded.py plan --repo /data/dev/openopt --history-junit /hf3fs-jd/data/tmp/openopt-release-20261010/full-merged/tests-wx7hh8vg/pytest.xml --history-junit /hf3fs-jd/data/tmp/openopt-release-20261010/merged-release-tools-frozen/tests-gylzh4zs/pytest.xml --history-junit /hf3fs-jd/data/tmp/openopt-release-20261010/merged-native-gates-v1/tests-0962z9dj/pytest.xml --shards 3 --shard-timeout 300 --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/shard-cluster/final-plan
```

The last input contains 264 testcases: 238 passed, 26 explicit CPU/CUDA skips,
JUnit wall 199.388 seconds. These are overlapping cost observations, not
additional full-regression passes. Later JUnit overrides exact old testcase
times; new unknown nodeids remain explicit. Also append the dedicated tool
JUnit once available so its roughly two-minute cost is not assumed zero.
Never execute an old worktree's plan as final current-main qualification.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/cluster_sharded_regression.py plan --repo /data/dev/openopt --plan <sharder-plan-path> --plan-sha256 <sharder-plan-pin> --image <existing-cached-image> --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/shard-cluster/final
sudo -n -u infra env KUBECONFIG=/home/infra/.kube/h3c-gpu-admin.conf PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/cluster_sharded_regression.py run --plan <cluster-plan-path> --plan-sha256 <cluster-plan-pin> --submit
sudo -n -u infra env KUBECONFIG=/home/infra/.kube/h3c-gpu-admin.conf PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/cluster_sharded_regression.py collect --plan <cluster-plan-path> --plan-sha256 <cluster-plan-pin>
sudo -n -u infra env PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/cluster_sharded_regression.py check --plan <cluster-plan-path> --plan-sha256 <cluster-plan-pin>
```

Repeat collect until the current stage passes, then explicitly run the next
stage. BLOCKED is not a retry prompt. Default namespace/PVC/service account/
priority match the existing devtest channel; `--node` can pin a coordinator-
selected node but must not be used to hide a failure. `--python` is a container
executable, not the host validation environment. Artifact paths must resolve
inside shared 3FS temporary storage.

The `worker` action is only a Job command. Existing sharder CLI remains
unchanged; the cluster wrapper retains `plan/run/check`, adds `collect`, and
uses `--plan --plan-sha256` for its frozen cluster plan after preparation.
The final combined `verification.json` uses the existing tests/source schema
and `coverage="tests"`; only CPU correctness is claimed. Host `check` reads
the accepted container replay rather than substituting host Python for it.
Historical 216-second predictions are not container costs or measured speedup.

## Development Acceptance

The development seat's CPU-only fake-transport tests use tiny, clean external
Git fixtures and real `verify.py` worker subprocesses. Fake Job/Pod/log samples
are explicitly labeled; their image digests and runtime fingerprint are not
cluster evidence. A separate local CPU runtime probe checks actual dependency
discovery/import behavior but is not a container probe.

Coverage includes exact collection/source/JUnit replay, sequential probe and
heaviest-shard admission, pending-state refusal, zero-GPU manifests, missing
runtime dependencies, Python/package drift, failed/extra/restarted Pods,
image/owner/source/sidecar tampering, rejected duplicate/missing stage Jobs,
partial transport failure and stopped subsequent admission. No test installs
dependencies or makes real Kubernetes calls.

Final dedicated development verification on 2026-10-10:
`/hf3fs-jd/data/tmp/openopt-release-20261010/shard-cluster/unit-v3/tests-fvpabaju/verification.json`
reports 25 passed, zero failures/errors/skips, pytest wall 122.87 seconds.
The same directory contains `pytest.xml`; append it after older cost inputs
when generating the final source plan. Its external pytest fixture preserves
all five fake Job receipts, actual worker subprocess reports, combined JUnit
and independent replay acceptance. Do not count its three tiny testcases as
optimizer/public-corpus qualification.

Earlier dedicated attempts remain under `unit-v1/tests-oqkj89qf/` (18 passed,
96.92 seconds) and `unit-v2/tests-f6hyeiz3/` (22 passed, 110.05 seconds).
Their counts overlap and predate later admission/tamper controls. No
controlled serial/cluster speed comparison or actual container execution
was performed by the development seat.

## Explicit Fixture V2 Transport

This replacement seat starts at main `5920c7f` and imports its own sharder
commit `48e1204` into private branch
`codex/release4-20261010-cluster-fixture-v2` (private cherry-pick `8f5f5b4`).
Only this cluster entry, its test file and this document are changed next.
The sharder/verify implementation, production optimizers and cluster are not
modified. The main coordinator must integrate `48e1204` before the new wrapper.

Version boundaries are explicit:

- Existing sharder/config v1 remains the default file-LPT path. It cannot claim
  fixture grouping, and `--fixture-aware` with a v1 plan is rejected.
- The sharder v2 plan needs its own `--fixture-aware --fixture-file ...` input.
  The cluster `plan` additionally needs `--fixture-aware`; omitting it rejects
  a v2 source plan. Subsequent run/collect/check/worker consume pinned contracts,
  not new command-line grouping overrides.
- The cluster config v2 and transport contract
  `pytest-source-pinned-fixture-transport-v2` preserve the complete host
  collection sidecar, audit, selected files, closure, constraints, isolated
  builtin policies, components, historical leaf costs and explicit targets.
  Nothing is reconstructed from nodeids alone.

The host pytest installation paths are not container runtime paths. The probe
maps each audited `_pytest` module to the container installation only after
exact pytest version, module name, installation origin and source SHA256 match.
The private numerical/runtime source is not copied, installed, or relaxed.
Unknown origin/module, missing dependencies, different source bytes or version
fail before execution. A compatible existing runtime is required; do not
install SDKs/packages or substitute the host Python.

The probe writes a byte-pinned `runtime-plan.json` and
`runtime-collection.json` outside the source tree. Mapping only replaces
runtime source paths and corresponding isolated-builtin origin paths.
Source pins, fixture dependency edges, connected components, assignments,
costs and budgets remain identical. Every shard recomputes the full v2 pytest
collection/audit before calling the original verify entry. The final check
uses that same pinned runtime plan; the host independently reconstructs its
allowed path-only transformation and checks all worker/collection/summary pins.
The summary retains schema 1 with `coverage="tests"`, adding
`plan_schema_version=2`, source transport plan pin and transport contract.

A genuine user session/module shared component cannot be broken up by this
transport. Overbudget shared components remain FAIL. This does not repair
the previously measured heavy `produced` component or expand any deadline.
Full repository collection in a real container remains coordinator-owned.

## Bounded Independent CPU Acceptance

The test-file entry partitions all wrapper tests into disjoint v1 and v2
batches. Each invokes the real existing verify entry under a 179-second
external process deadline, retains timeout/nonzero first cause, and checks
actual JUnit, source before/after, process/log paths and receipt hashes.
No actual Kubernetes method is used; a fake transport drives real tiny CPU
worker subprocesses.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_cluster_sharded_regression.py --batch v1 --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/cluster-fixture-v2/reaccept
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_cluster_sharded_regression.py --batch fixture-v2 --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/cluster-fixture-v2/reaccept
```

Initial v2 minimal acceptance is three passes/39.02 seconds at
`minimal-v1/tests-cm6fcw69/verification.json`.
The first disjoint bounded batches are v1 25 passes/123.45 seconds at
`bounded-v1/v1-51kx1m9a/verification/tests-nhq1jaef/verification.json`, and
v2 eight passes/62.81 seconds at
`bounded-v1/fixture-v2-8pje0t7v/verification/tests-y61oxgft/verification.json`.
Both source gates pass. These precede final origin/summary-binding controls;
they are not clean final-source evidence. Paths are below
`/hf3fs-jd/data/tmp/openopt-release-20261010/cluster-fixture-v2/`.

Controls include exact cross-file session-component preservation, full
function-fixture audit, unaudited wholefile preservation, host-to-runtime
path mapping, explicit opt-in rejection, version/module/hash/origin mismatch,
nodes-only sidecar rejection, rehashed constraint forgery rejected by real
fresh collection, mapping/plan/summary tampering, and complete three-shard
five-testcase JUnit replay. Tiny counts are test harness evidence, not five
optimizer instances, commercial qualification, container execution or speed.

Final precommit bounded verification includes the origin/summary controls:
v1 25 passes/120.10 seconds at
`bounded-v2/v1-iliqz61f/verification/tests-3detpfz5/verification.json`;
v2 nine passes/65.27 seconds at
`bounded-v2/fixture-v2-fqqedz9u/verification/tests-b_s5drjj/verification.json`.
Both parent `acceptance.json` receipts bind complete source, process, tests and
verification SHA; source consistency is PASS, failures/errors/skips zero.
The v2 process wall is 72.09 seconds, distinct from pytest wall.
Postcondition smoke is 80 passes/four explicit CUDA skips/6.17 seconds at
`post-smoke-v1/smoke-rsbhjm7l/verification.json`, source-consistency PASS.
These receipts bind dirty precommit source bytes; clean-source reacceptance
uses the same two bounded commands after this three-file commit, retaining
separate evidence rather than rewriting these receipts.

## Preparation Failure Receipts

The preparation-receipt repair starts from frozen `ba675eba`, independently
of main integration. It changes only this wrapper, its tests and this document.
Sharder grouping, cost estimates, CPU capacity, Job deadlines, concurrency,
runtime transport and admission state transitions are unchanged.

Every preparation that reaches the authorized external artifact allocation
gets its own `cluster-*/prepare.log` and `prepare.json`. Both successful and
failed CLI preparation records include the actual Python executable, PID,
CLI argument list, caller-pinned input-plan location/hash, elapsed time,
returncode, first cause, terminal verdict and log SHA256. Library calls are
explicitly labeled library invocations rather than fabricated CLI executions.
No credentials are read by the receipt code; logs contain CLI parameters and
tracebacks, so callers must not pass secrets as parameters.

A pinned source plan already marked FAIL is still rejected by the original
validator before clone/config/run-ledger creation. Its original `error` is
retained separately from the immediate `invalid_plan_status` gate error.
Missing files, wrong pins and opt-in mismatch retain their own actual errors.
Each attempt uses a fresh directory; re-running does not overwrite an earlier
receipt, log or failed input. Existing success config/plan/ledger bytes retain
their original semantics apart from the already expected output relocation,
new hashes and creation time. New diagnostic sidecars are not config fields
and are not runtime or regression qualification.

Allocation failure outside authorized shared 3FS, an inaccessible artifact
root, abrupt process termination, or a failure while writing the receipt can
still prevent a terminal receipt. Those failures are not success and have no
fallback to local storage. Argparse rejection before preparation is also
outside this receipt boundary. Do not claim broader durable-CLI coverage.

The existing CLI needs no new flags:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/cluster_sharded_regression.py plan --repo /data/dev/openopt-wt-release4-20261010-cluster-fixture-v2 --plan /hf3fs-jd/data/tmp/openopt-release-20261010/cluster-fixture-v2-review/frozen-ba-cost-plan/plan-pq45_4do/plan.json --plan-sha256 7fc799fab4f4b18fe9d5fbec02a3a6b888278f60cf63bb2ea70cd373a2c5b9a7 --fixture-aware --image m.daocloud.io/docker.io/lmsysorg/sglang:dev-dsv41 --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/cluster-prepare-receipt/real-budget-refusal
```

That historical ba input is FAIL, not an admissible final-source plan. The
command must return 1, retain its source first cause and emit the external
receipt coordinate; it must not create a cluster config or submit any Job.
The old 507.420-second mixed prediction has stale known histories and unknown
costs. Owner same-source remeasurement remains required; this repair never
lowers unknown estimates or extends a deadline.

The real CLI controls use external clean tiny Git fixtures, bounded caller
process logs and the actual wrapper CLI, not mocked `prepare`. They check FAIL
for a failed source plan, wrong hash, missing file and wrong opt-in; duplicate
failed calls must preserve old bytes. V1 and explicit v2 successful calls
compare exact config bytes with the original expected semantics and retain
full transport metadata. Neither successful CLI preparation admits a Job.

Development `minimal-v1/tests-_f85rxu7/verification.json`, below
`/hf3fs-jd/data/tmp/openopt-release-20261010/cluster-prepare-receipt/`, retains
five passing assertions but a final source-consistency FAIL caused by an edit
during testing. It is not accepted evidence. Subsequent fixed-source batches
use the existing 179-second test-file entry and separate external artifacts;
never rewrite the failed receipt or promote its pytest returncode to PASS.

The next combined v1 batch
`bounded-v2/v1-df26xw4u/acceptance.json` hit its 179-second process deadline
and remains FAIL/worker_timeout, with the partial log retained. V2 at
`bounded-v2/fixture-v2-_shab53s/acceptance.json` passed ten cases and source
consistency in 109.410349 process seconds on unchanged dirty development bytes.
Neither is clean-commit acceptance.

New CLI controls now use a separate `--batch prepare` (five cases); existing
v1 remains 25 cases, fixture-v2 contains ten cases including its CLI success
transport control. Name classification assigns every test to exactly one
batch without reducing production collection or changing any production
parallelism/deadline. All three batches retain the original 179-second gate:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_cluster_sharded_regression.py --batch prepare --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/cluster-prepare-receipt/reaccept
```

Fixed-source development prepare acceptance is five passes/zero skip at
`bounded-v3/prepare-jrwvbhxr/acceptance.json`, process wall 45.826756 seconds,
source consistency PASS over 285 files. This is dirty precommit evidence.
The actual historical budget refusal at
`real-budget-refusal/cluster-vtja_z5c/prepare.json` retains the original
507.420-second source error, validator error `invalid_plan_status`, CLI
returncode 1 and external log SHA256
`5c8eab0eb9de4696b9c9520fb79ba92e7792dfd500d283f24e0f537c283f0bea`.
Its receipt SHA256 is
`a48b24e1ca9e41033b4dd6009ce76d27c5c55b906b4f4461e37922cd9480639a`;
no config, source clone or run ledger is produced. This is a real local CLI
rejection of historical input, not current-source cost remeasurement.
After commit, clean reacceptance runs all three disjoint batches under
`clean-v4/`, preserving development negatives unchanged.

Clean `9c46c8e` reacceptance retains v2 ten passes at
`clean-v4/fixture-v2-trzdemdf/acceptance.json` (114.536962 process seconds),
and prepare five passes at `clean-v4/prepare-4gueyrhn/acceptance.json`.
The v1 batch `clean-v4/v1-asblh9im/acceptance.json` again timed out at 179
seconds and remains FAIL. No complete-suite PASS is claimed for that attempt.
Both earlier v1 timeouts ran alongside another batch; contention is a possible
factor, not a measured root-cause attribution.

Explicit local test-only `--batch v1-a` and `--batch v1-b` now partition the
original sorted v1 test-function targets by even/odd index. Their union is
exactly the unchanged `--batch v1` selection, with zero overlap. The tiny
fixtures are function-scoped; no module/session user fixture is split.
Production grouping, admission, concurrency and deadlines remain unchanged.
For this bounded acceptance use v1-a, v1-b, fixture-v2 and prepare, and check
their actual JUnit identities against the complete test-file collection.
Each invocation still has its own original 179-second gate. New clean evidence
lives under `clean-v5/`; failed v4 evidence is never relabeled.

## Archived Execution Contract Repair

The diagnostic commit `9c46c8e` did not fix archived acceptance. Nash's
unchanged five-case replay retained one positive and four false accepts
at `cluster-prepare-receipt-review/frozen-9c46/coordinated-irvu_z4v/`.
Those failures remain historical red evidence, separate from preparation
diagnostics and RSOC numerical/GPU evidence.

`intended_manifest` now regenerates the exact Job manifest and stage item
from caller-pinned config/plan, phase/id, and the current accepted runtime
and report pins. The archived name/output/manifest location must match that
stage item, and the rehashed archived manifest must equal the regenerated
manifest. Both prior-stage `verify_accepted` and active-stage `collect` use
this check. An internally consistent rewritten archive cannot become its
own expected execution. A rewritten report pin also invalidates a previously
accepted check Job's command before later summary replay.

`terminal_job` independently compares `command`, `args`, `env`, and `envFrom`
lists exactly for both Job and Pod. A missing list is equivalent only to
an explicit empty list; null, extra arguments, altered/duplicate/extra
environment entries, valueFrom additions, and unsubmitted envFrom sources
are not equivalent. Reviewed defaulting outside those fields remains under
the existing subset check; CPU resources, UID, image digest, restarts,
termination, original source/JUnit and all numerical gates are unchanged.

Owner controls include a baseline-positive archived replay, all four Nash
mutation shapes plus envFrom with synchronized rehashes, unchanged config,
stage path/name substitutions, twelve Job/Pod execution mutations and
active-stage refusal before any transport query. The existing v1/v2 real
CPU worker chains remain necessary regression coverage. No real Kubernetes
call, GPU execution, numerical solve replay, deadline increase or main-tree
edit is part of this repair. Independent closure requires Nash's unchanged
controls against the new committed source, not old receipts or owner claims.

Same-byte precommit owner evidence below `cluster-prepare-receipt/`:
`manifest-targeted-v1/tests-1m_mvrai/verification.json` reports 14 passes,
zero skip/fail/error, 14.21 pytest seconds and source consistency PASS.
`manifest-chain-v1/tests-ul8m_6df/verification.json` reports both original
v1/v2 real CPU worker chains passing in 76.51 pytest seconds with source
consistency PASS. These are development receipts, not clean-source
qualification. The three explicit files also passed the existing scanner
selftest, credential-literal and shape gates. Postcommit receipts must bind
the new commit and unchanged bytes, with the independent controls kept intact.
