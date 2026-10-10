# Fixed13 Independent Numerical Acceptance V1

Only scripts/accept_rotated_cuda.py, tests/test_accept_rotated_cuda.py and this
document belong to this task. Base acc199306936922b6d75330884310a58cde84a9b.
The separate cluster review red gate and unchanged counterexamples remain
retained. This does not repair that tool or authorize GPU submission.
The 76-case extreme campaign is a separate denominator and tool.

## Arithmetic And Bounds

Algorithm fixed13_original_rational_decimal_v1 never imports Torch, NumPy,
the optimizer, the producer, or archived executable source. It never solves,
projects through production code, or manufactures absent evidence.
All finite represented binary64 coordinates are decoded exactly as rational
numbers. Original Ax+s-b, Px+q+A.T*z, individual component tolerances, ordered
cone checks, per-block signed-product complementarity, diagonal PSD range,
primal/dual objectives and signed gap use exact rational operations.
Only the frozen diagonal {0,1} Hessians, n in {1,3}, m in {3,4}, one zero
equality plus RSOC(3), or single RSOC(3/4), are supported. Other inputs fail.
No eigensolver or regularized inverse substitutes for dual availability.
RSOC checks both signs and required-small-head, not a global norm surrogate.
Original eps_abs/eps_rel are unchanged: model binary64 1e-8/0, binary32
3e-5/0; projection 1e-12 with binary64 3e-14 or binary32 3e-6.

Six nearest-point oracles use a direct RSOC coordinate elimination at bounded
Decimal precision 800, checked independently at precision 1000 for the same
represented binary64 result. Comparison uses each component, atol=0 and
the producer's unchanged rtol, including the tiny head. Interior preservation
is bitwise. Projection also checks the original cone, the negative residual
in the self-dual cone, and Moreau orthogonality with homogeneous relative
3e-14/3e-6 signed-product limits. This is bounded test-only arithmetic,
not arbitrary-precision optimizer support, a generic solver or performance.

All original 9 LP/QP checks (8 for the projection feasibility witness) and
three cone evidence objects must have complete exact membership and identical
boolean outcomes to independent arithmetic. Recorded floating metric bytes
remain caller pinned; independent rational residuals are separately named.
This is NOT approximate dictionary matching and does NOT claim bitwise replay
of an unversioned BLAS metric algorithm. Numeric-replay is research only.

## Evidence Contract

Exactly 13 identities from tests.test_rotated_conic_cuda:
four original-model dtype pairs, six fixed projection IDs and three original
rejections. The rejection baseline must pass every original gate; the
candidate must reproduce the exact mutation and specified failed checks.
No requirement makes a legitimate rejected candidate an accepted solution.

Each exact JUnit case has exactly path and sha256 properties for one unique
rotated-conic.json within the pinned result's pytest-tmp. Complete identities
and file membership are bidirectional, with zero failure/error/skip.
No sidecar from another result, CPU placement, omitted original vector/model,
extra target or frozen configuration mutation is accepted.

Caller pins use an external JSON manifest:

```json
{
  "schema_version": 1,
  "run_root": "/absolute/run",
  "files": {
    "run": {"path": "/absolute/run/run.json", "sha256": "64 lowercase hex"},
    "result": {"path": "/absolute/run/case-0/cuda-check-ID/result.json", "sha256": "64 lowercase hex"}
  }
}
```

The complete required files are run, result, junit, submitted_job, status,
pod_log, pytest_log, execution_log, telemetry, source_manifest, outer_result.
Each binding has exactly path and sha256. Caller additionally pins the
manifest bytes with --pins-sha256. seal writes a new external bundle.json;
accept additionally requires --bundle and --bundle-sha256 and repeats all
checks before exact full-bundle comparison. Both require clean tool source.
No old allowlist is widened.

Archive checks reconstruct the exact command/environment independently of
submitted and actual Job/Pod. Extra args/envFrom or coordinated environment
changes reject. Only fieldRef apiVersion absent versus v1 and GPU resource
integer 1 versus string "1" have explicit narrow normalizations.
Whole execution specs also compare exactly after enumerated API defaults
(DNS/scheduler/grace period, serviceAccount alias, termination-message defaults,
Pod scheduling defaults/two standard tolerations and the exact standard
serviceAccount projected mount). Unknown defaults, extra workingDir or injected
mount/source fields reject. The projected token is a schema/path description;
no credential value is read or retained.
Single GPU, sole Pod, owner UID, no restart, completion/exit, image digest,
node/time, actual Pod completion log, inner/outer result bytes, full JUnit,
all telemetry sample count/order/time/UUID and runtime placement are checked.
All frozen sources must equal run manifest, run commit tree and original
commit bytes; all src Python files plus producer also equal scope-current
bytes and each sidecar's complete hash membership/loaded sources.
An auditor absent from the historical GPU snapshot is explicitly reported as
a later CPU audit, not retroactive same-commit GPU execution.

## Historical Input And Limits

Historical run owner-e3d5511-rsoc-cuda-v1/release-regression-6zusdrx_:
run SHA db76ee1680d1b8cbe5b199a4ea5434e49b2c9c28e5146fee552c3ec1f68c0532;
case-0/cuda-check-rhrgvjq6/result.json SHA
4847170c52e669e8152fcd0d2e63bf552e263c28805b1d865eda01769952946c.
Raw execution has 13 passed, zero skipped, 49 telemetry samples.
Do not trust its previous execution seal or proof flags as numerical authority.

numeric-replay uses a distinct pinned manifest with exactly schema_version=1,
result_root and junit binding. It only reads thirteen frozen original
sidecars. It never emits a seal or archive/source qualification. Missing
evidence fails loudly. Tests are self-contained synthetic CPU controls;
historical checks require OPENOPT_RUN_ROTATED_EVIDENCE_TESTS=1 and explicit
paths. Missing opt-in means skip, not qualification.
The separate complete archive test requires OPENOPT_RUN_ROTATED_ARCHIVE_TESTS=1,
OPENOPT_ROTATED_PINS and OPENOPT_ROTATED_PINS_SHA256. Mathematical history uses
OPENOPT_ROTATED_RESULT_ROOT and OPENOPT_ROTATED_JUNIT_SHA256. Missing required
variables after opting in fail, rather than silently skip.

## Actual Receipts And Retained First Causes

All paths below are relative to the external evidence prefix
/hf3fs-jd/data/tmp/openopt-release-20261010/fixed13-numeric-audit/.

- historical-inventory-v1/inventory.json is inventory only, not acceptance.
  Manifest pins SHA a0d802d9e496df48c6ad81cf016758047f8d07860c93af75d72b1e7784128b51;
  numeric pins SHA 1c6a42a7f18b00d98cf3c2955a8139a4037f1adcc819c2ef5078a5a67dfbbddf;
  JUnit SHA 7fe57255ea0a93b7533005544f8c79d89283e85e9dd3c96c328e95987a8bfee2.
- historical-numeric-v1/numeric-replay.json retained 6/13 pass and seven rejects:
  the first tool model contract mistakenly encoded -eye(3)'s negative-zero
  entries as positive zero. The frozen A bytes remain unchanged. The repaired
  exact model contract includes -0.0, rather than normalizing or approximating
  dictionaries. historical-numeric-v4/numeric-replay.json has 13/13 PASS,
  four original LP/QP positives, six independent projection positives and
  three correctly rejected candidates with accepted unmutated baselines.
- development-v2/tests-a8n9lem_/verification.json is FAIL despite 37 unit passes:
  source was edited while source-consistency validation was in flight. It is
  not clean evidence. Later verification kept the tree stationary.
- development-v3-optin/tests-tw_t5zd2/verification.json retained one FAIL and
  39 passes. The tool first misread the telemetry's second column as GPU index.
  Frozen benchmark_general_qp.py explicitly samples
  uuid,utilization.gpu,memory.used,power.draw. The repaired parser validates
  those actual finite ranges and one stable UUID, not a made-up index column.
- development-v4-optin/tests-ah97tcp9/verification.json has 40 passes,
  zero skips/failures/errors, pytest 45.58 seconds and source-consistency PASS.
  This development receipt is not a clean committed freeze.
- scan-v1/scan.json has three explicit files, known credential literal hits=0
  and credential assignment/private-key shape hits=0. Needles stayed in memory.

Clean committed receipts are produced separately under clean-full40,
clean-default, clean-smoke, historical-seal and historical-accept. No named
directory here is a PASS merely because it is named; inspect its actual JSON.
The historical mathematical and archive audit is a later CPU read of actual
frozen CUDA vectors, not a claim that the new auditor executed in that Pod.

Example full archive command, with caller-reviewed pins:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python \
  scripts/accept_rotated_cuda.py seal --pins /absolute/pins.json \
  --pins-sha256 CALLER_SHA256 --artifact-dir /absolute/fresh-audit-directory
```

Acceptance repeats the same command with mode accept, --bundle pointing to the
sealed bundle.json and --bundle-sha256 supplied independently by the caller.
Neither command uses an old proof PASS as an input criterion.

Commercial performance, adversarial runtime attestation, general conic scope,
76 extremes, transfer-inclusive speedup, multi-GPU, licensed baselines and
full release qualification remain UNVERIFIED.

## Fixed13 Identity And Type Repair

Repair base 75b54762e9fd37e334f9f34064bd7436c55c1250; the three-file scope
remains unchanged. The independent ead32bdd suite and its original
43-pass/11-fail receipt are retained, not edited or used as a developer
qualification run. Main's separate reviewer must replay those unchanged
54 tests against the eventual repair freeze.

Submitted and actual resources must be batch/v1 Job and the actual Pod must
be v1 Pod. Its sole controller owner must name batch/v1 Job with the same
UID/name. ImageID must match a nonempty whitespace-free image prefix plus
exactly @sha256: and 64 lowercase hexadecimal digits. Runtime-qualified
prefixes such as docker-pullable:// remain legal. This syntactic digest
check is not a signature check, registry validation or remote attestation.
No new Pod-list top-level metadata is required of the existing portable
fixture. All actual per-resource identities remain mandatory.

Schemas, Job retry/completion/parallelism/success/failure/active counters,
container exit/restart, result returncode/test counts, telemetry counts/rc,
runtime capability and enumerated numeric API defaults distinguish JSON
integers from booleans and integral floats. Missing submitted Job parallelism
and completions retain the existing Kubernetes defaults of integer one;
missing terminal failed/active retain integer zero. Present malformed fields
do not receive these defaults. No command, model, oracle, source binding,
tolerance, negative baseline or fixed13 denominator was changed.

Owner red controls against the original gate:
/hf3fs-jd/data/tmp/openopt-release-20261010/fixed13-type-repair/
red-owner-v1/tests-_nheyvr8/verification.json and pytest.xml.
Actual terminal: 14 failed, zero skips, 39.46 seconds. Ten complete synthetic
archives were audited and caller-repinned after mutation, covering owner
apiVersion, Job kind, digest, exit/restart/succeeded/retry/parallelism/
completion booleans and coordinated inner/outer result booleans. The other
four reds were sidecar/evidence/analytic/cone boolean schemas. These include
the independent original eleven failure classes plus nested schema aliases.
The additional owner tests retain the same rejection expectations and write
machine-readable rejection.json for each coordinated archive mutation.
Synthetic archives are CPU controls, not actual CUDA or origin evidence.

Validation outputs use fresh directories under the same fixed13-type-repair
prefix. Development/frozen verification.json requires actual test counts and
source_consistency PASS; directory names alone are never verdicts. Historical
seal/accept re-read the original caller-pinned real13 archive described above,
using the repaired clean auditor, not new GPU execution. Final receipt paths
and the clean commit are returned to main for independent review.

development-full-v1/tests-2ncyd_kx/verification.json finished at
2026-10-10T13:25:48.322093+00:00: PASS, 87 tests, zero failures/errors/skips,
pytest wall 95.10 seconds, source_consistency PASS. Source clean=false:
this is stationary development evidence, not the committed freeze.
Both explicit historical opt-ins ran, including complete real13 archive
and thirteen numerical sidecars. The ten synthetic coordinated rejections
each retain mutated-pins.json and rejection.json beneath pytest-tmp.
scan-v1/scan.json recorded zero literal/shape hits across the three files.
