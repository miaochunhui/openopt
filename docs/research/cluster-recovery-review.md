# Independent Cluster Recovery Review

## Findings

The historical findings below refer only to committed
`e618b68c5209794c8cb1ec4cef727e7e13ef1ada`, not the current production gate.
Only this document, `scripts/review_cluster_recovery.py` and its tests
are changed by this seat. All original evidence is unchanged.
No cluster command, solve, installation, credential lookup or push is performed.

Three historical P2 evidence-gate gaps were reproduced against copied run evidence.
The file/line references in this list refer to that historical commit:

1. `verify_cluster_recovery.py:83` accepts seven unique arbitrary case names.
   Replacing both failed/recovered selections with seven unrelated names and
   regenerating the report/pins still passes upstream replay. Exact expected
   identities and suite counts are required by the independent review.
2. `verify_cluster_recovery.py:162` reads the coordinator result but not the
   immutable `cuda-check-*/result.json`. A copied immutable FAIL and coordinator
   PASS still pass upstream report replay. Independent review rejects unequal
   result bytes and includes both in its input manifest.
3. `verify_cluster_recovery.py:139` compares manifest container zero with Pod
   container zero, not the captured actual Job template. A Job template with
   command `false` and an unchanged successful Pod still passes upstream replay.
   A jointly substituted non-frozen runner also passes. Independent review
   checks exact manifest/Job/Pod runner, source/output arguments, environment,
   identity, resources, one-container topology and no-retry policy.

These are evidence acceptance defects, not observed numerical regressions.
The original caller-pinned report remains valid under the independent stronger checks.
The primary owner was notified before completion; this seat does not repair
or replace the original recovery implementation.

The historical live producer (`qp_campaign.py:144`) omits Pod and Job UIDs.
Its `2026-10-10T09:54:07.908893+00:00` snapshot is summary-consistent with
the original failed container's exit/restart/finish conditions, but offline
identity authentication is **UNVERIFIED**, not reconstructed from an earlier
status capture. The two historical name-only attributions at line 90 also
remain unauthenticated by that summary. This seat never reads current cluster
state or claims those old observations establish current health.

## Real Evidence And Counterexamples

Caller report:

```text
/hf3fs-jd/data/tmp/openopt-release-20261010/failure-recovery-real-v3/recovery.json
cd911cf2cbe4f1488511157b02cb8c81c33b6ad740ea5f6437915bf4dbbb6610
```

Caller live:

```text
/hf3fs-jd/data/tmp/openopt-release-20261010/live-recovery-v1/live-095407908933.json
09bb2842415c903046258ecff9f26f59a4bfdc21fb3fd50fcd8dd9ed16deb6a1
```

The independent review additionally hashes all frozen source members and
checks them against each recorded Git commit and source-identity manifest.
All executed `src/openopt` modules, the target test and CUDA runner remain
byte-identical across the packaging fix, not merely the original four CORE
files. The exact required document appears only in the recovered snapshot.
The source snapshot's generated `source-hashes.json` is explicitly checked;
it is not mistaken for an unhashed executable source. Actual successful
telemetry must contain the declared number of successful nonempty samples.
All inputs are rechecked before returning; filesystem hashes are not remote
attestation and cannot authenticate maliciously fabricated observations.

At `2026-10-10T10:03:28.783617+00:00`, the coordinated experiment started at:

```text
/hf3fs-jd/data/tmp/openopt-release-20261010/recovery-review/coordinated-v2/review.json
```

It passed one unchanged copied positive control and rejected all seven
coordinated negatives after regenerating report/input/caller pins:

| Copied mutation | e618b68 replay | Independent review |
| --- | --- | --- |
| Seven unrelated test identities on both runs | Accepted | Rejected |
| Immutable FAIL, coordinator PASS | Accepted | Rejected |
| Job template command differs from Pod | Accepted | Rejected |
| Manifest/Job/Pod jointly use non-frozen runner | Accepted | Rejected |
| Immutable/coordinator telemetry count inflated | Accepted | Rejected |
| Failed run erased from newly pinned live summary | Accepted report only | Rejected |
| Report promoted to commercial/certified scope | Rejected | Rejected |

The live-erasure case is not a claim that upstream `verify` verifies live:
that API never takes a live input. It demonstrates why the combined independent
review must check preservation separately. Both original report/live hashes
and original report input descriptors were unchanged after the experiment.
CPU inspection of real GPU artifacts is not a new CUDA run or numerical solve.

Preflight: `preflight/smoke-hz43oq6o/verification.json` under the same root,
80 passed, four explicit unavailable-CUDA skips, provenance and CPU example PASS.
Two review-tool development failures remain retained: `real-v1` rejected the
generated snapshot hash manifest as unexpected; `coordinated-v1` rejected pytest
cache symlinks while cloning. The corrected tool checks the exact generated
manifest and excludes only pytest temporary/cache trees from copied fixtures.
Neither failure changed original evidence, solver code or GPU tasks.
CPU unit acceptance: `unit-v1/tests-_u3mw4b1/verification.json`, 66 passed,
zero failures/errors/skips, provenance and CPU example PASS. The on-disk
`test-counterexamples.json` supplies the final terminal counterexample verdict;
the incremental progress file is not a completion marker.

## CLI And Remaining Qualifications

Use the existing CPU validation environment and `infra` identity:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/review_cluster_recovery.py review --report REPORT --report-sha256 REPORT_SHA --live LIVE --live-sha256 LIVE_SHA --historical-commit e618b68c5209794c8cb1ec4cef727e7e13ef1ada --historical-verifier-sha256 784eba817bea9d2dfe48288827bfb4392068a564a01f6fba6cd65e4fb86b4bbc --artifact-dir NEW_EXTERNAL_DIRECTORY
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/review_cluster_recovery.py counterexamples --report REPORT --report-sha256 REPORT_SHA --live LIVE --live-sha256 LIVE_SHA --historical-commit e618b68c5209794c8cb1ec4cef727e7e13ef1ada --historical-verifier-sha256 784eba817bea9d2dfe48288827bfb4392068a564a01f6fba6cd65e4fb86b4bbc --artifact-dir NEW_EXTERNAL_DIRECTORY
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_review_cluster_recovery.py --target tests/test_verify_cluster_recovery.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/recovery-review/unit
```

Only a new, never-overwritten 3FS audit directory is written. Failure keeps
`failure.json` with first cause and exit code; both terminal paths keep
`execution.log`. Counterexamples retain every copied fixture and report.
The CLI checks all four approved historical Git blobs, not equality between
the current checkout and e618b68. Caller commit and verifier hash are required,
with no HEAD/ref/short-hash defaults. All four approved SHA256 values are
fixed in the independent tool; arbitrary caller-pinned toy code cannot become
the historical oracle.

## Integration And Research Dependencies

The integration tree starts at `d1fdcb72845bc25f32bf0d38014ddc7721f68b68`
and cherry-picks the original `c03a351` review as `c001975`. Before editing,
the current primary owner's three P2 patches were read: production recovery
adds exact JUnit identities, both result-byte comparison, Job/Pod execution
and complete executed-source checks; live adds actual Pod/Job identities.
Production now constructs schema 2. Historical schema-1 reports must not
force the production gate back to the old implementation or weaken it.
The committed owner fix `5fd60a1` is explicitly cherry-picked as `930e732`
in this integration tree solely as a read-only production dependency. The
primary owner should pick only the subsequent three-file integration commit,
not reapply that dependency or the original review commit.

The canonical `review` entry independently reconstructs the historical
schema-1 report, including exact summaries and ordered input inventory,
from the stronger `strict_leg` checks. It imports/calls neither the current
production gate nor the historical verifier. Every frozen source file still
must match its committed Git blob and recorded source identity; full execution
membership, identical result bytes, exact seven case identities, telemetry,
execution environment, missing/recovered document and final input re-read
checks remain mandatory. Summary changes, extra qualification fields and an
empty report input list cannot pass even after resealing the caller report pin.
The output binds the independent reviewer bytes separately from the historical
source context. Neither binding is remote execution attestation.

Only `counterexamples` executes the approved historical verifier. Its loader
reads the exact full-commit Git object, validates caller commit/hash and all
four fixed historical blobs, rechecks the verifier bytes before compilation,
then loads an isolated module with a `git:<commit>:<path>` filename.
It never imports `artifact/source`, modifies `sys.modules` production entries,
executes historical CUDA runners, invokes a solver or invokes a cluster client.
Each counterexample records the actual historical source descriptor. Thus
the old fake-name acceptance assertion remains a historical finding even
after the current production gate rejects it.

These are **non-hermetic research tests**, not portable source-only unit tests.
`tests/test_review_cluster_recovery.py` reads the caller-pinned original report,
live snapshot, both full frozen run directories and all their source/log/XML/
result/telemetry sidecars. The module-scoped counterexample fixture copies
real 3FS evidence into the verifier's external TMPDIR, excluding only pytest
temporary/cache trees. Tests do not download fixtures or silently skip absent
evidence. A full `scripts/verify.py tests` or container regression that collects
this file therefore needs the historical 3FS tree at the recorded absolute
paths, readable originals and a writable new external artifact/TMPDIR.

The checkout also needs read-only `git` and the actual trusted repository
objects for e618b68, `6c3dbef9e687278410fc074679e699210b3d75cc` and
`e566b0b8163f099a29a0d554fa3b87ad15e4fb67`, including their trees/blobs.
A worktree's `.git` file alone is insufficient if its common Git directory
is absent in the container. A source-only/ZIP/shallow snapshot without those
objects cannot reproduce this research suite; missing mounts/objects fail
loud rather than count as full CPU regression success. The smoke selection
does not collect this research test. Container coverage and mounts remain the
coordinator's explicit responsibility; this seat submits no containers.

This integration does not upgrade the original failure/live evidence,
commercial qualification, numerical CUDA status or current cluster health.
Historical live UID authentication remains UNVERIFIED.

Integration evidence root:
`/hf3fs-jd/data/tmp/openopt-release-20261010/recovery-review-integration/`.
The actual `real-v1/review.json` at `2026-10-10T10:20:21.652472+00:00`
records canonical RECOVERY_REVIEW_PASS, neither verifier executed, and the
original 7/7/3 denominator. `coordinated-v1/review.json` retains the copied
positive and all seven rejected negatives with authenticated old-Git source
descriptors. Both original report/live pins remain unchanged. These receipts
are CPU disk replay, not a new GPU solve.

The first integration `unit-v1/tests-uwg5_96y` has 81 passing pytest cases
but its final provenance verdict is FAIL: this seat edited tests/documentation
while it ran. That receipt is retained and is not final acceptance. The
corrected run freezes the sources through the end of verification. Added
combined tests also replay the same actual copied positive and five production
negative mutations through the repaired schema-2 gate; historical oracle
results remain independently bound to e618b68. Final acceptance receipts are
recorded in the handoff, not inferred from incremental progress.

`RECOVERY_REVIEW_PASS` grants only this packaging recovery and preservation
check. Numerical CUDA qualification is **NOT_GRANTED**, commercial qualification
and publication stay **UNVERIFIED**. Original seven failed tests remain FAIL;
the docfix run does not retroactively repair them or qualify a later final tree.
Native/refinement execution and original-model sidecar numerical replay remain
separate final-source acceptance gates owned by the coordinator.

Read-only commercial/publication prerequisite observations:

- Actual SDK discovery at `2026-10-10T10:03:51.407775+00:00`:
  `commercial-readiness-review/discovery/sdk-548d5ftv/result.json` under the
  release task root. Gurobi, CPLEX, MOSEK and cuOpt were not discovered in that
  environment; no license/network/solve attempt. This is environment-scoped,
  not a claim that no authorized installation exists anywhere.
- Actual runner discovery at `2026-10-10T10:05:04.468558+00:00`:
  `commercial-readiness-review/runner-discovery/discovery-3bsk_wt3/result.json`.
  Qualification is UNVERIFIED. Real approved usage and actual licensed
  matched-model runs with independent quality, failures and paired timing
  remain prerequisites, not satisfied by offline SDK substitutes.
- `publication/recovery-demo-20261010/local-verification.json` says PREPARED,
  deterministic byte readback PASS, online/commercial UNVERIFIED. A local ZIP
  is not deployed-source readback, browser rendering or authorized publication.
  Current project release/publication contracts keep those gates separate.
- Full cone/MIP/MIQP capability, representative industrial coverage,
  certificate robustness, transfer-inclusive competitive speed and genuine
  multi-GPU solver scaling are not established by a packaging fix or these
  seven tests. Do not assert universal dominance or exact certification.

The independent gate is delivered as an explicit CLI, not wired into a
permanent publication/dispatch hook. Production integration and the upstream
P2 repairs remain the primary owner's responsibility.
