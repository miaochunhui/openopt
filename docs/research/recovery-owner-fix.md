# Recovery Gate Revision

The coordinator repaired the three P2 gaps reproduced by the independent
historical recovery review. This gate closes only the retained missing-document
packaging failure, not refinement numerical CUDA qualification, commercial
qualification, performance, or online publication.

The production verifier requires the exact seven frozen refinement testcase
identities and consistent leaf-suite counts. It checks byte-identical immutable
and coordinator results, the actual manifest/Job/Pod frozen runner command,
environment, single container/GPU, and no-retry settings. Successful telemetry
must match the actual nonempty sample list. Every executed source member is
checked against the run and result source identities; all inputs are re-read
before returning. Live summaries retain the actual Pod UID and Job controller
UID and reject missing or ambiguous identities instead of inferring them.

The strengthened recovery report has schema version 2. Historical version 1
reports and their failed counterexamples are not overwritten or reissued as
current acceptance. Use a new output directory for each changed gate:

```text
scripts/verify_cluster_recovery.py record --failed-run FAILED_RUN --recovered-run RECOVERED_RUN --artifact-dir NEW_3FS_DIRECTORY
scripts/verify_cluster_recovery.py verify --report REPORT --sha256 CALLER_PIN
scripts/qp_campaign.py live --namespace devtest --artifact-root NEW_3FS_DIRECTORY --recovery REPORT CALLER_PIN
```

Run as infra through the existing interpreter and connection, never copying
connection material. Neither entry creates/deletes Jobs or changes services.
Original evidence remains under the external release task root.

Real recovery evidence on 2026-10-10:
`/hf3fs-jd/data/tmp/openopt-release-20261010/failure-recovery-owner-v4/recovery.json`,
SHA256 `21760dc5a4bb8e8f1481465de4810b124ebd600038cc7629f244df511a4ff67f`.
The unchanged-source original-model tests and the packaging recovery remain
separate: this report is RECOVERY_VERIFIED only. Independent replay completed
at 10:14:14 UTC. The 10:14:14 UTC live read reports four Ready nodes, zero active
project Pods, zero cordoned nodes and three retained attributed failures.

Initial revision unit verification:
`recovery-owner-fix-unit-v1/tests-1xqvw7gb/verification.json` under that root,
36 passed, zero failures/errors/skips, source consistency PASS, 1.49-second
pytest wall. Later explicit Pod-identity controls require a new test receipt;
this initial count is not final-revision or full-repository qualification.

Final code unit verification:
`recovery-owner-final-unit-v3/tests-cl0x5oef/verification.json`, completed
10:20:02 UTC: 40 passed, zero failures/errors/skips, source consistency PASS,
1.54-second pytest wall. The preceding related 53-case run additionally tested
campaign tools; these overlapping counts must not be added.
Post-edit smoke at `recovery-owner-smoke-v2/smoke-74guu1ag/verification.json`
reports 80 passed and four explicit CUDA skips.

The pinned independently copied historical controls were replayed by the
repaired production gate at 10:19:07 UTC. Receipt:
`/hf3fs-jd/data/tmp/openopt-release-20261010/recovery-owner-counterexamples-v1.json`,
verdict COUNTEREXAMPLES_REPAIRED_PASS. Its unchanged positive passes; replaced
selection, contradictory internal result, contradictory actual Job command,
joint wrong runner and inflated telemetry all reject for their expected
first causes. The independent original review pin is
`09ef5c214e6b9c631720b7424d12839992e205be4a9645796b0016306b8f10bb`.
Reproduce with the existing CPU-only receipt entry:

```text
tests/test_verify_cluster_recovery.py --review /hf3fs-jd/data/tmp/openopt-release-20261010/recovery-review/coordinated-v2/review.json --sha256 09ef5c214e6b9c631720b7424d12839992e205be4a9645796b0016306b8f10bb --output NEW_3FS_RECEIPT
```

The final strict-identity live snapshot at 10:17:28 UTC is
`live-recovery-owner-v5/live-101728739391.json` under the task root: PASS,
four Ready, zero active project tasks. This is a dated live observation,
not a permanently installed health guard or proof of numerical qualification.

The historical independent audit uses its own explicitly trusted committed
source coordinate to reproduce old acceptance defects. It must not execute
artifact snapshots or require the repaired production source to remain buggy.
Its integration and final source regression are separate checks.
