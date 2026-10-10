# Durable Replay Diagnosis

## Conclusion

The release4 failure is a crash-fixture synchronization error, not evidence
that restart added the wrong number of attempts. The old Barrier counted two
certification callbacks, not two distinct jobs. An expired lease allowed both
worker slots to execute job 1 before job 2 was claimed. Recovery then correctly
added one attempt to each job, producing `[3, 1]` instead of `[2, 2]`.

Only `tests/test_durable_shadow.py`, its multi-worker helper, and this document
are changed. No production source, lease duration, heartbeat interval, count
assertion, dependency, GPU, infrastructure or credential is changed. This is a
completion/refill seat, not an increase in simultaneous native agents.

## Original Failure And Causality

The original evidence root is
`/hf3fs-jd/data/tmp/openopt-release-20261010/precondition/full-v1/tests-_9xpb4am/`.
Its `pytest.log` retains the `[3, 1] == [2, 2]` assertion and exit code 1:
1 failed, 1653 passed, 36 skipped. Log SHA256:
`d2adb156590c48993a263f78c53258dbf1bdd28447d7748c7a03ca44581b840a`.
The original journal remains read-only and is not repaired or rewritten.

On October 10, 2026 at 08:47:47--08:47:49 UTC, its journal events show:

| Relative Seconds | Job | Event | Attempt / Fence |
| --- | --- | --- | --- |
| 0.000000 | 1 | ONLINE_READY | 0 / 0 |
| 0.051405 | 1 | LEASED, original owner | 1 / 1 |
| 0.563627 | 1 | LEASED, same original owner | 2 / 2 |
| 1.130677 | 2 | ONLINE_READY | 0 / 0 |
| 2.096296 | 1 | LEASED, recovery owner | 3 / 3 |
| 2.204579 | 2 | LEASED, recovery owner | 1 / 1 |

`DurableShadowCertificationService._schedule_next` fills free worker slots by
calling `SQLiteShadowJournal.claim_next`. That query admits ONLINE_READY jobs
or expired LEASED jobs in job-id order; each claim increments attempt and fence.
The scheduler can therefore reclaim job 1 while its first callback is blocked.
`_run_job` starts heartbeat renewal and invokes `_execute_shadow`, which calls
the fixture's `certify_exact`. Two calls for job 1 satisfy the old Barrier.
The journal proves job 2 was not even inserted when that second claim occurred.
`_complete` fences results by owner/token; restart executes each accepted claim
once. The failed result is consistent with this actual pre-crash state.

The observed 0.512-second interval between claims exceeds the 0.5-second lease.
The original run did not measure heartbeat scheduling, GIL stalls, transaction
lock wait or FULL/WAL commit latency separately. 3FS delay is a plausible
contributor, not a convicted sole cause. No source bug or source-write expansion
is established by this evidence.

## Fixture Repair And Negative Controls

The helper controls only the journal module's lease clock. Both real submitted
jobs enter their actual certification callbacks; thread-local execution context
records job id, ticket, owner, attempt and fence. One coordinator publishes JSON
after both entries, rather than two threads overwriting one marker.
Before SIGKILL, the parent checks distinct identities, both first attempts,
both LEASED states, live deadlines, and the live SQLite rows against the marker.
It advances journal time past the leases only after kill, then uses the real
service, claims, Clarabel certification, terminal commits and acknowledgements.
The fixed 0.6-second expiration sleep is removed; no sleep or timeout is enlarged.

Acceptance retains strict `[2, 2]` attempts and fences, exactly two LEASED
events and one SUCCEEDED event per job, no FAILED/FENCED_LATE event, repeat
collect result-hash equality, and zero outstanding jobs after acknowledgement.

`legacy-expired-first` is a fixed test-only fault injection: the first real
claim's clock advances to its lease deadline and heartbeat renewal is withheld.
It reproduces the original `[2, 0]` pre-crash state and `[3, 1] recovery.
The identity guard rejects it before recovery; a separate count negative control
actually recovers and proves the unchanged replay-once guard rejects `[3, 1]`.
No database attempt, fence or state is fabricated.

## Verification And Reproduction

All new evidence is under
`/hf3fs-jd/data/tmp/openopt-release-20261010/durable/`.
`baseline/tests-7tzx3562/` is an unchanged isolated rerun: 1 passed, which does
not erase the original intermittent failure.
`acceptance-v1/` retains a harness failure after 3 passing tests: pytest's
`*current` directory aliases were double-counted. Real-path deduplication and
case/hash validation corrected the harness; v1 is not accepted.

`acceptance-v2/acceptance.json` is PASS, finished at
2026-10-10T09:00:28.776620+00:00. It binds the original log/journal, intentional
red exit code 1 with `[3, 1]`, and three sequential independent pytest processes:

| Run | Directory Below acceptance-v2 | JUnit Tests / Failures / Errors / Skips |
| --- | --- | --- |
| 1 | green-1/tests-n3esi9g7 | 3 / 0 / 0 / 0 |
| 2 | green-2/tests-6hzxbm9u | 3 / 0 / 0 / 0 |
| 3 | green-3/tests-pxogp25o | 3 / 0 / 0 / 0 |

Each run includes `pytest.log`, `pytest.xml`, `verification.json`, and three
semantic `acceptance.json` files with entry/job/event evidence and source hashes.
CPU smoke: `smoke/smoke-6zgdazy4/verification.json`, 79 passed, 4 explicit
CUDA skips. Durable/journal/completion-order regression:
`regression/tests-s0isq2bs/verification.json`, PASS, 47 passed, no failures,
errors or skips, finished at 2026-10-10T09:01:37.325932+00:00.

From the durable worktree, a new evidence directory must be unused:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/helpers/durable_shadow_multi_crash_worker.py acceptance --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/durable/reacceptance
```

The repository helper runs controlled red then three sequential green/negative
control suites, checks nonempty JUnit counts, real-path evidence membership,
semantic cases and source hashes, and fails loud on any unexpected result.
It never reruns a failed suite automatically.

## Source Identity And Limits

Seat baseline: `82514ce95af1258bbeda5e3ada80c0ebf0eff184`.
SHA256 of tested bytes:

| Path | SHA256 |
| --- | --- |
| src/openpo/durable_shadow.py | e306b65814f6faa5436cffb3aa8d9fb7b7b5fc32d4e92f6bb461a11e8bf175f3 |
| src/openpo/shadow_journal.py | 7c6477abd2e8b6d04b6577bb01c676d88bbdcfa31a6f402b4aaa8a8b6447db7b |
| tests/test_durable_shadow.py | f715292b4c0dcf0cafb8baab5bb359a74503d99c184a105e3975d6e47922a449 |
| tests/helpers/durable_shadow_multi_crash_worker.py | 2097008f329b0a867c803eba972e5c8225aa4b0d8b8d50ddb416bc721ed8cd78 |

Production source hashes match the failed precondition worktree.
The tests validate crash replay under controlled lease expiry on real 3FS
SQLite storage; they do not qualify real-time heartbeat liveness under arbitrary
storage delay. Such qualification needs a separately coordinated instrumentation
and source write-set proposal if an actual contract violation is demonstrated.
CPU-only work does not qualify CUDA or performance. Full-repository regression
is not run by this seat; final merged acceptance belongs to the coordinator.

The memory router found no project-private openopt memory index. The applicable
QP campaign handoff and current release4 campaign/audit were read. The infra
router rejects absolute cross-repository edit targets; its relative logical
tests/document targets select correctness/execution rules only and grant no
infra write authorization. The actual write set is the three openopt paths above.
