# Source-Pinned Full CPU Regression

`scripts/verify_sharded.py` plans, runs and independently rechecks a complete
pytest CPU collection. This is a full CPU correctness gate, not CUDA,
commercial-solver, optimizer-performance or production qualification.

## Contract

- Planning requires a clean committed Git snapshot. It reuses
  `source_snapshot` and `check_source_consistency` from `scripts/verify.py`,
  including the complete tracked/unignored file set and every file SHA256.
- Pytest performs discovery with the repository configuration. There is no
  `test_*.py` glob or frozen list standing in for collection. A small explicit
  plugin records every actual nodeid; empty, duplicate or ambiguous JUnit
  identities fail. Explicit environment addopts and auto-loaded third-party
  plugins are disabled consistently for collection and execution.
- All tests in one file stay together. LPT assigns files by summed historical
  testcase times. Later supplied JUnit wins for the same exact pytest/JUnit
  identity; unknown tests are listed explicitly and use a declared prediction
  default. Predictions exclude collection, imports, provenance, examples and
  contention. They are not measured wall time and do not establish speedup.
- CPU partition count (`--shards`) and simultaneous worker count (`--workers`)
  are independent positive integers, each bounded at 32 and defaulting to 3.
  Neither limit is a GPU-card limit. Eight file partitions may run through
  three workers, admitting queued work only after completed work passes.
  Zero-duration observations still produce nonempty partitions.
- Planning and every consumer re-read pinned historical JUnit, recompute the
  file-LPT assignment and reject understated or inconsistent predictions.
  A whole file whose predicted testcase sum exceeds the shard deadline fails
  explicitly as indivisible; the planner never splits a file and risks its
  fixture dependencies. An assigned shard over the deadline also fails with
  its id, predicted cost and budget; increase partition count and replan.
  A finite configured timeout alone cannot turn an overbudget plan green.
  Predictions at the deadline do not include overhead or prove schedulability;
  leave measured headroom and retain runtime timeouts as failures.
- Bounded CPU workers execute the existing `verify.py run` through its
  CLI with explicit complete file targets. Each gets separate external
  logs, pytest cache, temporary files, JUnit and verification report.
  GPU visibility is explicitly disabled. No installation or cluster access
  is performed.
- Worker timeout is finite, greater than zero and at most 300 seconds.
  Timeout kills only the admitted worker process group, records the actual
  exit code and fails. There is no retry. On the first observed failed
  completion, queued admission stops; already admitted workers are drained
  with their original deadlines. Three already admitted workers cannot
  retroactively be described as stopped before admission.
- Acceptance requires every child PASS, identical source bytes and commit,
  exact target membership, zero failures/errors and exact two-way nodeid
  membership against the full collection. Explicit CPU skips are preserved,
  not counted as passes or CUDA proof. Source is checked before and after.
- Independent `check` requires caller-pinned plan and report hashes, re-reads
  every child JUnit and source report, verifies child hashes and membership,
  re-collects pytest, and compares current source again. A top-level rehashed
  forgery cannot replace child numerical counts or actual membership.

## Commands

Run after the coordinator integrates the tool and freezes a clean commit.
Use the existing validation interpreter; do not run the expensive full
repository suite in this development worktree.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify_sharded.py plan --repo /data/dev/openopt --history-junit /hf3fs-jd/data/tmp/openopt-release-20261010/full-merged/tests-wx7hh8vg/pytest.xml --history-junit /hf3fs-jd/data/tmp/openopt-release-20261010/merged-release-tools-frozen/tests-gylzh4zs/pytest.xml --shards 3 --shard-timeout 300 --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/sharded-regression/final
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify_sharded.py run --repo /data/dev/openopt --plan <printed-plan-path> --plan-sha256 <printed-plan-sha256> --workers 3 --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/sharded-regression/final
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify_sharded.py check --repo /data/dev/openopt --plan <printed-plan-path> --plan-sha256 <printed-plan-sha256> --report <printed-verification-path> --report-sha256 <printed-report-sha256> --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/sharded-regression/final
```

Every action creates a new external directory; prior failures are retained.
For a larger CPU suite, select `--shards 8` when planning and retain
`--workers 3` when running. Do not edit a pinned three-shard plan in place.
Append all later measured JUnit inputs, including the sharder and cluster
tool tests themselves, after overlapping older histories. No cost input is
an additional full-suite pass or a measured speedup.
`--log-dir` is an alias for `--artifact-dir`; logs remain beside machine-readable
results rather than in the source tree. PASS results contain
`coverage="tests"`, source/source-consistency and the same `tests` count schema
as `verify.py`. The combined `pytest.xml` preserves all original testcases
and their skips for a release renderer. Three worker walls overlap and must
not be summed as accepted campaign wall.

## Validation Scope

`tests/test_verify_sharded.py` runs the production `verify.py` subprocess entry
against a clean, external, tiny Git fixture. Real pytest collection and
execution include configured `check_*.py` discovery, parameter IDs containing
punctuation, a class method and an explicit skip. The tiny provenance/example
scripts are visibly labeled fixture-only and are not OpenPO/solver evidence.

Negative controls cover collection/hash/source conflicts, empty selection,
duplicate targets, missing/duplicate JUnit identities, incorrect counts,
partial failing workers, stopped admission and a real SIGKILL timeout.
The coordinator still owns the final clean-source full-repository run and
independent review. Historical full-suite timings are planning inputs only;
neither them nor a tiny fixture establish actual full-suite sharding benefit.

## Development Evidence

On 2026-10-10 the external dedicated regression completed with 30 passed,
zero failures/errors/skips in 56.04 seconds:
`/hf3fs-jd/data/tmp/openopt-release-20261010/sharded-regression/unit-v3/tests-ah0qqlei/verification.json`.
Its pytest temporary directory contains the real tiny-repository CLI chain,
child process/JUnit files and independent `acceptance.json`. Those four tiny
testcases include one deliberate skip and are not four optimizer problems.

The first attempt remains under `unit-v1/tests-s8vzo4r5/`, with 27 passed and
two failures: eager indexing into an empty failure list, and a negative-control
mutation that did not actually change its collected nodeid. Both causes were
fixed; the mutation now has an explicit inequality self-check.
`unit-v2/tests-e5npgr5o/` passed 29 tests before the process-command and real CLI
controls were added. Counts overlap; do not add these rounds together.

The added test file itself costs roughly 56 seconds in this run, not zero.
Unknown historical testcase cost is explicitly predicted until the final run
provides measurements. Concurrent imports and 3FS metadata contention may
make actual three-worker execution slower than file-LPT predictions. No
controlled serial-versus-sharded same-source speedup was measured here.

## CPU Capacity And Budget Acceptance

The 2026-10-10 capacity repair separates CPU partition count from concurrency,
each bounded at 32 with default 3. Dedicated related verification:
`/hf3fs-jd/data/tmp/openopt-release-20261010/sharder-capacity/final-related-v2/tests-kdhhk5sx/verification.json`
reports 67 passed, zero failures/errors/skips, pytest wall 234.68 seconds and
unchanged source. This includes the complete sharder suite and the unchanged
cluster wrapper suite. Append its `pytest.xml` to final-source cost histories.
Post-edit smoke `postcondition/smoke-u6iyk3s8/verification.json` under the same
artifact root reports 80 passed and four explicit CUDA skips.

The initial minimum acceptance `minimal-v1/tests-eyc2lq72/` passed five tests
in 29.45 seconds. Its real eight-partition/three-worker fixture executes nine
testcases including one deliberate skip, replays all eight process/JUnit
sidecars, and verifies exact collection, no overlap, nonempty partitions and
unchanged source. It is tiny CPU correctness evidence, not solver qualification
or measured acceleration. The final related suite repeats that acceptance.

The first dedicated attempt `unit-v1/tests-p86kur39/` retained 41 passes and
one failure: the collection-conflict fixture's prediction metadata was stale,
so the new provenance gate correctly rejected it before the intended fresh
collection gate. The fixture now recomputes its internal prediction first;
`fix-negative-v1/tests-b7li9z20/` passed the independent negative control.
No runtime gate was relaxed.

Read-only cost diagnosis on frozen source `3792a86`, using eight partitions
and appending `frozen-r1-proof-unit/tests-20scfhkx/pytest.xml`, retained FAIL:
`/hf3fs-jd/data/tmp/openopt-release-20261010/sharder-capacity/frozen-r1-cost-diagnostic/plan-8jhikr_y/plan.json`.
The exact cause is `file_prediction_exceeds_deadline` for
`tests/test_gpu_release_proof.py`, predicted leaf testcase sum 369.019 seconds
against a 300-second deadline. That file has zero unknown nodeids because its
cost was measured; the full collection still has 494 unknown nodeids.
The 95-case input spans two files: its leaf sum is 372.634 seconds, distinct
from the reported 374.78-second pytest wall. The planner uses leaf sums only.
No shard was executed from this failed plan.

The old eight-partition capacity failure and old roughly 487-second
three-partition plan remain preserved. Final r2 source must be replanned after
integration; increasing CPU partitions cannot repair a single indivisible
overbudget file. Safe fixture-aware node grouping or genuine test-file
refactoring requires a separate reviewed design. No cluster operation,
installation, existing wrapper change or performance claim is part of this
repair. The current wrapper admits all remaining Jobs after its smoke gate;
the host sharder `--workers` option is not a Kubernetes concurrency limit.
