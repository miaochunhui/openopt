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
- At most three CPU workers execute the existing `verify.py run` through its
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
