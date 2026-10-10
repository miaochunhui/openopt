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
- By default all tests in one file stay together. LPT assigns files by summed
  historical testcase times. Later supplied JUnit wins for the same exact pytest/JUnit
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
  explicitly as indivisible. Only an explicit v2 fixture-file selection can
  replace this partition unit with an audited component. An assigned shard
  over the deadline also fails with its id, predicted cost and budget;
  increase partition count and replan.
  A finite configured timeout alone cannot turn an overbudget plan green.
  Predictions at the deadline do not include overhead or prove schedulability;
  leave measured headroom and retain runtime timeouts as failures.
- Bounded CPU workers execute the existing `verify.py run` through its
  CLI with explicit file or exact node targets. Each gets separate external
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

## Explicit Fixture Components

Default planning remains schema v1/file-LPT. To request the restricted schema
v2 contract, pass both `--fixture-aware` and one or more repeated
`--fixture-file tests/example.py` arguments during `plan`. Empty, duplicate,
unknown or uncollected selections fail. This does not select a smaller test
suite: pytest still discovers every nodeid in the complete repository, and
unselected files remain indivisible whole-file groups. Existing pinned
`run --workers 3` and `check` interfaces are unchanged; neither action accepts
a new fixture selection that could silently change the plan.

The explicit collection plugin cross-checks selected test/fixture AST with
pytest's actual fixture closure, including autouse, usefixtures and transitive
dependencies. Class methods stay together. Function-scoped local fixtures
allow standalone tests and parameter cases to form separate units. Shared
class/module/session fixture identities union all their dependent units before
historical leaf-cost LPT; LPT never splits a connected component. Session
dependencies shared across selected files also union them. A visible user
session dependency crossing into an unselected file rejects admission and
names the dependent node; it is not silently treated as a disposable resource.

Two infrastructure exceptions are explicitly qualified for pytest 8.4.2:
`tmp_path_factory` from `_pytest.tmpdir` and `pytestconfig` from
`_pytest.fixtures`, both with actual session scope. The allowlist verifies
callable name/module, package-relative origin, runtime version and complete
implementation-file SHA256 before omitting a session edge. The recorded hashes
are `236918c090160c1f6b935e162ff44ab273a700221d5f40ac15892be75e4503ee`
and `41dfd72c1e6da619906bf84945694c1cb9b919bad18303278fac35630c8c7748`,
respectively. Unknown versions, origins or hashes fail; a user fixture sharing
either name is never exempt. This is not a generic plugin whitelist.

Static usage must also prove direct `mktemp` calls with a literal basename and
default/true `numbered`, without aliasing or passing the factory elsewhere.
The real tiny replay checks unique same-prefix allocations and private worker
basetemps. Config itself is mutable, not inherently immutable. Its exception
only permits direct `getoption`/`getini`/`get_verbosity` calls with constant
arguments inside equality/inequality assertions against literals. Aliases,
mutation, returned mutable-list escape and unmodelled uses fail. User shared
module/session fixtures still create connected components even if their
artifacts are immutable.

This is a restricted structural contract, not a general Python purity proof.
Dynamic scope/fixture lookup, unknown plugin fixtures, fixture overrides,
unmodelled xunit scope, wrapped tests and nonfunction parameter scopes reject
selected-file admission. Only explicitly recognised pytest builtin fixtures
are accepted; their implementation hashes and pytest version are recorded.
Conftest source is audited because it can inject fixtures/hooks into selected
files. Arbitrary global state, external resource semantics and helper side
effects cannot be proved independent by fixture scope alone. The coordinator
must review selected-file isolation; no automatic split-all option is exposed.

Each v2 plan pins its selected files, source/runtime hashes, complete collection
sidecar, actual closure, connected groups, historical JUnit and group leaf
costs. Run and independent check re-collect and compare the whole audit before
admission/acceptance. A rehashed plan cannot understate costs, split a pinned
group, omit/duplicate nodes or replace actual child JUnit membership.
`fixture_group_prediction_exceeds_deadline` retains the indivisible group id
and leaf cost. Unknown costs remain explicit predictions, not measurements;
neither fixture grouping nor a 300-second timeout establishes acceleration.

No cluster-wrapper change or cluster acceptance is included. The current
wrapper reconstructs a nodeid-only collection sidecar, whereas v2 requires the
fixture audit and runtime pins too. This round's v2 handoff is therefore
host-only; a wrapper adaptation is a separate coordinator task. A different
container runtime may also fail the local implementation-path/hash pins and
requires validation, not installation or a waiver of the fixture boundary.

## Real Split Diagnostic

Evidence root:
`/hf3fs-jd/data/tmp/openopt-release-20261010/sharder-fixture/`.
The initial broad-audit diagnostic retained 3297 collected nodeids and FAIL,
first cause `unmodelled_fixture_request` at `tests/test_conic.py:507`:
`full-audit-diagnostic-v1/tests-w1wmjhc1/pytest-tmp/test_full_collection_audit_dia0/full-audit/diagnostic.json`.
The outer diagnostic test passing only proves it detected the rejection;
it is not full-regression PASS. That broad-audit interface was replaced by
explicit file selection; unrelated files are no longer statically audited.
The post-builtin full-file audit repeats this FAIL with 3311 collected nodeids
and 95 explicit files, same first cause:
`builtin-boundary-v3/tests-_f4kue1q/pytest-tmp/test_development_full_audit_di0/all-file-audit/diagnostic.json`.
Its `complete_failure_inventory=false` is explicit. Both diagnostics captured
dirty development source hashes, not a clean final-source plan. The temporary
historical/whole-repository diagnostic tests were removed from the delivered
suite; their external evidence remains. No unsupported `request` usage was
silently accepted to make a blanket audit green.

Proof-file collection only (no proof tests executed):
`real-diagnostic-v1/tests-2l19iyr9/pytest-tmp/test_fixture_real_proof_closur0/proof-collection/cost-diagnostic.json`.
The 68 collected nodeids match frozen-r1 historical JUnit identities with zero
unknown costs. Whole-file leaf sum is 369.019 seconds. This early diagnostic
incorrectly joined 59 nodes through builtin `tmp_path_factory`, producing a
368.762-second component. Its over-conservative infrastructure edge is
superseded by the qualified exception above; the historical failure remains
on disk, not relabelled as a pass.

Full collection with just that proof file selected retained 3302 nodeids and
FAIL, first cause `session_dependency_outside_fixture_selection` for builtin
`tmp_path_factory`, dependent
`tests/mpo/test_benchmark_cli.py::test_parser_full_flag_set`:
`real-diagnostic-v1/tests-2l19iyr9/pytest-tmp/test_real_explicit_proof_full_0/full-proof-selection/diagnostic.json`.
That builtin boundary failure is also superseded, not a current user-session
blocker. Both diagnostics predate final tool-test additions and are not
final-source membership counts. No shards were dispatched from either
diagnosis. The broad audit is fail-fast and retains only its first rejection;
it does not supply a complete inventory of all unsupported files.

Corrected full collection with only the proof file selected:
`builtin-real-v2/tests-4t40w_a4/pytest-tmp/test_real_proof_builtin_correc0/real-proof-builtin/diagnostic.json`.
It records 3309 nodeids and selected-file audit PASS, but this is collection
evidence only, not execution or final frozen-source membership. The 68 proof
nodes have zero unknown historical costs. The genuine `produced(module)`
component contains exactly 41 dependent nodes, leaf sum **334.420 seconds**,
still above 300. The other 27 singleton groups total 34.599 seconds; the whole
file remains 369.019 seconds. User module sharing, not the builtin factory,
now explains the indivisible budget failure. Builtin qualification/hash,
each group membership, JUnit hash and development source hashes are recorded.
No proof tests or real repository shards were executed in this diagnosis.

The proof-file owner must first fix the actual guard/fixture narrow-copy costs
or design genuinely isolated helper/test groups, then remeasure leaf JUnit and
freeze/replan final source. At least 34.420 seconds of leaf cost must be removed
from the shared group to reach 300, with additional measured headroom needed
for imports, snapshots and examples. This diagnosis does not attribute that
cost to a particular helper without profiling. Read-only inspection finds
whole evidence-tree copies in `case_copy` and several tamper tests; these are
owner profiling candidates, not a measured causal attribution. Merely widening
fixture selection cannot erase the 334.420-second connected resource. No proof
test file, verifier, wrapper, cluster object, dependency or deadline was
changed here.

Minimum explicit-selection acceptance:
`narrow-v1/tests-b544lhfa/verification.json` reports six passed in 29.89 seconds.
Its clean external tiny repository executes eight partitions through three
workers, exact full membership and independent replay, with one deliberate
skip. The synthetic historical input assigns six cheap parameter nodes
60 seconds each, so the old whole-file plan fails at 360 seconds while v2
groups pass prediction. Those invented cost inputs test the budget mechanism;
they are not measured solver performance or real 360-second execution.
The first attempts remain under `minimal-v1` (AST root diagnostic accessed a
missing line number) and `minimal-v2` (a builtin class plugin was misidentified
as external); these implementation defects were fixed, not admission-waived.

Post-builtin boundary acceptance:
`builtin-boundary-v3/tests-_f4kue1q/verification.json` reports 11 passed,
zero failures/errors/skips in 69.83 seconds, with unchanged development
source. It checks official identity/scope/origin/version/hash admission and
negative controls for unsafe unique-temp usage, mutable config return escape
and a user fixture named `tmp_path_factory`. It also repeats real eight-shard,
three-worker run/replay with unique worker temp directories, and captures the
whole-file audit FAIL described above. Its outer PASS is tool/diagnostic
acceptance only. `latest-precondition/smoke-o4l_t92f/verification.json` records
80 passed, four explicit CUDA skips and unchanged source in 6.22 seconds.

## Delivered Tool Acceptance

The post-builtin complete dedicated sharder suite is
`final-builtin-v4/tests-acr5uwbh/verification.json`: 75 passed, zero failures,
errors or skips, pytest wall 169.51 seconds. Default v1/file-LPT and opt-in v2
CLI plan/run/check, shared module/session closure, narrow-file boundary,
budget forgery, source/audit/JUnit tamper, partial admission failure and actual
SIGKILL timeout controls all ran. Source consistency reports PASS over 269
files with no changed/missing/extra entries. The development snapshot was
dirty by design during acceptance; this is not a clean final-source full CPU
regression. Only this documentation's evidence paragraphs were appended
after the run; the tested script and tests stayed unchanged.

The v2 tiny replay is independently accepted at
`final-builtin-v4/tests-acr5uwbh/pytest-tmp/test_fixture_cheap_nodes_overb0/fixture-check/check-bzy4qgeu/acceptance.json`:
eight exact nodes, seven passes and one deliberate skip, eight nonempty
shards through three bounded workers, zero duplicate/missing/unexpected JUnit
identities, private basetemps and 12 unique same-prefix temp directories.
Its invented historical costs are budget-mechanism inputs, not speed evidence.

Unchanged wrapper/default compatibility is
`wrapper-default-builtin-v4/tests-urpornfn/verification.json`: 25 passed,
zero failures/errors/skips, pytest wall 123.97 seconds, source consistency
PASS. These CPU mocks and external fixture checks do not validate v2 cluster
dispatch. Post-edit smoke is
`final-postcondition/smoke-op6pq0ka/verification.json`: 80 passed and four
explicit CUDA skips in 6.48 seconds, source consistency PASS.
Append the two final dedicated `pytest.xml` inputs after earlier overlapping
histories when replanning final source. Do not sum these overlapping test
rounds or use their passing exit codes as final full-repository qualification.

## Clean Plan And Public V2 Handoff

Code/test commit `48e12046714e44c36942f162215f688ad9572db5` was clean when
the real proof-file plan was generated:
`clean-proof-budget-v5/plan-_39j_uig/plan.json`, SHA256
`aa9cc541d06de2ce6dbce2639dc675e700073593f25af18bad92c921344f1024`.
Full collection is 3310 nodes, source consistency PASS over 269 files.
Selected proof-file audit succeeds, with 68 nodes and zero unknown proof
costs, but plan verdict is FAIL:
`fixture_group_prediction_exceeds_deadline`, group 59, exactly 41 dependent
nodes, 334.420 seconds against 300. Only user `produced(module)` constrains
that group; the qualified builtin factory is not a grouping edge. No shards
were admitted. This is a clean-source budget rejection, not a full test run.

Public robustness source `08f3eac` was inspected in an independent clean
acceptance worktree with the delivered sharder cherry-picked, commit
`9b043e7c525bc53328294adfa9a025fc9e0d19f4`. No existing source file there was
edited and no public tests executed. Its complete 3535-node collection
includes exactly 77 public nodes. Explicit public-file opt-in retains FAIL:
`clean-public-audit-v5/plan-qf9valrj/collection/collection.json`,
`unmodelled_fixture_request` at
`tests/test_experiment_public_robustness.py:201`. This is the exact first
static rejection, not proof of all other constraints passing. Five visible
user fixture declarations (`local_real_evidence`,
`no_solver_calls_in_this_suite`, `real_metadata_test_copy`, `baseline`,
`experiment`) are function-scoped; nevertheless node-local reporting through
`request`, custom `LOCAL_EVIDENCE` decorators and helper side effects remain
outside the current isolation contract. Do not claim those 77 independent
groups have been admitted merely because fixture scope is function.

The coordinator's real histories, under
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/`, are:
`portable-default-v2/p2-unit-m1tldfv2/pytest.xml`, 68 passes and nine explicit
historical/local-evidence skips, leaf sum 244.085 seconds, SHA256
`bb3228fe76b5ca2a412bdc4b9268ca9b96f6e4f4d9d85f606dbc82a38a1631d2`;
`portable-optin-v2/p2-unit-ga31eyj5/pytest.xml`, 77 passes, leaf sum
281.862 seconds, SHA256
`0c43af485678ac00e735edeed6a0e13bfcf4a754b6507c790a91c14b0fe03998`.
These are distinct from roughly 245/283-second verification walls.
Use the history matching the actual evidence mode, rather than treating
skipped observations as measured opt-in execution costs.

Default whole-file cost ingestion is recorded in
`clean-public-wholefile-v5/plan-6dm57u3a/plan.json`, SHA256
`1304e8f2024f5dc7f754f1075d5d3eb9a9167784b84da988b32a6738259ce499`:
the public file stays intact at 244.085 seconds, source consistency PASS.
This cost-focused 32-partition plan supplied only the public history, leaving
3458 unknown nodes. Its PASS does not authorize final dispatch or claim
complete cost coverage. Keep the public file whole by default, review its
node-local reporting/isolation separately if splitting is still needed,
append all actual final-source histories and replan after owner cost repairs.
No deadline, wrapper, fixture owner file, runtime package or cluster object
was changed to obtain these observations.
