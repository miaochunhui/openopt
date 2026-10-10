# GPU Proof Loaded-Source Guard

## Scope And Trust

This is a CPU-tested loaded-source identity guard, not GPU qualification,
remote attestation, an exact certificate or commercial solver qualification.
The worktree baseline is `3792a86f9f5658642c5e250053a48d77c6425643`.
Only the producer, its tests and this document are changed. The outer acceptor,
dispatch, numerical checker, cluster and other agents' trees are untouched.
The producer/test files do not overlap Herschel's outer-audit write set.

Caller-pinned source membership and complete original arrays remain mandatory.
Artifact snapshot code is never imported or executed. The current producer
must be loaded from the explicit trusted root. Cached foreign modules are
rejected before ordinary imports, without removing or replacing `sys.modules`.
Changing `sys.path` alone cannot repair a foreign cache. A caller switching
trusted roots must use a fresh process.

`loaded_source_check(current_root, expected_sources=None)` returns `root`,
`modules` (actual paths and SHA256) and `callables` (source paths, hashes and
lines). Generated dataclass constructors additionally report their `<string>`
code filename and source declaration. `replay_case(..., current_root=None)`
preserves the old default root while allowing explicit root binding.
Production, case replay and result validation invoke the guard; seal and
bundle replay use result validation. Checks occur before and after production
and result replay. Original six candidate gates, both ray checks, raw model
pins, numerical thresholds and status requirements are unchanged.

## Bound Objects

- Every loaded `openopt` module has exact trusted-root file, actual byte hash,
  spec origin, package search path and `sys.modules` identity checks.
  Parent-package attributes must refer to that same cached module.
- All explicit checker functions/helpers/class methods and selected config
  validators are compared with code compiled from trusted bytes. Globals must
  be the live module dictionary; closures, contextmanager wrapper and runtime
  positional/keyword defaults are checked separately. Type-sensitive defaults
  distinguish `True` from numeric values and `None` from empty defaults.
- `QPConfig`, `OperatorQPConfig`, `ScalingConfig`, `NumericalEvidence` and
  `_Problem` constructors require frozen assignment-only dataclass code,
  expected fields/defaults, globals, object closure, plain type/base and
  unchanged constructor entry. No trusted class is executed to generate
  expected code. Cross-core captured config/checker/evidence aliases are bound.
- Module aliases `np`, `math`, `torch`, `contextmanager`, `asdict` and `dataclass`
  are checked against the trusted runtime dependencies. Selected callables'
  referenced module-level literal constants are checked against source AST.
  Current checker epsilon factors are code literals and `np.finfo` calls,
  not separately mutable module-level epsilon constants.

The trusted envelope includes Python, standard library, installed NumPy and
PyTorch and the caller-trusted producer/outer tools. This guard does not hash
third-party native binaries, attest an adversarial interpreter, or verify every
mutable dependency internal (for example replacing `np.finfo` inside the actual
NumPy module). It does not audit every solver callable or all module globals.
In particular, no-solve replay tests may replace solver/factory/hardware methods
with forbidden stubs. This contract remains supported.

New literal checker keyword defaults are read from the final caller-trusted
AST, rather than a hard-coded tolerance list. Unknown default expressions,
non-assignment-only/keyword-only dataclass generation or new decorators are
unsupported and fail closed. Future versioned opt-in arithmetic/config changes
must be tested after integration on the final tree; no changing Nash worktree
was used as this guard's acceptance source.

## Evidence

All following paths are relative to
`/hf3fs-jd/data/tmp/openopt-release-20261010/proof-loaded-source/`.
Each test gate stores `pytest.log`, JUnit and `verification.json` externally.

| Gate | Evidence | Actual outcome |
| --- | --- | --- |
| Foreign-loaded module P1 before fix | `before-red/tests-9b7bcy0v` | 12 real negatives failed to raise |
| Defaults/constructor before fix | `defaults-before-red/tests-mx13r6o3` | 6 real negatives failed to raise |
| Defaults/constructor first repair | `defaults-after-green/tests-po3m4jw6` | 7 passed, 3.92 s |
| File-set/byte fixture probe before repair | `case-copy-before-red/tests-t2sy5fpk` | 3 failed; membership lacked coordinates and same-length mutation was accepted |
| Expanded targeted batch | `current-p1-targeted/tests-xo2w1ypw` | 217 passed / 14 failed, 216.93 s |
| Targeted fixture repairs | `fixture-repair-green/tests-ema_883m` | 37 passed, 108.11 s |
| Global alias / parent cache before fix | `global-parent-before-red/tests-4j0n_1i9` | 3 real negatives failed to raise |
| Global alias / parent cache after fix | `global-parent-after-green/tests-nj9u2pxg` | 15 passed, 4.17 s |
| Code smoke before final alias gate | `p1-code-smoke/smoke-hcj9_ti_` | 80 passed / 4 explicit CUDA skips, 6.42 s |

The 14 expanded-batch failures were preserved, not promoted to green. Eight
were a conditional local `importlib` import in the new fresh-process fixture.
Six were CPython constructor-slot pollution caused by temporarily assigning
`__new__` on the shared evidence class. The repaired negative uses a detached
class; each direct mutation now proves the original constructor works after
restoration. The 37-case rerun exercises fresh-process positive/negatives,
constructor restoration, original nine-case replay, no-solve contract and
the affected full direction summaries. It is not a full-suite rerun.

The expanded batch's 90 defaults/constructor negatives reuse the same real
nine-case producer fixture: six mutations, gap/infeasible/unbounded and five
entry points (case/validate/seal/replay/production). Foreign preload similarly
covers four core modules, those three statuses and all five entry points.
Failures during production preserve a FAIL result and first-cause log.

Historical `full-v3/tests-5l2q6vk6` was 150 passed / 408.17 s, before cache and
fixture narrowing. `affected-v4/tests-efnaurjr` was 116 passed / 118.60 s before
the defaults/constructor correction. Neither is current final full coverage.
Latest whole-file timing is unmeasured; a 240--360 s scheduling estimate is
not a measured result and cannot authorize a 300 s whole-file cluster job.
Use the separately owned fixture-aware sharder rather than raising a deadline.

## Measured Local Cost

`current-p1-targeted/tests-xo2w1ypw/pytest-tmp/` holds machine-readable cost
records, including full original/copied file inventories and byte descriptors:

- `test_loaded_source_cost_and_li0/guard-cost.json`: cold 68.697 ms,
  warm median 14.303 ms, 11 modules, 23 callables, nine compilation entries.
  This measurement predates the final dependency/parent alias checks.
- `test_single_case_fixture_prese0/fixture-cost.json`: same produced evidence,
  narrowed copy plus byte verification 1.261 s, 76 files / 1,089,655 bytes;
  whole copy plus byte verification 7.359 s, 467 files / 1,366,858 bytes.
  This is one local ordered comparison, not a robust speed benchmark.
- Three `test_single_case_fixture_reje*/fixture-mutation.json` records prove
  missing file, extra file and same-length hash mutation rejection, including
  the actual selected file and copy calls.

`case_copy` includes all selected sidecar descriptors recursively, complete raw
model/vectors/extensions, every source file and result/execution/telemetry.
Exact membership and every copied descriptor must equal the original before
return. It never reduces/reconstructs arrays or relaxes numerical gates.
Fewer fixture files do not prove faster optimization.

Only exact-byte AST/code compilation is cached (bytes, filename, optimization
level; bounded 16 entries). Each guard invocation rereads and hashes actual
files and rechecks live objects, defaults, aliases and cache identity.
There is no live-identity success cache. Network-filesystem source/inventory
reads and repeated fresh-process fixture copies remain test-cost contributors.

## Final Acceptance

Final-source targeted and smoke logs are recorded under `final-guard-targeted/`
and `final-guard-smoke/` after the last manual edit. Their machine-readable
terminal outcomes, not this document's expected result, are authoritative.
No whole-file final run, GPU job, wheel requalification, outer all-nine
integration or independent Herschel review is claimed by this commit.
The main agent must integrate the commit, freeze its final current tree,
revalidate source-origin in a fresh process and conduct the separate actual
GPU/outer acceptance. Old snapshots cannot be relabeled as final-source runs.

## Sys Compatibility Follow-Up

The separately committed follow-up to `2cb3020` permits an actual trusted
`sys` import/alias. Source defaults and referenced global constants may read
only enumerated numeric fields of the trusted `sys.float_info` record; no
arbitrary AST attribute, expression or call is evaluated. Direct references
inside checker functions remain covered by code identity and the `sys` alias
check. Mutation inside the trusted runtime itself remains outside the guard's
claimed scope.

No changing Nash checker tree was imported. A fresh-process test adds an
explicit synthetic trusted checker declaration using `sys.float_info.epsilon`,
`min` and `max` to the external test source copy before import and source pin.
This is a compatibility fixture, never the evidence snapshot. One process
checks the positive declaration and four actual mutations (module alias,
global epsilon, keyword default and probe code), with unchanged source-map
and successful restoration after each mutation. It runs no solver.

`sys-compat-before-red/tests-o76w5cfc` preserved the real incompatibility:
one failed test, `unsupported loaded source dependency import`.
`sys-compat-after-green/tests-9u2_s9tr` records 16 passed, zero skips,
10.12 s, including the four rejection controls in the sys fixture.
Clean-commit follow-up validation is external under `sys-compat-clean-targeted/`
and `sys-compat-clean-smoke/`; read the actual report source commit, clean
status, byte hashes and terminal counts rather than assuming a PASS.
This follow-up does not establish final Nash integration, a whole-file timing
below three minutes, independent P1 closure or formal GPU qualification.

## Independent Fixture-Cost Follow-Up

The new branch `codex/release4-20261010-proof-fixture-cost` starts at clean
`fa25d31a18c5e8ee32f9994942924fdd1b8f484e`. The previous reviewed worktree is
untouched. This follow-up changes only this document and the producer tests;
`scripts/gpu_release_proof.py` and all numerical/source identity gates remain
unchanged. No GPU, cluster, installation or remote push is involved.

The opt-in `tests/test_gpu_release_proof.py --fixture-cost` entry records
actual profiler calls, source snapshots, external logs/JUnit and the separately
owned Helmholtz fixture audit. It requires explicit targets or a pinned,
same-source collection group; it never defaults to running the entire file.
`--no-profile` measures real selected execution without profiler overhead.
`--baseline-collection` checks the complete original parameterized nodeids,
not merely equal counts. New helper code is not executed as artifact snapshot
code. Historical JUnit is labelled historical, never current runtime proof.

Evidence below is relative to
`/hf3fs-jd/data/tmp/openopt-release-20261010/proof-fixture-cost/`:

| Controlled test change | Evidence | Actual outcome |
| --- | --- | --- |
| Clean precondition | `precondition/smoke-6mbmsjng` | 80 passed, 4 explicit CUDA skips |
| Original copy profile | `baseline-profile-v3` | 16 passed; wall 62.620 s |
| Sealed manifest reuse only | `manifest-profile-v1` | 16 passed; wall 57.269 s |
| Direct byte copy follow-up | `direct-profile-v1` | 16 passed; wall 49.968 s |
| Nine-case bytes and fresh checker compatibility | `copy-compat-v1/tests-y46pdu0d` | 15 passed; pytest 121.00 s |
| Reversible sealed mutations | `restoration-v1/tests-uk1wlcfe` | 9 passed; pytest 31.96 s |

The profiler proves calls occurred: the baseline had 2703 `shutil.copyfile`
calls / 27.590 cumulative seconds, 11 `case_copy` calls / 12.647 seconds,
and 36 live guard calls / 2.716 seconds. These cumulative times overlap and
must not be summed into wall time. Manifest reuse reduced original `read`
calls from 8582 to 7757, but only modestly reduced `case_copy` to 11.959 seconds.
The direct-copy run had the same 2703 copy calls and 36 guard calls.
Runs were ordered local samples, not randomized throughput qualification;
contention and filesystem variation remain possible. No producer optimization
or live identity cache was introduced.

`case_copy` selects from the real sealed fixture inventory instead of hashing
the same original files again. Its destination inventory still verifies every
selected byte and exact membership against the sealed original descriptors.
Fresh caller-trusted source copies and selected evidence copies use ordinary
independent files, not hard links, array reconstruction or solver replay.
The nine-case selfcheck independently inventories the complete original tree,
checks every copied case/model/extension byte, replays all nine original models,
mutates each private copy and verifies the original remains unchanged.
Three existing missing/extra/same-length-hash negatives retain their nodeids
and now mutate the actual copy helper.

The full-tree A/B selfcheck used the same real 467-file / 1,370,418-byte fixture:
shutil copy plus byte checking 6.922 seconds, direct copy plus byte checking
5.119 seconds. These are single ordered test-fixture measurements, not
optimizer performance. All inventories are external in
`test_fixture_copy_ab_preserves0/nine-case-copy-ab.json` under the 15-test run.

The four sealed disk-tamper tests use the genuine module-owned fixture
sequentially instead of cloning it four times. Each proves the full original
inventory equals the sealed inventory before mutation, catches the real replay
rejection, restores in `finally` and verifies every original file/byte afterwards.
A failed restoration fails loudly. The explicit module dependency stays visible
to Helmholtz; no resource edge is hidden and no fixture is shared across workers.

The original full collection contains 269 nodes: one genuine 214-node
`produced(module)` component and 55 singleton components. Its runtime AST/pytest
closure audit is in `baseline-collection-v3` and `manifest-collection-v1`.
The historical 68-node / 334.420-second component is not the current denominator.
Final-source collection and unprofiled selected-component runs are recorded
separately under `final-collection-*` and `final-group-*`. Read their actual
counts, source hashes and budget fields. No whole-file run is implied by a
profile or by a sum of overlapping targeted counts. The explicit cost goal
remains selected batches below 180 seconds and the heaviest actual component
at most 300 seconds; these are not waived or increased.

Two initial diagnostic startup mistakes occurred before pytest execution:
comparing a resolved 3FS alias with an unresolved root, and omitting repository
root from the script import path. They were corrected explicitly; neither
startup error was a solver failure or a passing test run.

### Fixed New-Checker Compatibility

The fresh-process compatibility test uses the fixed Git checker revision
`0c5524e8afbfa067889c1696b8617d17d4cc90ce`, SHA256
`b3f1018c31d908e2e52ff96a097cd76fbb552efc40481489489624c48cdfa583`.
It copies those caller-trusted bytes before pin/import, not from a changing
Nash tree or frozen evidence executable. The live identity includes
`_binary64_runtime`, `_jacobi` and `_reproducible_candidate`. All six original
candidate checks pass with the versioned arithmetic; default `arithmetic`
remains `"numpy"`.

Function-local `import ctypes` / `import ctypes.util` is covered by actual
helper code equality, not the module-level import allowlist. No allowlist was
expanded. Four real same-source mutations are rejected: runtime-helper code
containing those imports, new spectral-helper code, checker keyword default
and `REPRODUCIBLE_ARITHMETIC` global constant. The actual rejection coordinates
and callable identities are in `compatibility.json` under the 15-test run.
Mutation inside trusted runtime/ctypes dependencies remains outside the
guard's attestation scope. This fixture is not final Nash/RSOC integration:
the main agent must freeze the integrated current source and independently
repeat fresh-process/code/defaults/ray acceptance on that new source.

The fixed-revision probe is an explicit diagnostic option
`--checker-revision`, not a mandatory collected test: history-free source
packages must not require that Git object. The ordinary added test uses a
clearly labelled synthetic trusted checker with function-local ctypes imports,
one new helper, literal keyword default and module constant. It preserves all
four real mutation categories without claiming that synthetic code is Nash's
implementation. The original 269 nodeids are preserved; the final two added
nodeids are the nine-case copy selfcheck and
`test_function_local_imports_and_new_helper_defaults_rejected`.
