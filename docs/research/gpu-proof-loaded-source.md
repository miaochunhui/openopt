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
