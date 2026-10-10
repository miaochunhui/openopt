# HSD Loaded Repair Independent Review

Target: `6cefdcf0b1f2a05c76c26dceafbf10430bc67169`. This document and
`tests/test_hsd_loaded_repair_independent.py` are the only reviewer write paths.
Production, original tests, main, other worktrees, dependencies and cluster
are read-only. No GPU, installation, network or nested agent is used.

## Original Assertions And New Source

The harness reads the exact Git object
`9c6b69bda8e86ad8cd3b198ebb72afa155186bff:tests/test_hsd_lp_independent.py`,
SHA256 `74a7996bcb60d96b19cc6935736b18aaa62341f1d3f48c53384b2ddf5726aedd`.
Every original test function, decorator and assertion is executed unchanged.
The independent Fraction original-coordinate oracle is unchanged. The explicit
enclosing context binds the new owner root and five full source hashes; the
historical internal label is not evidence for the new owner.

Only the module-scoped counting fixture is adapted. Its historical projection
wrapper conflicts with the repaired strict imported-alias contract. The new
fixture observes real Python projection/solve calls and real C LU calls using
`sys.setprofile`, checks actual tensors and retains counts. It does not replace
any production function, code, imported alias or guard. The prior profile is
required to be empty and restored. Twelve genuine cold native CPU solves,
including a one-iteration failed candidate, feed the unchanged 45 assertions.
No positive is accepted merely because an altered module was rejected.

The old clean review remains FAIL, 43 pass/two true loaded-checker failures:
`hsd-independent/clean-full-v1/review-j3id7xaj/receipt.json`, SHA256
`438aa68a33aa98d18b81b2c6e2ae7e4a32723ced2c1c1c54eb6c7f0dcabd55c8`.
The author's byte-identical old-harness replay also remains FAIL, 20 pass/
25 fixture errors, zero failure/skip:
`hsd-loaded-repair/unchanged45-clean-v1/unchanged-ma249lmb/receipt.json`,
SHA256 `95f193844f1824c62236e0ebc37bb6fad978492379c9bd0cc3323691934a51df`.
Neither result is relabelled. New acceptance needs complete actual execution.

## Independent Controls

The original loaded candidate/Farkas controls load altered source in external
copies, restore trusted bytes and require rejection. A separate exact-input
test computes `A.T y = 1/2` for `A=[[-1],[1]], y=[1,1.5]`, requires the original
annihilation gate to reject and also checks a valid equality candidate.

New prepare/solve negatives exercise actual checker code, helper/default/
dataclass fields, constructor default instances, class behavior, native helper,
projector alias/wrapper/closure, cone/Fraction aliases, module cache/origin and
guard helpers. Post-iteration mutations test the retained closure after a real
LU call. An additional control changes the existing guard's code in place,
preserving its object identity; combined candidate and exact-half Farkas tests
must reject that substitution both after preparation and during a real solve.
Unexpected returns remain unconditional red, not xfail or skip. Observations
are saved before the rejection assertion, with source/code/model hashes and
independently recomputed mathematics.

## Receipt Entry

Use the existing CPU interpreter, CUDA hidden and one BLAS/intra-op thread.
The CLI requires clean complete reviewer source, records exact nonempty
collection and JUnit bidirectional membership, duplicate/suite counts, original
45 membership, observations, first cause and source before/after. Collection
has a 25-second budget and pytest 120 seconds; timeout is FAIL. Every run gets
a new external directory and never overwrites prior evidence.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. timeout --kill-after=5 170 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_hsd_loaded_repair_independent.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/hsd-loaded-repair-independent/clean-full-v1
```

Pre-smoke: 80 pass/four explicit local CUDA skips, 6.94 pytest seconds, clean
source PASS at `hsd-loaded-repair-independent/pre-smoke-v1/smoke-d59nwvtv`.
The owner's clean 104-node suite is author evidence, not independent closure.
Read the final external `receipt.json` for actual acceptance and clean reviewer
commit. Execution attestation, CUDA, public/general/commercial qualification and
speed claims remain UNVERIFIED even if this bounded CPU suite passes.

## Actual Findings

Development full-v2 is **FAIL: 96 nodes, 91 pass/five fail/zero error/skip**,
36.18 pytest seconds, unchanged source. Original 45 nodes all execute and pass;
both original load/restore controls reject. The 40 separate prepare/solve
closure negatives reject, including defaults/classes/aliases. The clean final
receipt must independently repeat these facts after the reviewer commit.

Evidence: `hsd-loaded-repair-independent/development-full-v2/review-pnx7jzis/receipt.json`,
SHA256 `131bc51cfdfba289af1cb0de68a3c61f34dd36aa59b08a54aa6c82b31fbb0fb4`.
The first development run at `development-full-v1/review-kiodwmyj` stays FAIL:
pytest 91 pass/five fail, 32.57 seconds. Its four combined observations exposed
the reviewer's tuple/list JSON readback comparison, and its summary encountered
an incorrect JUnit module identity expectation. Only these two harness defects
were corrected; every mathematical and rejection assertion remains unchanged.

**P1: retained guard object identity does not bind its current executable code.**
Prepare compares the guard's code to trusted source, but solve only checks
`_loaded_binding is self._binding_entry` before calling that mutable function.
Replacing the existing function's `__code__` with a no-op preserves identity
and disables the guard's own bootstrap, entry and post-solve checks. This is
a persistent in-process mutation, not an excluded restore-between-checks race.

Five actual controls return rather than reject:

- In-place guard code after prepare permits a complete one-iteration solve.
- Guard plus candidate checker code after prepare or during the first real LU
  returns `solved_candidate` at iteration four. Original equality `A=1,b=2,q=1`
  has `x=2.7473684210526317`; the independent oracle rejects primal equation,
  original slack cone, stationarity and objective gap.
- Guard plus Farkas checker code after prepare or during the first real LU
  returns `primal_infeasible_ray` at iteration two, `y=[1,1.5]`.
  Independent Fraction gives exactly `A.T y=1/2`, not annihilation.

The smallest owner-side requirement is an independent trusted guard-code
check before invoking the retained guard at each entry and exit; comparing
mutable function identity alone cannot meet the declared loaded-code contract.
The reviewer does not modify production. All five assertions stay red, saved
models/results/oracles and altered code hashes precede their assertions.
Do not promote the owner's 104-node PASS or the original45 PASS to closure.

The CLI also supports `--check-receipt FULL_EXTERNAL_RECEIPT_PATH`: it rechecks
clean complete source, exact 96-node collection/JUnit, original45, 50 observations,
artifact/history pins and acceptance without a new solve. An internally intact
FAIL receipt still returns one and prints `integrity=PASS acceptance=FAIL`.

## Caller-Pinned Owner Entry

The original reviewer `b99ae01152701968c0e1c25a772ff02c061d22de` stays frozen.
Its clean final result is FAIL: 96 nodes, 91 pass/five fail/zero error/skip,
original45 pass, source consistency true. Actual receipt:
`hsd-loaded-repair-independent/clean-full-v1/review-lw72l8p1/receipt.json`,
SHA256 `ed79cd4f852624bd10882a732050ebf6d0a1770e09e2017d1e329119ffb9323a`.
Final smoke is 80 pass/four explicit local CUDA skips, source PASS:
`clean-smoke-v1/smoke-suqdx505/verification.json`, SHA256
`ddf49f323421aed3a613d5d3230e6520d26ae99f648b5fac322784b7e84b2b62`.
Neither the owner104 PASS nor smoke removes the five genuine false accepts.

The separate `codex/release4-20261010-hsd-owner-map` worktree modifies only this
document and the independent test's entry mapping. It never edits the old
review tree or the new owner. With no owner arguments, the fixed-owner v1
contract remains. Explicit paired `--owner-root ABSOLUTE_ROOT --owner-commit
FULL_40_LOWERCASE_SHA` selects receipt v2, without implicit source relabelling.
The owner root must be the exact clean Git top-level and HEAD must equal that
caller commit. Five source pins are derived from the caller-fixed Git objects
and compared to disk. The reviewer and complete owner source snapshots remain
separate and must both stay unchanged. Subprocesses transport and revalidate the
full owner map, then check loaded module file/origin/cache identities. Unknown
or mismatched bindings reject; no production guard is bypassed.

Every original and added test function, decorator, counting fixture, hostile
control helper and mutation inventory is checked byte-for-byte against
`b99ae011`. This does not add nodes or change any mathematical/rejection
assertion. Historical45 still compiles from its independently fixed original
bytes. The five true failures remain unconditional failures. The prior clean
96-node FAIL is pinned into v2 receipts and never upgraded.

From the clean reviewer worktree, use the existing interpreter and CPU-only
environment shown above; substitute only the caller's clean owner path/SHA:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. timeout --kill-after=5 170 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_hsd_loaded_repair_independent.py --owner-root "$OWNER_ROOT" --owner-commit "$OWNER_COMMIT" --check-owner --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/hsd-owner-map/new-owner-entry
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. timeout --kill-after=5 170 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_hsd_loaded_repair_independent.py --owner-root "$OWNER_ROOT" --owner-commit "$OWNER_COMMIT" --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/hsd-owner-map/new-owner-full
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. timeout --kill-after=5 170 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_hsd_loaded_repair_independent.py --owner-root "$OWNER_ROOT" --owner-commit "$OWNER_COMMIT" --check-receipt "$NEW_RECEIPT"
```

`--check-owner` imports and collects only: exact nonempty 96-node membership
must match the pinned old receipt; a read-only profile traps any solve,
preparation, numerical checker or LU execution and reports zero calls. Its PASS
is mapping/collection acceptance only, with `accept_hsd_loaded_repair=false`.
It is not new mathematical or repair signoff. The full command still uses
25-second collection and 120-second pytest budgets; no deadline is increased.
Full acceptance requires 96/96 pass, original45 pass, all 50 observations, exact
collection/JUnit bidirectional membership with no duplicates, valid positive
12-solve/profile evidence and both complete clean source snapshots unchanged.
The five controls must actually reject at the original execution points; an
earlier import/fixture error cannot substitute for executing their assertions.
CUDA, remote execution and commercial qualification remain UNVERIFIED.

No-solve CLI selftest:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. timeout --kill-after=5 170 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_hsd_loaded_repair_independent.py --entry-selftest --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/hsd-owner-map/entry-selftest
```

It executes eight real CLI controls: six rejects (root-only, commit-only,
short commit, wrong HEAD, nested root and relative root), one collect-only
positive on the unchanged old clean reviewer used as a source root, and one
rehashed owner-transport reject. This is eight entry controls, not eight new
pytest nodes. Development evidence:
`hsd-owner-map/development-entry-v1/entry-selftest-0mb3ly1n/receipt.json`,
SHA256 `5e46a8d283770c4129506854d12bb118e52a76c78eb8e50d7553d0157a42fcca`;
eight controls PASS, exact96 collected, observed numerical calls zero.
Development smoke: `development-smoke-v1/smoke-9gjk9bje`, 80 pass/four explicit
CUDA skips, 6.71 pytest seconds. New owner freeze was not yet supplied for this
entry-only evidence; no new owner is signed off.
