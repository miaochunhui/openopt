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
