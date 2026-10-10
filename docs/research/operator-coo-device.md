# CUDA COO Structural Validation Repair

This is a preparation/placement repair, not a new solver, speedup, commercial
qualification or accepted CUDA execution. The exclusive write set is
`src/openopt/operators.py`, `tests/test_operator_coo_device.py` and this document.
Base commit is `f9cc655ba14b2804a97befa5755aad46417b4b77`; the original 53-node
producer, Placement, thresholds, numerical checker, presets and public exports
are unchanged. CPU COO and all CSR reconstruction retain their existing native
`check_invariants=True` path. QP/CG arithmetic, settings and acceptance defaults
are unchanged.

## Actual First Cause

Immutable historical run:
`/hf3fs-jd/data/tmp/openopt-release-20261010/chair-f9cc655-operator-cuda-admin-v2/release-regression-ra548ym9`.
Its case is `case-0/cuda-check-5wtqbktw`, completed at
`2026-10-10T15:15:45.354239+00:00`: 53 nodes, 44 pass, nine fail, zero skip,
exit1. Node h3c2, NVIDIA RTX PRO 4500 Blackwell, torch `2.13.0+cu130`, CUDA13.
This run remains FAIL and is not relabeled or backfilled.

All nine fail at original `operators.py:81`, `TensorOperator` COO reconstruction
using `torch.sparse_coo_tensor(..., check_invariants=True)`. The dispatcher emits
a CPU `tensor([0,0])`; unchanged `Placement` at producer line81 rejects it as
off-device. Failed nodes are four f64 models, four f32 models and the f64 zero-row
negative, all COO. Conversion refusals and non-COO nodes do not establish that
COO preparation works. The trace identifies the constructor/validation boundary,
not a need to weaken Placement or skip checks. No new GPU job was submitted.

Historical operator SHA256:
`d63165ea0a1a6c7f3325c74a77c132d3962feee0638541d016c360bf41c611c2`.
Unchanged original CUDA producer SHA256:
`3e8a051bd7ab72964defb41a67e043671c5905242bc423d82f062e41fd5da120`.
The new test CLI reads JSON/XML/source as bytes only, verifies actual source
SHA bindings, checks the nine COO failure texts and 53/9/0 JUnit membership,
and records hashes/bytes without executing any archived code. Its verdict is
`HISTORICAL_FAIL_DIAGNOSED`, never repaired execution PASS or remote attestation.
Its receipt is external at
`operator-coo-device/diagnosis-v1/diagnosis.json`, SHA256
`7236c0eaeb74c2878e509ee8e214823b8caaffd6ba1b16602d01b6d906489557`.

## Replacement Validator, Not Removed Safety

Only CUDA COO calls the new private `_checked_device_coo` before reconstruction:

- Require dense int64 indices of shape `(2,nnz)` on the original device.
- Require a 2-D, nonhybrid COO matrix with real float32/float64 metadata.
- Require dense 1-D values matching matrix dtype/device, index nnz and raw nnz.
- Check BOTH strict lower and upper original bounds for EACH axis on the input
  device, including zero-sized axes. Extreme int64 indices cannot wrap through
  a flattened-key arithmetic check because no flattened keys are constructed.
- Require finite original values. Every failed check raises before rebuilding,
  coalescing, sparse multiplication or transpose preparation.
- Rebuild with explicit original dtype/device and `is_coalesced=False` only
  AFTER validation. Unsorted/duplicate COO entries are legal; an input false
  coalesced claim is not trusted. The existing owned clone then coalesces and
  validates finite combined values, so duplicate-sum overflow is also rejected.

`check_invariants=False` on this one reconstruction is intentional: the complete
supported COO structural predicates were checked explicitly immediately before
the call. There is no global validator toggle, exception swallowing, host-index
transfer or sparse-to-dense substitution. Scalar boolean reductions synchronize
with the host; this cost is disclosed and not called GPU speedup. CPU/CSR keep
the old native validator and all their existing malformed-input tests.

The helper can be exercised directly on CPU to check its mechanism. Production
CPU routing does NOT switch to it. Such tests establish CPU structure/negative
controls only; they do not demonstrate CUDA kernel placement on torch2.13.

## Tests And Reproduction

Run as infra in the independent worktree, using the existing interpreter without
installation. All evidence roots below are external to the worktree.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python scripts/verify.py tests --target tests/test_operator_coo_device.py --target tests/test_qp_operators.py --target tests/test_operator_original_contract.py --target tests/test_qp_precondition.py --target tests/test_operator_original_cuda.py --target tests/test_sparse_contract_independent.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/operator-coo-device/replay
```

The unchanged sparse independent source-boundary test requires a CLEAN committed
tree, so use the frozen owner commit for this full command. New tests cover legal
duplicates/unsorted/zero nnz/empty axes/f32/f64, false coalesced flags, both axis
signs/bounds/extreme indices, NaN/Inf, raw metadata mutations, finite duplicate
overflow, unchanged CPU/CSR backend checks and malformed CSR. Original tests
retain original-model Fraction acceptance, conversion/overflow/underflow
refusals, callback device contracts and solver residual controls.

Pre-smoke: `operator-coo-device/pre-smoke-v1/smoke-i4djde09/verification.json`,
80 pass/four explicit local CUDA skips, 6.52s. The CPU pre-repair mechanism test
is retained at `pre-repair-red-v1/tests-to4o5nuq/verification.json`, one fail
because the new private helper did not exist. That is NOT a CUDA reproduction;
the actual CUDA first cause is the immutable nine failures described above.
Development unit-v2 is `unit-v2/tests-47sp50bg/verification.json`, 152 pass,
21 explicit default-off CUDA skips, 13.09s. Final clean-source receipts and
counts are supplied in the owner handoff, not inferred from these earlier runs.

Read-only historical diagnosis entry (no solve, no artifact code execution):

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /hf3fs-jd/data/tmp/openopt-validation-env/bin/python tests/test_operator_coo_device.py --diagnose-run /hf3fs-jd/data/tmp/openopt-release-20261010/chair-f9cc655-operator-cuda-admin-v2/release-regression-ra548ym9 --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/operator-coo-device/diagnosis-replay
```

Explicit selection of `test_cuda_coo_structure_entry` (12 nodes) and/or
`test_cuda_coo_invalid_entry` (nine nodes) requests actual CUDA: missing device
raises, never skips or falls back. Ordinary whole-file CPU regression skips
these explicitly default-off GPU nodes. The available shared worker entry
supports target/expression selection; no shared worker or environment is changed.
For the chair AFTER independent review, use the ORIGINAL unchanged formal entry:

```text
--target tests/test_operator_original_cuda.py::test_cuda_collection_entry
--expression test_cuda_collection_entry
```

Its original 53 nodes, strict placement/profiler and numerical acceptance remain
the required actual device regression. The new structural controls can separately
be selected with targets above or expression
`test_cuda_coo_structure_entry or test_cuda_coo_invalid_entry`.
Neither CPU mechanisms nor this document grants repaired CUDA acceptance. Only
the chair submits GPU work, and only after clean-source independent review.
Power-cone and previous sparse-CUDA worktrees remain frozen. No root, permission
helper, chmod, package install, push, publication, cluster action or nested agent
was used for this repair.
