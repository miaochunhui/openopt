# Independent Conic Type Repair Review

Frozen production target: `966a2b3520f21aa0fe9e9156b960756d9f56d3d7`.
Worktree: `/data/dev/openopt-wt-release4-20261010-conic-types-new-independent`.
The reviewer owns only this document and
`tests/test_conic_types_repair_independent.py`. No production, previous review,
main repository, GPU, dependency, network or nested-agent changes are permitted.

## Source And Historical Boundary

The new review pins the repaired direction module, original conic and numerical
evidence modules, native prepared implementation, verifier and owner test/doc
bytes. The direction module SHA256 is
`63b1d278f26d536903904fdf502c49f3fca914b52cf60249778219f08be3a517`.
Clean verification and replay compare the complete source snapshot, module
paths and function code origins. This is local source consistency, not remote
attestation or protection against arbitrary interpreter compromise.

The original independent commit
`09751dc06815038869926dc9092e30591984abf9` is not edited or imported as the new
review. Its old failure is not requalified. The owner replay remains FAIL at
`/hf3fs-jd/data/tmp/openopt-release-20261010/conic-types-repair/unchanged-independent-v1/replay-auvgoot7/replay.json`,
SHA256 `1c1f6b73a8c19dd902bae71f9275f743f34168e557689d83ee75d2918972f0ba`:
120 tests, 119 passed, one failed, zero errors/skips. Its mixed-q test uses
the old `build` function, which creates `np.array([-1., False], dtype=float64)`
BEFORE calling the checker. The checker therefore sees exactly `[-1., 0.]`.
The new review preserves that failure and independently demonstrates why it
cannot be used as a raw-input rejection test.

For q, b and y separately, the new test proves that a boolean-containing list
and a legitimate numeric list produce identical materialized float64 bytes.
Raw types are recorded before conversion. Raw lists must raise; correctly
materialized numeric arrays are accepted when the independent original-model
oracle passes. This is not permission to infer or certify their earlier origin.
Object/boolean arrays, nested lists/tuples and nested boolean tensors remain
outside the accepted direction-input contract.

## Independent Coverage

Twelve positive original mathematical examples cover coupled zero rows,
nonnegative/SOC/RSOC Farkas separators, ordered cone products, empty LP,
nonnegative/equality/SOC/RSOC recession, singular non-diagonally-dominant QP
and a mixed cone recession. Each has an actual wrong-sign negative control.
Fraction formulas independently check A.T*y, P*d, original base equations,
cone head signs and squared inequalities, and strict signed negative margins.
No owner fixture, projection, eigensolver or solver supplies the oracle.

Raw mixed booleans are passed directly for all P/q/A/b/y/d/x0/s0 fields,
including Python and NumPy true/false; the numeric fixture builder is NOT
used on their mutated raw fields. Nested model/witness booleans, both tolerance
fields, CPU double tensors, numeric list/tuple/float64 arrays and exactly
representable integers have explicit controls. Typed snapshots and original
bytes must stay unchanged across the direction call; CPU tensors receive no
gradients or implicit transfers. Unsupported devices reject before input
access. Meta tensors provide CPU-only no-transfer controls.

Tiny nonzero identities, missing/incorrect bases, negative heads, one-ULP
SOC/RSOC violations, equality at a strict margin, indefinite/nonsymmetric P,
nonfinite/inexact data, intermediate overflow, cancellation overflow,
underflow and bounded model-size failures must never become accepted evidence.
One separate native prepare-only probe checks caller/copy isolation and frozen
original bytes on CPU; it does not execute a native solve or extend native
boolean parsing. Existing conic candidate/reference/native input contracts
were not upgraded by the direction-only repair and are not independently
signed off as raw-bool rejecting.

## Reproduce And Evidence

All runtime data, typed observations, logs and JUnit stay outside the repository
on 3FS. Every run creates new directories and receipts exclusively. No previous
FAIL or receipt is overwritten. The CLI invokes the existing verifier, then
performs live pytest collection and exact bidirectional collection/JUnit
membership, duplicates and suite/leaf counts. It replays all twelve original
math cases, re-reads typed observations and requires zero false accepts,
zero failures/errors/skips and clean before/after source. Dropped/extra/
duplicated/coordinated-truncated membership has negative controls.

Use the existing CPU environment, hide CUDA and set all three BLAS/OpenMP
thread variables to one. Each verification must finish within 170 seconds;
no public solve, GPU job or network fetch is performed.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_conic_types_repair_independent.py --run --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/conic-types-new-independent/clean-final-v1
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_conic_types_repair_independent.py --audit-artifacts NEW_EXTERNAL_VERIFICATION_DIRECTORY
```

The second command is an alternative audit of an unaudited verification
directory, not an overwrite of the receipt produced by --run. It requires
the retained historical replay file and its exact pin; missing evidence fails
loud. Direct pytest numerical tests do not require the historical file.

Pre-review smoke:
`conic-types-new-independent/pre-smoke-v1/smoke-vtq6smi3/verification.json`,
80 passed/four explicit local CUDA skips, 7.12 pytest seconds, clean original
production source. Those smoke skips are not part of independent acceptance.
The first development run was
`conic-types-new-independent/development-v1/tests-oauh3moe/verification.json`,
136 passed, zero failures/errors/skips, 11.34 pytest seconds. Its review files
were uncommitted, so it is not the clean final source signoff.
After adding strict cast/prepared receipt re-read and five cast-boundary
metadata negative controls, development-v2 was
`conic-types-new-independent/development-v2/tests-ahw1plgw/verification.json`,
141 passed, zero failures/errors/skips, 11.48 pytest seconds. This too is
development evidence, not a clean signoff. Final frozen acceptance is the
actual `independent-review.json` below the fresh `clean-final-v1/review-*/tests-*`
directory; it contains exact counts, both nodeid lists, all hashes and
before/after clean source checks.
No terminal independent PASS is asserted before its actual clean receipt.

`solver_calls=0` refers to this supplied-direction/math and prepare-only
review, not the existing verifier's separate portfolio example or pre/post
smoke suites. Those auxiliary regressions perform CPU solves outside this
direction review and cannot be counted as conic ray production or GPU evidence.

The bounded direction audit is approximate supplied binary64 host numerical
evidence, not an exact underlying-real certificate, generic conic solver,
HSD capability, GPU execution, performance, industrial coverage or commercial
qualification. It cannot recover upstream type provenance from numeric bytes.
