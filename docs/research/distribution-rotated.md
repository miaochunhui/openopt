# Installed RSOC Distribution Acceptance

Scope: independently extend only `scripts/check_distribution.py`,
`tests/test_distribution_rotated.py` and this document from frozen base
`e3d5511b744073627e582219fc066f474d31d500` on 2026-10-10. Algorithms, public
exports, shared reports, main worktree, infrastructure and GPU are unchanged.

## Original Numerical Models

Both CPU cases use original variables x=(a,b,w), A=[(0,0,1);-I], b=(2,0,0,0),
cones=(Zero(1),RSOC(3)), so w=2 and 2ab>=w^2 with both heads nonnegative.
The LP has P=0, q=(1,2,0), analytical optimum x=(2,1,2), objective 4,
slack (0,2,1,2), dual (-2,1,2,-2).
The QP has P=diag(1,4,0), q=(0,0,1), the same optimum/slack, objective 6,
dual (-5,2,4,-4). The original primal equation, stationarity, cone boundary,
complementarity and zero signed gap can be recomputed directly.

Each installed probe actually executes native PyTorch ADMM and explicitly
selected Clarabel CPU reference on each same model: two native and two
Clarabel solves, zero commercial solves. Original arrays are frozen and
checked unchanged. Both observed solutions must pass a separately called
installed original-coordinate checker, including all nine checks and a
finite available signed gap. Native actual tensors must be CPU float64.
Analytical point/objective and paired native/reference results are checked.
Embedded solver evidence alone is never accepted. Wrong dual signs, planted
slack and wrong tail coordinates must fail the independent checker.

## Byte And License Contract

The coordinator freezes a strict schema1 source manifest before building:
source commit/tree/clean, every original openopt/openpo Python relative path
with SHA256 and byte length, 0BSD expression and original LICENSE descriptor.
JSON duplicate keys, nonfinite constants, traversal, unknown fields and bad
descriptors fail loudly. The worker receives the manifest path and a caller
SHA256, not a self-declared installed-source identity.

Build input is copied from the source snapshot into an external artifact
directory and rehashed before build. The wheel Python set and exact bytes
must equal the frozen snapshot. Wheel license metadata and original license
bytes are validated. Isolated installation uses only the new wheel with
`--no-index --no-deps`; build uses `--no-index --no-deps --no-build-isolation`.
Installation also uses `--no-compile`; unpinned package bytecode/native
executables are rejected, so a stale pyc cannot bypass original source bytes.
Pip cache and version/network checks are disabled for this offline run.
No dependencies, SDKs or GPU packages are installed/downloaded.

A fresh `python -I` process verifies installed Python set, all exact source
bytes and installed openopt 0BSD metadata/LICENSE before imports, rejects
cached checkout imports, and checks every loaded project module path/bytes.
The same byte check is repeated after numerical acceptance. The numerical
report includes original models, both observed solutions, independent
evidence and negative controls. The coordinator binds its report path/hash/
bytes and checks source consistency after build/install. These hashes permit
byte recomputation; they do not establish a complete dependency/legal audit,
signed execution attestation or unrelated license redistribution rights.

## Acceptance Entry

Artifacts remain under `/hf3fs-jd/data/tmp/openopt-release-20261010/wheel-rsoc-v2`.
Use the existing validation interpreter and hide CUDA; set OMP/OpenBLAS/MKL
threads to one. No source data or old cluster fixture is needed for these tests.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_distribution_rotated.py --target tests/test_distribution_license.py --target tests/test_distribution_sources.py --target tests/test_rotated_conic.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/wheel-rsoc-v2/cpu
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/check_distribution.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/wheel-rsoc-v2/wheel
```

Standalone legacy `--installed TARGET` remains accepted and exercises the
original probes plus RSOC, but reports `provenance=UNVERIFIED` without an
external caller-pinned descriptor. The coordinator supplies optional
`--source-manifest PATH`, `--source-manifest-sha256 PIN`, and external
`--probe-report PATH`; partial path/pin arguments reject. Bare imports cannot
claim source-qualified distribution acceptance. CPU success is not CUDA
execution, performance, all cones or commercial qualification. All failures
remain in their original external run directories; completion requires
actual verification/installed-probe/distribution JSON and clean committed
source consistency, not this document or a directory name.

## First Cause

First CPU run at `cpu-v1/tests-szp6y9aw/verification.json` retained one
failure and 86 passes, 12.51s. The duplicate-license negative fixture appended
a header AFTER the METADATA header/body separator; that is body text, not
a duplicate parsed header. The fixture now inserts the duplicate before the
separator. No production acceptance criterion or numerical tolerance was
relaxed. The actual native/reference LP/QP test passed in that first run.
