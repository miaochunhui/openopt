# Native PSD Primitive v1

## Contract And Scope

Base: `73613fce9faafdd622baf69dcf1ecb64b4cfa6f7`.
Only `src/openopt/psd_cone.py`, `tests/test_psd_cone.py` and this document
belong to this task. No existing conic/native/HSD/default/public export changes.
This is an opt-in SDP building block, not a general SDP solver, an LP/QP
extension, a commercial qualification or a measured GPU speedup.

Support: a single real dense strided exactly symmetric n-by-n matrix,
1<=n<=6, binary32/binary64, explicit `device="cpu"` or `"cuda:<index>"` and
explicit matching `dtype`. No batch, sparse, complex or autograd contract.
Tensor conversion, hidden symmetrization of input, NaN/Inf and unavailable
CUDA are rejected. Unsupported normalization which loses a nonzero original
coordinate or fails an exact represented-coordinate scaling roundtrip rejects;
overflowing projected coordinates reject. No installation
or third-party solver/factorization fallback exists.

`project_psd(X, device=..., dtype=...)` returns a `PSDProjection` containing
an owned matrix, algorithm description and independent evidence.
Projection is with respect to the Frobenius inner product tr(XY) on real
symmetric matrices. Mathematically Y=Q diag(max(lambda,0)) Q^T for X=QDQ^T.
Exact original PSD and negative-PSD membership provides identity/zero paths
without changing coordinates. Otherwise a represented power-of-two scaling
keeps the torch eigendecomposition/reconstruction on the requested device.
The independent host checker is explicitly disclosed, not a GPU-only claim.

`svec` stores the lower triangle in column order:
X00,sqrt(2)*X10,...,X11,sqrt(2)*X21,... . `smat` is the inverse convention.
Thus svec(X).svec(Y)=tr(XY) in real arithmetic; off-diagonal coordinates
have weight two in the Frobenius norm. Binary sqrt(2) rounds, so represented
roundtrip/isometry is numerical, not bitwise or rationally exact. The original
matrix, not an svec roundtrip, is the acceptance coordinate system.
In binary32 the sqrt(2) scalar is rounded to the target dtype before its
multiply/divide; the isometry is not a binary64-product oracle.

## Independent Original Matrix Gate

Checker version: `psd-principal-minors-moreau-v1`. The checker never calls
eigh/eigvalsh, projection, solve or iterative refinement. A disclosed host copy
preserves every represented binary32/binary64 value exactly as a bounded
Fraction. Direct rational elimination determines signs of **all** nonempty
principal minors, 2^n-1 per matrix. Testing only leading minors is insufficient
for semidefinite matrices and is not used.

Let G=Y-X, computed in these original rational coordinates. Exact Moreau
conditions for the self-dual PSD cone are Y>=0, G>=0, <Y,G>=0, equivalently
YG=0. The numerical gate checks:

- All principal minors of Y+diag(a+r*abs(Yii)) are nonnegative.
- All principal minors of G+diag(a+r*abs(Gii)) are nonnegative.
- Every component of YG has absolute value at most Lij.
- Absolute tr(YG) is at most sum_i Lii.

Here a=atol, r=rtol and
Lij = r*sum_k(abs(Yik*Gkj)+abs(Yik*Xkj)+abs(Xik*Gkj))
+ a*sum_k(abs(Yik)+abs(Gkj)) + n*a^2.
All sums/products/comparisons are evaluated directly as represented-coordinate
rationals. No squared norm overflows or underflows and no global max-entry
tolerance can hide a small negative diagonal beside a huge positive one.
Default a=0 and r=128*machine_epsilon of the requested dtype; explicit r must
be finite, nonnegative and <1. These are numerical residual limits, not
a forward-error bound for arbitrarily ill-conditioned projection problems.

Both original and shifted minor signs, `exact_psd`, numerical status and full
component ratios are retained. A shifted-minor pass with a negative original
minor is **not exact PSD or certification**. The eigenvalue clamp only proposes
a matrix; wrong candidates which fail the original gate raise
`PSDProjectionRejected` with the failed evidence. Input tensors are never
mutated. Rounding-sensitive candidates may reject instead of producing success.

## Validation And Remaining Work

CPU analytic positive/negative/mixed/boundary cases, Moreau conditions,
svec/Frobenius checks, near-zero/huge scales and corrupt spectral proposals
exercise the actual implementation. Default tests are portable and do not
require historical campaigns. The two CUDA targets require
`OPENOPT_RUN_PSD_CUDA_TESTS=1`; without opt-in they skip, and with opt-in but
missing CUDA they fail rather than skip or fall back. They do not count CPU
execution as GPU evidence. Existing CPU validation environment
only; no GPU submission, root/permission helper, dependencies or nested agents.
Owner freeze, CPU receipts and an independent review are prerequisites before
the main owner considers a fresh frozen GPU run. No GPU evidence is claimed here.

This limited module does not add affine SDP constraints, primal/dual SDP
iterates or status/ray checks, chordal sparse decomposition, general SDP
presolve, native SDP solver convergence, multi-GPU computation or fair
commercial performance evidence. Original-coordinate exact minor signs are
bounded checks of represented matrices, not an arbitrary-precision optimizer.

## Actual Owner Development Evidence

External root:
`/hf3fs-jd/data/tmp/openopt-release-20261010/native-psd/`.
Pre-edit smoke `pre-smoke-v1/smoke-smg_8ty9/verification.json` reports
80 pass/four explicit local CUDA skips, 6.63 pytest seconds, clean base source
and source consistency PASS. It is not CUDA or new-primitive qualification.

Original `development-v1/receipt.json` remains FAIL: 70 passed, one failed,
two CUDA skips. The first svec test computed a binary64 scalar product before
casting and falsely demanded equality with target-binary32 scalar arithmetic;
three of fifteen coordinates differed by one float32 ULP. The test now computes
the independently represented target-dtype coefficient as a Fraction before
the product. PSD/Moreau acceptance limits were not relaxed. Subsequent actual
`development-v2/receipt.json` is 74 passed/two CUDA skips, 3.10 pytest seconds.
`development-v3/receipt.json` adds full 1..6 principal-minor denominator,
collection/JUnit exact membership and explicit CUDA opt-in semantics.
`development-scan-v3/receipt.json` checks all three named files with zero
literal/shape hits plus one observed synthetic sensitivity hit and zero after
clearing. These development receipts bind dirty source bytes, not a later
clean commit; the fresh clean receipt must name its own full SHA.

Reproduce from the frozen worktree with the existing validation Python:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=src:. /hf3fs-jd/data/tmp/openopt-validation-env/bin/python \
  tests/test_psd_cone.py --artifact-dir <new-external-task-directory>
```

`--scan-only` selects the scan instead. Both modes require a new external
directory and write `receipt.json`/`review.log`; CPU review records actual
runtime, exact collection/JUnit and explicit skip identities. CUDA targets
remain mechanisms only until independently reviewed and genuinely run by the
main GPU owner; this CLI always hides CUDA and cannot grant GPU qualification.
