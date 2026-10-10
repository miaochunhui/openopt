# Reproducible Original QP Checker V1

## Contract Before Implementation

This document freezes the proposed opt-in arithmetic contract before code
implementation. Its first commit is a design receipt, not implemented
correctness, CUDA acceptance or commercial qualification.

Algorithm ID: `reproducible_binary64_v1`. The existing
`check_qp_candidate(..., arithmetic="numpy")` default retains its previous
NumPy arithmetic and evidence. Refinement defaults retain that same checker.
No ordinary QP solver, facade, cluster or dispatch file is changed.

Supported experimental size: dense real original models with
`1 <= n <= 8`, `0 <= m <= 64`, P=(n,n), A=(m,n), x=(n,), y=(m,).
Unsupported sizes raise before spectral work; they do not silently use the
default checker. Sparse, complex, nonfinite model/candidate values and
implicit device transfers are rejected. Bounds alone may use the existing
signed infinities. There is no arbitrary-size or arbitrary-precision claim.

The checker uses the same original P/q/A/l/u and candidate x/y as the legacy
gate. It retains 1e-8 default atol/rtol, individually scaled feasibility and
stationarity components, strict infinite-bound multiplier signs,
complementarity, dual availability, signed absolute gap, and refinement's
nonworsening five-metric rule. The correction producer retains its existing
1e-10 atol/rtol. Arithmetic reproducibility is not an exact proof.

## Fixed Scalar Arithmetic

Arithmetic starts from copied binary64 values and uses CPython IEEE binary64
round-to-nearest/ties-to-even scalar operations. Runtime probes reject
non-binary64/non-nearest environments. No BLAS, LAPACK, NumPy reductions,
FMA, `fsum`, solver, inverse, least-squares or higher-precision numerical
accumulator is used in this path.

Every product and addition is a distinct scalar operation with finite-result
checks. Dot and support reductions start at +0 and visit increasing indices;
matrix-vector products visit rows then columns in increasing order.
Scalar subtraction, division, sign decisions, maxima and thresholds have
fixed evaluation order. Nonzero subnormal inputs/intermediates, multiplication
or division flushed to zero, overflow and division by zero raise fail-loud.
Exact cancellation to zero is allowed. No compensated or adaptive precision
is enabled. Unsupported arithmetic does not become a successful candidate.

Square root for Jacobi rotations is rounded to binary64 using the binary64
input's integer numerator/power-of-two denominator, bounded integer `isqrt`
and an exact midpoint comparison. The output is one binary64 value.
Integer representation work is bounded by the binary64 exponent range;
it is not an arbitrary-precision QP solve or residual calculation.
This avoids relying on libm square-root implementation differences.

## Bounded Spectral Work

P symmetry retains the existing `64 * eps * n * maxabs(P)` threshold.
The symmetrized matrix uses separately evaluated half-products and addition.
An independent implementation of cyclic Jacobi rotations visits
(p,q) lexicographically, uses the stable small-angle tangent branch, and
accumulates eigenvectors in the same scalar order. There are at most
32 sweeps. An off-diagonal magnitude at most `4 * eps * maxabs(Psym)`
is the convergence criterion, not a change to any candidate tolerance.

After convergence the checker independently recomputes
`Psym * V - V * diag(lambda)` and `V.T * V - I` in fixed arithmetic.
Their max residuals must be at most `128 * eps * n * maxabs(Psym)` and
`128 * eps * n`, respectively. Nonconvergence or failed spectral verification
raises an explicit unsupported/unresolved error. Eigenpairs are sorted by
(eigenvalue, original column index); signs are canonicalized by the first
maximum-magnitude eigenvector component. Repeated eigenvalues are permitted,
but no arbitrary rotation from another runtime is imported.

The existing spectral threshold `64 * eps * n * maxabs(lambda)` is retained.
A negative eigenvalue beyond it rejects a nonconvex model. A nonzero
eigenvalue in the unresolved band prevents dual objective/gap availability;
it is NOT snapped to zero, regularized or inverted. Only computed exact
zero eigenvalues are treated as null directions, subject to the independent
spectral residual checks. Singular/ill-conditioned evidence remains
approximate, not certified. The original range residual is still recomputed
using the original P, not the Jacobi working matrix.

All spectral projections, inverse-coordinate reconstruction, original range
residual, primal/dual objectives, support and derived thresholds use the fixed
scalar path. Fixing just the two objective dots is insufficient.

## Versioned Producer And Replay

`RefinementConfig.check_arithmetic` is explicit and defaults to `"numpy"`.
Opt-in baseline and attempted candidates use the same versioned checker;
device execution of factorization/correction stays strict PyTorch CPU/CUDA.
No new solver fallback is introduced. Rejection retains original x/y.

The frozen CUDA seven-test producer will explicitly select v1, freeze
algorithm metadata and producer Python/NumPy/build/environment provenance,
and retain complete baseline/refined/three-negative-control evidence.
Its new schema is distinct from historical schema 1. Existing CPU/default
producer behavior and old schema-1 replay remain available without promotion.
The new auditor selects arithmetic only from a strictly checked schema and
config; it never retries another arithmetic mode when exact evidence differs.

The seven CUDA test identities and complete denominator remain unchanged.
New schema sidecars bind their checker implementation hash through the
existing source set. The already required refinement doc remains required;
this additional design doc is not silently added to the dispatch allowlist.
Main integration must separately arrange any additional documentation
snapshot requirement if it chooses to bind this doc as execution source.

## Acceptance And Cost

CPU-first acceptance includes complete tiny original-model positive and
negative controls, unsupported dimensions/rounding/spectrum rejection,
bitwise unchanged default outputs, strict device tests, and exact JSON
evidence replay in independent processes with distinct BLAS/thread/SIMD
environment settings. Record configuration differences separately from
proven runtime library versions or observed BLAS kernel dispatch.
No new dependency is needed or installed.

Measure tiny-test cost before larger runs. Each local batch must remain
below the user's three-minute coordination threshold; the full long
extension/release suite is reserved for the main coordinator.
External logs/results use
`/hf3fs-jd/data/tmp/openopt-release-20261010/reproducible-checker/`.

The old frozen-3792 seven-sidecar seal remains FAIL. A new algorithm cannot
rewrite or retrospectively accept old evidence. Fresh clean same-source GPU
execution, fresh sidecars, exact cross-runtime replay and strict final seal
are required; only the main agent may submit that new GPU run.

## Mathematical References

Mathematical reference only, independently implemented under 0BSD; no
reference implementation is copied:

- Stanford CME335 Jacobi methods lecture:
  https://web.stanford.edu/class/cme335/lecture7.pdf
- Python 3.10 integer square root contract:
  https://docs.python.org/3.10/library/math.html#math.isqrt

The Jacobi reference motivates orthogonal plane rotations and the stable
small-angle branch. It does not establish this implementation's acceptance,
arbitrary dimensions, runtime portability or exact certification.
