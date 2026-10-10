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
zero eigenvalues are treated as null directions only when the fixed-arithmetic
original P times that eigenvector is exactly zero, as well as passing the
independent spectral residual checks. This conservative extra check prevents
computed zero from silently hiding original small positive curvature.
Singular/ill-conditioned evidence remains
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

The frozen CUDA seven-test producer explicitly selects v1, freezes
algorithm metadata and producer Python/NumPy/build/environment provenance,
and retain complete baseline/refined/three-negative-control evidence.
Its new schema is distinct from historical schema 1. Existing CPU/default
producer behavior and old schema-1 replay remain available without promotion.
The new auditor selects arithmetic only from a strictly checked schema and
config; it never retries another arithmetic mode when exact evidence differs.
Final-source refinement requires schema 2. Legacy schema 1 is restricted to
explicit development/historical replay and cannot be promoted to a fresh
final-source refinement qualification.

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

The two historical seven-sidecar diagnostic unit tests are strictly opt-in
via `OPENOPT_RUN_EXTENSION_EVIDENCE_TESTS=1`. Missing/unset/other values
explicitly skip those tests and grant no evidence qualification. Opt-in
does not imply matching NumPy/BLAS or checker source, and such differences
remain visible failures. Their exact first-cause assertions describe the
old recorded local environment, not a portable default regression oracle.
Synthetic/default unit tests do not require the historical campaign path.

## Implemented CPU Receipt And Remaining Gates

Contract-first commit: `1d7cfe4b94aa66dbe6cb02e7e1e59dee05da94c2`.
The implementation is confined to the seven explicitly authorized files.
It requires Linux CPython plus accessible `libm.fegetround`; `ctypes` is used
only to read the current rounding mode, not for candidate calculations or a
solver. Runtime probes require gradual underflow even though subnormal model
values and calculation intermediates are unsupported. The default NumPy
checker and refinement's default numerical path are unchanged.

Exact auditor evidence comparison checks types and binary64 `float.hex()`,
including signed zero; int/float and bool/number substitution are rejected.
The standalone sidecar CLI compares canonical JSON scalars and rejects the
same substitutions. Rejected candidate x/y must retain baseline bytes.
No relative/ULP allowance or alternate-arithmetic retry is present.

All receipt paths in this section are relative to the external artifact root
above. These are CPU development validations of explicitly hashed dirty
implementation bytes, not clean-final CUDA qualification:

| Receipt | Result | Pytest wall |
| --- | --- | --- |
| `numerical-final-v3/tests-jy21tys6/verification.json` | 168 PASS, 7 CUDA skips | 27.83 s |
| `bitwise-final-a/tests-ka73p8v0/verification.json` | 20 PASS | 94.03 s |
| `bitwise-final-b/tests-g1i7jb8e/verification.json` | 25 PASS | 115.22 s |
| `bitwise-final-c-v2/tests-wg731orb/verification.json` | 20 PASS, 2 opt-in skips | 31.91 s |
| `historical-optin-v1/tests-cda6ctba/verification.json` | FAIL, 2 failed | 3.45 s |

Successful receipts include source-consistency PASS with no changed, missing
or extra files during each run. The A/B receipts precede the additional
signed-zero rejection-byte gate; C-v2 explicitly tests that addition.
Counts cover different snapshots, overlap other runs, and are not summed.
The historical opt-in run intentionally remains red: the implementation
checker no longer matches the old sidecars' frozen checker SHA. Its first
test records all seven strict source failures before its historical 4/7
expectation fails; the second fails directly at loaded-checker binding.
This is neither a repaired historical replay nor a numerical qualification.
Unset, empty, `0`, and `true` opt-in values skip before any historical
filesystem lookup; only exact `1` enters that evidence-dependent path.

The latest two-process receipt is
`numerical-final-v3/tests-jy21tys6/pytest-tmp/test_reproducible_seven_cpu_pr0/seven-sidecar-cross-process.json`,
SHA256 `7d78106b8bc812b488af52719869a8efba082a9852c27833003d1980ea23a9b9`.
The bare-checker receipt in sibling
`test_reproducible_cross_proces0/cross-process-receipt.json`
has SHA256 `34817dd0e9485d1fa4e981be6390d2fe9f934680a77d6f87fef965db287348d4`.
Seven complete CPU-produced sidecars replay exactly in both independent
processes: four accepted candidates and three legitimate rejected/unknown
controls. Both processes' fifteen-evidence JSON SHA is
`9dbb2a4835c59e248ac0b4a5f083f4ba4b95a7805328d0da5b2f694ccde79785`.

Both arms use Python 3.10.20 from the existing validation interpreter,
NumPy 2.2.6 from
`/hf3fs-jd/data/tmp/openopt-validation-env/lib/python3.10/site-packages/numpy/__init__.py`,
NumPy-reported OpenBLAS 0.3.29 build and Torch 2.8.0+cu128.
Arm A uses OPENBLAS/OMP/MKL threads=1, `OPENBLAS_CORETYPE=Haswell`,
`NPY_DISABLE_CPU_FEATURES=AVX2,FMA3`; arm B uses threads=2,
`OPENBLAS_CORETYPE=Sandybridge`, disabled-feature list empty.
Both hide CUDA. NumPy reports AVX2/FMA3 disabled in A and available in B.
Different BLAS environment settings are observed, not an observed BLAS
kernel-dispatch trace, different installed NumPy versions or GPU runtimes.
The latest one-shot n=8,m=64 CPU checker cost is 0.02502534497762099 s,
recorded in `test_reproducible_n8_cost_and_0/n8-cost.json`; this is neither a
solver speedup nor representative workload performance.

Remaining gates, with owners and no retrospective promotion:

- Main mathematical reviewer: independently audit fixed-operation order,
  square-root rounding and conservative bounded spectral acceptance.
- GPU-proof guard owner: coordinate new `sys`, `ctypes`, `ctypes.util`
  imports in certificates. This seat does not edit `gpu_release_proof.py`.
- Main integrator: freeze clean integrated source, run the unchanged seven
  CUDA identities, then independently seal/accept schema-2 evidence at exact
  caller run/result/JUnit/source pins. Local CUDA tests are skipped, not green.
- Main runtime verifier: replay across the actual producer/local runtime
  distributions. Different NumPy versions, CPU architectures and CUDA
  environments are not established by this same-venv CPU experiment.
- Main release owner: coordinate the complete repository regression,
  generated report/design synchronization outside this seven-file scope,
  packaging, publication and commercial evidence. This tiny checker closes
  none of general-size, arbitrary-precision or commercial qualification.

Stable local reproduction, from this worktree or a clean integrated copy:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_qp_certificates.py --target tests/test_qp_refinement.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/reproducible-checker/reproduce-numerical
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENOPT_RUN_EXTENSION_EVIDENCE_TESTS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_accept_extension_cuda.py::test_real_seven_sidecar_exact_first_cause_diagnosis --target tests/test_accept_extension_cuda.py::test_real_seven_complete_numerical_gates_and_diagnostic_never_accepts --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/reproducible-checker/reproduce-historical-optin
```

The second command is an explicit historical diagnostic, expected FAIL on
this new checker source. Neither command submits GPU work. Fresh CUDA
selection remains exactly the three existing function targets
`test_cuda_refinement_frozen_model_accuracy_ab`,
`test_cuda_refinement_actual_residual_correction`,
`test_cuda_refinement_negative_controls`: parametrization totals 3+1+3=7.

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
