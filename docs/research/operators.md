# Experimental Sparse / Matrix-Free QP

This module is an independent experiment at `openopt.operators`. It does not
change `qp.py`, the dense Cholesky path or the package exports. Main integration
owns `__init__.py`, `docs/qp-design.md` and generated design/benchmark pages.
No GPU/performance/commercial qualification is implied by CPU correctness.

## Contract And Construction

Minimize `0.5 x' P x + q' x` subject to `l <= A x <= u`.
The operator contract is `shape`, `device`, `dtype`, `matvec(v)` and
`rmatvec(v)` (the real adjoint). Only float32/float64 and CPU/CUDA are supported.

```python
import torch
from openopt.operators import (
    TensorOperator, MatrixFreeOperator, OperatorQPProblem, OperatorQPConfig,
    prepare_operator_qp, solve_operator_qp,
)

# P and A can be dense tensors, sparse tensors or operators.
P_sparse = P.to_sparse_coo()  # alternatively P.to_sparse_csr()
A_sparse = A.to_sparse_coo()  # alternatively A.to_sparse_csr()
problem = OperatorQPProblem(P_sparse, q, A_sparse, l, u, assume_psd=True)
config = OperatorQPConfig(device="cpu", dtype=torch.float64)
result = solve_operator_qp(problem, config)
workspace = prepare_operator_qp(problem, config)
repeat = workspace.solve()  # zero start, not changing-data warm start

# TensorOperator owns a detached copy of its source matrix.
tensor_operator = TensorOperator(A_sparse)
# Matrix-free callbacks are borrowed, not copied or automatically transferred.
A_device = A.to(device=config.device, dtype=config.dtype)
matrix_free = MatrixFreeOperator(
    tuple(A_device.shape), A_device.device, A_device.dtype,
    lambda v: A_device @ v, lambda v: A_device.T @ v,
)
```

Tensor inputs are copied to the requested device/dtype during preparation.
COO is coalesced; CSR stays CSR for forward multiplication, with a cached COO
transpose for adjoint multiplication. Neither path calls `to_dense`, builds
`A' A`, or factors a dense system. Storage remains proportional to input
nonzeros, transpose nonzeros and working vectors. Unsupported sparse layouts,
hybrid sparse tensors and complex inputs fail explicitly.
Sparse structural invariants are checked before coalescing or invoking kernels,
including out-of-range COO/CSR indices; PyTorch's default unchecked sparse
constructor is not accepted as validation evidence.

Matrix-free metadata and every returned vector must match the config exactly.
Callbacks must be deterministic, genuinely linear, mutually adjoint, finite,
and must not mutate inputs or change their operator throughout preparation and
solves. Borrowed callback state has no ownership or arbitrary-state guarantee.
Finite zero/linearity/adjoint/symmetry/curvature probes catch some broken
operators; they cannot establish those properties for every vector.

## Convexity And Failure Boundary

Every problem requires literal `assume_psd=True`: the caller explicitly
declares P symmetric positive semidefinite. No full spectral PSD validation is
performed, even for dense inputs on this path. Matrix symmetry is checked for
tensor inputs without densification; observed negative curvature contradicts
the assumption and is rejected. Passing these diagnostics does **not** certify
convexity. Results always contain `experimental=True` and
`convexity_certified=False`, including results with status `solved`.

The fixed-rho ADMM x step solves
`(P + rho A' A + sigma I) x = sigma x_previous - q + A'(rho z - y)`
using unpreconditioned CG and the preceding x as the inner initial iterate.
There is no dense-core modification, adaptive rho, preconditioner, scaling,
presolve, conic/MIP support, infeasibility/unboundedness certificate or general
matrix-free correctness/performance claim.

Default config:

```python
OperatorQPConfig(
    device="cpu", dtype=torch.float64,
    rho=1.0, sigma=1e-6, relaxation=1.6,
    max_iterations=5000, check_interval=25,
    eps_abs=1e-7, eps_rel=1e-7,
    cg_max_iterations=1000, cg_eps_abs=1e-10, cg_eps_rel=1e-10,
)
```

For float32, explicitly choose attainable inner tolerances, for example
`cg_eps_abs=1e-7, cg_eps_rel=1e-7`, and compatible outer tolerances. Difficult or
ill-conditioned systems may fail; silently relaxing tolerances is prohibited.

CG accepts only a recomputed true L2 residual
`||rhs - system.matvec(x)|| <= cg_eps_abs + cg_eps_rel * ||rhs||`.
A small recurrence residual is not acceptance: a failed true-residual test
restarts the direction inside the same budget. Nonpositive curvature or an
exhausted inner budget raises `CGConvergenceError`, with reason, inner iteration,
true residual, tolerance, and ADMM iteration when called from QP. Callback,
placement, malformed-vector and nonfinite failures are also loud and retain
their cause; none produces a pretend solved result.

Outer `solved` means only approximate feasibility, stationarity and signed
normal-cone residuals pass configured absolute/relative checks under the caller
assumptions. Residuals are checked at the actual x, including at the final
iteration even if it is not a check interval. Exhaustion without those checks
passing is `max_iterations`, not solved/infeasible/unbounded certification.
Results include `x`, `y`, `iterations`, `objective`, `primal_residual`,
`dual_residual`, `normal_cone_residual`, total `cg_iterations`, and final
`inner_residual` / `inner_tolerance`. Every accepted inner solve is checked;
the final residual fields describe only the last inner system, not the
outer KKT residual or the total CG history.

An explicit unavailable/invalid CUDA request fails before input transfer.
Tensor conversion and sparse kernel errors propagate rather than falling back.
Matrix-free callbacks must already execute on the requested CUDA device.
This seat does not use local GPUs or submit cluster Jobs. The main campaign
owns actual CUDA and transfer-inclusive cold/prepared performance verification,
with at most its authorized three cards; CPU tests are not CUDA evidence.

## CPU Acceptance And Integration

The focused gate is the existing validation environment running
`scripts/verify.py --target tests/test_qp_operators.py`, with artifacts under
`/hf3fs-jd/data/tmp/openopt-qp3-operators-20261010/`. CUDA is hidden for local
verification. Failed runs must remain alongside changed-version reruns.

Tests compare dense, COO, CSR and matrix-free paths against the independent
dense QP implementation and a planted primal/dual KKT witness, computing
objective/feasibility/stationarity/normal-cone errors directly. They cover
float64/float32, equality/one-sided/empty constraints, zero-P LPs, ownership,
uncoalesced COO, repeatable zero starts, broken adjoints/nonlinearity, observable
false PSD declarations, inner exhaustion/nonpositive curvature, true-residual
negative controls, outer iteration limits and strict explicit CUDA failures.
Sparse tests forbid dense conversion/spectral validation/factorization and
inspect actual sparse-mm input layouts and vector-only right-hand sides.
An additional 257-variable diagonal witness is constructed directly from COO
nonzeros, CSR nonzeros or vector-only callbacks, without a dense matrix source.

Only `src/openopt/operators.py`, `tests/test_qp_operators.py` and this document
belong to the operators seat. Public export changes and project-wide report
synchronization remain the main integrator's write set.

The first focused run `tests-_54z7jxc` is retained as FAIL (pytest rc=1,
60 failed / 14 passed). Its 60 failures share a fixture self-check that wrongly
required a floating-point planted stationarity residual to equal literal zero
(float64 residual `5.55e-17`, float32 `1.49e-8`). The changed fixture checks a
four-machine-epsilon bound; no solver tolerance was relaxed for this repair.
The changed run `tests-5lk1gzea` passed all 74 then-current tests. Subsequent
acceptance adds direct-nonzero/vector-only witnesses, malformed sparse-index
negative controls and a true-residual CG restart-budget test.

Final CPU acceptance on 2026-10-10 at 07:49:29 UTC: `tests-vivbc9hn` contains
`verification.json` PASS, pytest XML with 80 tests / zero failures / zero
errors / zero skips, and a passing CPU example. Post-edit `smoke-euh2fdka`
contains PASS, 79 passed / four explicitly CUDA-unavailable skips. Both ran
with CUDA hidden, using the existing validation environment (PyTorch
2.8.0+cu128); these are correctness-only results, not GPU or speed evidence.
The original failed report/log/XML remain unchanged in `tests-_54z7jxc`.
