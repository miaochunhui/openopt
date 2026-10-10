# Opt-In Operator CG Preconditioning

This scoped extension changes only `src/openopt/operators.py`. The dense QP
core, public package exports and default operator path are unchanged.
Main integration owns `docs/qp-design.md` and generated design/report pages;
their historical unpreconditioned description must be synchronized there.
No GPU run, wall-clock speedup or commercial qualification is claimed.

## Contract

`conjugate_gradient(..., preconditioner=None)` keeps the original arithmetic
path. An explicit preconditioner implements `shape`, CPU/CUDA `torch.device`,
real float32/float64 `dtype` and `apply(r) -> M^-1 r`. Its action must be fixed,
deterministic, linear, symmetric positive definite, and must not mutate inputs.
These are caller declarations: positive sampled `r' M^-1 r` is not an SPD proof.
Callback state is borrowed; no automatic device transfer or silent fallback.
Shape/dtype/device, finite outputs and positive finite products are checked.
Action failures retain their first cause, true residual and inner/ADMM coordinate.

`JacobiPreconditioner(diag_K)` owns a detached diagonal copy. Entries must be
finite and strictly positive; zero/negative/nonfinite entries are rejected,
not clipped, replaced or shifted. Application divides by the diagonal rather
than caching reciprocals that could overflow. Output overflow is also loud.

For tensor P and A, use `OperatorQPConfig(cg_preconditioner="jacobi")`.
Preparation computes exactly
`diag_K = diag(P) + rho * column_squared_norms(A) + sigma`.
Dense, COO and CSR are supported. Sparse entries are coalesced before squaring,
so duplicates sum before contributing to `diag(A' A)`. Missing diagonal entries
and empty columns contribute zero; sigma remains the existing system parameter.
No sparse densification, dense Gram construction, basis-vector probing or
factorization is used. Sparse extraction uses O(nnz + n) storage. Negative
P diagonals contradict `assume_psd` and fail even if rho/sigma would mask them.
Nonfinite diagonal arithmetic, including overflow of squared A entries, fails.

For any matrix-free P or A, automatic Jacobi is rejected. The caller must
explicitly provide the complete system diagonal or another SPD inverse action:

```python
from openopt.operators import JacobiPreconditioner, OperatorQPConfig, solve_operator_qp

# diag_K belongs to the unchanged original-coordinate system, on config device/dtype.
result = solve_operator_qp(
    problem, OperatorQPConfig(),
    preconditioner=JacobiPreconditioner(diag_K),
)
```

An explicit action and automatic Jacobi cannot be selected simultaneously.
`prepare_operator_qp` and `PreparedOperatorQP` accept the same keyword.
Tensor Jacobi data is owned; arbitrary explicit callback state remains borrowed
and must not change across repeated prepared solves.

## Acceptance Boundary

PCG uses `r' M^-1 r` only in its recurrence. Convergence always requires the
original unpreconditioned true L2 residual
`||rhs-Kx|| <= cg_eps_abs + cg_eps_rel*||rhs||`.
The norm, tolerances, iteration budget and original KKT checks are unchanged.
A failed true-residual nomination restarts from the recomputed residual and
reapplies M^-1 inside the same budget. Exhaustion and nonpositive curvature
remain failures; an outer solve cannot promote a failed inner solve to solved.

Literal `assume_psd=True` is still mandatory. Existing operator probes are
unchanged, and passing them or supplying Jacobi never certifies convexity.
The experimental status and `convexity_certified=False` remain unchanged.
No scaling, adaptive rho, routing, certificate generation or GPU kernel is added.

## CPU Evidence

`tests/test_qp_precondition.py` compares default CG and full operator-QP results
bitwise against the frozen pre-change module at commit `a698c67`, in float32
and float64 across dense/COO/CSR/matrix-free representations.

A 48-variable float64 known-KKT problem uses a quadratic diagonal from 1 to
1e6 and four active/inactive/equality rows. Both unpreconditioned controls and
Jacobi arms must solve with the same tolerances. Independent NumPy calculations
check stationarity, feasibility, normal cone, primal/dual witness error and
objective error. Reports include actual CG counts, not latency or GPU claims.
COO/CSR/matrix-free witnesses prohibit densification at runtime; additional
4096-variable sparse extraction tests use no dense matrix source.

Negative controls cover at least negative/zero/nonfinite diagonal, squared-norm
overflow, malformed metadata/output, failed callbacks, nonpositive r'z, tiny
preconditioned norm with a failing true residual, stateful-system false recurrence
convergence and restart budget, missing matrix-free input, contradictory PSD
declarations, conflicting options and failed inner-to-outer propagation.

Artifacts are external under
`/hf3fs-jd/data/tmp/openopt-release-20261010/precondition/`.
Verification uses the existing validation interpreter with
`CUDA_VISIBLE_DEVICES=-1`; GPU correctness and benefit remain UNVERIFIED and
belong to the main agent's later acceptance. CPU iteration reduction on a
synthetic diagonal system is not a latency/performance-release qualification.

Explicit CUDA COO/CSR tests cover float32/float64 tensor placement, Jacobi CG
true residuals, independent CPU original-coordinate KKT checks, and a false
recurrence convergence negative control. CPU-only verification skips these
tests explicitly; the main agent owns unified 4500 K8s execution.

### Observed Accuracy Limit And Revalidation

The original focused-v1 run `tests-1v2l1rm_` is preserved as FAIL:
138 passed / four known-KKT failures. A both-arm replay with the original
`cg_eps_abs=1e-11`, `cg_eps_rel=1e-13`, outer `eps_abs=1e-9`,
`eps_rel=1e-11` is preserved in focused-v3 `tests-s43wstj7`.
The control solved in 75 outer / 1247 CG iterations. Jacobi reached 5000 outer
iterations and remained `max_iterations`, with independent NumPy normal-cone
residual `2.149721928246251e-8` against `1.004e-9` (excess
`2.049321928246251e-8`). Its maximum x error was `2.149721928246251e-8` at
coordinate 1; y error was `1.8450759031729547e-8` at coordinate 1; objective
error was `8.614733815193176e-9`. The planted witness stationarity is
`5.551115123125783e-17`. All four layouts reproduce these values.

This is an observed inner-accuracy/outer-accuracy mismatch, not evidence of
a floating-point floor or universal Jacobi failure. The original global inner
threshold was approximately `5.698e-8`, larger than the per-coordinate primal
threshold. No automatic precision adjustment was implemented or hidden.
Revalidation explicitly tightened inner absolute tolerance to `1e-12` and
relative tolerance to `1e-15`, leaving outer tolerances unchanged. Both arms
then solved in 50 outer iterations: control 1089 CG, Jacobi 33 CG; independent
normal-cone residuals `3.940014980940987e-12` and `1.4992795893675748e-10`.

Final focused-v4 `tests-ddbh9s0p` has JUnit `pytest.xml`: 148 passed, zero
failures/errors, six explicit CUDA skips. It includes the original negative
as a required `max_iterations` result, tightened same-tolerance comparisons,
default frozen bitwise parity and the overflow/curvature boundaries.
`pytest-tmp/test_original_inner_accuracy_n*/known-kkt-*.json` retains both
negative/control arms; `pytest-tmp/test_illconditioned_known_kkt_*/known-kkt-*.json`
retains corrected comparisons. Reports are written before acceptance assertions
and include vectors, witness arrays, thresholds, residual excess and coordinates.
The focused-v3 FAIL also retains a mistaken diagnostic assertion that initially
attributed the failing arm to control; the corrected replay establishes Jacobi.

Post-edit smoke `smoke-6havpyts` passed 79 tests with four explicit CUDA skips.
All these artifacts remain under the authorized precondition artifact root.
