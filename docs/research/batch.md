# Experimental Same-Shape Dense QP Batch

This is an opt-in research module, not the default dense solver and not a
commercial/performance qualification. Import directly from `openopt.batch`;
the parent integration may export these names separately. `qp.py` and the
package entry are unchanged by this work.

## API

```python
from openopt.batch import (
    BatchQPProblem, BatchQPResult, PreparedBatchQP,
    prepare_batch_qp, solve_batch_qp,
)
from openopt.qp import QPConfig

batch = BatchQPProblem.stack(problems)  # nonempty same-shape QPProblem sequence
config = QPConfig(device="cuda:0", linear_solver="cholesky")
prepared = prepare_batch_qp(batch, config)
result = prepared.solve()
cold_result = solve_batch_qp(batch, config)
```

Stacked input fields are `P[B,N,N]`, `q[B,N]`, `A[B,M,N]`, and `l/u[B,M]`.
`B,N` must be positive; `M=0` is supported. There is no implicit broadcasting.
`stack` requires identical source shapes, devices and dtypes; its Python
assembly is preparation, not a per-problem solve. Direct stacked inputs are
copied/detached into the explicitly requested CPU/CUDA device and float dtype.
Every workspace owns its input copies and factors. Repeated solves start from
zero and return independent tensors; no shared mutable iterate buffers.

`BatchQPResult` has `x[B,N]`, `y[B,M]`, and `status: tuple[str, ...]` on the host.
`iterations[B]` is int64 on the requested device. `objective[B]`,
`primal_residual[B]`, `dual_residual[B]` and `normal_cone_residual[B]` are floating
tensors on that device. There is no batch-wide `solved` status.
Statuses are `solved`, `max_iterations` and `numerical_error`.

## Algorithm And Termination

ADMM uses the same fixed parameters, zero start, relaxation and box projection
as the existing dense implementation. Preparation checks every item's
dimensions, finite matrices, bound ordering, symmetry and PSD; symmetry/PSD
tolerances are item-local, not set by the largest matrix in the batch.
One batched `eigvalsh` and one batched `cholesky_ex` operate on `[B,N,N]`.
Cold measurement must include source stacking if performed, transfers, owned
copies, spectral validation, factorization, iteration and host result delivery.
PSD validation is a deliberate cubic cold cost, not an unchecked fast path.
Nonfinite systems and factorization failures raise with bad-item coordinates;
explicit unavailable/out-of-range CUDA requests do not fall back to CPU.
The batch backend is fixed-rho only: a config exposing `adaptive_rho=True`
raises explicitly. A missing field or `False` remains compatible with the
shared `QPConfig`; adaptive settings are never silently ignored.

The solve has a Python iteration loop, never a Python problem-solve loop.
`bmm`, batched `cholesky_solve` (or two batched triangular solves), projections,
reductions and active-item masks run on the selected device. At each
`check_interval`, and at the final iteration, tensor operations compute
per-item primal, stationarity and normal-cone residuals/tolerances. One
aggregate host boolean checks whether any item remains active. Status names
are materialized with one final device-to-host vector transfer; numeric
results remain resident.

An item freezes at its first checked solved/numerical-error iteration. Its
stored iteration count and metrics belong to that frozen iterate, not the
batch's last iteration. Other items continue to the real limit. Nonfinite
iterate/metric/objective/tolerance values cannot pass through `inf <= inf`
as solved. Numerical-error output retains nonfinite evidence rather than
silently substituting defaults. Factor application still includes inactive
lanes with zero right-hand sides: no lane compaction or heterogeneous shape
scheduling is claimed.

`solved` is approximate KKT termination, not a certificate. `max_iterations`
does not distinguish infeasibility, unboundedness or slow convergence.
No sparse/conic/MIP support, adaptive penalties, presolve, warm start, automatic
device routing, universal GPU speedup or production qualification is asserted.

## Acceptance And Ownership

CPU tests compare each item against the unchanged scalar dense solver, both
factor applications and float32/float64, plus independently calculated KKT and
objective values. Controls cover mixed easy/infeasible and solved/unbounded
batches, true per-item iteration counts, equality/infinite bounds, empty or
bad shapes, NaN/Inf, nonconvex matrices, local PSD tolerance, stack mismatch,
explicit unavailable/out-of-range CUDA, system overflow and numerical-error
isolation. Call-shape tests require single batched spectral/factor operations
and batched factor applications; repeat solves may not redo preparation.
The independent KKT ruler rejects three corrupted solved-result variants
(primal, dual and objective). Optional adaptive-rho rejection has a matched
false/true control; injected factorization failure preserves bad-item indices.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python scripts/verify.py \
  --target tests/test_qp_batch.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-20261010-batch/targeted
```

Recorded CPU correctness evidence on 2026-10-10 (counts describe these runs,
not a frozen future acceptance threshold):

- Targeted with adaptive controls: 71 passed, 2 CUDA skips, zero failures/errors;
  `/hf3fs-jd/data/tmp/openopt-qp3-20261010-batch/targeted-adaptive/tests-2k3_cmgq/verification.json`.
- Post-edit smoke: 79 passed, 4 CUDA skips, zero failures/errors;
  `/hf3fs-jd/data/tmp/openopt-qp3-20261010-batch/post-smoke/smoke-ejq4k5lz/verification.json`.
- Both runs include JUnit counts, import provenance and a completed CPU example;
  no GPU execution or timing claim follows from these CPU results.

CUDA-residency tests are provided but skipped in this seat's CPU-only run.
GPU execution, transfer-inclusive cold/prepared latency, throughput, traces
and integration/report-page updates are the parent campaign's responsibility,
within its separately authorized shared limit. This seat launches no GPU,
cluster job, nested agent or Flash worker, and installs no dependency.
Only `src/openopt/batch.py`, `tests/test_qp_batch.py` and this document belong
to this seat's commit.
