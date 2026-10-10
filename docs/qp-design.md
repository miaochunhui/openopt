# Experimental Dense Convex QP

`openopt.QPProblem(P, q, A, l, u)` minimizes `0.5 x' P x + q' x` subject to
`l <= A x <= u`. The core has no portfolio semantics. `P` must be real, dense,
symmetric and positive semidefinite; `q` must be nonempty. Equality and one-sided
infinite bounds are supported. Zero quadratic terms permit experimental LPs.

`QPConfig` fixes CPU/CUDA device, float32/float64, positive penalty,
regularization and absolute/relative tolerances. An explicit CUDA request fails
if unavailable; input tensors are copied into an owned prepared workspace.
Nonfinite matrices, invalid dimensions/bounds, complex and sparse tensors fail.
Preparation includes a full spectral convexity validation, so dense cold setup
has cubic cost. This is not an optimized sparse/matrix-free design.

The prototype uses ADMM with a Cholesky factor of `P + rho A' A + sigma I`,
box projection and a signed dual vector. A solve starts from zero, including
repeated prepared solves. The default linear path is unchanged Cholesky solve.
The optional `linear_solver="triangular"` switches only the factor application
to two triangular solves; reverting this option restores the baseline without
altering data, parameters or files.

`solved` requires primal feasibility, stationarity and normal-cone residuals
below absolute/relative tolerances. It is an approximate numerical status, not
an exact certificate. `max_iterations` does not distinguish infeasible from
unbounded; no such certificate is advertised. No presolve, scaling, sparse
backend, conic scope, MIP solver, universal `minimize`, or automatic GPU routing
is implemented.

## Measurement Contract

`scripts/benchmark_general_qp.py` plants a known optimal primal/dual witness in
synthetic positive-definite QPs with active lower/upper rows and equalities.
Independent NumPy calculations check feasibility, stationarity, signed-bound
duality, objective error and solution error. OSQP and Clarabel independently
solve frozen identical arrays; cross-machine timings are not rankings.

Model-cold means validation, input copies, factorization, iterations and host
results. It excludes process scheduling/startup and uses one untimed warmup.
Prepared timing excludes setup but includes returned host results; it is not
a warm-started changing-data workload. Latency repetitions are sequential
measurements. Independent problems are separate one-card K8s Jobs; launching
three such Jobs proves parallel execution, not multi-GPU strong scaling.

CPU/GPU pairs alternate order. The triangular A/B uses alternating order plus
an identical baseline control, all on the same GPU. The default remains the
original path; observed small improvements are scoped work values, not a
general performance release. Nsight traces a separate prepared solve with host
results; profile instrumentation overhead must not be mixed with unprofiled
latency. Pod cgroup counters and GPU samples cover the combined CPU/GPU
experiment window, including CPU reference intervals.

Raw results, telemetry, frozen problems, source hashes and traces stay on 3FS.
`scripts/render_qp_report.py` reads them to generate the design/report HTML.
The shared runner refuses a launch exceeding three live cards and requires a
passing numerical/Nsight prerequisite before a larger launch.

## Limits And Next Work

One seeded dense problem per size is not a representative optimization corpus.
Ill-conditioning, degeneracy, certificates, diverse sparse structure, solver
parameter robustness and commercial runs remain unqualified. The measured
speedup is against this project's four-thread CPU prototype, not optimized
commercial or open-source implementations. Commercial SDKs were unavailable
in the selected existing runtime; no SDK/license installation was attempted.

The existing documentation platform has no openopt publication entry.
Adding a new publication scope would change shared infrastructure and is
outside this repository's no-side-effect boundary; report HTML is local,
not claimed published online.
