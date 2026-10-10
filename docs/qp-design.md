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

An opt-in `adaptive_rho=True` balances normalized primal/stationarity residuals,
clips rho to configured bounds and refactors explicitly. The unscaled multiplier
is preserved. Failed/nonfinite refactors fail loudly. Default rho and finite
normal-path outputs remain unchanged; nonfinite residual inputs, thresholds and
objective now fail explicitly rather than allowing an overflow false success.

`solved` requires primal feasibility, stationarity and normal-cone residuals
below absolute/relative tolerances. It is an approximate numerical status, not
an exact certificate. `max_iterations` does not distinguish infeasible from
unbounded; no such certificate is advertised. No presolve, scaling, sparse
backend, conic scope, MIP solver, universal `minimize`, or automatic GPU routing
is implemented by the dense default path.

## Optional Research Paths

`openopt.batch` provides same-shape stacked dense QPs with batched factor/solve,
per-item status, metrics and real checked iteration counts. Finished items freeze
while others continue. Cold includes stacking/validation/transfers; prepared
excludes setup. This path explicitly rejects adaptive rho instead of ignoring it.

`openopt.operators` provides dense/COO/CSR and borrowed matrix-free operators,
with CG true-residual acceptance. It never densifies sparse matrices or forms
the dense Gram system. The caller must declare `assume_psd=True`; probes do not
certify convexity. The path is experimental, unpreconditioned and slower on the
measured small sparse case. It is not a default routing replacement.

`openopt.certificates` independently checks supplied original-coordinate
candidates, numerical dual range/gap, Farkas directions and recession rays.
It does not generate directions or change solver termination status. Singular
spectral uncertainty remains unavailable. Approximate numerical evidence is
not an exact proof or a rigorous certified bound.

`openopt.corpus` supplies eight reproducible synthetic QP/LP stress families,
including singular, redundant, infeasible and unbounded models. These are not
a representative industrial/public corpus. Returned vectors are checked against
preconstructed witnesses, not against the solver's own residuals.

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
passing numerical/Nsight prerequisite before a larger legacy launch.

The v3 campaign adds independent pinned-byte acceptance: source membership,
problem/witness arrays, every saved host solution, dtype/eps, strict solver
status, latency statistics, controls, telemetry and profile tables are rechecked.
Performance admission now requires a sealed independently accepted correctness
bundle. The gate is enforced by the submission function, not only its CLI.
This offline integrity gate is not hostile-party authentication, a
cluster-wide lease or proof of commercial parity. Only the coordinator submits
Jobs; separate competing coordinators still need a shared scheduler.

Batch performance uses eight copies of one frozen problem, compared with eight
sequential prepared/cold scalar solves on the same GPU. It is not heterogeneous
batch qualification, commercial acceleration or a multi-GPU algorithm.
Input generation/imports are excluded from model-cold; model assembly, transfers,
model validation, factorization, iteration and host outputs are included.
The independent CPU quality oracle is run outside the timed interval, equally
for all arms. These are host-output-inclusive solver latencies, not total
production acceptance time including an independent oracle.

## Limits And Next Work

Historical v2 measurements belong to the frozen earlier source. Current v3
reports do not substitute changed source for those results. Several early v3
GPU results lacked replay sidecars/statuses; these are preserved but rejected
by independent acceptance, and corrected producers generated new evidence.

One seeded dense problem per size is not a representative optimization corpus.
Broad ill-conditioning/degeneracy coverage, solver-generated certificates,
diverse sparse structure, parameter robustness and commercial runs remain
unqualified. The measured
v3 batch speedup is against this project's sequential GPU execution. The
earlier v2 speedup was against its four-thread CPU prototype. Neither proves
commercial superiority. Current CPU comparisons against OSQP/Clarabel are
same-host, same-problem results, not representative solver rankings.
Commercial SDKs were unavailable in the selected existing runtime; no
SDK/license installation was attempted.

The existing documentation platform has no openopt publication entry.
Adding a new publication scope would change shared infrastructure and is
outside this repository's no-side-effect boundary; report HTML is local,
not claimed published online.
