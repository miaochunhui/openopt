# openopt Development Ledger

## Established Foundation

- Extract the complete OpenPO implementation without modifying source history.
- Preserve the `openpo` public API and provide an `openopt` compatibility entry.
- Provide independent correctness verification and explicit migration evidence.

These items are accepted only by current `scripts/verify.py` output, not by the
presence of files or old benchmark claims.

## Next Development Stages

The user selected 0BSD on 2026-10-10. Licensing is implemented in `LICENSE`,
package metadata and wheel acceptance; it does not relicense dependencies.
The product target and release gates are in `strategy.html`.

1. General model: an experimental real dense convex QP/LP contract is now
   implemented in `src/openopt/qp.py`, with explicit dtype/device, approximate
   result status and independent KKT validation. Extend the quadratic
   operator interface, constraints, dtype/device semantics and result status.
   Keep portfolio adapters outside the general core; design GPU-resident
   dense, sparse and matrix-free operators, batching and prepared workspaces.
2. Correctness and robustness: compare matched LP/QP with licensed Gurobi,
   CPLEX and MOSEK plus OSQP/Clarabel/HiGHS references. Cover infeasible,
   unbounded, degenerate, ill-conditioned, empty and non-finite inputs.
   Missing commercial runs remain UNVERIFIED, not passed.
3. GPU qualification: measure transfer-inclusive cold/hot latency, p50/p95,
   success rate, objective gap, primal/dual residuals, throughput, memory and
   hardware/software versions on identical paired cases. Keep explicit GPU
   requests strict and choose automatic CPU/GPU routing from measurements.
4. Convex conic scope: define SOCP and further cone contracts before claiming
   support; qualify against MOSEK and independent open references.
5. MILP/MIQP scope: specify bounds, integrality, optimality gaps, presolve,
   branch-and-bound/cuts and termination certificates. Compare licensed
   Gurobi/CPLEX and SCIP/HiGHS. A fast heuristic is not an exact MIP solver.
6. Distribution: clean-wheel acceptance and CPU/CUDA CI, followed by separate
   reproducible commercial/GPU qualification. No license server or SDK is
   bundled; commercial benchmark execution requires authorized installations.

## Explicit Non-Claims

- No universal `minimize`, sparse/conic/MIP backend or infeasibility/unboundedness
  certificate is implemented; the dense QP API is experimental.
- No universal GPU speedup or production qualification is asserted.
- No commercial solver parity or world-leading ranking is asserted.
- The user authorized this experiment's K8s Jobs with at most three 4500 cards.
  No service, credential, RBAC, networking or shared publication changes were made.
- Historical research decisions are not current runtime validation.

## Current Experiment

The 4500 experiment and failures are recorded externally under
`/hf3fs-jd/data/tmp/openopt-4500/`. `scripts/render_qp_report.py` generates current
design and benchmark pages from accepted result/verification artifacts.
Model-cold and prepared timings are separate. Comparisons to the project's CPU
prototype do not establish performance against OSQP/Clarabel/commercial solvers.
Three independent one-card Jobs do not establish a multi-GPU solver.

Remaining qualification: licensed commercial matched runs, representative
diverse/ill-conditioned/sparse corpora, certificates, actual multi-GPU scaling,
hardware lower-bound analysis and publication through an authorized project
entry. These are not promoted to DONE.
