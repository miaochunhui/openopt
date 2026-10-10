# openopt Working Entry

- This repository is the independent maintenance home for the imported OpenPO
  foundation. Do not modify infra or optimizer as a side effect of work here.
- `openopt` is the project/distribution name. Preserve `openpo` compatibility.
  Legacy models are portfolio-specific; `src/openopt/qp.py` is a separate
  experimental real convex QP/LP core, not a complete general optimizer.
  Scaling, preconditioned operators and numerical termination are opt-in.
  Conic and MILP reference backends are explicitly CPU-only; they do not
  establish native GPU capability, MIQP support or commercial qualification.
  The separate opt-in native dense zero/nonnegative/SOC prototype and QP
  polishing/refinement require original-model numerical checks. API presence
  or CPU tests do not establish CUDA execution, complete cone/MIP scope,
  arbitrary-precision guarantees or commercial qualification.
- Product direction: a general-purpose GPU-accelerated optimizer targeting
  commercial-grade correctness, robustness and performance. Gurobi and CPLEX
  are LP/QP/MIP baselines; MOSEK is a convex/conic baseline. This is a target,
  not an achieved ranking. The acceptance contract is in `docs/strategy.html`.
- Run `python3 scripts/iteration_gate.py smoke` before algorithm/performance
  work, and after edits. `python3 scripts/verify.py` is the complete local
  regression entry. Neither is a performance-release qualification.
- Changes to algorithms, solver parameters or benchmark methodology must update
  `docs/qp-design.md` and the generated `docs/design.html` /
  `docs/benchmark-report.html`. Use the applicable versioned report generator
  with external machine-readable evidence; do not manually edit generated HTML.
- Compare continuous LP/QP against OSQP and Clarabel. Compare future discrete
  optimization against SCIP and HiGHS, as well as licensed commercial
  baselines where available. Missing commercial evidence is UNVERIFIED,
  not a win or an open-source-only substitute for commercial qualification.
  Never promote heuristic solutions to exact/certified results without evidence.
- GPU is a first-class execution path. Keep explicit device requests strict;
  verify actual tensor placement, accuracy and end-to-end transfer-inclusive
  speedups separately. CPU support is for references and measured routing,
  not silent fallback from an explicit GPU request.
- Keep code/configuration/documentation here; put logs, test artifacts and
  benchmark results outside the worktree. Verification accepts `--artifact-dir`;
  the default is `/hf3fs-jd/data/tmp/openopt-verification`.
- Do not copy credentials, internal deployment configuration, proprietary
  datasets, commercial reference documents or source-repository history.
- The user selected maximally permissive licensing on 2026-10-10; this project
  uses BSD Zero Clause (`0BSD`), including its imported OpenPO implementation.
  Keep `LICENSE` and package metadata consistent. Dependencies and commercial
  SDKs retain their own licenses; do not relicense or vendor them implicitly.
