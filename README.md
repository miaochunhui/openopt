# openopt

A general-purpose GPU-accelerated open optimizer in development, targeting the
correctness, robustness and performance of leading commercial solvers.
Extracted from OpenPO into an independent repository.

**License:** [BSD Zero Clause (0BSD)](LICENSE). Commercial use, modification and
redistribution are permitted without copyleft or attribution conditions.
Dependencies and optional commercial solver SDKs retain their own licenses.

**Current status:** the portfolio research foundation now has a separate,
experimental domain-neutral dense convex QP/LP interface. It is not a complete
general-purpose optimizer release. Commercial parity remains unverified.
Measured GPU benefits and negative results are scoped to synthetic dense QPs
and this project's CPU prototype; see [the report](docs/benchmark-report.html).

## Commercial Target

Gurobi and IBM CPLEX are the LP/QP and mixed-integer comparison targets; MOSEK
is the convex/conic target. OSQP, Clarabel, HiGHS and SCIP remain reproducible
open-source references. No single solver is declared universally best.

The staged scope is general LP/convex QP, then conic optimization, then
MILP/MIQP. GPU-resident operators, batched solves and reusable workspaces are
the acceleration direction. Each supported problem class must independently
pass numerical and robustness gates before performance claims.
See [the target and acceptance contract](docs/strategy.html).

## Quick Start

Use an isolated Python 3.10+ environment. Install the appropriate PyTorch build
for your CPU/CUDA platform, then:

```bash
python3 -m pip install -e '.[test]'
python3 scripts/iteration_gate.py smoke
python3 scripts/verify.py
python3 examples/portfolio_qp.py --device cpu
```

For Linux CUDA/Triton support:

```bash
python3 -m pip install -e '.[test,cuda]'
python3 examples/portfolio_qp.py --device cuda:0
```

Explicitly requested unavailable CUDA devices fail rather than silently using
CPU. Test output lists CUDA skips separately. Verification logs and temporary
test data are stored outside the repository; use `--artifact-dir` to select a
different location.

## What Is Included

- Batched low-rank factor QP, bounded-simplex projection, PyTorch CPU/CUDA and
  Triton kernels.
- General linear constraints, primal-dual and ADMM solvers, warm/prepared
  workspaces, and exact CPU references.
- Portfolio-specific adapters, optional MOSEK integration, and multi-period
  portfolio research modules with their synthetic-data regression tests.
- Shadow certification, durable local journals and HTTP service library code.
  Nothing is deployed or started by importing the package.
- Independent verification, a runnable example, and selected benchmark tools.

Existing `from openpo import ...` imports continue to work.
`from openopt import ...` preserves the existing public API without
renaming classes or duplicating solver implementation:

```python
from openopt import FactorRiskModel, PortfolioProblem, SolveConfig, solve
```

The separate experimental interface is:

```python
from openopt import QPProblem, QPConfig, solve_qp, prepare_qp

# P/q/A/l/u are real dense tensors. Minimize .5*x.T@P@x + q.T@x.
result = solve_qp(QPProblem(P, q, A, l, u), QPConfig(device="cuda:0"))
```

Optional research APIs now include same-shape batched QPs, sparse/matrix-free
operators with checked CG, residual-balanced rho and independent numerical
evidence checks. They are opt-in and remain experimental.
See [the v3 measured report](docs/benchmark-report-v3.html) for scoped batch
benefit, negative results, independent acceptance and remaining commercial gaps.
Optional bounded Ruiz scaling restores original coordinates before independent
acceptance. Jacobi-preconditioned CG and generated numerical infeasibility or
recession directions are opt-in; objective-gap termination is independently
checked. Batch solves explicitly reject unsupported termination options.

Conic zero/nonnegative/SOC contracts have an explicitly selected Clarabel CPU
reference backend. MILP has an explicitly selected HiGHS CPU reference backend
with independent incumbent checks. These are not native GPU conic/MIP engines;
MIQP and other cone classes remain unsupported. Explicit CUDA requests never
route to these CPU references. Commercial LP/QP adapters require an authorized
installed solver and frozen-model replay; discovery alone is not qualification.
There is no universal `openopt.minimize`. See
[the exact boundary](docs/qp-design.md) and
[release completion gates](docs/release-contract.md).
GPU acceleration is not automatically preferable: small problems measured
slower on GPU. Explicit requests never silently fall back.

## Maintenance

`scripts/verify.py` runs import-provenance checks, API compatibility tests and
the complete migrated regression suite. `scripts/iteration_gate.py smoke` runs
the smaller correctness preflight. They report failures and actual test counts;
neither certifies speedups, production readiness or untested GPU hardware.

`scripts/migrate_openpo.py --check` checks the exact imported working snapshot.
Normal verification uses `--provenance`, validating the original import commit
without preventing subsequent independent source development.

`python3 scripts/check_distribution.py` builds a wheel, checks its module set
and 0BSD license payload/metadata, then runs an isolated import/solve test.
Its outputs also stay outside the tree.

The source snapshot and explicit import/exclusion lists are recorded in
`docs/migration-manifest.json`. `docs/MIGRATION.md` describes the boundary,
`docs/design.html` the architecture, and `docs/BACKLOG.md` the development path.

Selected research benchmarks are under `benchmarks/`. They use synthetic
problems, retain their original defaults, and may require additional
`.[benchmark]` dependencies. Run them with explicit device/output options.
They do not consume private production data or inherited historical results.
