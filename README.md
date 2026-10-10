# openopt

GPU-accelerated optimization foundation, extracted from OpenPO into an
independent repository.

**Current status:** the implementation is a portfolio-optimization research
foundation. A general-purpose optimization API is the next development stage,
not an already implemented feature. The imported source had no license
declaration; a public open-source release requires a rights-holder decision.

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
`from openopt import ...` exposes the same existing public API without
renaming classes or duplicating solver implementation:

```python
from openopt import FactorRiskModel, PortfolioProblem, SolveConfig, solve
```

There is no `openopt.minimize` or universal problem model yet. The API currently
uses portfolio concepts. GPU acceleration is not automatically preferable:
small or sparse problems may favor the CPU reference solvers.

## Maintenance

`scripts/verify.py` runs import-provenance checks, API compatibility tests and
the complete migrated regression suite. `scripts/iteration_gate.py smoke` runs
the smaller correctness preflight. They report failures and actual test counts;
neither certifies speedups, production readiness or untested GPU hardware.

`scripts/migrate_openpo.py --check` checks the exact imported working snapshot.
Normal verification uses `--provenance`, validating the original import commit
without preventing subsequent independent source development.

`python3 scripts/check_distribution.py` builds a wheel, checks its module set
and runs an isolated import/solve test. Its outputs also stay outside the tree.

The source snapshot and explicit import/exclusion lists are recorded in
`docs/migration-manifest.json`. `docs/MIGRATION.md` describes the boundary,
`docs/design.html` the architecture, and `docs/BACKLOG.md` the development path.

Selected research benchmarks are under `benchmarks/`. They use synthetic
problems, retain their original defaults, and may require additional
`.[benchmark]` dependencies. Run them with explicit device/output options.
They do not consume private production data or inherited historical results.
