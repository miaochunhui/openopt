# openopt Working Entry

- This repository is the independent maintenance home for the imported OpenPO
  foundation. Do not modify infra or optimizer as a side effect of work here.
- `openopt` is the project/distribution name. Preserve `openpo` compatibility;
  the current models are portfolio-specific, not a general optimization API.
- Run `python3 scripts/iteration_gate.py smoke` before algorithm/performance
  work, and after edits. `python3 scripts/verify.py` is the complete local
  regression entry. Neither is a performance-release qualification.
- Changes to algorithms, solver parameters or benchmark methodology must update
  `docs/design.html` and `docs/benchmark-report.html`.
- Compare continuous LP/QP against OSQP and Clarabel. Compare future discrete
  optimization against SCIP and HiGHS. Never promote heuristic solutions to
  exact/certified results without numerical evidence.
- Keep code/configuration/documentation here; put logs, test artifacts and
  benchmark results outside the worktree. Verification accepts `--artifact-dir`;
  the default is `/hf3fs-jd/data/tmp/openopt-verification`.
- Do not copy credentials, internal deployment configuration, proprietary
  datasets, commercial reference documents or source-repository history.
- Licensing is not yet decided. Do not add a license, SPDX identifier or
  open-source-release claim without the rights holder's explicit decision.
