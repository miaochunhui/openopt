# Imported Research Tools

These selected tools preserve the source snapshot's implementation and
methodology. They do not import old experiment outputs or production datasets.

- `compare_open_source.py`: factor QP compared against OSQP and Clarabel.
- `compare_constrained.py`, `generate_constrained_cases.py`: synthetic general
  linear constraints and reference comparisons.
- `compare_warm_start.py`: prepared state and warm-start research.
- `profile_router.py`: CPU/GPU crossover research; its CLI requires CUDA.
- `benchmark_solver.py`: exploratory throughput, not a qualification gate.
- `benchmark_trade_style_qp.py`: trade-style QP projection research.
- `benchmark_cntr_6000.py`: synthetic Barra-shaped proxy and the real problem
  builder exercised by the NPZ round-trip regression. No CNTR samples are used.
- `mpo_t1_benchmark.py`: synthetic multi-period research benchmark.

Use `--help` to inspect the retained CLI and explicitly choose devices and
output paths outside this worktree. `benchmark_solver.py` writes its JSON to
stdout; the verification wrapper captures output in its external log directory.

The old historical HTML renderer, canonical artifact pipeline, production
delivery bundles and deployment manifests are intentionally not included.

The retained synthetic benchmark's own KEEP label is only its original
research decision, not a qualification for the openopt project or real data.
