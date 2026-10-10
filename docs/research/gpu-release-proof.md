# Frozen GPU Numerical Producer

`scripts/gpu_release_proof.py` creates new evidence only. It does not change
solvers, submit Jobs, install dependencies, publish, or reaccept the earlier
integrated CUDA suite. Independent suite execution acceptance remains owned by
`accept_release_cuda.py`. Native conic and refinement producers are separate.

## Contract

The explicit `run --device cuda:0` path requires one visible RTX PRO 4500,
observed compute capability, Pod/node identity and cgroup/GPU runtime samples.
There is no CPU fallback. `cpu-smoke --device cpu` has a separate scope and
requires `--allow-cpu-smoke` when sealed/replayed; its verdict is never GPU PASS.
CPU smoke measures current-process rusage, explicitly not Pod cgroup usage;
the formal CUDA path still requires cgroup v2 quota and runtime CPU samples.

Each selected solve records the complete original `P/q/A/l/u`, host `x/y`,
generated directions and actual feasible base when applicable. Arrays are
non-pickled NPY bytes with shape/dtype/device/layout/component hashes. COO/CSR
matrices retain canonical sparse components. All `src/**/*.py` and the producer
are copied byte-for-byte and checked for bidirectional membership against the
explicit current source root. A model digest excludes observed device but binds
storage metadata and raw bytes.
Every solve also saves its actual resident workspace model. Unscaled resident
inputs must byte-match original storage; scaled inputs must match D/E/c within
only accumulated machine-arithmetic rounding, without relaxing original checks.

The nine fixed float64 synthetic coverage cases are basic/ill-conditioned
scaling, Jacobi COO/CSR, SPD/LP/singular gap, infeasible and unbounded. They do
not establish public/industrial coverage, float32 qualification, comprehensive
capabilities, performance or commercial superiority.

Scaling saves D/E/c and raw scaled x/y, checking exact original restoration.
Jacobi saves resident sparse P/A, the real diagonal, every CG rhs/initial/result
and its actual outer y/z/iteration,
actual device and true unpreconditioned residual. Replay checks the unshifted
original system and configured tolerance without adding a rounding allowance.
GPU-core sparse tensors are never densified. Small host-only spectral replay is
explicitly separate from the core.

Direction tracing observes the unchanged solver's real checkpoint arguments and
terminal iterate. Exported rays must equal normalized checkpoint differences;
the unbounded base must byte-equal the actual terminal x, not a planted feasible
witness. Directions/base are CPU exports by the existing solver, while their
checkpoint computation and x/y stay on the explicitly requested compute device.

Two independent checks are saved: the existing original-coordinate certificate
checker and standalone NumPy equations. The latter recompute feasibility,
stationarity, support/complementarity, PSD range, primal/dual objectives and gap,
or Farkas/recession/descent and base feasibility. Unresolved curvature,
missing vectors, failed/unknown solves and overbroad gates fail loudly.

## Run And Handoff

The principal GPU entry for the main coordinator's existing one-card Pod is:

```bash
python3 scripts/gpu_release_proof.py run --device cuda:0 --case all --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/gpu-proof/formal
```

The coordinator may use explicit repeated `--case` selections for independent
Jobs, at most three authorized cards overall. No cluster operation belongs to
this producer. The final line provides the immutable result path and SHA256.
Use that exact caller pin with `seal --result ... --result-sha256 ...`, explicit
`--current-root` and an external sibling `--artifact-dir`. Seal prints its
bundle path/hash. `replay --bundle ... --sha256 ...` independently reads every
saved model/vector and source byte; it never executes snapshot code, a solver
or a model factory. Shell placeholder examples are intentionally not supplied.

All outputs, failures, telemetry and execution logs are external under 3FS tmp.
Output directories are unique and artifacts are create-only. Seal inventories
all files, rejecting added, removed or modified evidence. Source changes require
a new run; repinning does not exempt numerical/config/placement checks.

The CPU unit tests execute the actual producer and replay and reject byte,
model, direction/base, checkpoint, tolerance, placement, source membership,
scaling restoration and CG mutations. A caller pin is an integrity anchor, not
a cryptographic hardware attestation or a legal/commercial qualification.
