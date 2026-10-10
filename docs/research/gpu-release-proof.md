# Frozen GPU Numerical Producer

`scripts/gpu_release_proof.py` creates new evidence only. It does not change
solvers, submit Jobs, install dependencies, publish, or reaccept the earlier
integrated CUDA suite. Independent suite execution acceptance remains owned by
`accept_release_cuda.py`. Native conic and refinement producers are separate.

## Contract

The explicit `run --device cuda:0` path requires one visible RTX PRO 4500,
observed compute capability, Pod/node identity and cgroup/GPU runtime samples.
It requires explicit `--expectations` and `--expectations-sha256`, checked before
hardware access and before solving. There is no CPU fallback. `cpu-smoke --device cpu` has a separate scope and
requires `--allow-cpu-smoke` when sealed/replayed; its verdict is never GPU PASS.
CPU smoke measures current-process rusage, explicitly not Pod cgroup usage;
the formal CUDA path still requires cgroup v2 quota and runtime CPU samples.

Successful CUDA production and numerical replay report
`verdict=SEMANTIC_NUMERICAL_PASS`, `numerical_verdict=PASS`, and
`execution_verdict=UNVERIFIED`. CPU production/replay preserve
`CPU_SMOKE_PASS` / `CPU_SMOKE_REPLAY_PASS` with execution also UNVERIFIED.
Bare seal reports SEALED, never execution PASS. Device/Pod metadata and hashes
within an artifact establish self-consistency, not origin or real execution.
Only the separately owned coordinator run audit may qualify execution after
binding independent cluster status, exact command, source, model manifest and
coordinator-collected runtime/completion logs. This is not remote attestation.

## Pre-Dispatch Manifest

The caller freezes `proof-expectations.json` at the run root before dispatch.
Its exact schema has three keys:

```json
{
  "schema_version": 1,
  "source_sha256": {"src/openopt/qp.py": "<sha256>", "scripts/gpu_release_proof.py": "<sha256>"},
  "cases": {"gap_lp": {"model_sha256": "<sha256>", "model": {"P": {}, "q": {}, "A": {}, "l": {}, "u": {}}}}
}
```

The abbreviated example omits full source membership and tensor descriptors;
it is not a runnable manifest. `source_sha256` must contain all entries from
`source_map(run/source)` with exact bidirectional current-source equality.
Each model is reviewed CPU float64, with the same canonical COO/CSR conversion
as `execute_case`: construct with `make_problem`, use `.to_sparse()` for P/A
for `jacobi_coo`, and then `.to_sparse_csr()` for `jacobi_csr`. Each tensor is a
`save_tensor` descriptor with complete shape/dtype/device/layout/components.
Component path/hash/bytes records are relative to the manifest parent, the
RUNroot; the coordinator uses `arrays/<case>-original-<field>-<component>.npy`.
No solutions or supplied directions belong in the expected models.

The manifest may contain a superset of one Job's selected cases. Run and replay
strictly look up each selected name and compare model digest plus every raw
component hash, dtype, layout and shape against this independently pinned
manifest. Original model matching is checked before solver preparation.
The report binds `expectations={path: absolute, sha256, bytes}` to the caller's
path/hash. Seal/replay require the caller to supply that same trusted path and
hash; they never obtain trust by reading the producer's claimed descriptor.
Replay reads and verifies expected raw arrays, never calls a factory or solver.
CPU smoke may omit expectations, but then qualifies only its saved-model
numerical semantics, not fixed-case model origin.

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

Direction production observes the unchanged solver's checkpoint arguments and
terminal iterate. Replay requires terminal iteration to be a frozen policy
checkpoint (or max iterations), nonadaptive rho to equal the frozen policy,
finite correctly shaped differences and zero previous iterates for the first
checkpoint. Exported rays must equal normalized checkpoint differences;
the unbounded base must byte-equal terminal x. Both Farkas and recession checks
are independently recomputed from original data and dx/dy/current x; both full
summaries, including metrics/thresholds/unavailable, must exactly match with
strict JSON types. Exactly one must pass and match the reported terminal status.
These checks reject impossible checkpoints and contradictory evidence; they
do not by themselves prove that a coordinated tuple was generated by execution.
That origin claim requires the external coordinator audit. Directions/base are CPU exports by the existing solver, while their
checkpoint computation and x/y stay on the explicitly requested compute device.

Two independent checks are saved: the existing original-coordinate certificate
checker and standalone NumPy equations. The latter recompute feasibility,
stationarity, support/complementarity, PSD range, primal/dual objectives and gap,
or Farkas/recession/descent and base feasibility. Unresolved curvature,
missing vectors, failed/unknown solves and overbroad gates fail loudly.

## Run And Handoff

The GPU entry uses `run --device cuda:0`, explicit `--case`, external
`--artifact-dir`, and the coordinator's pre-dispatch `--expectations` path and
`--expectations-sha256` pin. The command is assembled by the separately owned
dispatch tool; no cluster command or placeholder launch is supplied here.
One `OPENOPT_GPU_PROOF_RUNTIME` JSON line carries
`device/gpu_uuid/gpu_name/gpu_count/node/pod` for coordinator-collected logs.
GPU UUID comes from actual CUDA device 0 properties; every nvidia-smi sample
must refer to that same UUID/name. The completion line binds result path/hash.

The coordinator may use explicit repeated `--case` selections for independent
Jobs, at most three authorized cards overall. No cluster operation belongs to
this producer. The final line provides the immutable result path and SHA256.
Use that exact caller pin with `seal --result ... --result-sha256 ...`, explicit
`--current-root`, the independently trusted expectations path/hash, and an
external sibling `--artifact-dir`. Seal prints its
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

The four coordinated pre-seal regressions reproduce the audited substitutions:
whole-model/solution replacement is rejected against independent expected
arrays; CPU-as-CUDA cannot obtain execution PASS (missing expectations or an
explicit execution PASS claim is rejected); planted ray/base at an impossible
iteration is rejected; contradictory selected/opposite terminal summaries are
rejected. A self-consistent relabeling may pass numerical equations only and
must retain execution UNVERIFIED. Negative controls retain incorrect-x rejection
and original nine-case CPU replay, including strict no-solve/no-factory guards.
Original red audit evidence remains external under
`/hf3fs-jd/data/tmp/openopt-release-20261010/native-conic/review-producer/run-v1`.
