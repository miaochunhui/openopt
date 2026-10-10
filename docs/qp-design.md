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
an exact certificate. The unchanged default `max_iterations` does not distinguish
infeasible from unbounded. Opt-in `dual_gap_termination=True` adds independent
original-model dual-range/gap checks. Opt-in `direction_termination=True` derives
directions from ADMM checkpoints and independently checks them; unbounded status
also requires a saved feasible base. Unverified directions remain unknown.
No full presolve, native GPU MIP engine, MIQP, universal `minimize`, or
automatic GPU routing is implemented by the dense default path.

## Optional Research Paths

`openopt.batch` provides same-shape stacked dense QPs with batched factor/solve,
per-item status, metrics and real checked iteration counts. Finished items freeze
while others continue. Cold includes stacking/validation/transfers; prepared
excludes setup. This path explicitly rejects adaptive rho, dual-gap termination
and direction termination instead of ignoring unsupported options.

`openopt.operators` provides dense/COO/CSR and borrowed matrix-free operators,
with CG true-residual acceptance. It never densifies sparse matrices or forms
the dense Gram system. The caller must declare `assume_psd=True`; probes do not
certify convexity. Opt-in Jacobi PCG checks the true residual and preserves
COO/CSR sparsity; matrix-free preconditioning requires an explicit SPD inverse
action. Fewer CG iterations alone are not an end-to-end speed claim.
The historical small sparse case was slower. This is not a default replacement.

`openopt.scaling` applies opt-in bounded reversible Ruiz/cost scaling. Solutions
and multipliers are restored, then checked against the unscaled original model.
It does not replace original tolerances or silently fix a failed qualification.

`openopt.certificates` independently checks supplied original-coordinate
candidates, numerical dual range/gap, Farkas directions and recession rays.
The dense solver can now opt into these checks for generated checkpoint
directions and termination. Singular spectral uncertainty remains unavailable.
Approximate numerical evidence is not an exact proof or rigorous certified bound.

`openopt.conic` defines zero, nonnegative, second-order and rotated second-order
cones, with independent
cone/KKT/complementarity/gap checks and an explicitly selected Clarabel CPU
reference. `openopt.integer` defines MILP bounds/integrality, an explicit HiGHS
CPU reference and independent incumbent/gap checks. MIQP is rejected; neither
module silently falls back from CUDA or constitutes a native GPU solver.

`openopt.native_conic` is a separate opt-in dense PyTorch proximal ADMM
prototype for zero/nonnegative/SOC/RSOC products, with explicitly requested
device/dtype and independent original cone/KKT/range/gap acceptance.
The RSOC convention is `(a,b,w): a,b>=0, 2ab>=||w||^2`. The native Euclidean
projection uses a scaled orthogonal Lorentz transform, retains interior values
exactly, compensates the original signed-product difference near the polar
boundary, and retains the weight as a mantissa/exponent pair until each final
coordinate. Float32 projection uses device-local float64 intermediates; it does
not transfer projection work to the CPU. Original feasibility checks each
head's sign separately and compares the required-head inequality and unchanged
absolute/relative tolerance with bounded dyadic integer products from the
original binary64 inputs. Norm square-root roundoff cannot invent a boundary
deficit; neither opposite-head scale nor transformed coordinates may conceal
an original violation. This limited exact input comparison is not a conic
optimality certificate or arbitrary-precision solver.
Unavailable/nonrepresentable arithmetic fails loudly. The explicit Clarabel
CPU reference transforms the corresponding A/b rows and restores both s/z;
all candidate checks run in original coordinates. CPU tests do not establish
same-source CUDA execution or speedup. See `research/rotated-conic.md`.
It is not the Clarabel CPU reference and never invokes a third-party solver.
Its scope excludes HSD, additional cones, sparse conic operators and generated
conic directions; CPU regression does not qualify its CUDA path or performance.

`openopt.refinement` optionally guesses active bounds, factors the original
unregularized KKT matrix, and performs finite-precision residual corrections.
Explicit float32 factor/float64 residual work is supported, not arbitrary
precision. Every accepted candidate passes independent original-model checks
and cannot worsen baseline feasibility/stationarity/sign/complementarity/gap.
Singular systems remain unknown; wrong guesses are rejected and preserve
baseline values. Neither prototype is a default-path optimization.

`openopt.presolve` separately offers disabled-by-default exact zero-row removal
and +1 identity-equality fixing. Rational host auditing rejects nonrepresentable
fixed substitutions rather than perturbing the problem. Sealed original,
working and reduced models retain maps, shifts and objective offset; postsolve
restores primal and signed equality multipliers and reruns every original
feasibility/stationarity/sign/complementarity/dual-gap gate. Exact contradictory
zero rows may produce independently checked numerical Farkas evidence, not
exact certificates. Other presolve rules are unsupported. Host structural
auditing and replay are explicit transfer costs, not a CPU solve fallback or
a demonstrated performance benefit. See `research/presolve.md`.

The restricted MPS/QPS importer freezes original bytes, model arrays, objective
sense/offset and bounds. Unsupported constructs are rejected rather than
silently dropped. Acquisition and public-model qualification are separate
gates; parser tests do not establish representative industrial coverage.

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

The `release-proof` dispatch submits one explicitly selected numerical case
per one-card Job, at most three cards for this coordinator. Release GPU
submissions require clean committed and byte-stable source. Collection records
the immutable nested result and hash, rejecting missing, duplicate, CPU-only,
failed or mismatched case artifacts. `gpu_release_proof.py` saves complete
original/resident models, vectors, placement, scaling/CG/checkpoint traces and
independent replay. Collected proof requires the exact final
`SEMANTIC_NUMERICAL_PASS` marker, numerical PASS and execution UNVERIFIED;
collection does not promote producer self-consistency to execution qualification.
The separate caller-pinned coordinator audit must bind pre-dispatch original
models, clean source, actual Job/Pod UIDs and independently collected logs before
execution can pass. Scoped synthetic proof is neither a performance claim nor
remote attestation or commercial qualification.

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
Broad ill-conditioning/degeneracy coverage, general numerical certificate
qualification, diverse sparse structure, parameter robustness and commercial
runs remain unqualified. Opt-in implementations do not close those gates.
The measured
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
