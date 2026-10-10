# Explicit MILP CPU Reference Contract

This experimental interface models

```text
minimize       q.T @ x
subject to     l <= A @ x <= u
               lower <= x <= upper
               x[i] integer when integrality[i] == 1
```

`integrality` is a length-n numeric vector containing only 0 (continuous) and
1 (integer). Binary variables are integers with bounds [0, 1]; semi-continuous,
semi-integer, logical/SOS constraints and quadratic objectives are unsupported.
`MILPProblem.P` must be None, even for a zero Hessian. `solve_miqp` raises
`NotImplementedError`. No quadratic term is silently discarded.

## Backend And Input

Import directly from `openopt.integer`; the package facade is outside this
change's write set:

```python
from openopt.integer import MILPProblem, MILPConfig, solve_milp

problem = MILPProblem(
    q=[1.0, 0.25], A=[[2.0, 1.0]], l=[3.5], u=[float("inf")],
    lower=[0.0, 0.0], upper=[2.0, 2.0], integrality=[1, 0],
)
result = solve_milp(problem, MILPConfig(backend="highspy", device="cpu"))
# x = [1, 1.5], incumbent = dual_bound = 1.375.
```

The caller must explicitly supply `backend="highspy"` and a config. HiGHS is
an external CPU reference, not an Openopt branch-and-bound engine, native GPU
implementation, commercial solver substitute or performance claim. This module
contains no exponential enumeration, CUDA computation or automatic routing.
Explicit CUDA/non-CPU config or tensor input is rejected without consulting
CUDA availability or transferring to CPU. Missing highspy raises with the
original import exception; it never installs a dependency or chooses a fallback.
The existing validation environment provides the optional backend.

Inputs are copied from real dense numeric arrays or CPU tensors into float64.
The original arrays/tensors are not mutated. Empty row sets are supported;
empty decision vectors, malformed shapes, boolean integrality, nonnumeric,
complex/sparse data, nonfinite q/A/integrality, NaN bounds, lower +infinity,
upper -infinity and reversed bounds are rejected.
Allowed infinity is only lower -infinity or upper +infinity.
HiGHS reference range restrictions are explicit: finite q/bounds must have
magnitude <1e20; nonzero A coefficients must lie in (1e-12, 1e15).
The small-matrix threshold is explicitly 1e-12. Rejecting unsupported scales
prevents a finite bound becoming infinity or a nonzero coefficient being
silently dropped. The standalone checker is not limited to these backend scales.
Integer candidates outside [-2**53+1, 2**53-1] cannot have their integrality
resolved in float64 and are rejected.

## Results And Independent Checks

`x` is the validated incumbent vector, or None; `incumbent`/`objective` is its
independently recomputed q.T @ x, or None. `dual_bound` is the HiGHS MIP lower
bound, not a recomputed LP relaxation or an Openopt-certified bound. Missing
MIP lower bound (-infinity) is represented as None, never fake zero.
NaN/+infinity bounds and nonfinite claimed incumbent/objective raise.
For an all-continuous LP, the adapter instead computes the signed dual support
from row/column duals and independently checks stationarity and finite supports.
These floating-point checks are numerical evidence, not exact/interval proof.

`check_milp_candidate(problem, x, dual_bound=...)` does not call HiGHS or trust
its residuals/status. It independently evaluates each original constraint,
variable bound, distance to the nearest integer and q.T @ x. Feasibility and
integrality use separate absolute tolerances, default 1e-7; large unrelated
rows or bounds cannot mask another row's violation. A continuous component is
not rounded. Invalid candidates from the backend raise instead of becoming
successful feasible results. Arithmetic overflow, underflow and invalid
operations are fail-loud.

Gap fields use this explicit module convention, not HiGHS's `mip_gap`:

```text
signed_gap = incumbent - dual_bound
abs_gap    = abs(signed_gap)
rel_gap    = abs_gap / max(1, abs(incumbent))
gap_pass   = abs_gap <= config.abs_gap + objective_tol
             OR rel_gap <= config.rel_gap + objective_tol / max(1, abs(incumbent))
```

The default requested absolute/relative gaps are zero; the default absolute
objective consistency tolerance is 1e-7. A lower bound exceeding the incumbent
by more than this tolerance fails, rather than clipping a negative gap to zero.
HiGHS's native relative-gap denominator differs near zero: native gap limits
are passed unchanged, but public gaps/acceptance always follow the formula
above. This can be stricter than native termination. The checker validates
arithmetic/order of a supplied dual_bound, not its global mathematical validity.
A dishonest lower bound cannot be independently certified by scalar gap checks.

`status="optimal"` requires raw HiGHS kOptimal, a finite independently feasible
incumbent, an available lower bound, correct bound ordering and gap acceptance.
This means numerical optimality within the declared tolerances; it is not an
exact proof. If kOptimal lacks that evidence, the adapter raises.
Unknown/notset, objective-target, interrupt, memory and limit statuses return
`feasible` only with a validated incumbent, otherwise `unknown`; even zero gap
cannot upgrade them to optimal. Raw infeasible/unbounded are attributed to
HiGHS and have no solver-generated Openopt certificate/ray. An ambiguous
unbounded-or-infeasible status remains unknown. Backend errors raise with
operation/model status; warnings are surfaced. `backend_status`,
`backend_version`, `termination_reason` and qualification preserve provenance.

## Limits And Scope

`time_limit=None` and `node_limit=None` mean leave the respective native limit
unchanged. Finite nonnegative time limits (seconds) and nonnegative integer
node limits, including zero, are passed to `time_limit` and `mip_max_nodes`.
Time covers native solving, not Python validation/conversion/return checking,
and is not a hard process timeout. Node limits restrict branch-and-bound nodes,
not Python iterations, presolve or a universal amount of computation. An LP
has no branch-and-bound node budget. Presolve may solve a model before a limit
is reached. A positive limit permits optimal completion only if HiGHS truly
finishes and independent evidence passes.

HiGHS uses kSolutionLimit for some MIP node-limit stops; `node_limit` is reported
when the configured cap and actual node count establish it. Otherwise retain
`node_or_solution_limit`, never invent a precise cause. Raw kIterationLimit
retains `iteration_limit`. Deterministic real zero-limit tests disable presolve.
No status is inferred from wall time alone. Limits do not manufacture
infeasibility, unboundedness or optimality certificates.

Scope is exactly `src/openopt/integer.py`, `tests/test_integer.py`, and this
document in the assigned integer worktree. No main tree, infra, package facade,
shared report, GPU configuration, dependency, credential or nested agent edits.
Existing QP algorithms/parameters and benchmark methodology are unchanged.
MILP reference documentation is local to this module; broader design/report
integration remains the coordinator's write set.

The infra integrity gate passed. Its edit router rejected these cross-repository
targets with exit 2 because they are outside infra, not because no rules apply.
The explicit assignment and openopt AGENTS govern the three-file scope.
memory-router returned no private mapping; the latest repository handoff and
backlog were read. Historical qualifications are not reused as new evidence.

## Acceptance And Remaining Work

Use the existing environment, CPU only, and external artifacts:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py \
  --target tests/test_integer.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/integer/acceptance
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/iteration_gate.py \
  smoke --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/integer/postflight
```

The pre-edit smoke passed 79 tests with four explicit CUDA skips at
`integer/preflight/smoke-49xjyn7t/pytest.xml`. New-code evidence is separate.
The first targeted run at `integer/acceptance-v1/tests-ennrwni_` retained
155 passes and one failed test (pytest exit 1): the unbounded fixture expected
unbounded from a presolved MIP, but HiGHS returned unbounded-or-infeasible.
The adapter correctly preserved unknown. The fixture now disables presolve
for the explicit unbounded path and separately checks conservative handling
of the actual presolve status. This is not a solver-status promotion or retry
of unchanged code; the original failure artifact is retained.
Tests include analytic mixed continuous/integer and binary MILPs, signed bounds,
empty rows, zero objective, continuous LP duals, actual infeasible/unbounded,
real zero/positive limits, malformed/nonfinite data, scale rejection and
independent fractional/constraint/bound/gap negative controls.
Controlled backend mutations test statuses, missing/corrupt incumbents, objective
disagreement, inverted/missing bounds, option propagation, original error causes
and explicit backend/device/MIQP rejection. Mutations are not real limit
performance experiments. A final test writes `integer-acceptance.json` into
external pytest temporary storage and records its path in JUnit properties.
It includes UTC time, backend version, three source hashes and analytic plus
independent negative evidence. Counts and verdicts must be read from actual
`pytest.xml` and `verification.json`, not inferred from process return codes.

The corrected-code targeted verification at
`integer/acceptance-v2/tests-4thjozu3` passed 159 tests, zero failures/errors/skips,
using highspy 1.15.1. The post-edit smoke at
`integer/postflight-v2/smoke-whh405e8` passed 79 tests with four explicit CUDA
skips. These runs overlap preflight and are not added into a distinct-case count.
The final targeted verification uses a separate `integer/final` directory so
the source hashes include this finalized document. Real zero-limit stops
surface HiGHS kWarning with kTimeLimit/kSolutionLimit; these are preserved
warnings, not swallowed exceptions or optimal results. JUnit records the
acceptance property despite pytest's xunit2 compatibility warning.

Remaining UNVERIFIED: native GPU MIP, MIQP, independent global-bound/certificate
replay, licensed commercial/SCIP matched comparisons, industrial/public diverse
MILP corpora, performance qualification, full-repository and clean-wheel release
integration. HiGHS correctness tests do not close any of these gates.
Positive-limit tests establish successful completion of small analytic models;
limit-with-incumbent status semantics additionally have controlled backend
negative tests, not a timed industrial MILP incumbent experiment.
