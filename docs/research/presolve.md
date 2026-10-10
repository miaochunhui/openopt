# Conservative Exact QP Presolve

## Contract

This opt-in implementation is not commercial or full presolve. It does not
enable any existing solver path automatically and uses only the project's
PyTorch dense CPU/CUDA solver. Explicit CUDA requests fail if unavailable;
the rational structural audit and independent numerical replay execute on
the host, with explicit copies, not a hidden CPU solver fallback.

```python
from openopt.presolve import PresolveConfig, presolve_qp, solve_presolved_qp
from openopt.qp import QPConfig

options = PresolveConfig(enabled=True)
prepared = presolve_qp(problem, QPConfig(), options)
result = solve_presolved_qp(problem, QPConfig(), options)
assert result.status != "solved" or result.evidence.passed
```

`PresolveConfig()` is disabled. `prepared.original_problem` is a CPU float64
clone of the original float32/float64 tensors **before solver casting**.
The working and reduced models, variable/row maps, fixed values, row shifts
and objective constant are retained. A byte seal rejects mutation before
restoration and checking. This is accidental-tamper detection, not an
adversarial signature. Mixed input devices/dtypes, nonfinite input, invalid
bounds, nonsymmetric and nonconvex models fail loud.

## Exact Reductions

Recognized constraints are `l <= A x <= u`, with:

- Rows whose coefficients are **exactly** zero and contain zero in their
  bounds. No tolerance makes a small coefficient zero.
- Finite exact equalities on rows with one coefficient **exactly +1**.
  These are the strict identity-row representation of fixed variable bounds.
  Duplicate consistent equalities select one row to restore the multiplier.

General singleton equalities, negative identities, implicit bounds, conflicting
identity equalities and other presolve techniques are not reduced. Conflicts
remain in the model. Notes explain recognized but unsupported reductions.

For fixed variables `x_K=c` and free variables `z=x_F`, the reduced model is:

```text
P_hat = P_FF
q_hat = q_F + P_FK c
A_hat = A_:F
l_hat = l - A_:K c,  u_hat = u - A_:K c
offset = 0.5 c.T P_KK c + q_K.T c
```

Infinite bounds are masked before subtraction. Exact rational arithmetic
audits the generated finite shifts, bounds, linear terms and constant against
the original binary floating-point values. If solver casting loses input
values, arithmetic overflows/underflows or the transformed result is not
exactly representable, **all fixed substitutions are retained**, explicitly
noted. Zero-row removal remains independently available. This deliberately
conservative rule sacrifices reductions rather than silently perturbing the
problem. The host audit is not a performance optimization or GPU speedup.

Original empty-variable input is rejected, matching the existing QP contract.
An **internally reduced** zero-variable model is a defined constant model with
unique empty primal. It is never passed to the nonempty dense solver. Its
restored candidate still goes through the full original checker.

## Original Dual And Acceptance

For kept rows, `y_original[row]=y_reduced`. Removed feasible zero rows and
unselected duplicate identity equalities receive zero multipliers. For each
selected `+1` fixed equality row `r_j`, compute in original working coordinates:

```text
g = P x + q + A_kept.T y_kept
y_original[r_j] = -g[j]
```

This is the signed equality multiplier, unrestricted in sign. It cancels the
fixed coordinate of stationarity. No free-variable stationarity residual is
repaired. Multipliers are not claimed to be unique.

`prepared.restore(x_hat,y_hat)` checks byte integrity, candidate dimensions,
dtype, explicit device and finite arithmetic. It makes **no** optimality claim.
`prepared.check(x,y)` independently reruns `check_qp_candidate` on the
pre-cast original model, enforcing all existing gates: original feasibility,
stationarity, multiplier sign, complementarity, available unregularized dual
objective and signed duality gap. A reduced `solved` result becomes
`candidate_rejected` if **any** original gate fails. Iteration limits remain
limits; reduced solver rays become `reduced_direction_unverified`, not original
certificates. There is no planted witness or third-party solve in this module.

A contradictory exact zero row is retained. Its signed row direction is
lifted through selected fixed equality rows to annihilate the original `A.T`.
`check_primal_infeasibility` independently verifies it on the original model.
Only passed numerical evidence yields `infeasible_numerical`; a contradiction
smaller than the numerical separation gate remains `infeasible_unverified`.
Neither status is an exact certificate. No arbitrary enumeration is used.

## Validation And GPU Handoff

Run with the existing validation interpreter and external evidence:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/iteration_gate.py tests --target tests/test_qp_presolve.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/presolve/cpu
```

Coverage includes a coupled QP solved both reduced/restored and unreduced,
original signed dual and gap, objective offset, bitwise disabled paths,
constant models, LP/degeneracy/empty rows, exact and near-zero structure,
unsupported/conflicting/unrepresentable cases, preserved pre-cast inputs,
iteration limits and three categories of deliberately wrong candidates.
Mapping/model mutation, malformed/nonfinite data and explicit CUDA failure
are negative controls.

The submitting coordinator can select:

```text
target: tests/test_qp_presolve.py
expression: cuda_presolve
```

Eight CUDA cases span coupled, active-bound, constant and contradictory models
in float32/64.
Each writes an externally stored `full-model.npz` containing all original,
working and reduced P/q/A/l/u, complete maps/shifts/offset and actual x/y/ray.
`presolve-acceptance.json` binds this NPZ by SHA256, includes source hashes and
the independently recomputed original evidence; JUnit properties pin the
JSON path and hash. NPZ byte readback is checked with `allow_pickle=False`.
CPU tests use the same sidecar producer. CUDA tests prepared here are **not**
formal GPU evidence until the coordinator executes and independently accepts
them on the authorized K8s hardware.

This worktree does not change package exports, source design/report generators,
GPU deployment or publication. The coordinator owns integration of those
cross-scope files and final frozen-source/GPU acceptance.

## Retained Negative Results

The second CPU attempt failed during pytest collection because a parameterized
test omitted its `option` argument. Its exit code 2 and first cause remain in
`presolve/cpu-v2/tests-h1dkxmod/pytest.log` under the external release evidence
root. Only the test signature was fixed; models and tolerances were unchanged.

Deliberate unsupported substitutions (rounded cast, nonrepresentable bound
shift/objective constant, overflow and underflow) retain the original model.
A reduced solved QP with an unresolved original PSD spectrum is rejected even
when original stationarity passes: original dual objective/gap remain
unavailable. Tiny zero-row contradictions that fail the existing Farkas
separation margin are not promoted to a verified infeasibility status.
The negative-control JSON records three independently rejected actual
primal/multiplier mutations; no acceptance gate is relaxed.
