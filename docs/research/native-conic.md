# Native Dense Conic Prototype

This independent opt-in API implements dense real convex conic QP/LP for
ordered zero, nonnegative and Lorentz SOC blocks. Iteration, factorization
and projection use PyTorch on the explicitly selected CPU or CUDA device.
There is no third-party solver fallback, automatic routing or change to the
existing Clarabel reference API. The implementation is original project code
under 0BSD; dependencies retain their own licenses.

## Contract And Derivation

The original model is

```text
minimize f(x) = .5 x^T P x + q^T x
subject to A x + s = b, s in K, P symmetric PSD.
L(x,s,z) = f(x) + I_K(s) + z^T(Ax+s-b).
```

Proximal two-block ADMM with fixed positive rho and sigma gives:

```text
x+ = argmin_x f(x) + z^T(Ax+s-b)
              + rho/2 ||Ax+s-b||^2 + sigma/2 ||x-x_previous||^2
(P + rho A^T A + sigma I)x+
    = sigma x_previous - q + A^T(rho(b-s)-z)
s+ = projection_K(b - Ax+ - z/rho)
z+ = z + rho(Ax+ + s+ - b)
```

The proximal term makes the step invertible even for LP, singular curvature,
redundant equalities or empty constraints. It vanishes from the fixed-point
stationarity equation. It is not model regularization and cannot supply a
dual objective. Multipliers follow `z in K*`, not the negative convention.
At a fixed point the equations imply primal feasibility, stationarity and
`-z in N_K(s)`. For a consensus interpretation, use
`F(v,t)=f(v)+I_{Av+t=b}` and `G(x,s)=I_K(s)`. Apply ordinary two-block ADMM
to `(v,t)=(x,s)` in the fixed positive metric `diag(sigma I,rho I)`.
The free x projection sets x=v and its consensus multiplier remains zero.
Eliminating t=b-Av leaves exactly the updates above. Thus sigma/rho are
metric weights, and the quadratic f stays in the first proximal step.
When a primal-dual saddle point exists, this is the consensus ADMM setup,
not a new convergence theorem. This derivation does not assert a universal
finite convergence rate, strong duality without assumptions or an exact proof.

Zero projects to zero and its dual to the whole row space. Nonnegative
projection clamps below zero. For SOC `(t,u)` with `r=||u||`, return unchanged
when `r<=t`, zero when `r<=-t`, otherwise `((t+r)/2, (1+t/r)u/2)`.
Scaled norms and half sums avoid unnecessary overflow; unrepresentable
arithmetic fails loudly.

Original full symmetric PSD validation is done in CPU float64 before casting.
The raw original matrices are frozen separately from the device workspace.
Every checkpoint transfers candidate vectors explicitly to the independent
`check_conic_candidate` CPU checker. Only its complete original-coordinate
equation/cone/stationarity/block-complementarity/range/dual-gap checks can return
`solved`. Tolerances are never loosened based on dtype. Float32 users must
explicitly select attainable tolerances. Limits return `max_iterations`, never
infeasible or unbounded; unresolved curvature spectra remain rejected.

## API And Acceptance

```python
from openopt.native_conic import NativeConicConfig, prepare_native_conic

config = NativeConicConfig(enabled=True, device="cuda:0")
workspace = prepare_native_conic(problem, config)
result = workspace.solve()
```

Inputs are copied, repeated solves reset all state, and output tensors remain
on the requested device. Explicit CUDA requests fail if unavailable. CPU
validation, transfers and independent replay are intentional and are not
GPU-resident performance claims.

CPU tests compare exactly the same frozen model against real Clarabel, including
SOC boundary LP, mixed cones, singular and unconstrained QPs, zero objective LP,
redundant and unrestricted equalities and three randomly generated QPs without
planted optimum witnesses. Tests reject altered primal/slack/dual/model/objective,
forged evidence and placement claims, even after the attacker recomputes hashes.
They also reject nonfinite models, casting, system and iterate overflow and
preserve exceptions from factorization and independent checks.

Run the existing external-artifact verifier:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_native_conic.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/native-conic/cpu
```

The main GPU coordinator should select `tests/test_native_conic.py` with
expression `cuda`. Each numerical CUDA case emits JUnit properties
`native_conic_acceptance_path` and `native_conic_acceptance_sha256` pointing to
external JSON with full original P/q/A/b/cone data, host x/s/z, actual tensor
placement, device/dtype, config/tolerances, objective, independent evidence and
source hashes. It does not require Clarabel inside the Pod.

After pinning the JUnit-linked sidecar hash, independently replay it using:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_native_conic.py --replay /hf3fs-jd/data/tmp/EXPLICIT-RUN/native-conic.json --sha256 CALLER-PINNED-SHA256 --source-root /data/dev/openopt --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/native-conic/EXPLICIT-REPLAY
```

The replay reads saved model bytes, never a test factory or solver. It must also
be bound to the main coordinator's pinned run/source/JUnit/device audit; the
test-local replay alone does not prove cluster execution or prevent changing
all model/vector/hash fields together.

## Scope Limits And Sources

No HSD embedding, infeasibility rays, rotated/PSD/exponential/power cones,
sparse or matrix-free conic solve, presolve, scaling, automatic rho tuning,
multi-GPU solver or performance qualification is implemented. This prototype
does not close commercial qualification or general optimization coverage.
Only three scoped files are written by this seat. Main integration owns the
design ledger, generated reports, exports and actual GPU execution.

The requested primary mathematical sources are the Stanford-hosted PDOS and
SCS papers:

```text
https://web.stanford.edu/~boyd/papers/pdf/pdos.pdf
https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf
```

No source code from either project is copied. This method is proximal ADMM on
the original QP, not a claim to reproduce SCS/PDOS or their HSD algorithms.

On 2026-10-10 the PDFs were acquired sequentially through the established
Astrill fetcher at `--rate 1m --tries 1`, with both reported `c14_ok=1`.
The frozen PDOS paper (December 6, 2013), sections 3 and 6.1/6.2, supplies the
consensus interpretation and cone/equality projection formulas; its LP-only
method is not directly copied as a QP solver. The frozen SCS paper
(July 25, 2016), sections 2.3 and 3.2, explains the HSD algorithm that this
prototype deliberately does not implement.

```text
pdos.pdf: 275921 bytes
SHA256 327d2c933ffe097c13fe49e36c777f9501d0b4bb5c00c3c5cea8b8dbd000df18
scs_long.pdf: 394400 bytes
SHA256 1db3c2deb86cd54e43f1a03eb96a42f766b83134efcb7864ec2427e6cca5a2a2
```

PDFs and fetch logs stay outside Git under the seat's `native-conic/references`
artifact directory. These hashes freeze the obtained bytes; no caller-supplied
upstream authenticity hash was available.

## Diagnosis And Remaining Integration

The first test run retained one failed extreme-value fixture: PyTorch's default
float32 input already overflowed before projection. The fixture now explicitly
uses float64, leaving fail-loud projection intact. A separate controlled red
test proved Python-list float values were rounded before the original model
was frozen (`-1.00000003` became `-1.0`). Input freezing now preserves float64
before device casting; the strict float32 candidate remains rejected against
the original model. Device-cast loss of nonzero data is explicitly rejected.
Red artifacts are retained under `cpu-v1` and `list-precision-red`, not counted
as accepted tests.

The inherited shared release artifact root failed the global cross-user
permission check. The existing verifier successfully wrote and read this
seat's evidence as infra. This seat did not broaden root permissions or alter
other owners' evidence. Main coordination owns that shared-tree condition.
GPU execution, source/JUnit sealing and report integration remain with the
main agent; presence of CUDA tests is not a GPU execution result.
