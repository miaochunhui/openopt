# Rotated Conic CUDA Evidence Version 1

This producer contract is based on clean `5920c7f`. Only this document and
`tests/test_rotated_conic_cuda.py` are added. Main owns dispatch and Mendel owns
independent acceptance integration. No existing checker, solver, acceptance
tool or cluster object is changed. CPU mechanism tests are not CUDA evidence.

## Exact Selection And Identity

Select only these three function targets, or the file with `-k cuda_rotated`:

| Function | Explicit parameter IDs | Positive | Negative |
| --- | --- | ---: | ---: |
| `test_cuda_rotated_original_model` | `rotated_lp-float32`, `rotated_lp-float64`, `rotated_qp-float32`, `rotated_qp-float64` | 4 | 0 |
| `test_cuda_rotated_projection_extremes` | `interior-asymmetric-float64`, `interior-overflow-float64`, `near-polar-float64`, `small-head-float64`, `small-head-float32`, `tail-overflow-float64` | 6 | 0 |
| `test_cuda_rotated_original_rejections` | `wrong-equality-dual`, `planted-slack`, `wrong-dual-cone` | 0 | 3 |

The complete denominator is 13, with zero failure, error or skip required for
actual CUDA execution. Every member must have its complete exact identity;
function existence or aggregate counts alone are insufficient.
JUnit classname is `tests.test_rotated_conic_cuda`, and name is the function
plus the bracketed explicit ID. JSON `test_identity` is that classname, `::`,
and that complete name. CUDA-unavailable selection explicitly skips all 13;
those skips never satisfy GPU acceptance.

Each member writes one unique `rotated-conic.json` inside its external pytest
temporary directory. Exactly two JUnit user properties bind it:
`rotated_conic_acceptance_path` and `rotated_conic_acceptance_sha256`.
The latter hashes actual bytes, not a reconstructed dictionary.

## Schema 1

`family=rotated_conic_cuda`, `schema_version=1`, `verdict=PASS` denotes successful
test execution, including expected candidate rejection. It does not imply
every candidate passes. Exact case identity selects the outcome contract.

All cases contain the full original `problem` (`P/q/A/b`, ordered
`{kind,dimension}` cones, with `rsoc` distinct from `soc`), actual `x/s/z`,
all `NativeConicConfig` fields including `eps_abs/eps_rel`, explicit requested
device and dtype, status, iterations, complete original checker evidence and
three separate supplied-slack/original-slack/dual cone evidence objects,
observed runtime, tensor metadata, and exact `source_sha256` membership.
Every Python file under `src`, plus this producer test, is hashed.
The document is a contract, not an execution source dependency.
The producer verifies loaded conic/native/checker module locations and bytes;
this is source provenance, not adversarial runtime attestation.

`runtime` records Python implementation/version, executable, NumPy/PyTorch
versions, compiled CUDA version, availability, GPU name, device capability,
and OMP/OpenBLAS/MKL/CUDA-visible environment values. `tensor_metadata` records
actual shape/device/dtype/layout for original GPU P/q/A/b (binary64), prepared
P/q/A/b/factor and candidate x/s/z (requested dtype).
Host transfers for independent original checking are explicit.
CPU tests use `device=cpu`, `runtime.cuda_execution=false`, and distinct CPU
identities. Replay can require CUDA and rejects those CPU reports.

## Numerical Gates And Bounds

Both analytic models use a zero equality block followed by an original RSOC.
The LP fixes w=2 and minimizes a+b, with x=(sqrt(2),sqrt(2),2).
The QP fixes w=2 and minimizes (a^2+b^2)/2, with the same x.
Their independently derived original-coordinate duals are retained.
Float64 uses `eps_abs=1e-8`; float32 uses `eps_abs=3e-5`; both use `eps_rel=0`.
Native termination and independent original feasibility, componentwise
stationarity, every complementarity block, dual range and signed gap must all
pass. No transformed SOC model substitutes for this original model.

Projection positives exercise original device tensors at extreme magnitudes.
A bounded test-only Decimal analytic Euclidean projection oracle operates on
the exact represented input, at precision 800, for these six vectors only.
It is not solver arbitrary-precision support or a general optimizer.
Component comparisons use fixed `atol=0`, `rtol=3e-14` for float64 and
`3e-6` for float32; interior preservation is additionally bitwise.
The extreme vector is also a complete zero-objective original feasibility
model: P=q=A=0, b=s=projected, x=z=0 and original RSOC. The full existing
candidate gate and both head signs are checked independently.
This feasibility witness alone does not establish nearest-point optimality;
the separate unscaled oracle comparison does. An overflowing unscaled
projection-QP objective is not fabricated or silently rescaled.

Negative candidates start from a genuine native LP result. They flip the
equality dual sign, replace x with (0,0,2) while keeping planted feasible
slack, or reverse the RSOC dual block. Expected original failed checks are
respectively stationarity, primal equation plus original slack cone, and
dual cone. Full original evidence is recomputed, not patched.
Baseline arrays/results are not mutated. Rejected candidates remain in the
sidecar alongside the accepted baseline and the exact mutation description.

Independent replay checks frozen arrays without solving or invoking a test
model factory. Evidence comparison is exact, not approximate dictionary
comparison. Runtime differences can therefore reject old evidence; no old
run is retroactively qualified. Final run/source/JUnit pinning and clean
source execution still belong to main's separately reviewed acceptance tool.

## CPU And Dispatch Boundary

Default tests are self-contained; they require no historical campaign path.
The CLI external-log test explicitly skips without a 3FS basetemp; all other
CPU mechanism tests run in ordinary pytest temporary directories. That optional
skip never counts as CUDA qualification. No historical runtime is required.
CPU mechanisms exercise both models/dtypes, all six projection vectors, all
three rejects, source and placement tampering, no-solve replay and exact
selection. Collection and the explicitly skipped CUDA selection are recorded
externally. The CPU report never claims CUDA PASS.

Main and Mendel must align this exact identity/config/runtime/placement and
source-membership contract before fresh frozen GPU dispatch. Existing
`accept_extension_cuda.py` does not accept this new family automatically.
The current tools and allowlists are not widened by this producer commit.
Commercial qualification, certification, arbitrary scale, performance,
multi-GPU and all-conic coverage remain UNVERIFIED.

The independent review of historical `b525b85` found ten additional production
numerical failures. Main owns their repairs. This producer's bounded mechanism
tests do not cover or close those failures; neither `b525b85` nor `5920c7f`
is promoted to a numerical acceptance freeze. A new independently reviewed
main source is required before any CUDA dispatch/qualification.

## Commands

From the clean integrated root, CPU-only preparation uses the existing
validation environment with CUDA hidden:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python scripts/verify.py tests \
  --target tests/test_rotated_conic_cuda.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/rotated-conic-cuda/CPU_NEW
```

The test-file CLI requires `PYTHONPATH` to that same root's `src`. Its exact
collection command is `tests/test_rotated_conic_cuda.py --collect-contract
--artifact-dir EXTERNAL_NEW_DIRECTORY`. Frozen no-solve replay is
`tests/test_rotated_conic_cuda.py --replay ABSOLUTE_JSON --sha256 SHA256
--source-root TRUSTED_CURRENT_ROOT --artifact-dir EXTERNAL_NEW_DIRECTORY`.
CUDA evidence is required by default. Only CPU mechanism validation explicitly
adds `--cpu-mechanism`; it cannot produce a CUDA receipt.
Each CLI creates a new directory with acceptance.json and execution.log;
failures retain failure.log and exit 1. No run/seal is created by these CLIs.

## Preparation Receipts

External prefix:
`/hf3fs-jd/data/tmp/openopt-release-20261010/rotated-conic-cuda/`.
The actual development receipt
`mechanisms-v3/tests-bwj3mh1p/verification.json` records 30 CPU passes,
13 explicit CUDA skips, zero failures/errors and source-consistency PASS;
pytest wall time is 15.19 seconds. It records Python 3.10.20, NumPy 2.2.6,
PyTorch 2.8.0+cu128, compiled CUDA 12.8, CUDA unavailable.
`smoke-v1/smoke-eqv9ndgk/verification.json` records 80 passes/four CUDA skips.
`collection-v2/acceptance.json` records the exact thirteen selected identities;
it precedes the final additional CPU controls and is not a clean-source pin.
Clean committed-source preparation is recorded separately in `clean-mechanisms`,
`clean-smoke`, `clean-collection` and `clean-cuda-skip`. None is GPU execution.
The contract above is proposed for main/Mendel integration, not a claim that
Mendel independently signed or that the existing acceptance family supports it.
