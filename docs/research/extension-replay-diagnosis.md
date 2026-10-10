# Refinement Exact Replay Diagnosis

## Decision

The frozen `3792a86f9f5658642c5e250053a48d77c6425643` CUDA run has
7 passed tests, no failures/errors/skips. Current-runtime independent
original-model numerical gates pass 7/7: four accepted candidates and three
correctly rejected/unknown negative tests. Exact recorded evidence replay
passes only 4/7. Final acceptance remains FAIL. This change adds diagnosis,
not a numerical repair, seal, commercial qualification or exact certificate.

The old failed seal and all immutable GPU outputs remain untouched. A newer
auditor is not in that run snapshot and cannot retrospectively claim final
source. `diagnose-refinement` is explicitly separate from `seal`/`accept`;
its report is always `DIAGNOSTIC_ONLY`, `final_source=false`,
`qualified=false`, `certified=false`. Any exact mismatch or failed numerical
gate returns 1. Even return code zero grants no acceptance.

## Frozen Coordinates

All evidence below is under
`/hf3fs-jd/data/tmp/openopt-release-20261010/`.

- Run: `frozen-r1-gpu-refinement/release-regression-9wjw_yt4/run.json`,
  SHA256 `277659410a53dfbd639cbe5bc78dacc8d91ff61f7def5d7650a1c95bd27ff5ea`.
- Immutable result: the same run's
  `case-0/cuda-check-3rvbwpaq/result.json`,
  SHA256 `e9a69ce0fd3ba4712753deb80867564ef44975f161f71413e45b339ffba8acaf`.
- JUnit: the same check directory's `pytest.xml`,
  SHA256 `9c657481c4885748f4cfc9708d8111f378d923935188da4b5d5fa15b791a4db6`.
- Original seal failure:
  `frozen-r1-refinement-acceptance/seal-jc1pua6m/acceptance.json`.
- Detailed arithmetic experiment:
  `extension-replay-diagnosis/first-cause-v4/tests-3861p0d2/pytest-tmp/test_real_seven_sidecar_exact_0/real-seven-diagnosis.json`.

The full seven-case denominator is checked against frozen pytest
parametrization, not merely function existence. Caller run/result/JUnit
descriptors, exact property paths/hashes, distinct sidecar locations,
complete source membership and current/snapshot/loaded-checker bytes are
checked. Frozen tests are parsed, never imported, and no solve is replayed.

## All Seven Cases

All names below have the prefix `tests.test_qp_refinement::`.

| Case | Original model gates | Exact evidence | Candidate |
| --- | --- | --- | --- |
| `test_cuda_refinement_frozen_model_accuracy_ab[dtype0-factor_dtype0]` | PASS | FAIL | accepted |
| `test_cuda_refinement_frozen_model_accuracy_ab[dtype1-factor_dtype1]` | PASS | FAIL | accepted |
| `test_cuda_refinement_frozen_model_accuracy_ab[dtype2-factor_dtype2]` | PASS | PASS | accepted |
| `test_cuda_refinement_actual_residual_correction` | PASS | FAIL | accepted |
| `test_cuda_refinement_negative_controls[wrong_active_set]` | PASS | PASS | rejected |
| `test_cuda_refinement_negative_controls[singular_kkt]` | PASS | PASS | unknown |
| `test_cuda_refinement_negative_controls[wrong_sign]` | PASS | PASS | rejected |

The last three passes mean explicit rejection reasons and unchanged
baseline x/y were verified, not that those candidates became acceptable.
Accepted candidates pass the existing original feasibility, stationarity,
sign, complementarity, dual-availability and gap checks, nonworsening checks
and all three independently replayed wrong-x/wrong-y/wrong-model controls.
The correction case also retains its before-failed/after-passed and actual
decreasing linear-residual trace requirements. No original tolerance,
dtype, model or candidate was changed.

## Exact First Differences

Both float64 A/B cases have n=3, m=5, P shape (3,3), A shape (5,3).
The saved versus replayed baseline primal objective is:

```text
saved    -4.375007499974095  -0x1.18001f75092e7p+2
replayed -4.375007499974094  -0x1.18001f75092e6p+2
absolute difference 8.881784197001252e-16 (1 binary64 ULP)
```

Their baseline signed gap is respectively `-7.499971283131401e-06`
(`-0x1.f750868900000p-18`) versus `-7.499971282243223e-06`
(`-0x1.f750868800000p-18`). The same absolute difference becomes 1,048,576
ULPs at the much smaller gap magnitude through cancellation. Refined
evidence and all negative controls in these two cases agree exactly.

A read-only arithmetic experiment, retaining the same decoded inputs and
`P @ x`, shows forward sequential libm FMA dot products reproduce the saved
objective. Local NumPy matmul, nonfused sequential sum, `math.fsum` and
`np.sum` reproduce the replayed objective instead; reverse FMA also differs.
This establishes sensitivity to CPU binary64 reduction order/fusion. It
does not prove the producer used this particular FMA implementation, and
FMA is not an acceptance fallback.

The residual-correction case has n=2, m=1, P shape (2,2), A shape (1,2),
binary32 factorization and binary64 residual/candidate checks.
Baseline/refined evidence agrees exactly; ten negative-control fields differ:

| Control / Field | Saved | Replayed | Difference |
| --- | --- | --- | --- |
| wrong_model / dual_objective | 0.019039905934297857 | 0.019039905934297885 | 8 ULP |
| wrong_model / duality_gap | 0.45977011494252895 | 0.4597701149425289 | 1 ULP |
| wrong_x / primal_feasibility | 1.8999999999999995 | 1.9 | 2 ULP |
| wrong_x / primal_objective | 3.615344467640918 | 3.6153444676409183 | 1 ULP |
| wrong_x / complementarity | 1.455174470623322 | 1.4551744706233223 | 1 ULP |
| wrong_x / duality_gap | 3.655174470623322 | 3.6551744706233222 | 1 ULP |
| wrong_x / complementarity_max threshold | 2.723232925738145e-10 | 2.7232329257381453e-10 | 1 ULP |
| wrong_x / duality_gap threshold | 4.615344467640918e-10 | 4.6153444676409186e-10 | 1 ULP |
| wrong_x / primal_feasibility_max threshold | 3.2499999999999996e-10 | 3.25e-10 | 1 ULP |
| wrong_y / range_residual | 2.220446049250313e-16 | 0.0 | absolute 2^-52; not a 1-ULP difference at zero |

The binary32 A/B case and all three rejection cases replay exactly.
All seven decoded original model arrays are byte-identical between NumPy
and the producer's CPU-double-tensor decoding path; those two checker inputs
produce identical local differences. The saved candidate values are read
unchanged from the same JSON. This excludes a decoder-specific model change
or direct binary32 GPU checker as the explanation of the observed mismatch.
It does not establish the producer's unrecorded pre-serialization runtime.

## Runtime Provenance

The local interpreter is the existing
`/hf3fs-jd/data/tmp/openopt-validation-env/bin/python`, Python 3.10.20,
NumPy 2.2.6, PyTorch 2.8.0+cu128; CUDA is explicitly hidden.
NumPy's own `show_config(mode="dicts")` reports scipy-openblas 0.3.29,
USE64BITINT/DYNAMIC_ARCH/Haswell build configuration and available AVX2/FMA3.
These are NumPy build/available-feature facts, not a captured BLAS kernel
dispatch trace.

The GPU result freezes PyTorch 2.13.0+cu130 and CUDA 13.0 on h3c2. It does
NOT freeze the producer's NumPy version, package origin, BLAS/LAPACK build,
CPU dispatch, thread settings or rounding mode. The runtime uses the existing
container environment; no new inspection Job or installation was performed.
The exact pair of producer/local NumPy distributions therefore cannot be
reported from these artifacts. CPU arithmetic path sensitivity is reproduced;
a specific differing producer BLAS version remains UNDETERMINED.

## Required Separate Repair Scope

No checker change is authorized by this three-file diagnostic task.
A real repair must first specify and version one portable numerical checker
algorithm, rather than compare approximate dictionaries or select whichever
runtime matches old values. The concrete current calls to address are:

- `src/openopt/certificates.py:215`: two length-n objective dot reductions.
- `src/openopt/certificates.py:144` and `:197`: `A @ x`, `P @ x`, `A.T @ y`.
- `src/openopt/certificates.py:168`: box-support sum.
- `src/openopt/certificates.py:125`: symmetric eigenvalue/eigenvector analysis.
- `src/openopt/certificates.py:226`, `:231`, `:232`, `:236`, `:237`:
  eigenbasis projections, inverse reconstruction, range residual, dual dot
  and signed gap. Fixing only the objective dot cannot close all ten fields.

Candidate next write set, requiring explicit approval:
`src/openopt/certificates.py`, `tests/test_qp_certificates.py`,
`tests/test_qp_refinement.py`, `scripts/accept_extension_cuda.py`,
`tests/test_accept_extension_cuda.py`, and one versioned checker-design doc.
The producer's existing CPU frozen checker calls in
`tests/test_qp_refinement.py:73`, `:77`, `:83`, `:87`, `:91` must retain full
original checks and freeze the algorithm version plus NumPy/BLAS/runtime
provenance. If the changed checker affects refinement's in-process evidence,
`src/openopt/refinement.py` and its explicit source contract require separate
review; do not silently expand that scope.

New same-source GPU execution, fresh sidecars, original complete positive
and negative gates, exact replay in both runtimes and strict final-source
seal/accept are required. Old values/results remain historical failures;
they cannot be rewritten or accepted using the new checker.

## Validation And Reproduction

The pre-document comprehensive targeted run is
`extension-replay-diagnosis/unit-v1/tests-ebwjxrxf/verification.json`:
96 passed, no skips/failures/errors, source consistency PASS.
The final run including the caller-JUnit negative control and optional
external-artifact availability handling is
`extension-replay-diagnosis/final-unit/tests-9nu_b1g6/verification.json`:
98 passed, no skips/failures/errors, 215.67 seconds pytest wall time, source
consistency PASS. Final smoke at
`extension-replay-diagnosis/final-smoke/smoke-uv7lwld4/verification.json`
has 80 passed and four explicit CUDA-unavailable skips, source consistency
PASS. This document-only receipt update does not change tested Python bytes.
Real-sidecar tests explicitly skip if these historical external artifacts
are absent; such a skip is never CUDA or replay qualification.

Run the diagnostic only from a clean committed source tree, with these three
caller pins. The diagnostic reports all seven cases even when exact evidence
differs, writes `acceptance.json`, `failure.log`, `execution.log` in a fresh
external directory, and remains red for this real run:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/accept_extension_cuda.py diagnose-refinement --current-root "$PWD" --run /hf3fs-jd/data/tmp/openopt-release-20261010/frozen-r1-gpu-refinement/release-regression-9wjw_yt4/run.json --run-sha256 277659410a53dfbd639cbe5bc78dacc8d91ff61f7def5d7650a1c95bd27ff5ea --result /hf3fs-jd/data/tmp/openopt-release-20261010/frozen-r1-gpu-refinement/release-regression-9wjw_yt4/case-0/cuda-check-3rvbwpaq/result.json --result-sha256 e9a69ce0fd3ba4712753deb80867564ef44975f161f71413e45b339ffba8acaf --junit /hf3fs-jd/data/tmp/openopt-release-20261010/frozen-r1-gpu-refinement/release-regression-9wjw_yt4/case-0/cuda-check-3rvbwpaq/pytest.xml --junit-sha256 9c657481c4885748f4cfc9708d8111f378d923935188da4b5d5fa15b791a4db6 --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/extension-replay-diagnosis/real-cli
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py --target tests/test_accept_extension_cuda.py --target tests/test_accept_release_cuda.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/extension-replay-diagnosis/final-unit
```

Exact evidence negative controls cover 1 ULP mutation, bool/numeric
substitution, missing fields and false status. The caller-JUnit negative
has identical bytes and a valid hash but the wrong path; it must be rejected.
Existing source-binding, denominator, tamper and no-retrospective-promotion
tests remain active. These tests validate the diagnostic and strict gate,
not a repaired cross-runtime numerical algorithm.
