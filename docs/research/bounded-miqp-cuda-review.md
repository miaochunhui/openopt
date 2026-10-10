# Bounded MIQP CUDA Original Replay

The selected source is clean `af1ef6d9993357d3647170246fe3d8192392a633`.
This review does not change numerical algorithms, parameters or benchmarks.
The only production change resolves shared-directory aliases in the read-only
campaign live result lookup and rejects missing or duplicate Job/Pod matches.
An alias resolves to the same actual directory, not to a substitute result.

The one-card run is under the external release root
`/hf3fs-jd/data/tmp/openopt-release-20261010/`:
`resume-chair-af1ef6d-miqp-cuda-v2-existing-channel/release-regression-ryxdsfq2`.
Job `openopt-release-regression-ryxdsfq2-0` ran on h3c2, RTX PRO 4500.
The immutable CUDA result finished at `2026-10-10T13:22:27.130556+00:00`:
one actual selected test, zero failure/error/skip, worker wall
13.29080133792013 seconds, pytest suite wall 8.257 seconds,
exit code zero and no restart. The initial observer-channel Forbidden is
retained separately; no Job was created by that failed attempt.

The original model is the independently fixed scalar polynomial `x*x-x`,
integer `x in {0,1,2}`. The independent Fraction oracle checks original
arrays, signed tangent/support lower bounds, branch coverage, candidates,
pruning, final status and gap. It does not import the producer checker or
trust the saved descriptor as the original model. The proof SHA256 is
`824c8785e3ed8f8ecfd61c5d5df306bf1664cbdc2fa1274ee29ce678a6e11266`.
The replay also checks exact selected numerical source bytes, actual JUnit
identity and a caller-pinned proof, before/after input bytes and original
result agreement. It does not grant remote execution attestation.

Development readback at `2026-10-10T13:30:51.623625+00:00`:
`chair-cuda-review-real-v1/replay-0lrwqia7/independent.json`, PASS;
three visited nodes, two QP proposals, two final covered regions, objective
and signed gap both `0/1`. The CUPTI trace has 1,414 CUDA kernel events,
summed duration 2228.5179999999914 microseconds. This is an instrumented
tiny regression, not Nsight profiling, an end-to-end speedup, a hardware
utilization qualification or a multi-GPU solver. Execution and performance
proof verdicts deliberately remain UNVERIFIED.

The initial live read using the `/hf3fs-jd` alias failed with StopIteration
before a snapshot was written. The run records used the canonical `/3fs`
path. Only the lookup changed; no source/run/result/cluster object changed.
The changed-code live read at `2026-10-10T13:30:49.564401+00:00` is
`resume-chair-miqp-live-v2-alias-repair/live-133049564401.json`, PASS:
four Ready nodes, zero active project Pods, no cordons, API ready, and
three separately attributed historical failures retained.

Unit development receipt
`chair-cuda-review-unit-v1/tests-mbmcl0io/verification.json` records
18 passes, zero skip/failure/error, 5.22 pytest seconds. The pre-edit smoke
has 80 passes and four explicit local CUDA skips, 8.78 pytest seconds.
Clean freeze and independent code review are separate from these
development receipts; neither is implied by this document.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_branch_bound_cuda_independent.py --run EXTERNAL_RUN --proof-sha256 CALLER_PROOF_SHA256 --artifact-dir NEW_EXTERNAL_3FS_ROOT
```

General MIQP, conic/nonlinear/global scope, licensed commercial same-model
comparison, distribution-wide regression and authorized online readback
remain unfinished. A finite exact-DD subset and a tiny original replay
cannot establish universal superiority.
