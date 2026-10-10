# Independent RSOC Extreme Acceptance Review

## Verdict And Boundary

The independent gate review is FAIL, not GPU numerical FAIL. The actual
76-node archived run replays successfully; altered observed execution records
still receive `RSOC_EXTREMES_REPLAY_PASS` on the reviewed envFrom repair.
No producer, algorithm, threshold, main tree, existing auditor or cluster
resource was modified. Fixed thirteen-node evidence remains a separate account.

Independent worktree:
`/data/dev/openopt-wt-release4-20261010-rsoc-extremes-independent`.
Base commit: `acc199306936922b6d75330884310a58cde84a9b`.
Only this document and `tests/test_rsoc_extremes_acceptance_independent.py`
belong to this review write set.

The selected gate root is
`/data/dev/openopt-wt-release4-20261010-extremes-acceptance`.
Reviewed gate SHA256 after owner envFrom repair:
`217652d2b58675d349c380a326430a05103331f6c3743a747280036a8abe1c62`.
Read-only helper SHA256:
`9980fe81a4d5e8c85546f846c7f4a1a3e75dbf4205a12e515721fb111f3f84bc`.
The first two runs captured uncommitted gate bytes. The owner subsequently
froze identical bytes as clean commit
`6ab87e96b869506039af0b9e6f4c09066d5b8363`; this was checked independently.
The final runner requires that clean caller-selected commit as well as the
two byte pins and saves an external snapshot before loading.

## Concrete Counterexamples

On the repaired gate, each following single status mutation passes the complete
76-node acceptance path, despite differing from the submitted manifest:

- `pod-workingDir`: introduce `/unsubmitted-python-shadow` as working directory.
- `pod-source-mount`: change the actual `threefs` mount path to `/unsubmitted`.
- `pod-source-volume`: replace the actual `threefs` PVC with an empty directory.
- `pod-placement`: replace actual RTX PRO 4500 product selection with NVIDIA-A100.
- `pod-namespace`: move the observed Pod identity into `foreign-namespace`.
- `owner-kind`: replace the actual Pod controller reference kind with ConfigMap.
- `image-invalid-digest`: replace resolved image identity with `foreign@sha256:x`.
- `exit-bool`: replace terminal integer zero exitCode with JSON false.
- `restart-bool`: replace integer zero restartCount with JSON false.

The first three are P1 execution/source-binding gaps: unchanged command and
environment strings cannot bind the files resolved through a different volume,
mount or Python working directory. The placement/identity/image/schema gaps
also prevent an exact execution signoff. The status gate compared command,
image, environment and resource dictionaries, but not those configuration
fields; Python equality allows `False == 0` and a substring is not a digest.
The owner must define allowed Kubernetes defaults and compare relevant actual
Job/Pod configuration against the submitted contract without loosening gates.

The initial gate SHA256
`3830cc2062086bd69b91749601429d94a862b1f106f89cf3ade736e8346f719e`
also accepted actual Pod `envFrom`. The repaired SHA rejects both independently
tested actual Job and actual Pod envFrom injections. That closure is specific;
it does not close the other nine counterexamples.

## Machine Evidence

Original run, never modified:
`/hf3fs-jd/data/tmp/openopt-release-20261010/owner-acc1993-extremes-cuda-v1/release-regression-o4xde6y1`.
The test hardcodes independent run/result/JUnit/status/collection pins and
checks them before fixture copying. No factory or solve is used. The real
baseline replays all 76 sidecars; import controls reject Torch, solver, test
factory and GPU producer imports during acceptance.

Mutations use one module-scoped relocated copy with byte-preserved snapshot
sources. Absolute evidence paths are mechanically relocated. Variant status
pins are deliberately recomputed to exercise semantic binding rather than
merely demonstrate that a checksum changes. These are synthetic mutations of
real frozen artifacts, not new CUDA executions or independent origin evidence.
Original artifacts remain unchanged. Every variant and observed verdict is
external; hard failures remain failures, with no xfail/skip or relaxed oracle.

Initial bounded selector: 1 PASS, 10 FAIL, zero errors/skips, pytest 31.87s.
Receipt:
`/hf3fs-jd/data/tmp/openopt-release-20261010/rsoc-extremes-independent/initial-v1/review-ic5kd0_g/receipt.json`.
Receipt SHA256:
`f936d336df4b2b451ae5556694833955b931db040f192d943b1ccca10a04d531`.

EnvFrom repair wholefile: 19 PASS, 9 FAIL, zero errors/skips, pytest 36.21s.
Receipt:
`/hf3fs-jd/data/tmp/openopt-release-20261010/rsoc-extremes-independent/envfrom-v2-full/review-8phzi1g3/receipt.json`.
Receipt SHA256:
`c423e17e8356509a2f36f03f6078b6415608e13da73d87b9b51988490c3e9d0b`.
This run captures the reviewed gate/helper bytes. Rejection controls passed for
actual Job/Pod args/env, two-card drift, owner UID drift, nonzero exit/restart,
snapshot/current source edits, lost representable heads, original-input
substitution, CPU device labels and incorrect opposite projection.

## Rerun

From the independent worktree, use the existing environment:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python \
  tests/test_rsoc_extremes_acceptance_independent.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/rsoc-extremes-independent/frozen-rerun
```

The exact independent denominator is 28 pytest nodes, including actual baseline
and controls, not 28 fresh GPU executions. Review runner receipts record source
state, original pins, input snapshot bytes, logs, JUnit and observation hashes.
For a later owner freeze, pass explicit `--target-root`, `--target-sha256`
and `--target-commit`;
helper drift requires separate review, not a silent fallback.
Explicit isolated module loading selects caller-reviewed tool bytes; it does
not prove a production loaded-source guard rejected cached foreign modules.
Commercial, speedup, certificate and remote-attestation claims remain absent.
