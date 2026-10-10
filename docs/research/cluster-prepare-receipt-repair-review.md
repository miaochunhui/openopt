# Cluster Archived Receipt Repair Review

Subject: `8beb9313c3759d68b2c2fbcde4da3d6f29418dae`, tree
`9965ac0d3ee59f3f199799c3ab1f1bbbe93f558f`.
Read-only subject root:
`/data/dev/openopt-wt-release4-20261010-cluster-prepare-receipt`.
Independent tree:
`/data/dev/openopt-wt-release4-20261010-cluster-repair-independent`.

The only review writes are this document and the unchanged test copied from
Nash `3cd55160f0aed9244adf492c11e9cdab04f0fa4d`.
Test SHA256 is
`cd0b13a2a2afbcafcdcfc2f9a7415b26731e6f5ce44446dede7383de6e5ea5a6`.
Its four negative assertions, positive baseline and fixture builders are
byte-identical. No production, original Nash worktree, main, infrastructure,
credentials, installed dependencies or cluster objects were modified.

## Actual Findings

The frozen subject CLI replay passed on 2026-10-10, ending
`2026-10-10T12:37:41.514668+00:00`, rc0, 34.559467 seconds:

- Positive: ACCEPTED.
- Coordinated extra args: REJECTED, `archived_manifest_config_drift`.
- Coordinated changed PYTEST_ADDOPTS: REJECTED,
  `archived_manifest_config_drift`.
- Coordinated extra environment member: REJECTED,
  `archived_manifest_config_drift`.
- Actual Job/Pod extra args with unchanged submitted manifest: REJECTED,
  `cpu_execution_args_drift`.

Every case first passes its unchanged positive baseline. Every manifest,
Job/Pod and state hash is recomputed; referenced archive hashes are valid.
Config and saved baseline bytes remain unchanged. Rebuilt intended manifest
differs for the three coordinated mutations and remains equal for the
actual-only mutation. Thus rejection is semantic, not a stale-hash artifact.
All tiny fixture source snapshots are clean and use the pinned real producer
scripts, not mocks of the acceptance functions.

CLI receipt:
`/hf3fs-jd/data/tmp/openopt-release-20261010/cluster-prepare-receipt-repair-review/cli-frozen/coordinated-kyj5wltc/acceptance.json`.
The five individual case reports and their pins are linked inside it.
The external execution log is pinned with SHA256
`4f366dd0bb16499f8141fdc48cab2ca2a759c6dbb3cd03130cfba2db99f13813`.
Mechanism preflight independently passed four tests, zero skip/failure,
0.40 pytest seconds:
`cluster-prepare-receipt-repair-review/mechanism-preflight/tests-ruzd5t7q/verification.json`.
That preflight includes the new untracked test, not clean review qualification.

## Production Binding

Full changed producer diff and relevant call paths were read.
`manifest` reconstructs command and pinned plan/runtime/report arguments
from caller-pinned config; CPU environment membership and values are generated
from the known key list. `intended_manifest` compares the archived stage
name, phase, id, output and manifest path to the regenerated stage item, then
requires the rehashed archived manifest to equal the generated manifest.
Both prior accepted-stage replay and active collect use this binding.

`terminal_job` compares command, args, env and envFrom exactly for both
observed Job and Pod. Only absent versus an explicit empty list is equivalent;
null, extra arguments or environment entries are not. Existing subset checks
for unrelated Kubernetes defaulting, resource equality, UID, resolved image,
restart/exit state and original source/JUnit gates remain in place.
No deadline, shard concurrency or numerical criterion was relaxed.

Frozen producer script SHA256:
`8aee150a3fcb1f147d09f72d3af5fb6faf1cf2290e988227e8e47d2d3d64ff0f`.
Verifier SHA256:
`4b90e4962ebc4e96697744f04e0f91dc1ff9b50a5634a8f24e323cfe5ebc03b5`.
Sharder SHA256:
`f65c3aa6b3f90f95d4165cf672e93ffe3f8b73a5a69fe37324671e69897d12fb`.

## Clean Acceptance Entry

From the independent tree, run the unchanged opt-in nine-node pytest arm:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENOPT_RUN_CLUSTER_RECEIPT_EVIDENCE_TESTS=1 OPENOPT_CLUSTER_RECEIPT_REVIEW_COMMIT=8beb9313c3759d68b2c2fbcde4da3d6f29418dae timeout --kill-after=5 170 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_cluster_prepare_receipt_independent.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/cluster-prepare-receipt-repair-review/clean-final
```

Integration requires actual clean review source, source consistency PASS,
exactly nine JUnit identities, zero failures/errors/skips and unchanged test
SHA. Timeout is a failure, not an accepted partial run. `clean-final` receipts
must be physically read; the CLI rc0 alone is insufficient. The subject commit
in every case remains 8beb931, while the outer verifier binds the new review
commit and complete source bytes.

## Retained Boundaries

Old ba675eba and 9c46c8e CLI false accepts remain unchanged under
`cluster-prepare-receipt-review/baseline/coordinated-nexjwggh` and
`cluster-prepare-receipt-review/frozen-9c46/coordinated-irvu_z4v`.
The old explicit JUnit receipt
`cluster-prepare-receipt-review/frozen-9c46-junit/tests-o00d5p7f`
remains five passed/four failed/zero skipped, not repaired evidence.

This review tests synthetic CPU archives through real production acceptance.
It performs no Kubernetes query/create, actual container/runtime probe,
optimizer numerical solve, GPU execution, installation or speed measurement.
It closes only the four unchanged archive substitution counterexamples on
the stated source. Full v1/v2 worker chains, fresh real cluster admission,
container/runtime and final-source full CPU regression remain separate gates.
