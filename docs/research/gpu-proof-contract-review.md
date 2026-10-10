# Independent GPU Proof Contract Review

`scripts/review_gpu_proof_contract.py` reviews the producer/dispatch combination
without modifying either component, the existing auditor, or cluster state.
Only its `run` action executes the actual CPU producer. It requires a clean
checkout at the caller's full producer commit and `CUDA_VISIBLE_DEVICES=-1`.
The dispatch is read from a caller-pinned Git commit, not a moving worktree.
Only its five contract functions are loaded; module and submit/collect code
are not executed. Reviewed source, tests and producer documentation are saved
byte-for-byte with their Git object hashes.

## Gates

The actual dispatch freezes all nine expected original models before the CPU
solve. Original P/q/A/l/u and model digests, vectors, sparse storage,
scaling/CG/checkpoint data and full source snapshots remain in external
evidence. Original CPU evidence and pre-dispatch expectations are unchanged
throughout review and replay. Nine cases means nine genuine CPU solves, not
factory witnesses or GPU execution.

After production, all solver/preparation/CG, model factory, hardware, CUDA and
subprocess entrypoints are patched to fail loudly. Numeric seal/replay operates
only on saved original bytes and independently pinned expected bytes.

Four coordinated substitutions reissue changed sidecar/result/log hashes:

- CPU-as-CUDA remains `SEMANTIC_NUMERICAL_PASS` with execution `UNVERIFIED`.
  Missing independent expectations and a missing coordinator origin record
  are rejected. Bare replay is deliberately not GPU execution acceptance.
- Whole model, resident model, vectors and genuine numerical summaries are
  replaced with a simpler optimal QP. The trusted pre-dispatch model rejects it.
- A mathematically valid planted ray/base replaces terminal x/current x and
  checkpoint differences at impossible iteration 1. Frozen interval rejects it.
- Opposite terminal evidence is changed to pass alongside the selected
  unbounded direction. Independent full-summary recomputation rejects it.

The dispatch compatibility gate checks semantic collection, non-proof legacy
separation, CPU/overclaim/failed numerical/placement/exit/case/expectation
rejection, exact final completion marker/hash, duplicate/nonfinal marker
rejection, exact one-to-three selection and exact worker command including the
two expectations flags. These are contract checks, not Job-status acceptance.
The independently owned run auditor still must bind actual coordinator
collection, Job/Pod UIDs, terminal status, node, image and device UUID before
execution may become PASS. This is not remote attestation.

## Machine Reproduction

Both actions require full `--producer-commit` and `--dispatch-commit`,
`--producer-root` (clean matching HEAD), `--repo-root` (Git object source),
and external `--artifact-dir`. Use the existing validation interpreter with
`CUDA_VISIBLE_DEVICES=-1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and
`PYTHONDONTWRITEBYTECODE=1`.

`run` prints `GPU_PROOF_CONTRACT_REVIEW`, its external report, bundle and SHA256.
Use `replay --bundle <printed path> --sha256 <independently retained pin>` with
the same explicit source commits. Replay does not execute saved snapshot code;
it imports the verified clean producer checkout, forbids solving/factories,
recomputes all nine numeric cases and repeats all four mutations and dispatch
checks. Every file is inventory-pinned; added, removed or changed evidence,
source or reviewer bytes fail. Failed reviews cannot be replayed as PASS;
completed failures may reproduce the failed gate as described below.

The bounded unit tests self-check duplicate JSON, unsafe paths, source pins,
fixture denominator, wrong first causes and at least three inventory
corruptions. Real CPU results are supplied separately by the audit CLI, not
inferred from unit-test success. No speed, commercial parity, complete optimizer
capability or CUDA execution qualification is claimed.

## Loaded-Module Stop Gate

The negative control installs cached foreign `openopt.qp` and
`openopt.certificates` modules with counted replacement policy/direction
functions. They return unchanged numerical values; this isolates source-origin
binding from numerical correctness. `validate_result` must reject the foreign
loaded source, rather than merely compare current files on disk.
Actual replacement-function call counts prove the fixture entered the relevant
imports. An accepted foreign module makes the entire review FAIL even when
the nine numerical cases, four earlier regressions and dispatch contract pass.
Mendel owns the producer repair; this reviewer never changes the producer.

Failed completed reviews can be numerically reproduced, but their replay
verdict is `REPRODUCED_REVIEW_FAIL`, reviewed gate FAIL and exit code 1.
They cannot become a passing review by repinning the report. Incomplete failed
runs cannot replay. Without an independent coordinator origin anchor, all
outcomes retain execution `UNVERIFIED`.

## Observed Main Combination

Reviewed producer main merge:
`3792a86f9f5658642c5e250053a48d77c6425643`, from the clean
`/data/dev/openopt-wt-release4-20261010-frozen-r1` checkout.
Reviewed dispatch main merge:
`027ce450c278f5aab3eab13cfa70905dae118634`, read from Git objects.
This is not an acceptance of only the earlier `275cef7` / `242e0a1`
independent branch combination.

At 2026-10-10T10:11:02.648319+00:00, nine actual CPU solves and original-model
replays passed. The actual unbounded base equals terminal x,
`[0.0, 8.930516678874442]`, not a planted feasible witness.
All 25 dispatch checks passed. Model replacement, impossible planted
checkpoint, and contradictory direction summaries were rejected. Coordinated
CPU-as-CUDA passed numerical semantics only, with execution UNVERIFIED;
missing expectations and absent coordinator origin were rejected.

The overall gate is **FAIL**: foreign cached policy ran nine times and each
foreign direction checker ran twice, yet `validate_result` returned
`CPU_SMOKE_REPLAY_PASS`. The negative control reproduces source-binding P1
and does not claim a false mathematical answer. Mendel owns the repair.
Do not admit formal GPU proof until that repair has stable-source independent
acceptance. Byte-identical numerical core files alone do not waive this gate.

External report:
`/hf3fs-jd/data/tmp/openopt-release-20261010/proof-contract-review/main-v1/review-2si6pq98/review.json`
SHA256 `fcf8950ab485e6e8242b7bfd8e83d9849fd0d569aff7083d97a7305f959255b5`.
Inventory bundle, sibling `review-2si6pq98-bundle.json`,
SHA256 `e0339bccb24350bf935805ba610cbf9b2281b95d61b4e8e4ee2a58000ac8d74d`.
Independent replay at 2026-10-10T10:11:41.873703+00:00 returned
`REPRODUCED_REVIEW_FAIL`, reviewed gate FAIL and exit 1, with zero forbidden
calls. Its acceptance is under
`proof-contract-review/main-replay-v1/replay-x8813xbn/acceptance.json`
at the same external release root.
Acceptance SHA256:
`7c141ac61d7078eb4cd0ea96423fa317cb04d969036fb738ddd67385dde2f618`.

Reviewer verification at 2026-10-10T10:13:18.538522+00:00:
25 unit tests passed, no skips, source consistency PASS.
External `proof-contract-review/unit-final/tests-4b56frij/verification.json`,
SHA256 `c42e145ac10d3baae555b78db9dbffa0d9fccc6ecb9ec9a9e96223346efbbb1e`.
CPU smoke at 2026-10-10T10:13:20.807363+00:00:
80 passed and four explicit CUDA-unavailable skips, source consistency PASS.
External `proof-contract-review/smoke-final/smoke-ctr3xu12/verification.json`,
SHA256 `f24a4c0626d7076ea388432c27769ed263484a6f0ab911279386cf46f10976ad`.
These verification counts do not change the reviewed producer stop-gate.

Exact CPU replay command, from this review checkout:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/review_gpu_proof_contract.py replay --producer-root /data/dev/openopt-wt-release4-20261010-frozen-r1 --producer-commit 3792a86f9f5658642c5e250053a48d77c6425643 --repo-root /data/dev/openopt --dispatch-commit 027ce450c278f5aab3eab13cfa70905dae118634 --bundle /hf3fs-jd/data/tmp/openopt-release-20261010/proof-contract-review/main-v1/review-2si6pq98-bundle.json --sha256 e0339bccb24350bf935805ba610cbf9b2281b95d61b4e8e4ee2a58000ac8d74d --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/proof-contract-review/independent-replay
```

The expected exit is 1 for this reproduced P1; it is not a broken unit-test
suite or a GPU PASS. `run` uses the same source flags and CPU environment,
without `--bundle/--sha256`, and preserves a fresh unique evidence directory.
This review modified only its three newly assigned files; no producer,
dispatch, existing auditor, credentials or cluster objects were modified.
