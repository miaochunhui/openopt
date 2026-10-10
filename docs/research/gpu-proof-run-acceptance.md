# GPU Proof Run Acceptance

`scripts/accept_gpu_proof_run.py` is a CPU-only, fail-closed acceptance tool for
one to three distinct, single-case CUDA proof Jobs. It neither solves nor calls
model factories, CUDA, Kubernetes, or frozen snapshot code. It calls the trusted
current producer's numerical `seal` and `replay`.

## Stable Interface

The caller independently retains two pins: the pre-dispatch expectations SHA256
and the coordinator-collected `collection.json` SHA256. Reading a new hash from
the evidence being audited is not an independent origin check. Do not repin
mutated records to override a rejection.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B /data/dev/openopt-wt-release4-20261010-proof-run-audit/scripts/accept_gpu_proof_run.py seal --collection /ABS/RUN/collection.json --collection-sha256 CALLER_COLLECTION_SHA256 --expectations-sha256 PRE_DISPATCH_SHA256 --current-root /ABS/CLEAN_FROZEN_SOURCE --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/proof-run-audit/seal
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B /data/dev/openopt-wt-release4-20261010-proof-run-audit/scripts/accept_gpu_proof_run.py accept --bundle /ABS/SEAL/bundle.json --sha256 CALLER_BUNDLE_SHA256 --expectations-sha256 PRE_DISPATCH_SHA256 --current-root /ABS/CLEAN_FROZEN_SOURCE --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/proof-run-audit/accept
```

These are interface examples, not a claim that a particular GPU run has passed.
All evidence/output paths must resolve into the configured shared 3FS tmp root.
Each CLI invocation writes a new unique output directory, `acceptance.json`,
`execution.log`, and a traceback on failure. Failure returns 1, never an
UNVERIFIED status that can subsequently become PASS within the same invocation.
`seal` reports SEALED only after both execution and numerical checks pass.

The current source root must be independently trusted, clean, and at exactly
the dispatched commit/tree, including every tracked file's actual bytes. The
acceptance script may run from a different worktree. After new commits are
integrated, use a separate clean source root at the recorded dispatch commit.
Never import code from `run/source` or a producer-side evidence snapshot.

## Input Contract

`run.json` must be a SUBMITTED release-proof run. Its collection is identical
except for `collected_at`, `states`, and each Job's `result_path/result_sha256`.
All one to three requested cases must be distinct and present, with states
exactly PASS. A partial, duplicate, failed or mixed size-ladder run is rejected.

`run.source_identity` has exact fields `commit/tree/clean/sha256`; clean is true.
The all-repository source bytes, the selected `src/scripts/tests/examples`
snapshot plus the exact file `docs/research/refinement.md`, snapshot manifest,
and producer's `src/**/*.py + producer` snapshot
are checked bidirectionally. Added files, missing files, ignored modifications,
and changed coordinator bytes are rejected.

`run.expectations` is the absolute `path/sha256/bytes` descriptor for
`run/proof-expectations.json`. The SHA must equal the caller's pre-dispatch pin.
Its exact schema is:

```json
{
  "schema_version": 1,
  "source_sha256": {"src/...py": "sha256", "scripts/gpu_release_proof.py": "sha256"},
  "cases": {
    "gap_lp": {
      "model_sha256": "canonical producer model hash",
      "model": {"P": {}, "q": {}, "A": {}, "l": {}, "u": {}}
    }
  }
}
```

Each model entry uses the producer's full tensor descriptor, with component
paths relative to RUN under `arrays/`. Float64 CPU preparation is
done once before dispatch by the reviewed source-bound factory, never during
acceptance. COO/CSR original storage must match the producer's selected case;
do not convert it to dense to obtain a different canonical hash.
Acceptance loads every expected raw array, checks byte/hash/shape/dtype/layout,
recomputes canonical model hashes, and compares the saved sidecar whole model
and each tensor with this separately pinned mapping. Both producer seal/replay
receive that independently pinned expectations path and SHA explicitly.

The exact worker command is:

```text
python3 RUN/source/scripts/gpu_release_proof.py run --artifact-dir RUN/case-I --device cuda:0 --case CASE --expectations RUN/proof-expectations.json --expectations-sha256 PRE_DISPATCH_SHA256
```

Original Job, API Job template, and actual Pod must agree on that command,
source PYTHONPATH, image, execution environment, NVIDIA runtime and source
volume bindings. Only Kubernetes' service-account projected mount and
equivalent quantity/downward-API defaults are permitted. Each must request and
limit one RTX PRO 4500, use Never and backoffLimit=0, and have one container.
The actual Job must Complete/succeeded=1, failed=active=0. Exactly one owned Pod
must be Succeeded; Job/Pod UIDs, controller owner reference, namespace, node,
imageID/containerID, exitCode=0, restartCount=0 and container lifetime are bound.

Each `case-I` has exactly one immutable `cuda-proof-*/result.json`, one selected
case, explicit cuda:0 and returncode 0. The result's source map, timestamps and
actual node/Pod must agree with coordinator records. Coordinator-fetched Pod
logs must contain one runtime marker:

```text
OPENOPT_GPU_PROOF_RUNTIME {"device":"cuda:0","gpu_count":1,"gpu_name":"NVIDIA RTX PRO 4500 Blackwell","gpu_uuid":"GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx","node":"ACTUAL_NODE","pod":"ACTUAL_POD"}
```

UUID/name must match all one-card telemetry rows and result hardware. The final
nonempty Pod log line must be exactly the producer's SEMANTIC_NUMERICAL_PASS completion marker
with result path, timestamp and SHA; duplicate or trailing completion claims
are rejected. Missing independent runtime/terminal evidence is not accepted.

## Seal And Limits

The run bundle pins the complete run inventory, including original manifests,
terminal status, coordinator logs, expected arrays, source snapshots, models,
vectors, traces and every producer-side file. Per-case numerical bundles are
also byte/hash pinned. Acceptance repeats control checks and inventory checks
after numerical replay to reject observed concurrent mutation.

PASS reports separate `numerical_verdict` and `execution_verdict` and the exact
case denominator. Bare producer numerical PASS does not prove execution;
without coordinator evidence execution remains UNVERIFIED and this gate FAILS.
This is ordinary independently collected provenance, not remote attestation:
a caller who fabricates and repins every trusted input can fabricate a coherent
bundle. It does not prove commercial qualification, performance superiority,
the complete optimizer, native conic/MIP capability, or multi-GPU solver scaling.

The unit tests construct explicitly simulated Pod/GPU records and analytic
numeric solutions on CPU. Actual producer replay runs with solve/factory/CUDA
entry points forbidden. They test one/two/three denominators, coordinated
artifact relabel and whole-model replacement, terminal/log/source/array changes,
missing data and pinned-inventory tampering. Test success is CPU verifier
evidence only, never GPU execution qualification.
