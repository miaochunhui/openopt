# Bounded MIQP CUDA Original Replay

## Explicit V2 Reader Repair

The current reader adds `audit_v2(run, manifest, expected_pin)`. It is a
CPU-only research archive reader, not a production solver or remote execution
attestation. No cluster operation, installation, algorithm or export changes
are part of this repair. Only the reader, its new contract tests and this
document are changed. The earlier independent `b55aa30` suite and readonly
Fraction oracle remain byte-unchanged.

The original independent final receipt remains FAIL at
`bounded-cuda-reader-independent/clean-final-v1/tests-jaqrw8pq/independent-review.json`
under the external release root: 56 nodes, 43 pass, 13 fail, zero skip.
The first actual false return was boolean numerical enclosures. The thirteen
also include run/collection/Job, result node/selected-test-source, JUnit count
and unrelated/changed trace acceptance. Neither that archive nor its FAIL
receipt is backfilled with new origin authority or silently reclassified.
Original final receipt SHA256:
`b499296c59e02227edbfb96d7a8ee701e6aae4a07cc0b0fd19853ba28655b84e`.

V1 keeps its historical proof-only replay path, but successful output is now
`LEGACY_NUMERICAL_ONLY`, `archive_binding_verdict=UNVERIFIED`. The corrected
nonbool enclosure validation is shared. V1 has no v2 authority and cannot
serve as a positive v2 control. Its legacy metadata limitations remain
disclosed; the old independent suite cannot be repinned to manufacture green.

### Caller Manifest Contract

The caller supplies a manifest path and its full lowercase SHA256, selected
before replay. `audit_v2` never creates or regenerates this trusted pin. A
changed manifest, even with coordinated rewritten evidence descriptors,
fails under the old caller pin. The exact schema has no optional/default
members:

```json
{
  "schema_version": 2,
  "authority": "caller-selected-archive-only",
  "run_root": "/absolute/canonical/selected/run",
  "target": "tests/test_branch_bound.py::test_cuda_frozen_bound_tree",
  "source_commit": "af1ef6d9993357d3647170246fe3d8192392a633",
  "source_tree": "0dee304d077d76e0cc3b65cb037b5ee993365681",
  "job_name": "caller-selected-job",
  "node_name": "caller-selected-node",
  "pod_name": "caller-selected-pod",
  "files": {"ROLE": {"path": "relative/path", "sha256": "FULL_SHA256", "bytes": 1}},
  "selected_source": {"SOURCE": {"path": "source/relative/path", "sha256": "FULL_SHA256", "bytes": 1}}
}
```

`ROLE` stands for ALL nine required distinct members: `run`, `collection`,
`job`, `status`, `result`, `side_result`, `junit`, `proof`, `trace`.
`SOURCE` stands for ALL three selected source members:
`src/openopt/branch_bound.py`, `src/openopt/qp.py`,
`tests/test_branch_bound.py`. This is selected-source membership, NOT
whole-repository or loaded-runtime-source verification. Every descriptor
requires a positive nonbool integer byte count (at most 64 MiB), canonical
relative path and full SHA256. Files and child directories open no-follow;
duplicate paths, missing/extra members, duplicate JSON keys, NaN, empty,
nonregular, oversized, escaping or changed files fail loudly. The run root
may use the existing shared-directory alias; it must canonicalize to the
explicit manifest root.

After pin checking, the reader independently validates run/collection
source commit/tree/maps (the dispatched source map is a validated subset of
the full snapshot) and one identical size-one Job selection, canonical result
SHA and duplicate-result bytes; archived Job kind/name/namespace/completion,
single Pod controller UID, node, host, exit and restarts; exact pytest suite
name/counts/host, testcase source/name and proof/trace properties; and the
three selected source bytes. It replays the independently fixed original
polynomial and configuration through the readonly host Fraction oracle,
checking original arrays, exact-DD PSD, signed bounds, coverage, candidate
integrality, signed gap and status. Enclosures require actual finite numerical
scalars, never bool/string coercions, and contain the exact rational values.
The readonly oracle itself must match its independently frozen SHA256 before
mathematical replay. No artifact source is loaded as executable code.

Success is `ARCHIVE_NUMERICAL_PASS`, with numerical/archive binding PASS but
execution, performance and commercial verdicts UNVERIFIED. A caller who
explicitly pins a different legal-looking trace has only selected those
bytes; the reader cannot infer which process emitted it. Two positive
controls demonstrate that limitation instead of calling it GPU verified.
No remote-origin, execution, loaded-source or speedup guarantee is added.

### Acceptance Entry

Default contract tests are self-contained, no solves/factory/cluster reads.
They first freeze an explicit v2 synthetic archive, then change proof,
metadata or trace while coordinating attacker-owned summaries and hashes;
the original caller manifest remains unchanged and must reject every mutant.
A second set repins invalid content explicitly to test semantic rejection,
not merely hashing. A genuine v2 positive, CLI completion, v1 downgrade,
schema, enclosure and symlink controls are all separate. The actual historical
archive test is opt-in with all three caller fields; missing partial selection
fails loudly and default skip does not count as historical qualification.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_branch_bound_cuda_independent.py --target tests/test_bounded_cuda_reader_contract.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/bounded-cuda-reader-repair/clean-final-v2
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_bounded_cuda_reader_contract.py --audit-artifacts EXTERNAL_VERIFICATION_DIRECTORY
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_branch_bound_cuda_independent.py --run EXTERNAL_RUN --evidence-manifest CALLER_MANIFEST --evidence-manifest-sha256 CALLER_FULL_SHA256 --artifact-dir NEW_EXTERNAL_3FS_ROOT
```

Actual archive selection uses `OPENOPT_BOUNDED_READER_RUN`,
`OPENOPT_BOUNDED_READER_MANIFEST`, `OPENOPT_BOUNDED_READER_MANIFEST_SHA256`.
No manifest is automatically produced for an old archive by this reader.
Author acceptance checks actual collection/JUnit membership, observations,
false accepts and clean source before/after in `reader-contract.json`.
Independent review of the new clean freeze remains required.

Preflight: `bounded-cuda-reader-repair/preflight-v1/smoke-py8mc0ws`, 80 pass,
four explicit local CUDA skips, 6.72 seconds. First development full contract:
`bounded-cuda-reader-repair/development-v1/tests-j42ecs00`, 118 pass, one
unselected historical-archive skip, 60.74 seconds. That run preceded the
v1 compatibility/downgrade revision and is not the final clean acceptance.
The v1 revision replay at `development-v2/tests-dp2qjr0k` passed the same
118 tests/one unselected-archive skip in 54.34 seconds. Live read of the
historical input schema then identified its `size=1` Job field and source-map
subset contract, rather than the initial synthetic `case=0`/equal-map fixture.
Both fixture and reader now use the actual structural contract; negative
controls cover bool size and coordinated source-subset mismatch. No old
archive pin, bytes or verdict was rewritten.
Schema-aligned development run `development-v3/tests-i5qef501`: 123 pass,
one unselected historical-archive skip, 52.93 seconds. Separate final CLI
failure/trace-overflow controls at `last-controls-v1/tests-2bria5fw`: two pass,
4.84 seconds. Read the clean final `reader-contract.json`, not these dirty
development receipts, for author acceptance and exact source/membership.

The optional local `--scan-sources EXTERNAL_DIRECTORY` release helper uses
the existing readonly infra credential scanner and compares a local needle
only in memory, without displaying any value. It is not a portable default
pytest dependency and does not modify credentials or infra. The source scan
at `security-v1/source-security-scan.json` measured three files, zero literal
and zero shape hits. New security helper bytes are part of the final source
snapshot; security results do not establish mathematical or GPU qualification.

## Historical V1 Evidence (Not V2 Acceptance)

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
# Historical v1 CLI syntax only; current CLI requires the explicit v2 fields above.
```

General MIQP, conic/nonlinear/global scope, licensed commercial same-model
comparison, distribution-wide regression and authorized online readback
remain unfinished. A finite exact-DD subset and a tiny original replay
cannot establish universal superiority.
