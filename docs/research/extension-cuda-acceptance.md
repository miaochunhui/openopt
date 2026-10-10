# Extension CUDA Acceptance

This is a CPU-only audit of frozen single-target CUDA regression execution and
original-coordinate numerical candidates. It is not a commercial qualification,
exact certificate, general solver capability claim or performance comparison.
The implementation is independent 0BSD code and does not import frozen tests or
call any solver. Existing `openopt.conic.check_conic_candidate` and
`openopt.certificates.check_qp_candidate` supply the independent numerical gates.

## Supported Denominators

The contract is versioned against the native/refinement tests integrated at
`c451dcf`, with each dispatch using the existing `release-regression` stage and
exactly one target:

| Suite | Target | Exact selection | Selected | Actual CUDA tests | CPU rejection tests |
| --- | --- | --- | ---: | ---: | ---: |
| native_conic | tests/test_native_conic.py | cuda | 21 | 19 | 2 |
| refinement | tests/test_qp_refinement.py | cuda_refinement | 7 | 7 | 0 |

Native includes 16 model/dtype positives, three CUDA numerical tamper controls
and two CPU-only explicit CUDA request rejection controls. The latter never
claim tensor placement or receive a CUDA candidate sidecar.

Refinement includes three dtype/factor A/B positives, one actual residual
correction positive and three wrong-active-set/singular-KKT/wrong-sign negatives.
`-k cuda` on refinement selects eight tests, not seven, and is deliberately
rejected by this seven-case contract. Use `-k cuda_refinement`.

The auditor parses, but never executes, frozen test AST. It derives complete
pytest parameter IDs using a restricted literal/torch-dtype decoder and checks
the exact function, parameter-domain and count contract. Dynamic IDs,
factories, unknown decorators, added/removed selected functions and parameter
substitutions require a reviewed contract update. Observed JUnit identities must
equal the complete expected set, not merely name existing functions. Zero
failure/error/skip and all existing execution/log/telemetry gates remain required.

## Binding And Replay

Both `seal` and `accept` require explicit caller-pinned absolute paths and SHA256
for immutable result, run and JUnit. `accept` additionally requires the extension
bundle path and SHA256. A coordinator result mirror is not accepted. All
descriptors bind path, SHA256 and bytes; existing execution validation checks
snapshot/source-manifest/run/job/result/JUnit/log/telemetry consistency.

Every real CUDA testcase must have exactly its family-specific two JUnit
properties: `<family>_acceptance_path` and `<family>_acceptance_sha256`. They bind
one unique sidecar inside that result's `pytest-tmp`, matching the execution
bundle proof descriptor. Aliases, escaped paths, shared sidecars, unknown
properties, missing pins and incomplete positive/negative denominators fail.

Sidecar source membership is exact, every source matches current and run
snapshot bytes, and normal source paths also match the run manifest. Current
Git source must be entirely clean and required paths tracked at HEAD. Loaded
checker bytes must match current/run scope sources, preventing a different
installed checker from quietly accepting a frozen model. Unrelated current core
differences are explicitly reported, not quietly promoted to full-core PASS.
The run commit is informational: the authoritative comparison is actual scoped
bytes, allowing later unrelated integrations without disguising any scope drift.
The auditor has its own clean committed source tree and separately sealed hashes.
Default final-source seal/accept additionally requires both identical auditor
files in the run snapshot. An auditor absent at execution time cannot
retrospectively promote a historical run to final-source acceptance.

Refinement additionally needs `source/docs/research/refinement.md`. When the
execution snapshot manifest narrowly includes this exact document, it must
match that manifest, run sources and actual snapshot bytes. For a legacy
manifest excluding the document, only explicit development replay may bind it
as a supplemental snapshot descriptor in the extension bundle. Final-source
requires the document in the run manifest. Neither path allows all docs or
silently edits the legacy manifest. The producer must
copy it into the snapshot before dispatch because its test hashes the document.
Missing supplemental document fails; do not fabricate it after execution.
Both auditor sources must be clean and committed in the auditor's tree.
Main owns producer/snapshot integration and GPU dispatch.

Native replay decodes full P/q/A/b/cones/x/s/z and checks independent original
KKT/gap/evidence/objective, dtype and placement/config consistency. Empty A
decodes to `(0,n)`. For its three numerical tamper tests, the auditor also seals
the actual sibling `tampered.json`, requires it equal exactly the frozen test's
single-vector `+1` mutation and independently verifies its rejection. Neither a
saved PASS nor a negative test name alone substitutes for this negative replay.

Refinement replay decodes full P/q/A/l/u, baseline/refined x/y and tolerances,
rechecks both saved evidence objects and all three saved numerical controls.
An accepted candidate must pass original KKT/gap and not worsen any of the five
original metrics. A rejected/unknown candidate is a legitimate negative test
outcome when baseline values are retained and the expected explicit reason is
present. The singular case must be unknown; wrong active set/sign must be
rejected. Do not require each candidate to be accepted, or each mutation of an
already rejected baseline to fail. The correction case additionally requires
the frozen mixed-precision correction trace; trace itself is producer evidence,
not a replay of linear algebra or a speed measurement.

## Stable CLI

Use the existing CPU validation environment with `CUDA_VISIBLE_DEVICES=-1`.
Run from a clean integrated source tree. The following placeholders are explicit
caller evidence coordinates, not auto-selected latest files:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python \
  /data/dev/openopt/scripts/accept_extension_cuda.py seal \
  --suite native_conic \
  --result /hf3fs-jd/data/tmp/RUN/case-0/cuda-check-ID/result.json \
  --result-sha256 RESULT_SHA256 \
  --run /hf3fs-jd/data/tmp/RUN/run.json --run-sha256 RUN_SHA256 \
  --junit /hf3fs-jd/data/tmp/RUN/case-0/cuda-check-ID/pytest.xml \
  --junit-sha256 JUNIT_SHA256 \
  --current-root /data/dev/openopt \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/extension-audit/seal
```

For refinement use `--suite refinement`. `seal` prints a pinned bundle
coordinate. `accept` uses the same result/run/JUnit arguments and replaces
`seal --suite native_conic` with:

```text
accept --bundle /hf3fs-jd/data/tmp/.../seal-ID/bundle.json --sha256 BUNDLE_SHA256
```

Each invocation creates a new external output directory containing
`acceptance.json` and `execution.log`; failure also retains first-cause traceback
in `failure.log` and exits 1. Successful seal is `SEALED`, successful execution
and numerical replay is `PASS`, while `commercial=UNVERIFIED`,
`certified=false`, `qualified=false`, `performance=UNVERIFIED` remain explicit.
No K8s operation, installation, credentials, push or solve is performed.

For historical successful native runs used during tool development only,
both invocations may explicitly use `--development-replay`. Reports then contain
`final_source=false` and `development_replay=true`; this cannot be consumed as
same-source final acceptance. Omission of that flag on either invocation fails
for a development bundle. Failed execution is rejected even in development
mode; missing refinement documents cannot be reconstructed after the run.

## CPU Validation

`tests/test_accept_extension_cuda.py` creates real clean Git repositories and
complete synthetic run/job/result/JUnit/sidecar fixtures under external pytest
basetemp. Fixture source includes a top-level exception, proving frozen test
modules must not be imported. Synthetic telemetry is not actual GPU evidence.

Negative controls include independently rehashed x, multiplier, original-model
and tolerance mutations; source membership/placement drift; exact property
pin and shared-path substitution; missing negative testcase with all counts,
logs and caller pins resealed; dynamic/same-count parameter changes; current
dirty source; supplemental document drift; and missing/mutated native negative
artifacts. Public CLI seal/accept PASS and caller-pin FAIL retain real report
files and exit codes. GPU results remain main's independent integration task.
