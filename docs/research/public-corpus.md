# Offline Public MPS/QPS Import

## Scope

`scripts/public_qp_corpus.py` imports explicitly declared external local files
with the existing **highspy CPU reader**, without downloads, dependency
installation, solves, presolve, GPU submissions, or facade/core changes.
It is an importer and frozen-evidence verifier, not a public/industrial corpus
qualification or optimizer benchmark. No public corpus was acquired by this
seat. Hand-authored test MPS files are **synthetic format fixtures**, even though
they are real files consumed by HiGHS; they count as zero industrial coverage.

Caller-supplied source URLs and license statements are declarations, not
verified authority, dataset authenticity, or redistribution permission. Project
0BSD does not relicense datasets. `origin=public` is recorded as `UNVERIFIED`;
`origin=synthetic` is `SYNTHETIC_NOT_COVERAGE`. All qualification counts remain
zero pending the main coordinator's independent public-data acceptance.

## Input And Reproduction

Provide an external JSON manifest and explicit external `--local-dir`. Paths
are relative to that directory, not glob-discovered; empty/duplicate selections,
traversal, symlink escapes, missing provenance, and input hash mismatches fail.
Uncompressed ASCII `.mps` and `.qps` inputs are supported. QPS input bytes are
stored unchanged as `raw.mps` so the existing reader selects its MPS parser.

```json
{
  "schema_version": 1,
  "cases": [{
    "case_id": "caller-selected-case",
    "path": "problem.qps",
    "source_url": "https://dataset.example.org/problem.qps",
    "license": "Exact dataset license statement and attribution supplied by owner",
    "input_sha256": "REPLACE_WITH_64_LOWERCASE_HEX_DIGITS",
    "origin": "public"
  }]
}
```

URLs are metadata only and are never fetched. URLs containing userinfo, query
strings or fragments are rejected to avoid recording access credentials.
Choose a new, nonexistent output directory beneath `/hf3fs-jd/data/tmp/`.
Existing evidence is never overwritten, including failed runs.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/public_qp_corpus.py import --manifest /hf3fs-jd/data/tmp/openopt-release-20261010/public-corpus/input.json --local-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-corpus/local --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-corpus/import-v1
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/public_qp_corpus.py verify --corpus /hf3fs-jd/data/tmp/openopt-release-20261010/public-corpus/import-v1 --manifest-sha256 CALLER_PINNED_CORPUS_JSON_SHA256
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_public_qp_corpus.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-corpus/tests-v1
```

The import command prints the exact corpus manifest hash for independent
pinning. Replay is read-only. Exit 2 retains errors/tracebacks; an import failure
also leaves `corpus.json` with `ERROR`, current case, and `failure.log`.

## Semantics And Frozen Evidence

The record audit accepts one objective `N` row, `L/G/E` constraint rows, named
column pairs, one RHS set, one bounds set, `LO/UP/FX/FR/MI/PL`, and `QUADOBJ`
single-triangle or exactly symmetric full `QMATRIX`. Every dense coefficient,
coordinate name/order, row/variable bound, objective sense and offset is
compared **exactly** to the reader model before acceptance. Duplicate records,
unknown/empty sections, multiple sets, integer markers/bounds, semicontinuous
variables, SOS, QSECTION and **all RANGES constructs** are rejected. This is a
deliberately conservative subset, not silent general-format degradation.
Negative `UP` with an implicit lower bound is rejected as ambiguous; explicit
`LO/MI` is required. Reader warnings and dropped tiny coefficients fail.

The original model is `0.5*x.T@P@x + q.T@x + offset` in unchanged x coordinates.
MPS objective-row RHS is independently audited as `offset = -RHS[objective]`.
Original `A,l,u` and separate `variable_lower,variable_upper` are frozen.
`solver_arrays()` appends identity bound rows with explicit row mapping, and
normalizes maximization by sign only. There is no variable shift or scaling.
The current core has no objective-offset parameter: consumers **must** add the
returned normalized offset and multiply by `objective_sign` when reporting an
original objective. `original_objective()` computes it independently.

Convexity is checked using CPU float64 `eigvalsh` on sense-normalized P.
Every negative eigenvalue is refused, without projection or convexification.
This numerical check is not an exact PSD certificate; near-singular valid
models can conservatively fail due to roundoff. Maximization requires negative
semidefinite original P. A 128 MiB estimated dense-array budget is enforced
before allocation; `--max-dense-bytes` explicitly changes it. Large sparse
corpora are not qualified by this dense importer.

Each case contains unchanged `raw.mps`, reader log, `arrays.npz` and
`metadata.json`: input URL/license/origin/SHA, original local path, coordinates,
reader versions/options, numerical convexity scope and per-array hashes.
Canonical array SHA256 is calculated over sorted compact JSON
`{"name":NAME,"shape":[...],"dtype":"<f8"}`, NUL, then C-order little-endian
float64 bytes. Scalar offset, infinities and shapes are included. A semantic
model hash binds these hashes plus coordinate names, sense and model name.
NPZ archive bytes are separately hashed; archive hashes are integrity evidence,
not a promise of cross-runtime ZIP byte reproducibility.

`corpus.json` binds metadata hashes, the importer source hash and the unchanged
input-manifest hash. NPZ bytes/shapes/dtypes are reread before a case is frozen.
Replay requires a **caller-pinned** corpus SHA, checks all raw/archive/array
hashes and membership, rereads JSON/NPZ without pickle, and independently
re-audits original records against frozen arrays. Even self-resealed wrong
offsets/bounds/Hessians/coordinate names are rejected against raw semantics.
Hashes alone cannot prove source authenticity: an independent caller must pin
trusted source bytes and assess provenance/license separately.

## Acceptance Boundary

CPU tests generate actual MPS files only under external pytest temporary
directories. They execute highspy and real import/verify subprocesses, preserve
JSON/log/JUnit evidence, and test integer/nonconvex/ranged/unsupported inputs,
input and frozen byte corruption, self-resealed semantic corruption, empty
selection, no-overwrite, original offset/bounds, QPS extension, LP and concave
maximization. No reader substitution or optional highspy skip is used.

Public-data acquisition, license verification, representative industrial
coverage, exact PSD proof, large sparse loading, solver correctness, commercial
comparisons and performance remain **UNVERIFIED**. Main-coordinator integration
and real public-problem qualification are separate acceptance steps.

## Seat Evidence (2026-10-10)

The final targeted CPU run finished at **2026-10-10 08:55:00 UTC**:
**183 passed, zero failures/errors/skips** (62 importer tests plus 121 existing
synthetic-corpus regression tests; this is not the complete repository suite).
Authoritative report, JUnit and logs are under
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-corpus/tests-v2/tests-rk7rnbio/`.
The `verification.json` was reread with `verdict=PASS`, CUDA unavailable, and
highspy 1.15.1. The first 55-test run remains separately preserved in `tests-v1`.

Exact source SHA256 of the tested importer:
`6b328d68eba43271aa44c2ec96bde31e186d1b2a7a14488147bd41f78f22eafa`;
tests source:
`b45ae8f4643643fcbefd626c68320c0f17672ac9a22f26a29352010549ec6335`.
The post-edit CPU smoke passed **79 tests**, with four explicit CUDA-unavailable
skips; its report/log/JUnit are in
`final-smoke/smoke-c0516fi_/` under the same seat artifact root. Those CUDA paths
and the full repository suite remain unverified by this seat.

Actual CLI freeze/replay evidence is in
`tests-v2/tests-rk7rnbio/pytest-tmp/test_cli_actual_process_import0/cli/`
relative to the seat artifact root above. Its `corpus.json` SHA256 is
`6eaf2087551ceb50d53f3edbd97d65d199050a6a2360d948e5e0eb0168b3f67d`.
The synthetic input byte SHA256 is
`590524093f79a3034e23038ec8519d6d5e5bfc71b8d7064dfa719fabd66a2a6c`;
semantic model SHA256 is
`821328f7ccad9bc95e4e2b42b100cd9791e5f87b5bbbb6131befa1e954d63e01`.
`cli-verify.log` reports `VERIFIED`, case_count=1 and zero public/industrial
qualification. `cli-negative.log` retains the actual wrong-pin traceback
`frozen corpus manifest SHA256 mismatch`; that subprocess returned 2.
These are format evidence, **not public-data evidence**.

No public real-problem source has been locked or downloaded in this scope.
This seat retains the exclusive acquisition entry; before any acquisition it
must read the complete infra network rules and use only the authorized
astrill-fetch channel with aggregate bandwidth below 8 MB/s. The coordinator
owns qualification runner/integration and must not repeat parsing/downloads.
