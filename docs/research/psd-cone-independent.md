# Independent Bounded PSD Review

Owner freeze: `2278294820a133c23d4f3fa632bd51ec24789463`.
Owner: `/data/dev/openopt-wt-release4-20261010-native-psd`.
Reviewer: `/data/dev/openopt-wt-release4-20261010-psd-cone-independent`,
branch `codex/release4-20261010-psd-cone-independent`, from that owner commit.
Only this document and `tests/test_psd_cone_independent.py` may change.
Production, owner mathematics/tests, main and all old archives remain read-only;
this task does not access historical GPU archives at all.
No permission helper/check mode, root, chmod, install, cluster, GPU, push or
nested agent is used. New evidence uses Python tempfile in an existing ordinary
infra-writable parent. A failed write must stop, never trigger permission repair.

## Independent Arithmetic

All represented matrix coordinates are decoded directly as Fractions.
Determinants use a signed-permutation subset dynamic program, not the owner's
pivot elimination, Torch spectrum/determinants, a solve or a production oracle.
Every nonempty principal subset is enumerated in size/index order for orders
1 through 6, respectively 1, 3, 7, 15, 31, 63 minors per matrix.
Original and component-locally shifted minors are compared exactly, including
singular PSD, zero, dense indefinite and nonleading-negative matrices.

With G=Y-X, original Fraction coordinates independently recompute YG, trace(YG),
the documented component-local absolute/relative limits and every check bit.
No svec roundtrip is used for matrix acceptance. Complete minor signs, exact/
numerical labels, component ratios and trace ratios must match actual evidence.
Analytic integer Householder matrices give exact PSD Y/G and exact YG=0 for
all 1..6 orders. Actual native CPU proposals are checked separately against
these analytic matrices and unchanged original numerical gates.
Rounded-negative-minor numerical PASS must remain explicitly not exact PSD;
strict zero-tolerance inspection must reject it.

Controls include wrong zero/identity/residual sign, extra positivity,
nonleading negative minors, huge/small mixtures, signed-product overflow/
underflow, native spectrum corruption/error, original asymmetric/unsupported
inputs, boolean/nonfinite tolerances and unavailable explicit CUDA. Invalid
CUDA is a local rejection control, not an actual CUDA execution.
Packing checks cover exact column order and represented sqrt(2) conventions
without claiming rationally exact svec isometry. Every preexisting src file
and original three owner files remains pinned; default public API is unchanged.

## Run And Readback

From the reviewer root, use the existing CPU runtime:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python \
  tests/test_psd_cone_independent.py
```

The CLI creates a unique `psd-independent-*` evidence directory under
`/hf3fs-jd/data/tmp/openopt-release-20261010`. Collection is bounded to 30
seconds and pytest to 170 seconds; timeout is retained FAIL. No historical
archive is copied or mutated. The runner records full source before/after,
exact nonempty collection/JUnit bidirectional membership, duplicate/missing/
extra zero, complete counts and each node's SHA-pinned observation.
Failure remains a real failed assertion, not xfail or omitted admission.
`--scan-only` scans only these two files, without disclosing credential values.
`--check-receipt PATH --receipt-sha256 CALLER_PIN` reads a frozen terminal
without rerunning mathematics, verifies artifacts/source/JUnit and reports
integrity separately from acceptance. No dirty-source run is clean signoff.

Preedit smoke on the clean owner baseline:
`psd-cone-independent-preedit/smoke-_cqnv6yu/verification.json` under that
external root, 80 passed/four explicit local CUDA skips, 6.44 pytest seconds,
source consistency PASS. It is not new-primitive or CUDA qualification.
Independent development/frozen receipts are supplied after actual runs;
owner self-reported PASS is never substituted for independent acceptance.

Development full `psd-independent-o87h9ne1/receipt.json` completed 128 passed,
zero failed/errors/skips, 60.29 pytest seconds. Its overall verdict remains
FAIL solely because the reviewer files were uncommitted, with source consistency
PASS; it is not clean signoff. SHA256:
`149bb799fbe8b8fb582170f0da463b1bbcacf96ea616129a29b1297b29b6fb30`.
Later corrections make forward differences exact between represented original
and proposed coordinates, and explicitly prove a zero-trace/nonzero-component
negative cannot pass just trace. They do not change owner gates.

This scope is a bounded dense symmetric matrix projection primitive only.
Actual CUDA, full SDP constraints/status/directions, remote execution binding,
commercial comparisons, speedup and general optimizer qualification remain
UNVERIFIED. The earlier fixed13 review remains BLOCKED_ARCHIVE_CORRUPTED,
not repaired or upgraded by this separate task.
