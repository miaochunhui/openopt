# SBOM Reader Schema Binding Repair

Base producer: `3d6b5da403dd7296447fefbf17c1f1f5004b8bad`.
Only `scripts/build_sbom.py`, the new reader test and this document are owned.
Original author33 and independent12 assertions and historical artifacts remain
unchanged. The independent12 result was 11 PASS / 1 FAIL: a rehashed receipt
with a wrong schema declaration was accepted by the original reader.

The repair binds the receipt and evidence schema declarations to the separately
supplied caller schema pin. Existing offline standard validation still binds
that pin to the official frozen schema bytes. Existing source, metadata,
wheel and installed membership checks remain in force. Receipt/evidence source,
verdict and legal/runtime boundaries must agree; no generation scope changes.

Tests use real disposable Git, dependency METADATA and wheel fixtures only.
No new real-project SBOM generation, dependency installation, network, SDK
import, credential discovery, solve or GPU execution is needed. Unknown licenses
remain NOASSERTION, legal qualification UNVERIFIED and runtime NOT_TESTED.

Use the existing interpreter with explicit current-root import binding:

```bash
sudo -n -u infra env PYTHONPATH=/data/dev/openopt-wt-sbom-reader-schema-binding-20261011 \
  /home/infra/anaconda3/bin/python -B \
  /data/dev/openopt-wt-sbom-reader-schema-binding-20261011/tests/test_sbom_reader_schema_binding.py \
  --artifact-dir NEW_EXTERNAL_DIRECTORY
```

Original author33 uses the same command with `tests/test_build_sbom.py`.
Collection and body retain the existing 45s/170s bounds. Verification, process
log, process status, exact collection and JUnit are retained externally.
Repair tests do not constitute independent acceptance; Helm transports the
unchanged original12 to the final frozen producer.

The repair adds ten reader lines only. Non-reader module AST and the complete
original author test bytes are mechanically compared against the base commit.
Development controls: original33 PASS and new16 PASS, both zero skips. The
new rejection control first produced a real 1 PASS / 1 FAIL, rc1, no timeout
at `sbom-reader-schema-binding/pre-repair-red-v1`; preserve that directory and
the independent original12 failure. Artifact roots are under
`/hf3fs-jd/data/tmp/openopt-release-20261011/`.

After the final commit, execute the two suites into fresh
`sbom-reader-schema-binding/clean-original33-v1` and
`sbom-reader-schema-binding/clean-controls-v1`. Do not edit these source files
after execution. The reader checks a caller-pinned verification, real process
status, exact collection/JUnit membership, counts and current source snapshot:

```bash
sudo -n -u infra env PYTHONPATH=/data/dev/openopt-wt-sbom-reader-schema-binding-20261011 \
  /home/infra/anaconda3/bin/python -B \
  /data/dev/openopt-wt-sbom-reader-schema-binding-20261011/tests/test_sbom_reader_schema_binding.py \
  --read-tests EXISTING_TEST_DIRECTORY --verification-sha256 ACTUAL_RECEIPT_SHA256
```

Use `--scan-owned --scan-output NEW_EXTERNAL_FILE` on the same entry for the
three explicit changed files. Existing scanner selftest output is captured;
credential needle values remain in memory and are never printed or recorded.
Only counts and file hashes are retained. This known-content scan is not proof
of unknown-secret absence. The original real source/wheel artifact is not
regenerated or reissued as evidence from this repair.
