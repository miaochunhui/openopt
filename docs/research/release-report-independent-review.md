# Independent Release Report Review

## Scope And Snapshot

Read-only review of main's `scripts/render_release_report.py`,
`tests/test_render_release_report.py` and `docs/research/release-report.md`.
Only this document is written in the independent worktree. The executable
doctest appendix uses the existing renderer and test helpers, never solves,
downloads, publishes or alters original input/output evidence.

The selected real input manifest is
`/hf3fs-jd/data/tmp/openopt-release-20261010/report/owner-current-inputs-v2/input-manifest.json`,
SHA256 `b103a0a6bac885aeec1dd3daea20feea91258f42dbeb2288f55cb79c9f2e3e66`.
Main HEAD at review is `8f198608219bf9f64a04055b3d93d570af590805`;
the three reviewed files are uncommitted owner changes, not that HEAD's bytes.
Producer SHA256 is
`f1a711c19ae0f76ccdf6bb63d37fffa05e2f11c72e514e5fd74d7a87cb5c3c4b`,
test SHA256
`02ef20718190526a15295dfbab969d961d671ffeab812f8662ca8eff204c1f81`,
and documentation SHA256
`aff7f4ae69d7426050d901349d8e60b427f74b674e6117c4c54f7ed7292e0c75`.

## Findings

P2 at `scripts/render_release_report.py:731`: complete robustness coverage is not anchored to case x predeclared-arm
membership. Recomputed counts only agree with the supplied rows, so a caller
can omit failed rows and reseal all summaries, reducing the true denominator.
This is not tampering under the original caller pin: it is a gate accepting a
new, internally consistent but incomplete selected evidence record. Require
an independently pinned plan/corpus/arm contract, or label the projection
explicitly as unverified input rows rather than full-denominator acceptance.

P2 at `scripts/render_release_report.py:662`: successful extension counts use generic regression arithmetic rather than
their independent acceptance contract. Native/refinement `execution.junit`
in `scripts/accept_release_cuda.py` rejects failure/error/skip, and
`scripts/accept_extension_cuda.py` checks exact expected testcase membership.
Presolve `scripts/accept_presolve_cuda.py` has COUNT=8/0/0/0 and checks the full
eight-case set. A resealed native/presolve acceptance with a skip must not
render the same successful independent acceptance claim. Ordinary CPU pytest
skips are a separate policy and should not be globally banned.

P2 at `scripts/render_release_report.py:873`: the retained real-v1 robustness projection does not expose its version or
known incomplete settings/runtime binding. It is historical numerical
evidence with five new UNQUALIFIED and one numerically QUALIFIED row, not a
v2 worker execution or complete settings qualification. Renderer PASS and a
displayed numeric row qualification must not close that known integrity gap.
Explicitly carry `public-robustness-v1` and full-settings/runtime UNVERIFIED
into public text/scalars. No v2 real run has been produced by this seat.

Owner-discovered overall verdict issue is fixed in this snapshot: producer
success is PASS, not QUALIFIED, and overall status is recomputed from rows.
The appendix verifies both acceptance of all-pass PASS and rejection of false
overall PASS without performing a backend solve.

## Limited Pass

The actual caller-pinned owner output is checked byte-for-byte against current
generation, not merely parsed. Presolve source is read from its explicitly
pinned run, matched against acceptance pins, not backfilled from main HEAD.
Actual presolve/native independent acceptance is PASS; refinement and proof
contract review remain FAIL. Commercial/online remain UNVERIFIED.
Current real output shows all six new robustness rows, denominators 2/2/2,
numerical counts Clarabel 1/2, Openopt 0/2, OSQP 0/2 and overall UNQUALIFIED.
These facts do not resolve the findings above.

## Executable Evidence

Run only the doctest document with external cache/temp/JUnit. Generated copies
are negative-test evidence, never replacement acceptance or solver receipts.
The JSON receipt records source-before/after hashes, all original output hashes,
each mutation result and original model/corpus/source anchors.

```bash
sudo -n -u infra mkdir -p /hf3fs-jd/data/tmp/openopt-release-20261010/release-report-independent-review/new-check
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B -m pytest -q --doctest-glob=release-report-independent-review.md docs/research/release-report-independent-review.md --basetemp=/hf3fs-jd/data/tmp/openopt-release-20261010/release-report-independent-review/new-check/tmp -o cache_dir=/hf3fs-jd/data/tmp/openopt-release-20261010/release-report-independent-review/new-check/cache --junitxml=/hf3fs-jd/data/tmp/openopt-release-20261010/release-report-independent-review/new-check/pytest.xml
```

The deliberately observed `ACCEPTED` mutants are open findings, not a test
pretending the renderer rejected them. The doctest checks the review's claims,
not acceptance of the release.

Actual review completed at `2026-10-10T10:46:28.702641+00:00`: one doctest
passed in 2.59 seconds, six controls, original source/input/output bytes
unchanged and JSON roundtrip verified. The receipt is
`/hf3fs-jd/data/tmp/openopt-release-20261010/release-report-independent-review/v2/tmp/release_report_independent_rev0/review.json`,
SHA256 `ba2cd4b2c6a7ac37de5d2caafeff070a45070b75e4c70f9ed362590ffed9c5eb`.
JUnit is `v2/pytest.xml`. The first `v1/pytest.xml` records one failed doctest:
its external temporary parent directory had not been created, so no mutation
was evaluated. That failed evidence is retained, not reported as gate failure.
The following fixture changes only copied top-level records; it does not
assert that edited rows are valid numerical candidates or new backend runs.

>>> import importlib.util, json, hashlib, sys
>>> from copy import deepcopy
>>> from pathlib import Path
>>> main = Path("/data/dev/openopt")
>>> sys.path.insert(0, str(main))
>>> def load(name, path):
...     spec = importlib.util.spec_from_file_location(name, path)
...     module = importlib.util.module_from_spec(spec)
...     spec.loader.exec_module(module)
...     return module
>>> report = load("independent_report_review", main / "scripts/render_release_report.py")
>>> helper = load("independent_report_helpers", main / "tests/test_render_release_report.py")
>>> helper.report = report
>>> expected = {"scripts/render_release_report.py": "f1a711c19ae0f76ccdf6bb63d37fffa05e2f11c72e514e5fd74d7a87cb5c3c4b", "tests/test_render_release_report.py": "02ef20718190526a15295dfbab969d961d671ffeab812f8662ca8eff204c1f81", "docs/research/release-report.md": "aff7f4ae69d7426050d901349d8e60b427f74b674e6117c4c54f7ed7292e0c75"}
>>> before = {name: report.sha((main / name).read_bytes()) for name in expected}
>>> assert before == expected, "owner source drift: review a new pinned snapshot"
>>> root = getfixture("tmp_path")
>>> manifest = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/report/owner-current-inputs-v2/input-manifest.json")
>>> pin = "b103a0a6bac885aeec1dd3daea20feea91258f42dbeb2288f55cb79c9f2e3e66"
>>> selected = report.parse(manifest.read_bytes(), "review")
>>> assert report.sha(manifest.read_bytes()) == pin
>>> rendered = report.generate(manifest, pin)
>>> owner = manifest.parent.parent / "owner-current-render-v2"
>>> assert all(report.read(report.shared_path(owner / name, "review"), "review")[0] == payload for name, payload in rendered.items())
>>> scalars = json.loads(rendered["summary.json"])["scalars"]
>>> assert scalars["extension_presolve_source"] == "3792a86f9f5658642c5e250053a48d77c6425643" != selected["source_sha"]
>>> assert scalars["extension_refinement_status"] == scalars["proof_review_status"] == "FAIL"
>>> assert scalars["robustness_status"] == scalars["robustness_original_status"] == "UNQUALIFIED"
>>> assert scalars["robustness_planned"] == scalars["robustness_attempted"] == 6
>>> assert sum(scalars[f"robustness_{i}_denominator"] for i in range(3)) == 6
>>> receipt = {"kind": "independent-report-review-not-release-acceptance", "source_before": before, "manifest_sha256": pin, "output_sha256": {name: report.sha(payload) for name, payload in rendered.items()}, "controls": {}, "solver_calls": 0}
>>> def mutate(name, role, change):
...     folder = root / name
...     folder.mkdir()
...     copied = deepcopy(selected)
...     item = next(item for item in copied["inputs"] if item["id"] == role)
...     value = json.loads(Path(item["path"]).read_bytes())
...     change(value)
...     payload = report.encoded(value)
...     leaf = folder / (role + ".json")
...     leaf.write_bytes(payload)
...     item.update(path=str(leaf), sha256=report.sha(payload))
...     path = folder / "input-manifest.json"
...     payload = report.encoded(copied)
...     path.write_bytes(payload)
...     try:
...         result = report.generate(path, report.sha(payload))
...         outcome = {"result": "ACCEPTED", "manifest_sha256": report.sha(payload), "scalars": {k: v for k, v in json.loads(result["summary.json"])["scalars"].items() if k.startswith(("robustness_", "extension_"))}}
...     except report.Rejected as error:
...         outcome = {"result": "REJECTED", "code": error.code, "manifest_sha256": report.sha(payload)}
...     receipt["controls"][name] = outcome
...     return outcome
>>> def drop_failed(value):
...     value["rows"] = [row for row in value["rows"] if row["verdict"] == "QUALIFIED"]
...     value["planned_denominator"] = value["attempted_count"] = len(value["rows"])
...     value["numeric_verdict"] = "PASS"
...     value["arm_counts"] = {arm: {"denominator": sum(row["arm"] == arm for row in value["rows"]), "QUALIFIED": sum(row["arm"] == arm for row in value["rows"]), "UNQUALIFIED": 0, "FAIL": 0, "BUDGET_EXHAUSTED": 0} for arm in {row["arm"] for row in value["rows"]}}
>>> dropped = mutate("drop-five-failed-rows", "public_robustness", drop_failed)
>>> assert dropped["result"] == "ACCEPTED" and dropped["scalars"]["robustness_planned"] == 1 and dropped["scalars"]["robustness_status"] == "PASS"
>>> def add_skip(value):
...     value["tests"]["skipped"] = 1
>>> assert mutate("presolve-pass-one-skip", "extension_presolve", add_skip)["result"] == "ACCEPTED"
>>> assert mutate("native-pass-one-skip", "extension_native", add_skip)["result"] == "ACCEPTED"
>>> assert mutate("false-overall-pass", "public_robustness", lambda value: value.update(numeric_verdict="PASS"))["code"] == "robustness_false_overall_status"
>>> def false_presolve_pin(value):
...     value["pins"]["run"]["sha256"] = "0" * 64
>>> assert mutate("wrong-presolve-run-pin", "extension_presolve", false_presolve_pin)["code"] == "presolve_acceptance_run_mismatch"
>>> def all_pass(value):
...     for row in value["rows"]:
...         row["verdict"] = "QUALIFIED"
...     for count in value["arm_counts"].values():
...         count.update(QUALIFIED=count["denominator"], UNQUALIFIED=0)
...     value["numeric_verdict"] = "PASS"
>>> assert mutate("producer-all-pass-enum", "public_robustness", all_pass)["result"] == "ACCEPTED"
>>> page = rendered["benchmark-report-v4.html"].decode()
>>> assert "public-robustness-v1" not in page and "settings" not in page and "runtime" not in page
>>> receipt["retained_v1_limitation_not_displayed"] = True
>>> original = next(item for item in selected["inputs"] if item["id"] == "public_robustness")
>>> real = json.loads(Path(original["path"]).read_bytes())
>>> receipt["real_anchors"] = {key: real[key] for key in ("version", "baseline_sha256", "corpus_sha256", "public_acceptance_sha256", "source_hashes")}
>>> assert real["version"] == "public-robustness-v1" and report.sha(Path(original["path"]).read_bytes()) == original["sha256"]
>>> receipt["source_after"] = {name: report.sha((main / name).read_bytes()) for name in expected}
>>> assert receipt["source_after"] == before
>>> assert report.sha(manifest.read_bytes()) == pin
>>> assert all(report.read(report.shared_path(owner / name, "review"), "review")[0] == payload for name, payload in rendered.items())
>>> receipt["checked_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
>>> receipt["verdict"] = "REVIEW_REPRODUCED_WITH_OPEN_P2_FINDINGS"
>>> evidence = root / "review.json"
>>> _ = evidence.write_bytes(report.encoded(receipt))
>>> assert json.loads(evidence.read_bytes()) == receipt
>>> print("INDEPENDENT_REVIEW", receipt["verdict"], len(receipt["controls"]), "controls")
INDEPENDENT_REVIEW REVIEW_REPRODUCED_WITH_OPEN_P2_FINDINGS 6 controls
