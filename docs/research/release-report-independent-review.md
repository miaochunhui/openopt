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

Historical transcript below retains the original assertions and byte pins.
Its prompts are prefixed with `|` so the current doctest cannot accidentally
execute old assertions against repaired main. The original executable
transcript is preserved in commit `ea4e938a86357a224e42d7b95cfda86eb83241b1`;
rerunning it requires that exact old owner-source byte snapshot, not HEAD.

| >>> import importlib.util, json, hashlib, sys
| >>> from copy import deepcopy
| >>> from pathlib import Path
| >>> main = Path("/data/dev/openopt")
| >>> sys.path.insert(0, str(main))
| >>> def load(name, path):
...     spec = importlib.util.spec_from_file_location(name, path)
...     module = importlib.util.module_from_spec(spec)
...     spec.loader.exec_module(module)
...     return module
| >>> report = load("independent_report_review", main / "scripts/render_release_report.py")
| >>> helper = load("independent_report_helpers", main / "tests/test_render_release_report.py")
| >>> helper.report = report
| >>> expected = {"scripts/render_release_report.py": "f1a711c19ae0f76ccdf6bb63d37fffa05e2f11c72e514e5fd74d7a87cb5c3c4b", "tests/test_render_release_report.py": "02ef20718190526a15295dfbab969d961d671ffeab812f8662ca8eff204c1f81", "docs/research/release-report.md": "aff7f4ae69d7426050d901349d8e60b427f74b674e6117c4c54f7ed7292e0c75"}
| >>> before = {name: report.sha((main / name).read_bytes()) for name in expected}
| >>> assert before == expected, "owner source drift: review a new pinned snapshot"
| >>> root = getfixture("tmp_path")
| >>> manifest = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/report/owner-current-inputs-v2/input-manifest.json")
| >>> pin = "b103a0a6bac885aeec1dd3daea20feea91258f42dbeb2288f55cb79c9f2e3e66"
| >>> selected = report.parse(manifest.read_bytes(), "review")
| >>> assert report.sha(manifest.read_bytes()) == pin
| >>> rendered = report.generate(manifest, pin)
| >>> owner = manifest.parent.parent / "owner-current-render-v2"
| >>> assert all(report.read(report.shared_path(owner / name, "review"), "review")[0] == payload for name, payload in rendered.items())
| >>> scalars = json.loads(rendered["summary.json"])["scalars"]
| >>> assert scalars["extension_presolve_source"] == "3792a86f9f5658642c5e250053a48d77c6425643" != selected["source_sha"]
| >>> assert scalars["extension_refinement_status"] == scalars["proof_review_status"] == "FAIL"
| >>> assert scalars["robustness_status"] == scalars["robustness_original_status"] == "UNQUALIFIED"
| >>> assert scalars["robustness_planned"] == scalars["robustness_attempted"] == 6
| >>> assert sum(scalars[f"robustness_{i}_denominator"] for i in range(3)) == 6
| >>> receipt = {"kind": "independent-report-review-not-release-acceptance", "source_before": before, "manifest_sha256": pin, "output_sha256": {name: report.sha(payload) for name, payload in rendered.items()}, "controls": {}, "solver_calls": 0}
| >>> def mutate(name, role, change):
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
| >>> def drop_failed(value):
...     value["rows"] = [row for row in value["rows"] if row["verdict"] == "QUALIFIED"]
...     value["planned_denominator"] = value["attempted_count"] = len(value["rows"])
...     value["numeric_verdict"] = "PASS"
...     value["arm_counts"] = {arm: {"denominator": sum(row["arm"] == arm for row in value["rows"]), "QUALIFIED": sum(row["arm"] == arm for row in value["rows"]), "UNQUALIFIED": 0, "FAIL": 0, "BUDGET_EXHAUSTED": 0} for arm in {row["arm"] for row in value["rows"]}}
| >>> dropped = mutate("drop-five-failed-rows", "public_robustness", drop_failed)
| >>> assert dropped["result"] == "ACCEPTED" and dropped["scalars"]["robustness_planned"] == 1 and dropped["scalars"]["robustness_status"] == "PASS"
| >>> def add_skip(value):
...     value["tests"]["skipped"] = 1
| >>> assert mutate("presolve-pass-one-skip", "extension_presolve", add_skip)["result"] == "ACCEPTED"
| >>> assert mutate("native-pass-one-skip", "extension_native", add_skip)["result"] == "ACCEPTED"
| >>> assert mutate("false-overall-pass", "public_robustness", lambda value: value.update(numeric_verdict="PASS"))["code"] == "robustness_false_overall_status"
| >>> def false_presolve_pin(value):
...     value["pins"]["run"]["sha256"] = "0" * 64
| >>> assert mutate("wrong-presolve-run-pin", "extension_presolve", false_presolve_pin)["code"] == "presolve_acceptance_run_mismatch"
| >>> def all_pass(value):
...     for row in value["rows"]:
...         row["verdict"] = "QUALIFIED"
...     for count in value["arm_counts"].values():
...         count.update(QUALIFIED=count["denominator"], UNQUALIFIED=0)
...     value["numeric_verdict"] = "PASS"
| >>> assert mutate("producer-all-pass-enum", "public_robustness", all_pass)["result"] == "ACCEPTED"
| >>> page = rendered["benchmark-report-v4.html"].decode()
| >>> assert "public-robustness-v1" not in page and "settings" not in page and "runtime" not in page
| >>> receipt["retained_v1_limitation_not_displayed"] = True
| >>> original = next(item for item in selected["inputs"] if item["id"] == "public_robustness")
| >>> real = json.loads(Path(original["path"]).read_bytes())
| >>> receipt["real_anchors"] = {key: real[key] for key in ("version", "baseline_sha256", "corpus_sha256", "public_acceptance_sha256", "source_hashes")}
| >>> assert real["version"] == "public-robustness-v1" and report.sha(Path(original["path"]).read_bytes()) == original["sha256"]
| >>> receipt["source_after"] = {name: report.sha((main / name).read_bytes()) for name in expected}
| >>> assert receipt["source_after"] == before
| >>> assert report.sha(manifest.read_bytes()) == pin
| >>> assert all(report.read(report.shared_path(owner / name, "review"), "review")[0] == payload for name, payload in rendered.items())
| >>> receipt["checked_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
| >>> receipt["verdict"] = "REVIEW_REPRODUCED_WITH_OPEN_P2_FINDINGS"
| >>> evidence = root / "review.json"
| >>> _ = evidence.write_bytes(report.encoded(receipt))
| >>> assert json.loads(evidence.read_bytes()) == receipt
| >>> print("INDEPENDENT_REVIEW", receipt["verdict"], len(receipt["controls"]), "controls")
INDEPENDENT_REVIEW REVIEW_REPRODUCED_WITH_OPEN_P2_FINDINGS 6 controls

## Repaired Snapshot V3

This is a new review and new external report, not revision of historical
v2 output or acceptance of the original failed numerical attempts. Current
producer/test/doc pins are checked below before and after execution.
Owner's repaired scope requires complete plan/corpus/result membership and
extension success counts 21/8/7 with no failure/error/skip. Settings/runtime
remain UNVERIFIED, including historical v1, rather than supplied by this
renderer. This closes projection integrity only, not numerical execution,
cluster/hardware attestation, legal permission or commercial qualification.

Preparation carries every v2 descriptor unchanged except the explicitly
authorized live-v8 role and regenerated fixed-time metadata, and appends only
the caller-pinned real-v1 plan and corpus JSON. No arrays/raw MPS or corpus
archive are included in public output. r1 GPU source remains
`3792a86f9f5658642c5e250053a48d77c6425643`, r2 numerical source remains
`8f198608219bf9f64a04055b3d93d570af590805`, and real robustness remains v1.
The private manifest is not public. New live-v8 is a selected pinned snapshot,
not proof of service state at render time. Owner's prior live-v7 failure was
reported as missing recovery pin, not a new cluster failure; it is retained,
never used as source acceptance here.

Rerun the same document-only pytest entry above using a new external test
directory. The first execution prepares `report/owner-current-inputs-v3` and
`report/owner-current-render-v3`; subsequent checks validate exact bytes and
do not overwrite them. Rejected mutation fixtures are confined to pytest tmp.

Actual repaired review finished `2026-10-10T10:54:41.030253+00:00`,
one doctest passed in 2.73 seconds, seven REJECTED controls, source-before/
after equality, all old input/output bytes unchanged and JSON readback.
Receipt:
`/hf3fs-jd/data/tmp/openopt-release-20261010/release-report-independent-review/repaired-v3/tmp/release_report_independent_rev0/repaired-review.json`,
SHA256 `1bf4c82420ef649ec7a677ef5f50356015e4f2d8cbdff88a9e8117b0f69f672d`.
Same directory has `review.log`; parent `repaired-v3/pytest.xml` has actual
counts. The three original P2 findings are closed only for this repaired
byte snapshot and the selected original contracts. Caller-pinned changed
plan/corpus is a new scope, not validation of a different corpus or run.
This renderer still trusts pinned independent acceptance outcomes; it does
not replay CUDA numerical sidecars or attest current hardware itself.

Prepared private manifest:
`/hf3fs-jd/data/tmp/openopt-release-20261010/report/owner-current-inputs-v3/input-manifest.json`,
SHA256 `b94fee512220c5b29cc0505583dfa3361a127c698dc07bbb904646da487d066b`.
Rendered output:
`/hf3fs-jd/data/tmp/openopt-release-20261010/report/owner-current-render-v3/`;
public manifest SHA256
`7a29bf6ac5daff209a5cf585ab1b0922b4d86f8e4fa2de78b9d5f07f3cd76d5a`,
summary SHA256
`e72eff793cfa18905f6bc10bd800a37c17c4230aac90bbfc596df66fc00e22f0`.
The two HTML pages contain no dataset arrays/raw bytes. Private report-bundle
and input manifest are not publication selections.

Independent CLI exact-output check also returned PASS:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B /data/dev/openopt/scripts/render_release_report.py --manifest /hf3fs-jd/data/tmp/openopt-release-20261010/report/owner-current-inputs-v3/input-manifest.json --manifest-sha256 b94fee512220c5b29cc0505583dfa3361a127c698dc07bbb904646da487d066b --output-dir /hf3fs-jd/data/tmp/openopt-release-20261010/report/owner-current-render-v3 --check
```

>>> import importlib.util, json, sys
>>> from copy import deepcopy
>>> from pathlib import Path
>>> main = Path("/data/dev/openopt")
>>> def load_repaired(name, path):
...     spec = importlib.util.spec_from_file_location(name, path)
...     module = importlib.util.module_from_spec(spec)
...     spec.loader.exec_module(module)
...     return module
>>> fixed = load_repaired("independent_repaired_report", main / "scripts/render_release_report.py")
>>> expected_fixed = {"scripts/render_release_report.py": "705b441630d2ed4c4da63063b02913c1cebbbd068e754d49bd451724f4e3274c", "tests/test_render_release_report.py": "d63dc402dc58d1c8b542b1d6ea560283c736649c9cd31e05b98b3551369ab141", "docs/research/release-report.md": "5b5597af7ec0a137917917c33be1fe01cd2e36495f952f83b368695e19abf5f4"}
>>> source_before = {name: fixed.sha((main / name).read_bytes()) for name in expected_fixed}
>>> assert source_before == expected_fixed, "repaired owner source drift"
>>> root = getfixture("tmp_path")
>>> base = Path("/hf3fs-jd/data/tmp/openopt-release-20261010")
>>> old_manifest = base / "report/owner-current-inputs-v2/input-manifest.json"
>>> old_payload = old_manifest.read_bytes()
>>> assert fixed.sha(old_payload) == "b103a0a6bac885aeec1dd3daea20feea91258f42dbeb2288f55cb79c9f2e3e66"
>>> old = json.loads(old_payload)
>>> def input_hash(item):
...     return fixed.sha(fixed.read(fixed.shared_path(item["path"], "review"), "review")[0])
>>> original_hashes = {item["id"]: input_hash(item) for item in old["inputs"]}
>>> assert original_hashes == {item["id"]: item["sha256"] for item in old["inputs"]}
>>> old_owner = base / "report/owner-current-render-v2"
>>> original_output_hashes = {path.name: fixed.sha(path.read_bytes()) for path in old_owner.iterdir()}
>>> new_live = {"id": "live", "path": str(base / "live-owner-integration-v8/live-105158878531.json"), "sha256": "42dabcddd34ffb59af5b5f445c5b9116289bf22c8eb99e316bbec602224e19d3"}
>>> added = [{"id": "public_robustness_plan", "path": str(base / "public-robustness/real-v1/plan.json"), "sha256": "80e432831b618b709588fee004b473c8b41106550b6fe3251abcf0d66a30b732"}, {"id": "public_robustness_corpus", "path": str(base / "public-robustness/real-v1/baseline/frozen/corpus.json"), "sha256": "3a434edfa8165167f56be25732840f03bbace1a5f5b7d1de7cdab989673d3f0f"}]
>>> supplied = [deepcopy(item) if item["id"] != "live" else new_live for item in old["inputs"] if item["id"] != "metadata"] + added
>>> assert all(input_hash(item) == item["sha256"] for item in supplied)
>>> modified = "2026-10-10T18:51:58+08:00"
>>> prepared = base / "report/owner-current-inputs-v3"
>>> if not prepared.exists():
...     v3_manifest, v3_pin = fixed.prepare_inputs(prepared, [(item["id"], item["path"], item["sha256"]) for item in supplied], old["source_sha"], modified)
... else:
...     v3_manifest = prepared / "input-manifest.json"
...     v3_pin = fixed.sha(v3_manifest.read_bytes())
>>> v3 = json.loads(v3_manifest.read_bytes())
>>> assert v3["source_sha"] == old["source_sha"] and v3["modified_at"] == modified
>>> expected_descriptors = [{"id": item["id"], "path": str(fixed.shared_path(item["path"], "review")), "sha256": item["sha256"]} for item in supplied]
>>> assert [item for item in v3["inputs"] if item["id"] != "metadata"] == expected_descriptors
>>> meta_item = next(item for item in v3["inputs"] if item["id"] == "metadata")
>>> meta = json.loads(Path(meta_item["path"]).read_bytes())
>>> assert meta == {"version": 4, "source_sha": old["source_sha"], "modified_at": modified, "regression_scope": "intermediate_snapshot", "cpu_binding": {"kind": "scope_only"}, "online_publication": "UNVERIFIED", "commercial_qualification": "UNVERIFIED"}
>>> generated = fixed.generate(v3_manifest, v3_pin)
>>> assert generated == fixed.generate(v3_manifest, v3_pin)
>>> output = base / "report/owner-current-render-v3"
>>> _ = fixed.write_bundle(output, generated, check=output.exists())
>>> _ = fixed.write_bundle(output, generated, check=True)
>>> summary = json.loads(generated["summary.json"])
>>> values = summary["scalars"]
>>> assert values["robustness_version"] == "public-robustness-v1"
>>> assert values["robustness_settings_binding"] == values["robustness_runtime_binding"] == "UNVERIFIED"
>>> assert values["robustness_status"] == values["robustness_original_status"] == "UNQUALIFIED"
>>> assert values["robustness_planned"] == values["robustness_attempted"] == 6
>>> assert values["extension_native_tests"] == 21 and values["extension_presolve_tests"] == 8
>>> assert values["extension_refinement_status"] == values["proof_review_status"] == "FAIL"
>>> assert values["extension_presolve_source"] == values["gpu_source_sha"] == "3792a86f9f5658642c5e250053a48d77c6425643" != v3["source_sha"]
>>> assert values["regression_scope"] == "intermediate_snapshot" and values["commercial_status"] == values["online_status"] == "UNVERIFIED"
>>> assert values["live_read_at"] == json.loads(Path(new_live["path"]).read_bytes())["read_at"]
>>> review = {"kind": "repaired-report-independent-review-not-qualification", "source_before": source_before, "manifest_path": str(v3_manifest), "manifest_sha256": v3_pin, "old_manifest_sha256": fixed.sha(old_payload), "old_output_sha256": original_output_hashes, "output_sha256": {name: fixed.sha(payload) for name, payload in generated.items()}, "explicit_inputs": v3["inputs"], "controls": {}, "solver_calls": 0, "legal_permission_verified": False}
>>> def repaired_mutation(name, edits):
...     folder = root / name
...     folder.mkdir()
...     selected = deepcopy(v3)
...     for role, change in edits.items():
...         item = next(item for item in selected["inputs"] if item["id"] == role)
...         value = json.loads(Path(item["path"]).read_bytes())
...         change(value)
...         leaf = folder / (role + ".json")
...         payload = fixed.encoded(value)
...         leaf.write_bytes(payload)
...         item.update(path=str(leaf), sha256=fixed.sha(payload))
...     payload = fixed.encoded(selected)
...     path = folder / "input-manifest.json"
...     path.write_bytes(payload)
...     try:
...         fixed.generate(path, fixed.sha(payload))
...         outcome = {"result": "ACCEPTED", "manifest_sha256": fixed.sha(payload)}
...     except fixed.Rejected as error:
...         outcome = {"result": "REJECTED", "code": error.code, "manifest_sha256": fixed.sha(payload)}
...     review["controls"][name] = outcome
...     assert outcome["result"] == "REJECTED", name
...     return outcome["code"]
>>> def reduce_failed(value):
...     value["rows"] = [row for row in value["rows"] if row["verdict"] == "QUALIFIED"]
...     value["numeric_verdict"] = "PASS"
...     value["planned_denominator"] = value["attempted_count"] = len(value["rows"])
...     value["arm_counts"] = {arm: {"denominator": sum(row["arm"] == arm for row in value["rows"]), "QUALIFIED": sum(row["arm"] == arm for row in value["rows"]), "UNQUALIFIED": 0, "FAIL": 0, "BUDGET_EXHAUSTED": 0} for arm in {row["arm"] for row in value["rows"]}}
>>> assert repaired_mutation("drop-five-failed", {"public_robustness": reduce_failed}) == "robustness_result_incomplete"
>>> def reduce_plan(value):
...     value["rows"] = value["rows"][:1]
>>> assert repaired_mutation("coordinated-drop-result-plan", {"public_robustness": reduce_failed, "public_robustness_plan": reduce_plan}) == "robustness_plan_incomplete"
>>> def skip_one(value):
...     value["tests"]["skipped"] = 1
>>> assert repaired_mutation("native-one-skip", {"extension_native": skip_one}) == "extension_independent_denominator"
>>> assert repaired_mutation("presolve-one-skip", {"extension_presolve": skip_one}) == "extension_independent_denominator"
>>> def wrong_model(value):
...     value["rows"][0]["model_sha256"] = "0" * 64
>>> assert repaired_mutation("coordinated-wrong-model", {"public_robustness": wrong_model, "public_robustness_plan": wrong_model}) == "robustness_plan_incomplete"
>>> assert repaired_mutation("false-overall-pass-v3", {"public_robustness": lambda value: value.update(numeric_verdict="PASS")}) == "robustness_false_overall_status"
>>> def wrong_pin(value):
...     value["pins"]["run"]["sha256"] = "0" * 64
>>> assert repaired_mutation("wrong-presolve-pin-v3", {"extension_presolve": wrong_pin}) == "presolve_acceptance_run_mismatch"
>>> public_files = json.loads(generated["manifest.json"])["files"]
>>> assert {item["path"] for item in public_files} == {"benchmark-report-v4.html", "design-v4.html", "summary.json", "claims.json"}
>>> for name, payload in generated.items():
...     if name != "report-bundle.json":
...         assert b"raw_sha256" not in payload and b"model_sha256" not in payload
...         assert b"primal_vector" not in payload and b"cone_z" not in payload
>>> assert all(input_hash(item) == item["sha256"] for item in old["inputs"])
>>> assert {path.name: fixed.sha(path.read_bytes()) for path in old_owner.iterdir()} == original_output_hashes
>>> review["source_after"] = {name: fixed.sha((main / name).read_bytes()) for name in expected_fixed}
>>> assert review["source_after"] == source_before
>>> assert old_manifest.read_bytes() == old_payload
>>> review["checked_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
>>> review["verdict"] = "PASS_PROJECTION_SCOPE_ONLY"
>>> leaf = root / "repaired-review.json"
>>> _ = leaf.write_bytes(fixed.encoded(review))
>>> assert json.loads(leaf.read_bytes()) == review
>>> log = root / "review.log"
>>> _ = log.write_text("INDEPENDENT_REVIEW verdict=PASS_PROJECTION_SCOPE_ONLY controls=7 solver_calls=0 old_bytes_unchanged=true\n")
>>> print("REPAIRED_REVIEW", review["verdict"], len(review["controls"]), "controls")
REPAIRED_REVIEW PASS_PROJECTION_SCOPE_ONLY 7 controls
