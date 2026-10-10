"""Negative controls plus real subprocess acceptance on an external tiny repo."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from scripts import verify_sharded as tool


def git(repo, *args):
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def xml(path, nodes, *, seconds=1, skipped=()):
    suite = ET.Element("testsuite", tests=str(len(nodes)), errors="0", failures="0",
                       skipped=str(len(skipped)))
    for node in nodes:
        classname, name = tool.junit_key(node)
        case = ET.SubElement(suite, "testcase", classname=classname, name=name, time=str(seconds))
        if node in skipped:
            ET.SubElement(case, "skipped", message="fixture skip")
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(path)
    return path


@pytest.fixture
def nodes():
    return ["tests/test_a.py::test_a", "tests/check_b.py::TestB::test_b[id.x::y]",
            "tests/test_c.py::test_c"]


@pytest.fixture
def mini_repo(tmp_path):
    repo = tmp_path / "repo"
    for directory in ("scripts", "examples", "tests", "src", "benchmarks"):
        (repo / directory).mkdir(parents=True, exist_ok=True)
    # The exact production verify entry executes real pytest and source checks.
    shutil.copyfile(tool.ROOT / "scripts/verify.py", repo / "scripts/verify.py")
    shutil.copyfile(Path(tool.__file__), repo / "scripts/verify_sharded.py")
    (repo / "scripts/migrate_openpo.py").write_text(
        'print("TINY_FIXTURE_PROVENANCE_ONLY")\n', encoding="utf-8")
    (repo / "examples/portfolio_qp.py").write_text(
        'print("TINY_FIXTURE_EXAMPLE_ONLY")\n', encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths=["tests"]\n'
        'python_files=["test_*.py", "check_*.py"]\n', encoding="utf-8")
    (repo / "tests/test_a.py").write_text(
        'import pytest\n@pytest.mark.parametrize("value", [1,2], ids=["v.1", "v::2"])\n'
        'def test_a(value):\n    assert value in (1,2)\n', encoding="utf-8")
    (repo / "tests/check_b.py").write_text(
        'class TestB:\n    def test_custom_discovery(self):\n'
        '        assert 2 + 3 == 5\n', encoding="utf-8")
    (repo / "tests/test_c.py").write_text(
        'import pytest\n@pytest.mark.skip(reason="tiny fixture explicit skip")\n'
        'def test_c():\n    raise AssertionError("skip must be preserved")\n', encoding="utf-8")
    git(repo, "init")
    git(repo, "config", "user.name", "Openopt fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "add", "scripts/verify.py", "scripts/verify_sharded.py",
        "scripts/migrate_openpo.py", "examples/portfolio_qp.py", "pyproject.toml",
        "tests/test_a.py", "tests/check_b.py", "tests/test_c.py")
    git(repo, "commit", "-m", "Freeze tiny external regression fixture")
    assert git(repo, "status", "--porcelain") == ""
    return repo


@pytest.fixture
def tiny_plan(mini_repo, tmp_path):
    directory = tool.plan_run(mini_repo, tmp_path / "plans", [], shard_timeout=90)
    path = directory / "plan.json"
    return mini_repo, path, tool.digest(path)


def test_lpt_uses_latest_real_test_time_and_unknowns(nodes, tmp_path):
    first = xml(tmp_path / "first.xml", nodes, seconds=10)
    second = xml(tmp_path / "later.xml", nodes[:1], seconds=50)
    additional = "tests/accept_extra.py::test_extra"
    assignments, history, unknown = tool.lpt(nodes + [additional], [first, second])
    assert assignments[0]["targets"] == ["tests/test_a.py"]
    assert assignments[0]["predicted_test_s"] == 50
    assert sum(item["predicted_test_s"] for item in assignments) == 71
    assert unknown == [additional]
    assert history[1]["sha256"] == tool.digest(second)
    assert {node for item in assignments for node in item["nodeids"]} == set(nodes + [additional])


@pytest.mark.parametrize("value", [0, -1, 301, float("inf"), float("nan")])
def test_timeout_outside_finite_gate_rejected(value):
    with pytest.raises(tool.GateError, match="timeout"):
        tool.deadline(value)


@pytest.mark.parametrize("shards", [0, 33, 1.5, True])
def test_invalid_capacity_rejected(nodes, shards):
    with pytest.raises(tool.GateError, match="shards"):
        tool.lpt(nodes, [], shards)


@pytest.mark.parametrize("workers", [0, 33, 1.5, True])
def test_invalid_worker_capacity_rejected(tmp_path, workers):
    with pytest.raises(tool.GateError, match="workers_must_be_1_32"):
        tool.execute(tool.ROOT, tmp_path / "missing-plan", "unused", tmp_path / "run", workers)
    assert not (tmp_path / "run").exists()


def test_32_cpu_shards_zero_duration_files_are_nonempty(tmp_path):
    nodes = [f"tests/test_{index}.py::test_ok" for index in range(32)]
    history = xml(tmp_path / "zero.xml", nodes, seconds=0)
    assignments, _, unknown = tool.lpt(nodes, [history], shards=32)
    assert len(assignments) == 32 and not unknown
    assert all(len(item["targets"]) == len(item["nodeids"]) == 1 for item in assignments)
    assert tool.capacity(32, "workers") == 32
    tool.prediction_budget(assignments, 300)


@pytest.mark.parametrize("seconds,files,match", [
    (301, 1, "file_prediction_exceeds_deadline"),
    (160, 2, "shard_prediction_exceeds_deadline"),
])
def test_wholefile_and_shard_over_budget_rejected(tmp_path, seconds, files, match):
    nodes = [f"tests/test_{index}.py::test_ok" for index in range(files)]
    history = xml(tmp_path / "cost.xml", nodes, seconds=seconds)
    assignments, _, _ = tool.lpt(nodes, [history], shards=1)
    with pytest.raises(tool.GateError, match=match):
        tool.prediction_budget(assignments, 300)


def test_prediction_exact_deadline_is_not_extended(tmp_path):
    nodes = ["tests/test_at_limit.py::test_ok"]
    history = xml(tmp_path / "limit.xml", nodes, seconds=300)
    assignments, _, _ = tool.lpt(nodes, [history], shards=1)
    tool.prediction_budget(assignments, 300)
    with pytest.raises(tool.GateError, match="exceeds_deadline"):
        tool.prediction_budget(assignments, 299)


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_unknown_prediction_invalid_rejected(nodes, value):
    with pytest.raises(tool.GateError, match="prediction"):
        tool.lpt(nodes, [], unknown_s=value)


def test_junit_count_duplicate_and_nonfinite_rejected(nodes, tmp_path):
    path = xml(tmp_path / "counts.xml", nodes)
    root = ET.parse(path)
    root.getroot().find("testsuite").set("tests", "9")
    root.write(path)
    with pytest.raises(tool.GateError, match="count"):
        tool.parse_junit(path)
    xml(path, nodes + nodes[:1])
    with pytest.raises(tool.GateError, match="duplicate"):
        tool.parse_junit(path)
    xml(path, nodes, seconds="nan")
    with pytest.raises(tool.GateError, match="time"):
        tool.parse_junit(path)


def test_empty_junit_and_too_few_files_rejected(nodes, tmp_path):
    with pytest.raises(tool.GateError, match="empty_junit"):
        tool.parse_junit(xml(tmp_path / "empty.xml", []))
    with pytest.raises(tool.GateError, match="too_few"):
        tool.lpt(nodes[:1], [], shards=3)


def test_plan_includes_non_test_prefix_and_real_nodeids(tiny_plan):
    repo, path, _ = tiny_plan
    plan = tool.read(path)
    assert plan["source"]["clean"] is True
    assert plan["source"]["commit"] == git(repo, "rev-parse", "HEAD")
    assert len(plan["collection"]["nodeids"]) == 4
    assert any(node.startswith("tests/check_b.py::") for node in plan["collection"]["nodeids"])
    assert any("v::2" in node for node in plan["collection"]["nodeids"])
    assert len(plan["unknown_nodeids"]) == 4
    assert plan["source_consistency"]["verdict"] == "PASS"


@pytest.mark.parametrize("mutation,match", [
    ("empty", "empty_or_duplicate"),
    ("omit", "shard_collection"),
    ("overlap", "shard_collection"),
    ("target", "target_nodeid"),
    ("duplicate_file", "duplicate_shard_file"),
])
def test_plan_membership_corruptions_rejected(tiny_plan, mutation, match):
    _, path, _ = tiny_plan
    plan = tool.read(path)
    if mutation == "empty":
        plan["collection"]["nodeids"] = []
    elif mutation == "omit":
        plan["shards"][0]["nodeids"].pop()
    elif mutation == "overlap":
        plan["shards"][1]["nodeids"].append(plan["shards"][0]["nodeids"][0])
        plan["shards"][1]["targets"].append(plan["shards"][0]["targets"][0])
    elif mutation == "target":
        plan["shards"][0]["targets"] = ["tests/absent.py"]
    else:
        plan["shards"][0]["targets"].append(plan["shards"][0]["targets"][0])
    with pytest.raises(tool.GateError, match=match):
        tool.validate_plan(plan)


def test_full_tiny_run_and_independent_replay(tiny_plan, tmp_path):
    repo, path, pin = tiny_plan
    directory = tool.execute(repo, path, pin, tmp_path / "runs", workers=3)
    report_path = directory / "verification.json"
    report = tool.read(report_path)
    assert report["verdict"] == "PASS"
    assert report["tests"] == {"tests": 4, "failures": 0, "errors": 0, "skipped": 1}
    assert report["coverage"] == "tests" and report["speedup"] is None
    assert report["source"] == tool.read(path)["source"]
    assert all(item["actual_worker_wall_s"] > 0 for item in report["shards"])
    assert git(repo, "status", "--porcelain") == ""
    audit_dir = tool.check(repo, path, pin, report_path, tool.digest(report_path),
                          tmp_path / "checks")
    assert tool.read(audit_dir / "acceptance.json")["verdict"] == "PASS"
    # A rewritable top-level hash does not make a forged child count valid.
    changed = copy.deepcopy(report)
    changed["tests"]["tests"] += 1
    tool.save(report_path, changed)
    with pytest.raises(tool.GateError, match="combined_report_count"):
        tool.check(repo, path, pin, report_path, tool.digest(report_path),
                   tmp_path / "tampered-check")
    audit = next((tmp_path / "tampered-check").glob("check-*/acceptance.json"))
    assert tool.read(audit)["verdict"] == "FAIL"


def test_eight_cpu_shards_three_workers_full_run_replay(mini_repo, tmp_path):
    files = []
    for index in range(5):
        path = f"tests/test_extra_{index}.py"
        (mini_repo / path).write_text("def test_extra():\n    assert 7 == 7\n")
        files.append(path)
    git(mini_repo, "add", *files)
    git(mini_repo, "commit", "-m", "Freeze eight indivisible CPU test files")
    planned = tool.plan_run(mini_repo, tmp_path / "eight-plan", [], shards=8, shard_timeout=90)
    plan_path = planned / "plan.json"
    plan = tool.read(plan_path)
    assert len(plan["shards"]) == 8 and len(plan["collection"]["nodeids"]) == 9
    assert all(item["targets"] and item["nodeids"] for item in plan["shards"])
    assert len({target for item in plan["shards"] for target in item["targets"]}) == 8
    assert any(node.startswith("tests/check_b.py::") for node in plan["collection"]["nodeids"])
    run = tool.execute(mini_repo, plan_path, tool.digest(plan_path), tmp_path / "eight-run", workers=3)
    report_path = run / "verification.json"
    report = tool.read(report_path)
    assert report["verdict"] == "PASS" and report["workers"] == 3
    assert report["tests"] == {"tests": 9, "failures": 0, "errors": 0, "skipped": 1}
    assert report["not_admitted"] == report["failures"] == [] and report["speedup"] is None
    events = []
    for item in report["shards"]:
        process = tool.read(Path(item["artifact_dir"]) / "process.json", item["process_sha256"])
        assert process["returncode"] == 0 and process["timeout"] is False
        events.extend([(process["started_at"], 1), (process["finished_at"], -1)])
    flying, peak = 0, 0
    for _, change in sorted(events):
        flying += change
        peak = max(peak, flying)
    assert flying == 0 and 1 <= peak <= 3
    checked = tool.check(mini_repo, plan_path, tool.digest(plan_path), report_path,
                         tool.digest(report_path), tmp_path / "eight-check")
    audit = tool.read(checked / "acceptance.json")
    assert audit["verdict"] == "PASS" and audit["tests"] == report["tests"]
    assert audit["source_consistency"]["verdict"] == "PASS"
    assert git(mini_repo, "status", "--porcelain") == ""


def test_overbudget_plan_saved_fail_and_forged_cost_stops_admission(mini_repo, tmp_path):
    history = xml(tmp_path / "overbudget.xml",
                  ["tests/test_a.py::test_a[v.1]"], seconds=301)
    with pytest.raises(tool.GateError, match="file_prediction_exceeds_deadline"):
        tool.plan_run(mini_repo, tmp_path / "overbudget", [history], shard_timeout=300)
    path = next((tmp_path / "overbudget").glob("plan-*/plan.json"))
    failed = tool.read(path)
    assert failed["verdict"] == "FAIL" and "file_is_indivisible" in failed["error"]
    assert failed["collection"]["nodeids"] and failed["shards"]
    assert failed["history"][0]["sha256"] == tool.digest(history)
    assert max(item["predicted_test_s"] for item in failed["shards"]) > 300
    forged = copy.deepcopy(failed)
    forged["verdict"] = "PASS"
    for item in forged["shards"]:
        item["predicted_test_s"] = 1
        item["file_test_s"] = {target: 1 for target in item["targets"]}
    tool.save(path, forged)
    with pytest.raises(tool.GateError, match="file_prediction_exceeds_deadline"):
        tool.execute(mini_repo, path, tool.digest(path), tmp_path / "forged-run", workers=3)
    report = tool.read(next((tmp_path / "forged-run").glob("run-*/verification.json")))
    assert report["verdict"] == "FAIL" and "exceeds_deadline" in report["error"]
    assert not list((tmp_path / "forged-run").glob("run-*/shard-*"))


def test_understated_prediction_and_history_drift_rejected(tiny_plan, tmp_path):
    repo, path, _ = tiny_plan
    plan = tool.read(path)
    plan["shards"][0]["predicted_test_s"] = 0
    with pytest.raises(tool.GateError, match="prediction_assignment_mismatch"):
        tool.validate_plan(plan)
    history = xml(tmp_path / "original.xml", ["tests/test_a.py::test_a[v.1]"])
    directory = tool.plan_run(repo, tmp_path / "history-plan", [history])
    bound = tool.read(directory / "plan.json")
    xml(history, ["tests/test_a.py::test_a[v.1]"], seconds=299)
    with pytest.raises(tool.GateError, match="prediction_history_pin_mismatch"):
        tool.validate_plan(bound)


def test_hash_source_and_collection_drift_stop_before_admission(tiny_plan, tmp_path):
    repo, path, pin = tiny_plan
    with pytest.raises(tool.GateError, match="hash_mismatch"):
        tool.execute(repo, path, "0" * 64, tmp_path / "wrong-pin")
    (repo / "tests/test_a.py").write_text("def test_changed():\n    assert True\n")
    with pytest.raises(tool.GateError, match="source_not_clean"):
        tool.execute(repo, path, pin, tmp_path / "dirty")
    git(repo, "add", "tests/test_a.py")
    git(repo, "commit", "-m", "Change fixture test")
    with pytest.raises(tool.GateError, match="source_drift"):
        tool.execute(repo, path, pin, tmp_path / "new-commit")
    assert all(not list(directory.glob("run-*/shard-*"))
               for directory in (tmp_path / "wrong-pin", tmp_path / "dirty", tmp_path / "new-commit"))


def test_collection_conflict_rejected_before_workers(tiny_plan, tmp_path):
    repo, path, _ = tiny_plan
    plan = tool.read(path)
    old = plan["collection"]["nodeids"][0]
    new = old.split("::", 1)[0] + "::test_absent"
    assert new != old
    plan["collection"]["nodeids"][0] = new
    plan["shards"], plan["history"], plan["unknown_nodeids"] = tool.lpt(
        plan["collection"]["nodeids"], [], shards=3,
    )
    tool.save(path, plan)
    with pytest.raises(tool.GateError, match="collection_drift"):
        tool.execute(repo, path, tool.digest(path), tmp_path / "bad-collection")
    assert not list((tmp_path / "bad-collection").glob("run-*/shard-*"))


def test_partial_failure_stops_queued_admission(mini_repo, tmp_path):
    (mini_repo / "tests/test_a.py").write_text(
        "def test_failure():\n    assert False, 'intentional admission negative'\n")
    git(mini_repo, "add", "tests/test_a.py")
    git(mini_repo, "commit", "-m", "Freeze deliberate failing fixture")
    history = xml(tmp_path / "cost.xml", ["tests/test_a.py::test_failure"], seconds=80)
    directory = tool.plan_run(mini_repo, tmp_path / "fail-plan", [history], shard_timeout=90)
    path = directory / "plan.json"
    with pytest.raises(tool.GateError, match="shard_failure"):
        tool.execute(mini_repo, path, tool.digest(path), tmp_path / "fail-run", workers=1)
    report = tool.read(next((tmp_path / "fail-run").glob("run-*/verification.json")))
    assert report["verdict"] == "FAIL" and report["not_admitted"] == [1, 2]
    assert len(report["failures"]) == 1 and report["failures"][0]["id"] == 0
    process = tool.read(next((tmp_path / "fail-run").glob("run-*/shard-0/process.json")))
    assert process["returncode"] == 1 and process["timeout"] is False
    verify_report = tool.read(next((tmp_path / "fail-run").glob(
        "run-*/shard-0/tests-*/verification.json")))
    assert verify_report["verdict"] == "FAIL"


def test_real_timeout_preserved_and_never_pass(tmp_path):
    worker = tmp_path / "blocking.py"
    worker.write_text("import time\ntime.sleep(10)\n")
    logs = tmp_path / "timeout"
    logs.mkdir()
    with pytest.raises(tool.GateError, match="worker_timeout"):
        tool.command_logged([os.sys.executable, str(worker)], tmp_path, logs, 0.05,
                            os.environ.copy())
    process = tool.read(logs / "process.json")
    assert process["timeout"] is True and process["returncode"] == -9
    assert "returncode=-9 timeout=True" in (logs / "process.log").read_text()


def test_artifacts_in_repo_rejected(mini_repo):
    with pytest.raises(tool.GateError, match="inside_repository"):
        tool.new_artifacts(mini_repo / "artifacts", mini_repo, "bad-")


def test_empty_pytest_collection_is_failure(mini_repo, tmp_path):
    (mini_repo / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths=["tests"]\npython_files=["nothing*.py"]\n')
    git(mini_repo, "add", "pyproject.toml")
    git(mini_repo, "commit", "-m", "Freeze empty collection negative")
    with pytest.raises(tool.GateError, match="process_failed_rc=5"):
        tool.plan_run(mini_repo, tmp_path / "empty", [], shards=1)
    plan = tool.read(next((tmp_path / "empty").glob("plan-*/plan.json")))
    assert plan["verdict"] == "FAIL" and "rc=5" in plan["error"]


def test_missing_duplicate_junit_member_rejected(nodes, tmp_path):
    artifacts = tmp_path / "shard"
    child = artifacts / "tests-fixture"
    child.mkdir(parents=True)
    source = {"clean": True, "commit": "a" * 40, "tree": "b" * 40,
              "sha256": {"a.py": "c" * 64}}
    consistency = tool.verifier(tool.ROOT).check_source_consistency(source, source)
    plan = {"repo": str(tool.ROOT), "source": source, "collection": {"nodeids": nodes},
            "shard_timeout_s": 90}
    item = {"id": 0, "targets": ["tests/test_a.py"], "nodeids": nodes[:1],
            "predicted_test_s": 1}
    process = {"returncode": 0, "timeout": False, "wall_s": 1, "timeout_s": 90,
               "pid": 1, "command": tool.shard_command(tool.ROOT, item, artifacts)}
    tool.save(artifacts / "process.json", process)
    report = {"verdict": "PASS", "mode": "tests", "coverage": "explicit_targets",
              "targets": item["targets"], "source": source, "source_consistency": consistency,
              "tests": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
              "cuda_available": False}
    tool.save(child / "verification.json", report)
    xml(child / "pytest.xml", nodes[1:2])
    with pytest.raises(tool.GateError, match="nodeid_mismatch"):
        tool.accept_shard(plan, item, artifacts)
    xml(child / "pytest.xml", nodes[:1] * 2)
    with pytest.raises(tool.GateError, match="duplicate_junit"):
        tool.accept_shard(plan, item, artifacts)
    xml(child / "pytest.xml", nodes[:1])
    process["command"] = ["not-the-admitted-verify-entry"]
    tool.save(artifacts / "process.json", process)
    with pytest.raises(tool.GateError, match="process_command_mismatch"):
        tool.accept_shard(plan, item, artifacts)


def test_combine_requires_exact_full_collection(nodes):
    plan = {"shards": [{"id": 0}, {"id": 1}], "collection": {"nodeids": nodes}}
    counts = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    with pytest.raises(tool.GateError, match="partial"):
        tool.combine(plan, [{"id": 0, "nodeids": nodes[:1], "tests": counts}])
    with pytest.raises(tool.GateError, match="full_collection"):
        tool.combine(plan, [{"id": 0, "nodeids": nodes[:1], "tests": counts},
                            {"id": 1, "nodeids": nodes[:1], "tests": counts}])


@pytest.mark.parametrize("fixture_aware", [False, True])
def test_cli_real_tiny_plan_run_check(mini_repo, tmp_path, fixture_aware):
    root = tmp_path / "cli"
    root.mkdir()

    def invoke(*args):
        folder = root / f"invocation-{len(list(root.glob('invocation-*')))}"
        folder.mkdir()
        command = [os.sys.executable, "-B", str(tool.__file__), *args,
                   "--repo", str(mini_repo), "--artifact-dir", str(root / "outputs")]
        return tool.command_logged(command, tool.ROOT, folder, 90,
                                   tool.cpu_env(tool.ROOT, folder))

    extra = ("--fixture-aware", "--fixture-file", "tests/test_a.py") if fixture_aware else ()
    invoke("plan", "--shards", "3", "--shard-timeout", "90", *extra)
    path = next((root / "outputs").glob("plan-*/plan.json"))
    pin = tool.digest(path)
    invoke("run", "--plan", str(path), "--plan-sha256", pin, "--workers", "3")
    report = next((root / "outputs").glob("run-*/verification.json"))
    invoke("check", "--plan", str(path), "--plan-sha256", pin,
           "--report", str(report), "--report-sha256", tool.digest(report))
    audit = tool.read(next((root / "outputs").glob("check-*/acceptance.json")))
    assert audit["verdict"] == "PASS" and audit["tests"]["tests"] == 4
    assert tool.read(path)["schema_version"] == (2 if fixture_aware else 1)
    assert git(mini_repo, "status", "--porcelain") == ""


def freeze_test_file(repo, source, path="tests/test_a.py"):
    (repo / path).write_text(source, encoding="utf-8")
    git(repo, "add", path)
    git(repo, "commit", "-m", "Freeze fixture dependency acceptance input")


def test_fixture_components_transitive_cross_file_and_class(tmp_path):
    nodes = ["tests/test_a.py::test_a", "tests/test_a.py::test_b",
             "tests/test_b.py::test_c", "tests/test_c.py::TestC::test_d",
             "tests/test_c.py::TestC::test_e", "tests/test_c.py::test_f"]
    audit = {"contract": tool.FIXTURE_CONTRACT, "nodes": [
        {"nodeid": node, "constraints": constraints} for node, constraints in zip(nodes, [
            ["module:a", "session:shared"], ["module:a"], ["session:shared"],
            ["class:c"], ["class:c"], [],
        ])]}
    groups = tool.fixture_groups(nodes, audit)
    assert [group["nodeids"] for group in groups] == [nodes[:3], nodes[3:5], nodes[5:]]
    assignments, _, unknown = tool.lpt(nodes, [], shards=3, components=groups)
    assert unknown == nodes
    assert all(item["nodeids"] for item in assignments)
    assert len({node for item in assignments for node in item["nodeids"]}) == len(nodes)
    assert max(item["predicted_test_s"] for item in assignments) == 3


@pytest.mark.parametrize("source,reason", [
    ('import pytest\n@pytest.fixture(scope=lambda **kw: "module")\n'
     'def shared():\n    return 1\n', "dynamic_fixture_scope"),
    ('import pytest\n@pytest.fixture\n'
     'def shared(request):\n    return request.getfixturevalue("other")\n',
     "dynamic_fixture_lookup"),
    ('import pytest\n@pytest.fixture\n'
     'def shared(request):\n    lookup = request\n    return lookup\n',
     "unmodelled_fixture_request"),
    ('pytest_plugins = ["unknown"]\n', "dynamic_fixture_environment"),
    ('def pytest_generate_tests(metafunc):\n    pass\n', "dynamic_pytest_hook"),
    ('def setup_module():\n    pass\n', "unmodelled_xunit_scope"),
])
def test_static_unknown_fixture_environments_fail_closed(tmp_path, source, reason):
    path = tmp_path / "test_dynamic.py"
    path.write_text(source)
    with pytest.raises(tool.GateError, match=reason):
        tool.static_fixture_file(path)


def test_fixture_module_dependency_overbudget_saved_failure(mini_repo, tmp_path):
    freeze_test_file(mini_repo,
        'import pytest\n@pytest.fixture(scope="module")\n'
        'def base():\n    return 1\n@pytest.fixture\n'
        'def derived(base):\n    return base\n'
        'def test_left(derived):\n    assert derived == 1\n'
        'def test_right(derived):\n    assert derived == 1\n')
    nodes = ["tests/test_a.py::test_left", "tests/test_a.py::test_right"]
    history = xml(tmp_path / "module-cost.xml", nodes, seconds=160)
    with pytest.raises(tool.GateError, match="fixture_group_prediction_exceeds_deadline"):
        tool.plan_run(mini_repo, tmp_path / "module-plan", [history], shards=3,
                      fixture_aware=True, fixture_files=["tests/test_a.py"])
    path = next((tmp_path / "module-plan").glob("plan-*/plan.json"))
    plan = tool.read(path)
    assert plan["schema_version"] == 2 and plan["verdict"] == "FAIL"
    assert "fixture_group_is_indivisible" in plan["error"]
    group = next(group for group in plan["grouping"]["groups"] if nodes[0] in group["nodeids"])
    assert group["nodeids"] == nodes
    rows = plan["grouping"]["audit"]["nodes"]
    assert all(any(entry["scope"] == "module" for entry in row["closure"])
               for row in rows if row["nodeid"] in nodes)
    assert any(item["predicted_test_s"] == 320 for item in plan["shards"])
    assert not list((tmp_path / "module-plan").glob("**/shard-*"))
    forged = copy.deepcopy(plan)
    forged["verdict"] = "PASS"
    for item in forged["shards"]:
        item["predicted_test_s"] = 1
        item["group_test_s"] = {key: 1 for key in item["group_test_s"]}
    tool.save(path, forged)
    with pytest.raises(tool.GateError, match="fixture_group_prediction_exceeds_deadline"):
        tool.execute(mini_repo, path, tool.digest(path), tmp_path / "forged-fixture-run")
    assert not list((tmp_path / "forged-fixture-run").glob("run-*/shard-*"))


def test_fixture_session_dependency_joins_across_files(mini_repo, tmp_path):
    freeze_test_file(mini_repo,
        'import pytest\n@pytest.fixture(scope="session")\n'
        'def common():\n    return 1\n@pytest.fixture\n'
        'def indirect(common):\n    return common\n', "tests/conftest.py")
    freeze_test_file(mini_repo, "def test_left(indirect):\n    assert indirect == 1\n")
    freeze_test_file(mini_repo, "def test_right(common):\n    assert common == 1\n",
                     "tests/check_b.py")
    directory = tool.plan_run(mini_repo, tmp_path / "session-plan", [], shards=2,
                              fixture_aware=True,
                              fixture_files=["tests/check_b.py", "tests/test_a.py"])
    plan = tool.read(directory / "plan.json")
    group = next(group for group in plan["grouping"]["groups"]
                 if "tests/test_a.py::test_left" in group["nodeids"])
    assert group["nodeids"] == ["tests/check_b.py::test_right", "tests/test_a.py::test_left"]
    assert any("owner:session" in constraint for constraint in group["constraints"])
    tool.validate_plan(plan)


@pytest.mark.parametrize("source,reason", [
    ('import pytest\n@pytest.fixture(scope=lambda **kw: "module")\n'
     'def shared():\n    return 1\n'
     'def test_dynamic(shared):\n    assert shared == 1\n', "dynamic_fixture_scope"),
    ('import pytest\n@pytest.fixture\n'
     'def shared(request):\n    return request.getfixturevalue("tmp_path")\n'
     'def test_dynamic(shared):\n    assert shared\n', "dynamic_fixture_lookup"),
    ('def test_unknown(record_property):\n    assert record_property\n',
     "unknown_builtin_fixture"),
    ('import pytest\n@pytest.mark.parametrize("value", [1, 2], scope="module")\n'
     'def test_scoped_parameter(value):\n    assert value\n', "nonfunction_param_scope"),
    ('import pytest\n@pytest.fixture\n'
     'def value(request):\n    return request.param\n'
     '@pytest.mark.parametrize("value", [1, 2], indirect=True, scope="module")\n'
     'def test_indirect_parameter(value):\n    assert value\n', "nonfunction_param_scope"),
    ('def decorator(fn):\n    return fn\n'
     '@decorator\n'
     'def test_decorated():\n    assert True\n', "unknown_test_decorator"),
    ('import pytest\npytest.fixture = lambda fn: fn\n'
     '@pytest.fixture\n'
     'def value():\n    return 1\n'
     'def test_dynamic(value):\n    assert value\n', "dynamic_fixture_environment"),
])
def test_fixture_collection_unknown_saved_fail_no_admission(mini_repo, tmp_path, source, reason):
    freeze_test_file(mini_repo, source)
    with pytest.raises(tool.GateError, match="process_failed_rc"):
        tool.plan_run(mini_repo, tmp_path / "unknown-plan", [], fixture_aware=True,
                      fixture_files=["tests/test_a.py"])
    plan_path = next((tmp_path / "unknown-plan").glob("plan-*/plan.json"))
    failed = tool.read(plan_path)
    collected = tool.read(plan_path.parent / "collection/collection.json")
    assert failed["verdict"] == "FAIL" and reason in collected["fixture_error"]
    assert collected["nodeids"] and not list(plan_path.parent.glob("shard-*"))
    assert reason in (plan_path.parent / "collection/process.log").read_text()


def test_fixture_cheap_nodes_overbudget_file_eight_shards_real_run_replay(mini_repo, tmp_path):
    freeze_test_file(mini_repo,
        'import pytest\n@pytest.fixture\n'
        'def local_value():\n    return 1\n'
        '@pytest.mark.parametrize("number", range(6))\n'
        'def test_many(number, local_value, tmp_path_factory, pytestconfig):\n'
        '    assert number >= 0 and local_value == 1\n'
        '    first = tmp_path_factory.mktemp("same-prefix")\n'
        '    second = tmp_path_factory.mktemp("same-prefix")\n'
        '    assert first != second and first.parent == second.parent\n'
        '    assert pytestconfig.getini("testpaths") == ["tests"]\n'
        '    first.joinpath("owner").write_text(str(number))\n')
    expensive_nodes = [f"tests/test_a.py::test_many[{number}]" for number in range(6)]
    history = xml(tmp_path / "measured-cost.xml", expensive_nodes, seconds=60)
    assert sum(row["time_s"] for row in tool.parse_junit(history)[0]) == 360
    with pytest.raises(tool.GateError, match="file_prediction_exceeds_deadline"):
        tool.plan_run(mini_repo, tmp_path / "old-file-plan", [history], shards=3)
    directory = tool.plan_run(mini_repo, tmp_path / "fixture-plan", [history], shards=8,
                              shard_timeout=90, fixture_aware=True,
                              fixture_files=["tests/test_a.py"])
    path = directory / "plan.json"
    plan = tool.read(path)
    assert plan["schema_version"] == 2 and plan["grouping"]["opt_in"] is True
    assert plan["verdict"] == "PASS"
    assert len(plan["shards"]) == len(plan["grouping"]["groups"]) == 8
    assert len(plan["collection"]["nodeids"]) == 8
    assert all(item["targets"] and item["nodeids"] for item in plan["shards"])
    all_nodes = [node for item in plan["shards"] for node in item["nodeids"]]
    assert sorted(all_nodes) == sorted(plan["collection"]["nodeids"])
    assert len(all_nodes) == len(set(all_nodes))
    assert max(item["predicted_test_s"] for item in plan["shards"]) == 60
    assert set(plan["grouping"]["audit"]["isolated_builtins"]) == {
        "tmp_path_factory", "pytestconfig"}
    assert all(not any("tmp_path_factory" in value or "pytestconfig" in value
                       for value in group["constraints"])
               for group in plan["grouping"]["groups"])
    assert plan["unknown_nodeids"] == [
        "tests/check_b.py::TestB::test_custom_discovery", "tests/test_c.py::test_c"]
    bad = copy.deepcopy(plan)
    bad["grouping"]["groups"][0]["nodeids"].append(expensive_nodes[0])
    with pytest.raises(tool.GateError, match="fixture_group_recompute_mismatch"):
        tool.validate_plan(bad)
    bad = copy.deepcopy(plan)
    bad["shards"][0]["targets"].append(bad["shards"][1]["targets"][0])
    with pytest.raises(tool.GateError, match="target_nodeid_conflict"):
        tool.validate_plan(bad)
    bad = copy.deepcopy(plan)
    bad["grouping"]["opt_in"] = False
    with pytest.raises(tool.GateError, match="opt_in_required"):
        tool.validate_plan(bad)
    bad = copy.deepcopy(plan)
    bad["shards"][0]["group_test_s"] = {"0": 0}
    with pytest.raises(tool.GateError, match="prediction_assignment_mismatch"):
        tool.validate_plan(bad)
    run_dir = tool.execute(mini_repo, path, tool.digest(path), tmp_path / "fixture-run", workers=3)
    report_path = run_dir / "verification.json"
    report = tool.read(report_path)
    assert report["verdict"] == "PASS"
    assert report["tests"] == {"tests": 8, "failures": 0, "errors": 0, "skipped": 1}
    assert report["plan_schema_version"] == 2 and report["speedup"] is None
    events = []
    for item in report["shards"]:
        process = tool.read(Path(item["artifact_dir"]) / "process.json",
                            item["process_sha256"])
        assert process["returncode"] == 0 and process["timeout"] is False
        events.extend([(process["started_at"], 1), (process["finished_at"], -1)])
    flying, peak = 0, 0
    for _, change in sorted(events):
        flying += change
        peak = max(peak, flying)
    assert flying == 0 and 1 <= peak <= 3
    checked = tool.check(mini_repo, path, tool.digest(path), report_path,
                         tool.digest(report_path), tmp_path / "fixture-check")
    audit = tool.read(checked / "acceptance.json")
    assert audit["verdict"] == "PASS" and audit["tests"] == report["tests"]
    assert audit["source_consistency"]["verdict"] == "PASS"
    assert git(mini_repo, "status", "--porcelain") == ""
    private_dirs = [path for path in run_dir.glob("shard-*/tests-*/pytest-tmp/same-prefix*")
                    if path.is_dir() and not path.is_symlink()]
    assert len(private_dirs) == 12
    assert len({path.parent for path in private_dirs}) == 6
    assert sorted(path.joinpath("owner").read_text()
                  for path in private_dirs if path.joinpath("owner").is_file()) == [
                      str(number) for number in range(6)]


def test_fixture_audit_drift_and_sidecar_tamper_stop_before_admission(mini_repo, tmp_path):
    directory = tool.plan_run(mini_repo, tmp_path / "audit-plan", [], fixture_aware=True,
                              fixture_files=["tests/test_a.py"])
    path = directory / "plan.json"
    plan = tool.read(path)
    sidecar = Path(plan["collection"]["path"])
    collection = tool.read(sidecar)
    collection["fixture_audit"]["nodes"][0]["constraints"].append("fixture:forged")
    collection["fixture_audit"]["nodes"][0]["constraints"].sort()
    tool.save(sidecar, collection)
    with pytest.raises(tool.GateError, match="hash_mismatch"):
        tool.execute(mini_repo, path, tool.digest(path), tmp_path / "sidecar-run")
    assert not list((tmp_path / "sidecar-run").glob("run-*/shard-*"))
    plan["collection"]["sha256"] = tool.digest(sidecar)
    plan["grouping"]["audit"] = collection["fixture_audit"]
    plan["grouping"]["groups"] = tool.fixture_groups(
        collection["nodeids"], collection["fixture_audit"])
    plan["shards"], plan["history"], plan["unknown_nodeids"] = tool.lpt(
        collection["nodeids"], [], shards=3, components=plan["grouping"]["groups"])
    tool.save(path, plan)
    with pytest.raises(tool.GateError, match="fixture_audit_drift"):
        tool.execute(mini_repo, path, tool.digest(path), tmp_path / "forged-audit-run")
    assert not list((tmp_path / "forged-audit-run").glob("run-*/shard-*"))


def test_fixture_selection_preserves_unaudited_wholefiles(mini_repo, tmp_path):
    freeze_test_file(mini_repo,
        'import pytest\n@pytest.fixture\n'
        'def local(request):\n    request.node.user_properties.append(("fixture", "wholefile"))\n'
        'def test_whole_one(local):\n    assert local is None\n'
        'def test_whole_two(local):\n    assert local is None\n', "tests/test_c.py")
    directory = tool.plan_run(mini_repo, tmp_path / "narrow-plan", [], shards=3,
                              fixture_aware=True, fixture_files=["tests/test_a.py"])
    plan = tool.read(directory / "plan.json")
    rows = plan["grouping"]["audit"]["nodes"]
    assert all(not row["audited"] for row in rows
               if row["nodeid"].startswith("tests/test_c.py::"))
    group = next(group for group in plan["grouping"]["groups"]
                 if "tests/test_c.py::test_whole_one" in group["nodeids"])
    assert group["nodeids"] == ["tests/test_c.py::test_whole_one",
                               "tests/test_c.py::test_whole_two"]
    target = next(item for item in plan["shards"]
                  if "tests/test_c.py::test_whole_one" in item["nodeids"])
    assert "tests/test_c.py" in target["targets"]
    tool.validate_plan(plan)


def test_fixture_session_edge_outside_selection_rejected(mini_repo, tmp_path):
    freeze_test_file(mini_repo,
        'import pytest\n@pytest.fixture(scope="session")\n'
        'def common():\n    return 1\n', "tests/conftest.py")
    freeze_test_file(mini_repo, "def test_left(common):\n    assert common == 1\n")
    freeze_test_file(mini_repo, "def test_right(common):\n    assert common == 1\n",
                     "tests/check_b.py")
    with pytest.raises(tool.GateError, match="process_failed_rc"):
        tool.plan_run(mini_repo, tmp_path / "boundary-plan", [], fixture_aware=True,
                      fixture_files=["tests/test_a.py"])
    path = next((tmp_path / "boundary-plan").glob("plan-*/plan.json"))
    collection = tool.read(path.parent / "collection/collection.json")
    assert "session_dependency_outside_fixture_selection" in collection["fixture_error"]
    assert "tests/check_b.py::test_right" in collection["fixture_error"]
    assert tool.read(path)["verdict"] == "FAIL"


@pytest.mark.parametrize("selection", [[], ["tests/absent.py"],
                                      ["tests/test_a.py", "tests/test_a.py"]])
def test_fixture_explicit_selection_required(mini_repo, tmp_path, selection):
    with pytest.raises(tool.GateError):
        tool.plan_run(mini_repo, tmp_path / "selection-plan", [], fixture_aware=True,
                      fixture_files=selection)
    path = next((tmp_path / "selection-plan").glob("plan-*/plan.json"))
    assert tool.read(path)["verdict"] == "FAIL"
    assert not list(path.parent.glob("shard-*"))


@pytest.mark.parametrize("name,module,scope", [
    ("tmp_path_factory", "_pytest.tmpdir", "session"),
    ("pytestconfig", "_pytest.fixtures", "session"),
])
def test_builtin_isolation_identity_and_hash_gate(name, module, scope, tmp_path, monkeypatch):
    import importlib

    official = getattr(importlib.import_module(module), name)
    function = official._get_wrapped_function()
    path = Path(function.__code__.co_filename)
    fixture = SimpleNamespace(func=function, scope=scope)
    accepted = tool.builtin_isolation(name, fixture, path, pytest.__version__, pytest.__file__)
    assert accepted["origin"] == str(path.resolve())
    assert accepted["sha256"] == tool.digest(path)
    assert accepted["scope"] == "session" and accepted["module"] == module
    fixture.scope = "module"
    with pytest.raises(tool.GateError, match="builtin_isolation_not_qualified"):
        tool.builtin_isolation(name, fixture, path, pytest.__version__, pytest.__file__)
    fixture.scope = scope
    with pytest.raises(tool.GateError, match="builtin_isolation_not_qualified"):
        tool.builtin_isolation(name, fixture, tmp_path / "forged.py",
                               pytest.__version__, pytest.__file__)
    with pytest.raises(tool.GateError, match="builtin_isolation_not_qualified"):
        tool.builtin_isolation(name, fixture, path, "unqualified", pytest.__file__)
    monkeypatch.setattr(tool, "digest", lambda ignored: "unqualified-source-hash")
    with pytest.raises(tool.GateError, match="builtin_isolation_not_qualified"):
        tool.builtin_isolation(name, fixture, path, pytest.__version__, pytest.__file__)


@pytest.mark.parametrize("source,reason", [
    ('def test_temp(tmp_path_factory):\n'
     '    tmp_path_factory.mktemp("same", numbered=False)\n', "builtin_temp_not_unique"),
    ('def test_temp(tmp_path_factory):\n'
     '    factory = tmp_path_factory\n', "builtin_isolation_usage_unproven"),
    ('def test_config(pytestconfig):\n'
     '    pytestconfig.option.verbose = 9\n', "builtin_isolation_usage_unproven"),
    ('def test_config(pytestconfig):\n'
     '    pytestconfig.addinivalue_line("usefixtures", "shared")\n',
     "builtin_config_not_readonly"),
    ('def test_config(pytestconfig):\n'
     '    pytestconfig.getini("testpaths").append("shared")\n',
     "builtin_config_result_may_escape"),
    ('def test_config(pytestconfig):\n'
     '    config_value = pytestconfig.getini("testpaths")\n',
     "builtin_config_result_may_escape"),
])
def test_builtin_isolation_unsafe_usage_rejected(mini_repo, tmp_path, source, reason):
    freeze_test_file(mini_repo, source)
    with pytest.raises(tool.GateError, match="process_failed_rc"):
        tool.plan_run(mini_repo, tmp_path / "unsafe-builtin-plan", [], fixture_aware=True,
                      fixture_files=["tests/test_a.py"])
    path = next((tmp_path / "unsafe-builtin-plan").glob("plan-*/plan.json"))
    collection = tool.read(path.parent / "collection/collection.json")
    assert reason in collection["fixture_error"]
    assert tool.read(path)["verdict"] == "FAIL"


def test_user_fixture_named_builtin_is_never_exempt(mini_repo, tmp_path):
    freeze_test_file(mini_repo,
        'import pytest\n@pytest.fixture(scope="session")\n'
        'def tmp_path_factory():\n    return 7\n', "tests/conftest.py")
    freeze_test_file(mini_repo,
        "def test_left(tmp_path_factory):\n    assert tmp_path_factory == 7\n")
    freeze_test_file(mini_repo,
        "def test_right(tmp_path_factory):\n    assert tmp_path_factory == 7\n",
        "tests/check_b.py")
    with pytest.raises(tool.GateError, match="process_failed_rc"):
        tool.plan_run(mini_repo, tmp_path / "shadowed-plan", [], fixture_aware=True,
                      fixture_files=["tests/test_a.py"])
    path = next((tmp_path / "shadowed-plan").glob("plan-*/plan.json"))
    collection = tool.read(path.parent / "collection/collection.json")
    assert any(reason in collection["fixture_error"] for reason in (
        "unresolved_or_overridden_fixture", "session_dependency_outside_fixture_selection"))
    assert tool.read(path)["verdict"] == "FAIL"
