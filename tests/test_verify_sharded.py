"""Negative controls plus real subprocess acceptance on an external tiny repo."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import subprocess
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


@pytest.mark.parametrize("shards", [0, 4, 1.5])
def test_invalid_capacity_rejected(nodes, shards):
    with pytest.raises(tool.GateError, match="shards"):
        tool.lpt(nodes, [], shards)


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
    for shard in plan["shards"]:
        shard["nodeids"] = [new if node == old else node for node in shard["nodeids"]]
    tool.save(path, plan)
    with pytest.raises(tool.GateError, match="collection_drift"):
        tool.execute(repo, path, tool.digest(path), tmp_path / "bad-collection")
    assert not list((tmp_path / "bad-collection").glob("run-*/shard-*"))


def test_partial_failure_stops_queued_admission(mini_repo, tmp_path):
    (mini_repo / "tests/test_a.py").write_text(
        "def test_failure():\n    assert False, 'intentional admission negative'\n")
    git(mini_repo, "add", "tests/test_a.py")
    git(mini_repo, "commit", "-m", "Freeze deliberate failing fixture")
    history = xml(tmp_path / "cost.xml", ["tests/test_a.py::test_failure"], seconds=100)
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


def test_cli_real_tiny_plan_run_check(mini_repo, tmp_path):
    root = tmp_path / "cli"
    root.mkdir()

    def invoke(*args):
        folder = root / f"invocation-{len(list(root.glob('invocation-*')))}"
        folder.mkdir()
        command = [os.sys.executable, "-B", str(tool.__file__), *args,
                   "--repo", str(mini_repo), "--artifact-dir", str(root / "outputs")]
        return tool.command_logged(command, tool.ROOT, folder, 90,
                                   tool.cpu_env(tool.ROOT, folder))

    invoke("plan", "--shards", "3", "--shard-timeout", "90")
    path = next((root / "outputs").glob("plan-*/plan.json"))
    pin = tool.digest(path)
    invoke("run", "--plan", str(path), "--plan-sha256", pin, "--workers", "3")
    report = next((root / "outputs").glob("run-*/verification.json"))
    invoke("check", "--plan", str(path), "--plan-sha256", pin,
           "--report", str(report), "--report-sha256", tool.digest(report))
    audit = tool.read(next((root / "outputs").glob("check-*/acceptance.json")))
    assert audit["verdict"] == "PASS" and audit["tests"]["tests"] == 4
    assert git(mini_repo, "status", "--porcelain") == ""
