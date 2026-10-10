"""CPU-only fake transport controls; no real Kubernetes mutation is allowed."""

from __future__ import annotations

import copy
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import cluster_sharded_regression as tool
from scripts import verify_sharded as shard


def git(repo, *args):
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def tiny(tmp_path):
    repo = tmp_path / "repo"
    for name in ("scripts", "src", "benchmarks", "tests", "examples"):
        (repo / name).mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("verify.py", "verify_sharded.py", "cluster_sharded_regression.py"):
        relative = f"scripts/{name}"
        shutil.copyfile(tool.ROOT / relative, repo / relative)
        files.append(relative)
    for relative, content in {
        "pyproject.toml": '[tool.pytest.ini_options]\ntestpaths=["tests"]\n',
        "scripts/migrate_openpo.py": 'print("TINY_FIXTURE_PROVENANCE_ONLY")\n',
        "examples/portfolio_qp.py": 'print("TINY_FIXTURE_EXAMPLE_ONLY")\n',
        "tests/test_a.py": "def test_a():\n    assert 1 == 1\n",
        "tests/test_b.py": "def test_b():\n    assert 2 == 2\n",
        "tests/test_c.py": "def test_c():\n    assert 3 == 3\n",
    }.items():
        (repo / relative).write_text(content, encoding="utf-8")
        files.append(relative)
    git(repo, "init")
    git(repo, "config", "user.name", "CPU fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "add", *files)
    git(repo, "commit", "-m", "Freeze tiny external mock-transport fixture")
    plan_dir = shard.plan_run(repo, tmp_path / "original-plan", [], shard_timeout=90)
    plan_path = plan_dir / "plan.json"
    output = tool.prepare(repo, plan_path, shard.digest(plan_path), tmp_path / "cluster",
                          "existing-image:fixture")
    config_path = output / "cluster-plan.json"
    return repo, config_path, shard.digest(config_path)


@pytest.fixture
def fixture_tiny(tmp_path):
    repo = tmp_path / "fixture-repo"
    for name in ("scripts", "src", "benchmarks", "tests", "examples"):
        (repo / name).mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("verify.py", "verify_sharded.py", "cluster_sharded_regression.py"):
        relative = f"scripts/{name}"
        shutil.copyfile(tool.ROOT / relative, repo / relative)
        files.append(relative)
    for relative, content in {
        "pyproject.toml": '[tool.pytest.ini_options]\ntestpaths=["tests"]\n',
        "scripts/migrate_openpo.py": 'print("FIXTURE_TRANSPORT_PROVENANCE_ONLY")\n',
        "examples/portfolio_qp.py": 'print("FIXTURE_TRANSPORT_EXAMPLE_ONLY")\n',
        "tests/conftest.py": (
            'import pytest\n@pytest.fixture(scope="session")\n'
            'def shared():\n    return object()\n@pytest.fixture\n'
            'def dependent(shared):\n    return shared\n'),
        "tests/test_a.py": (
            'import pytest\n@pytest.fixture\n'
            'def independent(tmp_path):\n    return tmp_path\n'
            'def test_shared_a(dependent):\n    assert dependent is not None\n'
            'def test_independent_a(independent):\n    assert independent.is_dir()\n'),
        "tests/test_b.py": (
            'def test_shared_b(dependent):\n    assert dependent is not None\n'
            'def test_independent_b():\n    assert 2 == 2\n'),
        "tests/test_c.py": 'def test_unaudited():\n    assert 3 == 3\n',
    }.items():
        (repo / relative).write_text(content, encoding="utf-8")
        files.append(relative)
    git(repo, "init")
    git(repo, "config", "user.name", "Fixture v2 transport")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "add", *files)
    git(repo, "commit", "-m", "Freeze cross-file session and independent fixture input")
    plan_dir = shard.plan_run(
        repo, tmp_path / "original-fixture-plan", [], shard_timeout=90,
        fixture_aware=True, fixture_files=["tests/test_a.py", "tests/test_b.py"],
    )
    plan_path = plan_dir / "plan.json"
    output = tool.prepare(
        repo, plan_path, shard.digest(plan_path), tmp_path / "fixture-cluster",
        "existing-image:fixture", fixture_aware=True,
    )
    config_path = output / "cluster-plan.json"
    return repo, config_path, shard.digest(config_path)


def fake_runtime(repo):
    return {
        "python": sys.version, "executable": sys.executable,
        "implementation": "CPython", "packages": shard.verifier(repo).versions(),
        "modules": {"fixture": {"file": "FAKE_TRANSPORT_ONLY"}},
        "git": "fake-transport-git", "cuda_available": False,
    }


class FakeKubernetes:
    """Record calls and optionally run real CPU workers in the tiny fixture."""

    def __init__(self, config, execute=False, fail_create_at=None):
        self.config = config
        self.execute = execute
        self.fail_create_at = fail_create_at
        self.calls = []
        self.created = {}
        self.pending = False

    def query(self, args, artifacts):
        self.calls.append(args)
        artifacts.mkdir(parents=True, exist_ok=False)
        if args[0] == "create":
            if self.fail_create_at == len(self.created):
                raise shard.GateError("intentional_fake_transport_create_failure_rc=42")
            spec = shard.read(args[2])
            name = spec["metadata"]["name"]
            receipt = copy.deepcopy(spec)
            receipt["metadata"]["uid"] = f"fixture-uid-{name}"
            self.created[name] = receipt
            if self.execute:
                command = spec["spec"]["template"]["spec"]["containers"][0]["command"]
                values = dict(zip(command[4::2], command[5::2]))
                # command begins python, -B, script, worker; remaining args are pairs.
                phase = values["--phase"]
                tool.worker(Path(values["--plan"]), values["--plan-sha256"], phase,
                            Path(values["--output"]),
                            int(values["--shard-id"]) if "--shard-id" in values else None,
                            Path(values["--runtime"]) if "--runtime" in values else None,
                            values.get("--runtime-sha256"),
                            Path(values["--report"]) if "--report" in values else None,
                            values.get("--report-sha256"))
            shard.save(artifacts / "fixture-receipt.json", receipt)
            return receipt
        if args[1] == "job":
            receipt = copy.deepcopy(self.created[args[2]])
            receipt["status"] = {} if self.pending else {
                "succeeded": 1, "conditions": [{"type": "Complete", "status": "True"}]}
            shard.save(artifacts / "fixture-job.json", receipt)
            return receipt
        name = args[3].split("=", 1)[1]
        receipt = self.created[name]
        if self.pending:
            return {"items": []}
        pod = {
            "metadata": {"name": name + "-pod", "uid": "fixture-pod-" + name,
                         "ownerReferences": [{"uid": receipt["metadata"]["uid"],
                                              "controller": True}]},
            "spec": copy.deepcopy(receipt["spec"]["template"]["spec"]),
            "status": {"phase": "Succeeded", "containerStatuses": [{
                "restartCount": 0, "imageID": "fixture://sha256:" + "a" * 64,
                "state": {"terminated": {"exitCode": 0}}}]},
        }
        result = {"items": [pod]}
        shard.save(artifacts / "fixture-pod.json", result)
        return result

    def logs(self, name, artifacts):
        artifacts.mkdir()
        (artifacts / "stdout.log").write_text("FAKE_TRANSPORT_ONLY\n")


def test_prepare_freezes_full_clean_source_and_source_plan(tiny):
    repo, path, pin = tiny
    config, plan = tool.load_config(path, pin)
    assert plan["source"] == shard.snapshot(repo)
    assert shard.snapshot(Path(plan["repo"])) == plan["source"]
    assert git(Path(plan["repo"]), "remote") == ""
    assert config["gpu_count"] == 0
    assert config["smoke_id"] == 0
    assert len(plan["collection"]["nodeids"]) == 3


def prepare_cli(repo, plan_path, plan_pin, root, flags=()):
    process = root / "caller"
    process.mkdir(parents=True)
    command = [
        sys.executable, "-B", str(tool.ROOT / "scripts/cluster_sharded_regression.py"),
        "plan", "--repo", str(repo), "--plan", str(plan_path),
        "--plan-sha256", plan_pin, "--image", "existing-image:fixture",
        "--artifact-dir", str(root / "preparation"), *flags,
    ]
    failure = None
    try:
        shard.command_logged(command, tool.ROOT, process, 30, shard.cpu_env(tool.ROOT, process))
    except shard.GateError as error:
        failure = error
    result = shard.read(process / "process.json")
    assert result["timeout"] is False, "CLI timeout is not an expected rejection"
    receipts = list((root / "preparation").glob("cluster-*/prepare.json"))
    assert len(receipts) == 1
    receipt = shard.read(receipts[0])
    assert receipt["command"] == command and receipt["invocation"] == "CLI"
    assert shard.digest(receipt["log"]) == receipt["log_sha256"]
    assert receipt["returncode"] == result["returncode"]
    if failure is not None:
        assert "process_failed_rc=1" in str(failure)
    else:
        assert result["returncode"] == 0
    return receipts[0], receipt


@pytest.mark.parametrize("mutation,reason", [
    ("failed_plan", "original_overbudget_shared_fixture"),
    ("wrong_pin", "hash_mismatch"),
    ("missing_plan", "FileNotFoundError"),
    ("wrong_optin", "explicit_opt_in_required"),
])
def test_prepare_real_cli_rejection_retains_firstcause_without_admission(tiny, tmp_path,
                                                                        mutation, reason):
    repo, config_path, _ = tiny
    config = shard.read(config_path)
    plan_path, pin = Path(config["source_plan"]), config["source_plan_sha256"]
    flags = ()
    if mutation == "failed_plan":
        plan = shard.read(plan_path, pin)
        plan.update(verdict="FAIL", error="GateError: original_overbudget_shared_fixture")
        plan_path = tmp_path / "failed-source-plan.json"
        shard.save(plan_path, plan)
        pin = shard.digest(plan_path)
    elif mutation == "wrong_pin":
        pin = "0" * 64
    elif mutation == "missing_plan":
        plan_path = tmp_path / "missing-source-plan.json"
    else:
        flags = ("--fixture-aware",)
    original_pin = shard.digest(plan_path) if plan_path.is_file() else None
    before = shard.snapshot(repo)
    path, receipt = prepare_cli(repo, plan_path, pin, tmp_path / "refusal", flags)
    assert receipt["verdict"] == "FAIL" and receipt["returncode"] == 1
    assert reason in receipt["first_failure"]["error"]
    assert receipt["source_plan_sha256"] == pin
    assert receipt["first_failure"]["coordinate"] == str(plan_path.resolve())
    assert not (path.parent / "cluster-plan.json").exists()
    assert not (path.parent / "run.json").exists()
    assert not (path.parent / "source").exists()
    assert not list(path.parent.glob("job-*.json"))
    log = Path(receipt["log"]).read_text()
    assert reason in log and "verdict=FAIL returncode=1" in log
    if mutation == "failed_plan":
        assert "invalid_plan_status" in receipt["first_failure"]["gate_error"]
        retained_pin, log_pin = shard.digest(path), shard.digest(receipt["log"])
        _, second = prepare_cli(repo, plan_path, pin, tmp_path / "second-refusal")
        assert second["first_failure"]["error"] == receipt["first_failure"]["error"]
        assert shard.digest(path) == retained_pin and shard.digest(receipt["log"]) == log_pin
    assert shard.snapshot(repo) == before
    assert (shard.digest(plan_path) if plan_path.is_file() else None) == original_pin


def test_prepare_real_cli_success_preserves_original_config_semantics(tiny, tmp_path):
    repo, old_path, _ = tiny
    old = shard.read(old_path)
    path, receipt = prepare_cli(
        repo, Path(old["source_plan"]), old["source_plan_sha256"], tmp_path / "success")
    assert receipt["verdict"] == "PASS" and receipt["first_failure"] is None
    config_path = path.parent / "cluster-plan.json"
    config, plan = tool.load_config(config_path, receipt["cluster_plan_sha256"])
    expected = copy.deepcopy(old)
    expected.update(
        artifacts=str(path.parent), plan=str(path.parent / "plan.json"),
        plan_sha256=shard.digest(path.parent / "plan.json"), created_at=config["created_at"],
    )
    assert config == expected
    assert config_path.read_bytes() == shard.encoded(expected)
    original = shard.read(old["source_plan"], old["source_plan_sha256"])
    original.update(repo=str(path.parent / "source"), collection=plan["collection"])
    assert plan == original
    state = shard.read(path.parent / "run.json")
    assert state == {
        "state": "PREPARED", "jobs": [], "events": [], "first_failure": None,
        "config_sha256": shard.digest(config_path),
    }
    assert not list(path.parent.glob("job-*.json"))


def test_manifest_zero_gpu_no_install_no_token_no_retry(tiny):
    _, path, _ = tiny
    config = shard.read(path)
    spec, _ = tool.manifest(config, "probe", None)
    pod = spec["spec"]["template"]["spec"]
    assert "runtimeClassName" not in pod
    assert pod["automountServiceAccountToken"] is False
    assert pod["restartPolicy"] == "Never" and spec["spec"]["backoffLimit"] == 0
    assert spec["spec"]["activeDeadlineSeconds"] <= 300
    container = pod["containers"][0]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert {entry["name"]: entry["value"] for entry in container["env"]}[
        "NVIDIA_VISIBLE_DEVICES"] == "void"
    assert not any(word in " ".join(container["command"]) for word in ("pip install", "apt "))
    assert container["command"][0] == "python3"
    assert "ttlSecondsAfterFinished" not in spec["spec"]


def test_no_submit_flag_never_constructs_transport(tiny, monkeypatch):
    _, path, pin = tiny
    monkeypatch.setattr(tool, "Kubernetes", lambda _: pytest.fail("transport constructed"))
    with pytest.raises(shard.GateError, match="explicit_submit"):
        tool.admit(path, pin)


def test_create_failure_preserves_firstcause_and_blocks_reentry(tiny):
    _, path, pin = tiny
    client = FakeKubernetes(shard.read(path), fail_create_at=0)
    with pytest.raises(shard.GateError, match="rc=42"):
        tool.admit(path, pin, True, client)
    state = shard.read(path.parent / "run.json")
    assert state["state"] == "BLOCKED" and "rc=42" in state["first_failure"]["error"]
    first = state["first_failure"]
    with pytest.raises(shard.GateError, match="blocked"):
        tool.admit(path, pin, True, client)
    assert shard.read(path.parent / "run.json")["first_failure"] == first
    assert len(client.calls) == 1


def test_complete_job_missing_worker_is_not_pass(tiny):
    _, path, pin = tiny
    client = FakeKubernetes(shard.read(path))
    tool.admit(path, pin, True, client)
    with pytest.raises(FileNotFoundError):
        tool.collect(path, pin, client)
    state = shard.read(path.parent / "run.json")
    assert state["state"] == "BLOCKED"
    assert len(client.created) == 1


def test_real_cpu_workers_over_fake_cluster_full_chain(tiny, monkeypatch):
    _, path, pin = tiny
    monkeypatch.setattr(tool, "runtime_probe", fake_runtime)
    client = FakeKubernetes(shard.read(path), execute=True)
    assert tool.admit(path, pin, True, client)["state"] == "PROBE"
    client.pending = True
    assert tool.collect(path, pin, client)["state"] == "PROBE"
    assert len(client.created) == 1
    with pytest.raises(shard.GateError, match="stage_not_ready"):
        tool.admit(path, pin, True, client)
    client.pending = False
    assert tool.collect(path, pin, client)["state"] == "PROBE_PASSED"
    assert len(client.created) == 1
    assert tool.admit(path, pin, True, client)["state"] == "SMOKE"
    assert len(client.created) == 2
    assert tool.collect(path, pin, client)["state"] == "SMOKE_PASSED"
    assert tool.admit(path, pin, True, client)["state"] == "SHARDS"
    assert len(client.created) == 4
    assert tool.collect(path, pin, client)["state"] == "SHARDS_PASSED"
    assert tool.admit(path, pin, True, client)["state"] == "CHECK"
    assert tool.collect(path, pin, client)["state"] == "PASS"
    audit = tool.check(path, pin)
    assert audit["tests"] == {"tests": 3, "failures": 0, "errors": 0, "skipped": 0}
    assert all(args[0] in ("create", "get") for args in client.calls)
    assert tool.main(["check", "--plan", str(path), "--plan-sha256", pin]) == 0
    state = shard.read(path.parent / "run.json")
    duplicate = copy.deepcopy(state)
    duplicate["jobs"].append(copy.deepcopy(duplicate["jobs"][0]))
    tool.save(path.parent / "run.json", duplicate)
    with pytest.raises(shard.GateError, match="duplicate_stage_job"):
        tool.check(path, pin)
    tool.save(path.parent / "run.json", state)
    partial = copy.deepcopy(state)
    partial["jobs"] = [job for job in partial["jobs"] if job["phase"] != "check"]
    tool.save(path.parent / "run.json", partial)
    with pytest.raises(shard.GateError, match="membership_mismatch"):
        tool.check(path, pin)
    tool.save(path.parent / "run.json", state)
    summary = path.parent / "verification.json"
    raw = shard.read(summary)
    raw["tests"]["tests"] += 1
    shard.save(summary, raw)
    with pytest.raises(shard.GateError, match="hash_mismatch"):
        tool.check(path, pin)
    assert state["first_failure"] is None


def test_runtime_missing_dependency_failure_saved_before_shard(tiny, monkeypatch):
    _, path, _ = tiny
    config = shard.read(path)

    def missing(_):
        raise RuntimeError("missing highspy; deliberate probe negative")

    monkeypatch.setattr(tool, "runtime_probe", missing)
    output = path.parent / "missing-runtime"
    with pytest.raises(RuntimeError, match="missing highspy"):
        tool.worker(Path(config["plan"]), config["plan_sha256"], "probe", output)
    result = shard.read(output / "worker.json")
    assert result["verdict"] == "FAIL" and "missing highspy" in result["error"]
    assert not list(path.parent.glob("shard-*"))


def test_python_and_package_drift_failure_preserved(tiny, monkeypatch):
    _, path, _ = tiny
    config = shard.read(path)
    plan = shard.read(config["plan"])
    runtime = fake_runtime(Path(plan["repo"]))
    probe = path.parent / "probe-reference.json"
    shard.save(probe, {"verdict": "PASS", "phase": "probe", "runtime": runtime})
    changed = copy.deepcopy(runtime)
    changed["python"] = "different container interpreter"
    changed["packages"]["torch"] = "different package"
    monkeypatch.setattr(tool, "runtime_probe", lambda _: changed)
    output = path.parent / "drift-worker"
    with pytest.raises(shard.GateError, match="runtime_drift"):
        tool.worker(Path(config["plan"]), config["plan_sha256"], "shard", output, 0,
                    probe, shard.digest(probe))
    assert shard.read(output / "worker.json")["verdict"] == "FAIL"
    assert not (path.parent / "shard-0").exists()


@pytest.mark.parametrize("mutation,match", [
    ("uid", "uid_mismatch"), ("gpu", "manifest_drift"), ("failed", "job_failed"),
    ("restart", "restarted"), ("image", "image_drift"), ("extra", "extra_pod"),
    ("exit", "rc=137"),
])
def test_terminal_job_corruptions(tiny, mutation, match):
    _, path, _ = tiny
    config = shard.read(path)
    spec, item = tool.manifest(config, "probe", None)
    client = FakeKubernetes(config)
    client.created[item["name"]] = copy.deepcopy(spec)
    client.created[item["name"]]["metadata"]["uid"] = "expected-uid"
    job = client.query(["get", "job", item["name"], "-o", "json"], path.parent / "fake-job")
    pods = client.query(["get", "pods", "-l", f"job-name={item['name']}", "-o", "json"],
                        path.parent / "fake-pods")
    if mutation == "uid":
        job["metadata"]["uid"] = "wrong"
    elif mutation == "gpu":
        pods["items"][0]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = 1
    elif mutation == "failed":
        job["status"]["failed"] = 1
    elif mutation == "restart":
        pods["items"][0]["status"]["containerStatuses"][0]["restartCount"] = 1
    elif mutation == "image":
        pods["items"][0]["status"]["containerStatuses"][0]["imageID"] = "fixture://sha256:" + "b"*64
    elif mutation == "extra":
        pods["items"].append(copy.deepcopy(pods["items"][0]))
    else:
        pods["items"][0]["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = 137
    with pytest.raises(shard.GateError, match=match):
        tool.terminal_job(job, pods, spec, "expected-uid", "fixture://sha256:" + "a"*64)


def execution_archives(spec):
    job = copy.deepcopy(spec)
    job["metadata"]["uid"] = "expected-uid"
    job["status"] = {
        "succeeded": 1, "conditions": [{"type": "Complete", "status": "True"}],
    }
    pods = {"items": [{
        "metadata": {"uid": "expected-pod", "ownerReferences": [
            {"uid": "expected-uid", "controller": True}]},
        "spec": copy.deepcopy(spec["spec"]["template"]["spec"]),
        "status": {"phase": "Succeeded", "containerStatuses": [{
            "restartCount": 0, "imageID": "fixture://sha256:" + "a" * 64,
            "state": {"terminated": {"exitCode": 0}}}]},
    }]}
    return job, pods


@pytest.mark.parametrize("placement", ("job", "pod"))
@pytest.mark.parametrize("mutation", (
    "args", "envFrom", "env_value", "env_member", "duplicate_env", "null_args",
))
def test_terminal_execution_inputs_strict_without_unrelated_defaulting_rejection(
        placement, mutation):
    spec = {
        "metadata": {"name": "synthetic"},
        "spec": {"template": {"spec": {"containers": [{
            "resources": {"requests": {"cpu": "1"}, "limits": {"cpu": "1"}},
            "command": ["python3", "-B", "worker"], "env": [
                {"name": "PYTEST_ADDOPTS", "value": ""}],
        }]}}},
    }
    job, pods = execution_archives(spec)
    job["spec"]["template"]["spec"]["containers"][0]["args"] = []
    pods["items"][0]["spec"]["containers"][0]["envFrom"] = []
    pods["items"][0]["spec"]["dnsPolicy"] = "ClusterFirst"
    baseline = copy.deepcopy((job, pods))
    assert tool.terminal_job(job, pods, spec, "expected-uid")["exit_code"] == 0
    container = (job["spec"]["template"]["spec"] if placement == "job"
                 else pods["items"][0]["spec"])["containers"][0]
    if mutation == "args":
        container["args"] = ["--not-submitted"]
    elif mutation == "envFrom":
        container["envFrom"] = [{"configMapRef": {"name": "synthetic-not-submitted"}}]
    elif mutation == "env_value":
        container["env"][0]["value"] = "-k test_a"
    elif mutation == "env_member":
        container["env"].append({"name": "UNSUBMITTED", "value": "1"})
    elif mutation == "duplicate_env":
        container["env"].append(copy.deepcopy(container["env"][0]))
    else:
        container["args"] = None
    assert (job, pods) != baseline
    with pytest.raises(shard.GateError, match="execution_.*_drift|manifest_drift"):
        tool.terminal_job(job, pods, spec, "expected-uid")


def test_archived_manifest_coordinated_rehash_preserves_config_and_rejects(tiny):
    _, path, pin = tiny
    config, plan = tool.load_config(path, pin)
    spec, item = tool.manifest(config, "probe", None)
    job, pods = execution_archives(spec)
    hardware = tool.terminal_job(job, pods, spec, "expected-uid")
    output = Path(item["output"])
    (output / "collection").mkdir(parents=True)
    collection_path = output / "collection/collection.json"
    shard.save(collection_path, {"nodeids": plan["collection"]["nodeids"]})
    worker_path = output / "worker.json"
    shard.save(worker_path, {
        "verdict": "PASS", "phase": "probe", "plan_sha256": config["plan_sha256"],
        "source": plan["source"], "source_consistency": shard.verifier(
            Path(plan["repo"])).check_source_consistency(plan["source"], plan["source"]),
        "collection_sha256": shard.digest(collection_path),
    })
    job_path, pods_path = path.parent / "archived-job.json", path.parent / "archived-pods.json"
    shard.save(item["manifest"], spec)
    shard.save(job_path, job)
    shard.save(pods_path, pods)
    item.update(
        uid="expected-uid", accepted=True, hardware=hardware,
        manifest_sha256=shard.digest(item["manifest"]), worker_sha256=shard.digest(worker_path),
        cluster_evidence={"job": str(job_path), "job_sha256": shard.digest(job_path),
                          "pods": str(pods_path), "pods_sha256": shard.digest(pods_path)},
    )
    state = {"state": "PROBE_PASSED", "jobs": [item], "events": [], "first_failure": None,
             "config_sha256": pin, "image_id": hardware["image_id"]}
    tool.stage_members(config, plan, state)
    tool.verify_accepted(config, plan, state)
    baseline = copy.deepcopy((spec, job, pods, state))
    for mutation in ("args", "env_value", "env_membership", "actual_args_only", "envFrom"):
        intended, observed, placed, changed_state = copy.deepcopy(baseline)
        containers = [observed["spec"]["template"]["spec"]["containers"][0],
                      placed["items"][0]["spec"]["containers"][0]]
        if mutation != "actual_args_only":
            containers.append(intended["spec"]["template"]["spec"]["containers"][0])
        for container in containers:
            if mutation in ("args", "actual_args_only"):
                container["args"] = ["--not-submitted"]
            elif mutation == "env_value":
                next(entry for entry in container["env"] if entry["name"] ==
                     "PYTEST_ADDOPTS")["value"] = "-k test_a"
            elif mutation == "env_membership":
                container["env"].append({"name": "UNSUBMITTED", "value": "1"})
            else:
                container["envFrom"] = [{"configMapRef": {"name": "synthetic-not-submitted"}}]
        shard.save(item["manifest"], intended)
        shard.save(job_path, observed)
        shard.save(pods_path, placed)
        changed = changed_state["jobs"][0]
        changed["manifest_sha256"] = shard.digest(item["manifest"])
        changed["cluster_evidence"].update(job_sha256=shard.digest(job_path),
                                         pods_sha256=shard.digest(pods_path))
        assert shard.digest(path) == pin
        assert shard.read(item["manifest"], changed["manifest_sha256"]) == intended
        assert shard.read(job_path, changed["cluster_evidence"]["job_sha256"]) == observed
        assert shard.read(pods_path, changed["cluster_evidence"]["pods_sha256"]) == placed
        tool.stage_members(config, plan, changed_state)
        reason = ("execution_args_drift" if mutation == "actual_args_only"
                  else "archived_manifest_config_drift")
        with pytest.raises(shard.GateError, match=reason):
            tool.verify_accepted(config, plan, changed_state)
    shard.save(item["manifest"], spec)
    shard.save(job_path, job)
    shard.save(pods_path, pods)
    for key in ("output", "manifest", "name"):
        changed_state = copy.deepcopy(state)
        changed_state["jobs"][0][key] += "-unsubmitted"
        with pytest.raises(shard.GateError, match="archived_stage_item_drift"):
            tool.intended_manifest(config, changed_state, changed_state["jobs"][0])
    tool.verify_accepted(config, plan, state)


def test_active_manifest_coordinated_rehash_blocks_collection_before_query(tiny):
    _, path, pin = tiny
    client = FakeKubernetes(shard.read(path))
    state = tool.admit(path, pin, True, client)
    item = state["jobs"][0]
    spec = shard.read(item["manifest"], item["manifest_sha256"])
    spec["spec"]["template"]["spec"]["containers"][0]["args"] = ["--not-submitted"]
    shard.save(item["manifest"], spec)
    item["manifest_sha256"] = shard.digest(item["manifest"])
    tool.save(path.parent / "run.json", state)
    before = list(client.calls)
    with pytest.raises(shard.GateError, match="archived_manifest_config_drift"):
        tool.collect(path, pin, client)
    stopped = shard.read(path.parent / "run.json")
    assert stopped["state"] == "BLOCKED"
    assert "archived_manifest_config_drift" in stopped["first_failure"]["error"]
    assert client.calls == before and len(client.created) == 1


def test_pending_not_pass_and_source_drift_stops(tiny):
    repo, path, pin = tiny
    config = shard.read(path)
    spec, _ = tool.manifest(config, "probe", None)
    job = copy.deepcopy(spec)
    job["metadata"]["uid"] = "expected"
    assert tool.terminal_job(job, {"items": []}, spec, "expected") is None
    (repo / "tests/test_a.py").write_text("changed\n")
    with pytest.raises(shard.GateError, match="source_not_clean"):
        tool.load_config(path, pin)


def test_reject_host_environment_and_local_artifacts(tiny, tmp_path):
    repo, path, _ = tiny
    config = shard.read(path)
    with pytest.raises(shard.GateError, match="host_environment"):
        tool.prepare(repo, Path(config["source_plan"]), config["source_plan_sha256"],
                     tmp_path / "host-env", "image:existing",
                     python="/hf3fs-jd/data/tmp/openopt-validation-env/bin/python")
    with pytest.raises(shard.GateError, match="shared_3fs"):
        tool.shared(Path("/data/dev/forbidden-artifacts"))


def test_runtime_probe_calls_versions_before_import(monkeypatch):
    class Missing:
        DEPENDENCIES = {"definitely-missing-openopt-test-package": "1"}

        def versions(self):
            raise RuntimeError("missing SDK-free dependency fixture")

    monkeypatch.setattr(shard, "verifier", lambda _: Missing())
    monkeypatch.setattr(tool.importlib, "import_module",
                        lambda _: pytest.fail("must fail before imports"))
    with pytest.raises(RuntimeError, match="missing SDK-free"):
        tool.runtime_probe(tool.ROOT)


def test_missing_probe_preserves_partial_inventory_before_failure(tiny, monkeypatch):
    _, path, _ = tiny
    config = shard.read(path)
    inventory = {"python": "fixture-python", "dependencies": {
        "highspy": {"status": "MISSING", "version": None}}}

    def missing(_):
        raise tool.RuntimeProbeError("missing highspy", inventory)

    monkeypatch.setattr(tool, "runtime_probe", missing)
    output = path.parent / "partial-probe"
    with pytest.raises(tool.RuntimeProbeError, match="missing highspy"):
        tool.worker(Path(config["plan"]), config["plan_sha256"], "probe", output)
    result = shard.read(output / "worker.json")
    assert result["runtime_probe_inventory"] == inventory
    assert result["verdict"] == "FAIL"


def test_real_versions_failure_keeps_missing_inventory(monkeypatch):
    class Missing:
        DEPENDENCIES = {"definitely-missing-openopt-test-package": "1"}

        def versions(self):
            raise RuntimeError("missing definitely-missing-openopt-test-package")

    monkeypatch.setattr(shard, "verifier", lambda _: Missing())
    with pytest.raises(tool.RuntimeProbeError) as raised:
        tool.runtime_probe(tool.ROOT)
    inventory = raised.value.inventory
    assert inventory["dependencies"]["definitely-missing-openopt-test-package"] == {
        "status": "MISSING", "version": None}
    assert inventory["python"] == sys.version
    assert inventory["commercial_qualification"].startswith("UNVERIFIED")


def test_worker_failure_keeps_dependency_firstcause_when_job_failed(tiny):
    _, path, pin = tiny
    client = FakeKubernetes(shard.read(path))
    tool.admit(path, pin, True, client)
    state = shard.read(path.parent / "run.json")
    output = Path(state["jobs"][0]["output"])
    output.mkdir()
    shard.save(output / "worker.json", {
        "verdict": "FAIL", "error": "RuntimeProbeError: missing highspy",
        "runtime_probe_inventory": {"dependencies": {"highspy": "MISSING"}},
    })
    with pytest.raises(shard.GateError, match="missing highspy"):
        tool.collect(path, pin, client)
    state = shard.read(path.parent / "run.json")
    assert state["state"] == "BLOCKED"
    assert "missing highspy" in state["first_failure"]["error"]
    assert len(client.created) == 1


def test_job_name_and_actual_heaviest_gate(tiny):
    _, path, _ = tiny
    config = shard.read(path)
    config["artifacts"] = str(path.parent.with_name("cluster-a_b_cdef"))
    spec, _ = tool.manifest(config, "probe", None)
    assert "_" not in spec["metadata"]["name"]
    # Gate selection comes from measured plan costs, not implicit shard 0.
    plan = shard.read(config["plan"])
    plan["shards"][2]["predicted_test_s"] = 900
    selected = min(plan["shards"], key=lambda item: (
        -item["predicted_test_s"], item["id"]))["id"]
    assert selected == 2


def test_prior_probe_corruption_blocks_before_create(tiny, monkeypatch):
    _, path, pin = tiny
    monkeypatch.setattr(tool, "runtime_probe", fake_runtime)
    client = FakeKubernetes(shard.read(path), execute=True)
    tool.admit(path, pin, True, client)
    tool.collect(path, pin, client)
    probe_path = path.parent / "worker-probe/worker.json"
    result = shard.read(probe_path)
    result["runtime"]["python"] = "tampered"
    shard.save(probe_path, result)
    with pytest.raises(shard.GateError, match="hash_mismatch"):
        tool.admit(path, pin, True, client)
    state = shard.read(path.parent / "run.json")
    assert state["state"] == "BLOCKED" and len(client.created) == 1
    assert state["first_failure"]["coordinate"] == "prior_stage_evidence"


def test_partial_parallel_admission_preserves_receipt_and_stops(tiny, monkeypatch):
    _, path, pin = tiny
    config, plan = tool.load_config(path, pin)
    fake_jobs = [
        {"phase": "probe", "id": None, "uid": "fixture-probe", "name": "fixture-probe",
         "accepted": True},
        {"phase": "shard", "id": config["smoke_id"], "uid": "fixture-smoke",
         "name": "fixture-smoke", "accepted": True},
    ]
    tool.save(path.parent / "run.json", {
        "state": "SMOKE_PASSED", "config_sha256": pin, "jobs": fake_jobs, "events": [],
        "first_failure": None, "runtime_sha256": "a" * 64,
    })
    # This control isolates transport failure from previously tested evidence replay.
    monkeypatch.setattr(tool, "verify_accepted", lambda *_: None)
    client = FakeKubernetes(config, fail_create_at=1)
    with pytest.raises(shard.GateError, match="rc=42"):
        tool.admit(path, pin, True, client)
    state = shard.read(path.parent / "run.json")
    assert state["state"] == "BLOCKED" and len(client.created) == 1
    assert len(state["jobs"]) == 3
    assert len([call for call in client.calls if call[0] == "create"]) == 2
    with pytest.raises(shard.GateError, match="blocked"):
        tool.admit(path, pin, True, client)
    assert len(client.calls) == 2


def test_actual_local_cpu_runtime_probe_not_container_evidence():
    result = tool.runtime_probe(tool.ROOT)
    assert result["python"] == sys.version and result["executable"] == sys.executable
    assert result["cuda_available"] is False
    assert set(result["packages"]) == set(shard.verifier(tool.ROOT).DEPENDENCIES)
    assert all(value["status"] == "FOUND" for value in result["inventory"]["dependencies"].values())
    assert result["inventory"]["commercial_qualification"].startswith("UNVERIFIED")


def test_fixture_v2_explicit_optin_and_full_metadata_not_nodes_rebuild(fixture_tiny, tmp_path):
    repo, path, pin = fixture_tiny
    config, plan = tool.load_config(path, pin)
    original = shard.read(config["source_plan"], config["source_plan_sha256"])
    assert config["schema_version"] == plan["schema_version"] == 2
    assert config["fixture_transport_contract"] == tool.FIXTURE_TRANSPORT
    assert shard.read(plan["collection"]["path"]) == shard.read(original["collection"]["path"])
    assert plan["grouping"] == original["grouping"] and plan["shards"] == original["shards"]
    audit = plan["grouping"]["audit"]
    assert audit["runtime_sha256"] and audit["isolated_builtins"]["tmp_path_factory"]
    shared = next(group for group in plan["grouping"]["groups"]
                  if "tests/test_a.py::test_shared_a" in group["nodeids"])
    assert shared["nodeids"] == [
        "tests/test_a.py::test_shared_a", "tests/test_b.py::test_shared_b",
    ]
    assert any(set(shared["nodeids"]).issubset(item["nodeids"]) for item in plan["shards"])
    with pytest.raises(shard.GateError, match="explicit_opt_in_required"):
        tool.prepare(repo, Path(config["source_plan"]), config["source_plan_sha256"],
                     tmp_path / "missing-optin", "existing-image:fixture")
    assert tool.main([
        "check", "--plan", str(path), "--plan-sha256", pin, "--fixture-aware",
    ]) == 1


def test_fixture_v2_prepare_real_cli_success_preserves_full_transport(fixture_tiny, tmp_path):
    repo, old_path, _ = fixture_tiny
    old = shard.read(old_path)
    path, receipt = prepare_cli(
        repo, Path(old["source_plan"]), old["source_plan_sha256"],
        tmp_path / "v2-success", ("--fixture-aware",))
    assert receipt["verdict"] == "PASS" and receipt["first_failure"] is None
    config_path = path.parent / "cluster-plan.json"
    config, plan = tool.load_config(config_path, receipt["cluster_plan_sha256"])
    expected = copy.deepcopy(old)
    expected.update(
        artifacts=str(path.parent), plan=str(path.parent / "plan.json"),
        plan_sha256=shard.digest(path.parent / "plan.json"), created_at=config["created_at"],
    )
    assert config == expected
    assert config_path.read_bytes() == shard.encoded(expected)
    original = shard.read(old["plan"], old["plan_sha256"])
    original.update(repo=str(path.parent / "source"), collection=plan["collection"])
    assert plan == original
    assert shard.read(plan["collection"]["path"])["fixture_audit"] == original["grouping"]["audit"]
    assert not list(path.parent.glob("job-*.json"))
    assert shard.read(path.parent / "run.json")["state"] == "PREPARED"


def test_fixture_v2_runtime_install_path_mapping_preserves_all_semantics(fixture_tiny, tmp_path):
    _, path, _ = fixture_tiny
    plan = shard.read(shard.read(path)["plan"])
    mapping = tool.local_runtime_mapping(plan)
    moved = [dict(item, runtime_path=str(tmp_path / "pod-site" / Path(item["host_path"]).name))
             for item in mapping]
    effective, collection = tool.mapped_plan(plan, moved, tmp_path / "pod-collection.json")
    assert effective["shards"] == plan["shards"]
    assert effective["grouping"]["groups"] == plan["grouping"]["groups"]
    assert effective["history"] == plan["history"]
    audit = effective["grouping"]["audit"]
    assert set(audit["runtime_sha256"]) == {item["runtime_path"] for item in moved}
    assert audit["isolated_builtins"]["tmp_path_factory"]["origin"] in audit["runtime_sha256"]
    assert collection["fixture_audit"] == audit
    assert collection["nodeids"] == plan["collection"]["nodeids"]
    assert effective["collection"]["sha256"] == tool.value_pin(collection)
    bad = copy.deepcopy(moved)
    bad[0]["sha256"] = "0" * 64
    assert bad != moved
    with pytest.raises(shard.GateError, match="mapping_identity_mismatch"):
        tool.mapped_plan(plan, bad, tmp_path / "bad-collection.json")


@pytest.mark.parametrize("mutation,reason", [
    ("version", "pytest_version_drift"),
    ("module", "runtime_module_mismatch"),
    ("hash", "runtime_source_drift"),
    ("origin", "runtime_origin_drift"),
    ("sidecar", "full_collection_mismatch"),
])
def test_fixture_v2_runtime_negatives_stop_before_execution(fixture_tiny, monkeypatch, mutation, reason):
    _, path, _ = fixture_tiny
    config = shard.read(path)
    plan = shard.read(config["plan"])
    if mutation == "version":
        monkeypatch.setattr(pytest, "__version__", "unqualified-version")
    elif mutation == "module":
        plan["cluster_fixture_transport"]["runtime_files"][0]["module"] = "_pytest.capture"
    elif mutation == "hash":
        real_digest = shard.digest
        runtime_file = tool.local_runtime_mapping(plan)[0]["runtime_path"]

        def changed_digest(value):
            return "0" * 64 if str(value) == runtime_file else real_digest(value)

        monkeypatch.setattr(shard, "digest", changed_digest)
    elif mutation == "origin":
        from types import SimpleNamespace
        original_import = tool.importlib.import_module
        entry = plan["cluster_fixture_transport"]["runtime_files"][0]
        misplaced = path.parent / "outside-pytest.py"
        shutil.copyfile(entry["host_path"], misplaced)
        assert shard.digest(misplaced) == entry["sha256"]

        def relocated_module(name, *args, **kwargs):
            if name == entry["module"]:
                return SimpleNamespace(__name__=name, __file__=str(misplaced))
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(tool.importlib, "import_module", relocated_module)
    else:
        sidecar = shard.read(plan["collection"]["path"])
        sidecar.pop("fixture_audit")
        shard.save(plan["collection"]["path"], sidecar)
        plan["collection"]["sha256"] = shard.digest(plan["collection"]["path"])
    shard.save(config["plan"], plan)
    monkeypatch.setattr(tool, "runtime_probe", fake_runtime)
    monkeypatch.setattr(shard, "launch_shard", lambda *_: pytest.fail("no shard admission"))
    output = path.parent / "negative-worker"
    with pytest.raises(shard.GateError, match=reason):
        tool.worker(Path(config["plan"]), shard.digest(config["plan"]), "probe", output)
    result = shard.read(output / "worker.json")
    assert result["verdict"] == "FAIL" and reason in result["error"]
    assert not list(path.parent.glob("shard-*"))


def test_fixture_v2_rehashed_constraint_forgery_fails_real_pod_collection(fixture_tiny, monkeypatch):
    _, path, _ = fixture_tiny
    config = shard.read(path)
    plan = shard.read(config["plan"])
    audit = plan["grouping"]["audit"]
    audit["nodes"][1]["constraints"].append("fixture:invented-shared")
    audit["nodes"][1]["constraints"].sort()
    collection = {"nodeids": plan["collection"]["nodeids"], "fixture_audit": audit}
    shard.save(plan["collection"]["path"], collection)
    plan["collection"]["sha256"] = shard.digest(plan["collection"]["path"])
    plan["grouping"]["groups"] = shard.fixture_groups(collection["nodeids"], audit)
    plan["shards"], plan["history"], plan["unknown_nodeids"] = shard.lpt(
        collection["nodeids"], [], 3, components=plan["grouping"]["groups"],
    )
    plan["cluster_fixture_transport"]["host_audit_sha256"] = tool.value_pin(audit)
    shard.validate_plan(plan)
    shard.save(config["plan"], plan)
    monkeypatch.setattr(tool, "runtime_probe", fake_runtime)
    output = path.parent / "forged-audit-worker"
    with pytest.raises(shard.GateError, match="fixture_audit_drift"):
        tool.worker(Path(config["plan"]), shard.digest(config["plan"]), "probe", output)
    result = shard.read(output / "worker.json")
    assert result["verdict"] == "FAIL" and "fixture_audit_drift" in result["error"]
    assert shard.read(output / "collection/collection.json")["nodeids"] == collection["nodeids"]
    assert not list(path.parent.glob("shard-*"))


def test_fixture_v2_real_cpu_workers_full_transport_replay(fixture_tiny, monkeypatch):
    _, path, pin = fixture_tiny
    monkeypatch.setattr(tool, "runtime_probe", fake_runtime)
    client = FakeKubernetes(shard.read(path), execute=True)
    for stage, terminal in (
        ("PROBE", "PROBE_PASSED"), ("SMOKE", "SMOKE_PASSED"),
        ("SHARDS", "SHARDS_PASSED"), ("CHECK", "PASS"),
    ):
        assert tool.admit(path, pin, True, client)["state"] == stage
        assert tool.collect(path, pin, client)["state"] == terminal
    audit = tool.check(path, pin)
    assert audit["tests"] == {"tests": 5, "failures": 0, "errors": 0, "skipped": 0}
    config, plan = tool.load_config(path, pin)
    state = shard.read(path.parent / "run.json")
    probe = shard.read(path.parent / "worker-probe/worker.json", state["runtime_sha256"])
    effective, _, effective_pin = tool.execution_plan(config, plan, probe)
    assert effective["grouping"]["groups"] == plan["grouping"]["groups"]
    assert effective["shards"] == plan["shards"]
    summary = shard.read(path.parent / "verification.json", state["report_sha256"])
    assert summary["plan_sha256"] == effective_pin
    assert summary["source_transport_plan_sha256"] == config["plan_sha256"]
    assert summary["plan_schema_version"] == 2
    assert summary["source_consistency"]["verdict"] == "PASS"
    assert len(client.created) == 5
    shard_jobs = [job for job in state["jobs"] if job["phase"] == "shard"]
    shared_nodes = {"tests/test_a.py::test_shared_a", "tests/test_b.py::test_shared_b"}
    shared_owners = [
        job for job in shard_jobs
        if shared_nodes.intersection(shard.read(Path(job["output"]) / "worker.json")["shard"]["nodeids"])
    ]
    assert len(shared_owners) == 1
    for job in shard_jobs:
        result = shard.read(Path(job["output"]) / "worker.json", job["worker_sha256"])
        fresh = shard.read(Path(job["output"]) / "collection/collection.json",
                           result["collection_sha256"])
        assert fresh["fixture_audit"] == effective["grouping"]["audit"]
    receipt = {
        "kind": "real_cpu_fixture_with_fake_cluster_not_container_qualification",
        "source": plan["source"], "source_consistency": summary["source_consistency"],
        "tests": audit["tests"], "host_audit_sha256": plan["cluster_fixture_transport"]["host_audit_sha256"],
        "runtime_mapping": probe["fixture_runtime_mapping"],
        "runtime_plan_sha256": effective_pin, "source_plan_sha256": config["plan_sha256"],
        "shared_component_owner_count": len(shared_owners), "job_count": len(client.created),
        "speedup": None,
    }
    receipt_path = path.parent / "fixture-transport-acceptance.json"
    shard.save(receipt_path, receipt)
    assert json.loads(receipt_path.read_text()) == receipt
    summary_path = path.parent / "verification.json"
    corrupted_summary = copy.deepcopy(summary)
    corrupted_summary["source_transport_plan_sha256"] = "0" * 64
    shard.save(summary_path, corrupted_summary)
    corrupted_state = copy.deepcopy(state)
    corrupted_state["report_sha256"] = shard.digest(summary_path)
    tool.save(path.parent / "run.json", corrupted_state)
    with pytest.raises(shard.GateError, match="archived_manifest_config_drift"):
        tool.check(path, pin)
    shard.save(summary_path, summary)
    tool.save(path.parent / "run.json", state)
    tampered = copy.deepcopy(probe)
    tampered["fixture_runtime_mapping"][0]["sha256"] = "0" * 64
    with pytest.raises(shard.GateError, match="mapping_identity_mismatch"):
        tool.execution_plan(config, plan, tampered)
    effective["shards"][0]["nodeids"].append("tests/false.py::test_false")
    shard.save(Path(probe["execution_plan"]), effective)
    with pytest.raises(shard.GateError, match="hash_mismatch"):
        tool.check(path, pin)


def accept_batch(batch, artifact_root):
    """Bound each real CPU batch below 180 seconds, retaining process and JUnit."""
    from scripts import verify
    root = tool.shared(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{batch}-", dir=root))
    process = output / "process"
    process.mkdir()
    prepare_tests = {
        "test_prepare_real_cli_rejection_retains_firstcause_without_admission",
        "test_prepare_real_cli_success_preserves_original_config_semantics",
    }
    family = "v1" if batch in ("v1-a", "v1-b") else batch
    targets = [
        f"tests/test_cluster_sharded_regression.py::{name}"
        for name in sorted(globals()) if name.startswith("test_")
        and ("fixture-v2" if name.startswith("test_fixture_v2_")
             else "prepare" if name in prepare_tests else "v1") == family
    ]
    if batch in ("v1-a", "v1-b"):
        targets = targets[0::2] if batch == "v1-a" else targets[1::2]
    report = {
        "kind": "bounded-real-cpu-fake-cluster-tests",
        "batch": batch, "verdict": "RUNNING", "targets": targets,
        "deadline_s": 179, "started_at": tool.stamp(),
        "qualification": "CPU fixture correctness only; no cluster/container/speed claim",
    }
    try:
        shard.require(targets and batch in ("v1", "v1-a", "v1-b", "fixture-v2", "prepare"),
                      "empty_or_invalid_test_batch")
        before = verify.source_snapshot(tool.ROOT)
        report["source"] = before
        command = [sys.executable, "-B", "scripts/verify.py", "tests",
                   "--artifact-dir", str(output / "verification")]
        for target in targets:
            command.extend(["--target", target])
        report["process"] = shard.command_logged(
            command, tool.ROOT, process, 179, shard.cpu_env(tool.ROOT, process))
        shard.require(report["process"]["wall_s"] < 180, "cpu_batch_over_180_seconds")
        receipts = list((output / "verification").glob("tests-*/verification.json"))
        shard.require(len(receipts) == 1, "batch_verification_membership_mismatch")
        receipt = shard.read(receipts[0])
        shard.require(receipt["verdict"] == "PASS" and receipt["targets"] == targets
                      and receipt["source"] == before, "batch_verification_not_bound")
        records, counts = shard.parse_junit(receipts[0].parent / "pytest.xml")
        shard.require(records and counts == receipt["tests"] and not any(
            counts[key] for key in ("failures", "errors", "skipped")), "batch_junit_not_pass")
        report.update(tests=counts, verification=str(receipts[0]),
                      verification_sha256=shard.digest(receipts[0]),
                      source_consistency=verify.check_source_consistency(
                          before, verify.source_snapshot(tool.ROOT)))
        shard.require(report["source_consistency"]["verdict"] == "PASS", "batch_source_drift")
        report["verdict"] = "PASS"
    except Exception as error:
        report.update(verdict="FAIL", first_failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at"] = tool.stamp()
        shard.save(output / "acceptance.json", report)
        print(f"CLUSTER_CPU_BATCH verdict={report['verdict']} batch={batch} "
              f"receipt={output / 'acceptance.json'}", flush=True)
    return output


def check_batches(paths, pins, artifact_root):
    """Reaccept complete disjoint clean-source CPU batches against real collection."""
    output = shard.new_artifacts(tool.shared(artifact_root), tool.ROOT, "batch-check-")
    audit = {"verdict": "RUNNING", "kind": "full-wrapper-cpu-batch-replay",
             "started_at": tool.stamp(), "qualification": "CPU only; no K8s/GPU/speed proof"}
    try:
        assert len(paths) == len(pins) == 4, "require four caller-pinned batch receipts"
        source = shard.snapshot(tool.ROOT)
        collection = output / "collection"
        collection.mkdir()
        nodes = [node for node in shard.collect(tool.ROOT, collection)
                 if node.split("::", 1)[0] == "tests/test_cluster_sharded_regression.py"]
        assert nodes, "empty wrapper collection"
        actual, observed_batches, counts = [], [], Counter()
        for path, pin in zip(paths, pins):
            receipt = shard.read(path, pin)
            assert receipt["verdict"] == "PASS" and receipt["source"] == source
            assert receipt["source_consistency"] == shard.consistent(tool.ROOT, source)
            process = shard.read(path.parent / "process/process.json")
            assert process == receipt["process"] and process["returncode"] == 0
            assert process["timeout"] is False and process["timeout_s"] == 179
            verification = shard.read(receipt["verification"], receipt["verification_sha256"])
            assert verification["verdict"] == "PASS" and verification["source"] == source
            assert verification["source_consistency"] == receipt["source_consistency"]
            assert verification["targets"] == receipt["targets"]
            assert verification["cuda_available"] is False
            records, batch_counts = shard.parse_junit(
                Path(receipt["verification"]).parent / "pytest.xml")
            assert batch_counts == receipt["tests"] == verification["tests"]
            assert not any(batch_counts[key] for key in ("failures", "errors", "skipped"))
            actual.extend(record["key"] for record in records)
            counts.update(batch_counts)
            observed_batches.append(receipt["batch"])
        assert Counter(observed_batches) == Counter(("v1-a", "v1-b", "fixture-v2", "prepare"))
        assert len(actual) == len(set(actual)), "duplicate wrapper JUnit membership"
        assert Counter(actual) == Counter(shard.junit_key(node) for node in nodes), \
            "complete wrapper collection/JUnit bidirectional mismatch"
        audit.update(verdict="PASS", source=source, source_consistency=shard.consistent(
            tool.ROOT, source), tests=dict(counts), collection_count=len(nodes),
            missing=[], extra=[], duplicates=[],
            receipts=[{"path": str(path), "sha256": pin} for path, pin in zip(paths, pins)])
    except Exception as error:
        audit.update(verdict="FAIL", first_failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        audit["finished_at"] = tool.stamp()
        shard.save(output / "acceptance.json", audit)
        print(f"CLUSTER_CPU_BATCH_CHECK verdict={audit['verdict']} "
              f"receipt={output / 'acceptance.json'}", flush=True)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--batch", choices=("v1", "v1-a", "v1-b", "fixture-v2", "prepare"))
    choice.add_argument("--check-batch", type=Path, action="append")
    parser.add_argument("--check-batch-sha256", action="append")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.batch is not None:
        assert args.check_batch_sha256 is None, "batch execution does not consume replay pins"
        accept_batch(args.batch, args.artifact_dir)
    else:
        check_batches(args.check_batch, args.check_batch_sha256 or [], args.artifact_dir)
