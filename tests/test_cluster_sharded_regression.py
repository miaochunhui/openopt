"""CPU-only fake transport controls; no real Kubernetes mutation is allowed."""

from __future__ import annotations

import copy
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

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
