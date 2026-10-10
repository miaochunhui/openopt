"""Independent archived cluster evidence controls, without Kubernetes access.

The CLI pins a committed subject, creates an external tiny source fixture using
its exact production scripts, and runs the real archived acceptance functions.
Fixtures are synthetic CPU transport evidence, never CUDA or container proof.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "scripts/verify.py",
    "scripts/verify_sharded.py",
    "scripts/cluster_sharded_regression.py",
)
CASES = ("positive", "args", "env_value", "env_membership", "actual_args_only")
DEFAULT_ARTIFACTS = Path(
    "/hf3fs-jd/data/tmp/openopt-release-20261010/cluster-prepare-receipt-review"
)


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def save(path, value):
    Path(path).write_bytes(encode(value))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *arguments):
    result = subprocess.run(
        ["git", *arguments], cwd=repo, capture_output=True, check=False, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"git_failed_rc={result.returncode}: {result.stderr.decode()}")
    return result.stdout


def subject_pin(repo, commit):
    full = git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    return {
        "commit": full,
        "tree": git(repo, "rev-parse", f"{full}^{{tree}}").decode().strip(),
        "sha256": {
            name: hashlib.sha256(git(repo, "show", f"{full}:{name}")).hexdigest()
            for name in SCRIPTS
        },
    }


def load_tool(repo):
    sys.path.insert(0, str(repo / "scripts"))
    for name in ("verify_sharded", "_receipt_review_cluster"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        "_receipt_review_cluster", repo / SCRIPTS[-1],
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == repo / SCRIPTS[-1]
    assert Path(module.shard.__file__).resolve() == repo / SCRIPTS[1]
    return module


def fixture(subject_repo, pin, output):
    repo = output / "tiny-repo"
    for name in ("scripts", "src", "benchmarks", "tests", "examples"):
        (repo / name).mkdir(parents=True, exist_ok=True)
    paths = list(SCRIPTS)
    for name in SCRIPTS:
        (repo / name).write_bytes(git(subject_repo, "show", f"{pin['commit']}:{name}"))
        assert digest(repo / name) == pin["sha256"][name]
    content = {
        "pyproject.toml": '[tool.pytest.ini_options]\ntestpaths=["tests"]\n',
        "scripts/migrate_openpo.py": 'print("SYNTHETIC_TRANSPORT_ONLY")\n',
        "examples/portfolio_qp.py": 'print("SYNTHETIC_TRANSPORT_ONLY")\n',
        "tests/test_a.py": "def test_a():\n    assert 1 == 1\n",
        "tests/test_b.py": "def test_b():\n    assert 2 == 2\n",
        "tests/test_c.py": "def test_c():\n    assert 3 == 3\n",
    }
    for name, value in content.items():
        (repo / name).write_text(value, encoding="utf-8")
        paths.append(name)
    git(repo, "init")
    git(repo, "config", "user.name", "Independent transport review fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "add", "--", *paths)
    git(repo, "commit", "-m", "Freeze synthetic source fixture with pinned real gates")
    tool = load_tool(repo)
    plan_dir = tool.shard.plan_run(repo, output / "original-plan", [], shard_timeout=90)
    plan_path = plan_dir / "plan.json"
    folder = tool.prepare(
        repo, plan_path, digest(plan_path), output / "cluster",
        "synthetic-image:never-submitted",
    )
    config_path = folder / "cluster-plan.json"
    config, plan = tool.load_config(config_path, digest(config_path))
    expected, item = tool.manifest(config, "probe", None)
    job = copy.deepcopy(expected)
    job["metadata"]["uid"] = "synthetic-job-uid"
    job["status"] = {
        "succeeded": 1, "conditions": [{"type": "Complete", "status": "True"}],
    }
    pod = {
        "metadata": {
            "name": item["name"] + "-pod", "uid": "synthetic-pod-uid",
            "ownerReferences": [{"uid": job["metadata"]["uid"], "controller": True}],
        },
        "spec": copy.deepcopy(expected["spec"]["template"]["spec"]),
        "status": {
            "phase": "Succeeded", "containerStatuses": [{
                "restartCount": 0, "imageID": "synthetic://sha256:" + "a" * 64,
                "state": {"terminated": {"exitCode": 0}},
            }],
        },
    }
    pods = {"items": [pod]}
    hardware = tool.terminal_job(job, pods, expected, job["metadata"]["uid"])
    worker_output = Path(item["output"])
    (worker_output / "collection").mkdir(parents=True)
    collection_path = worker_output / "collection/collection.json"
    tool.shard.save(collection_path, {"nodeids": plan["collection"]["nodeids"]})
    worker_path = worker_output / "worker.json"
    tool.shard.save(worker_path, {
        "verdict": "PASS", "phase": "probe", "plan_sha256": config["plan_sha256"],
        "source": plan["source"],
        "source_consistency": tool.shard.verifier(Path(plan["repo"])).check_source_consistency(
            plan["source"], plan["source"]),
        "collection_sha256": digest(collection_path),
        "runtime": {"kind": "synthetic-not-a-runtime-qualification"},
    })
    job_path, pods_path = folder / "observed-job.json", folder / "observed-pods.json"
    tool.shard.save(item["manifest"], expected)
    tool.shard.save(job_path, job)
    tool.shard.save(pods_path, pods)
    item.update(
        uid=job["metadata"]["uid"], accepted=True, hardware=hardware,
        manifest_sha256=digest(item["manifest"]), worker_sha256=digest(worker_path),
        cluster_evidence={
            "job": str(job_path), "job_sha256": digest(job_path),
            "pods": str(pods_path), "pods_sha256": digest(pods_path),
        },
    )
    state = {
        "state": "PROBE_PASSED", "jobs": [item], "events": [], "first_failure": None,
        "config_sha256": digest(config_path), "image_id": hardware["image_id"],
    }
    tool.stage_members(config, plan, state)
    tool.verify_accepted(config, plan, state)
    assert config["gpu_count"] == 0
    return tool, config_path, config, plan, state, expected, job, pods


def mutate(case, expected, job, pods):
    submitted = expected["spec"]["template"]["spec"]["containers"][0]
    observed = job["spec"]["template"]["spec"]["containers"][0]
    placed = pods["items"][0]["spec"]["containers"][0]
    containers = (observed, placed) if case == "actual_args_only" else (
        submitted, observed, placed)
    for container in containers:
        if case in ("args", "actual_args_only"):
            assert "args" not in container
            container["args"] = ["--review-injected-unsubmitted-argument"]
        elif case == "env_value":
            entry = next(item for item in container["env"] if item["name"] == "PYTEST_ADDOPTS")
            assert entry["value"] == ""
            entry["value"] = "-k test_a"
        elif case == "env_membership":
            container["env"].append({"name": "OPENOPT_REVIEW_UNSUBMITTED_ENV", "value": "1"})
        elif case != "positive":
            raise ValueError(f"unknown_case: {case}")
    if case != "positive":
        assert observed == placed


def replay_case(subject_repo, pin, output, case):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "case": case, "subject": pin,
        "kind": "synthetic-cpu-archives-real-production-gate",
        "cuda_execution": False, "cluster_execution": False,
        "qualification": "NOT numerical, commercial, performance or cluster qualification",
        "expected": "ACCEPTED" if case == "positive" else "REJECTED",
    }
    try:
        tool, config_path, config, plan, state, expected, job, pods = fixture(
            subject_repo, pin, output)
        report["positive_before_mutation"] = "ACCEPTED"
        report["fixture_source"] = plan["source"]
        baseline = copy.deepcopy({
            "manifest": expected, "job": job, "pods": pods, "state": state,
        })
        save(output / "baseline.json", baseline)
        baseline_pin = digest(output / "baseline.json")
        item = state["jobs"][0]
        before = {
            "config": digest(config_path), "manifest": item["manifest_sha256"],
            **item["cluster_evidence"],
        }
        mutate(case, expected, job, pods)
        tool.shard.save(item["manifest"], expected)
        tool.shard.save(item["cluster_evidence"]["job"], job)
        tool.shard.save(item["cluster_evidence"]["pods"], pods)
        item["manifest_sha256"] = digest(item["manifest"])
        item["cluster_evidence"]["job_sha256"] = digest(item["cluster_evidence"]["job"])
        item["cluster_evidence"]["pods_sha256"] = digest(item["cluster_evidence"]["pods"])
        state_path = Path(config["artifacts"]) / "review-archived-state.json"
        save(state_path, state)
        after = {
            "config": digest(config_path), "manifest": item["manifest_sha256"],
            **item["cluster_evidence"],
        }
        assert before["config"] == after["config"]
        assert digest(output / "baseline.json") == baseline_pin
        rebuilt, _ = tool.manifest(config, "probe", None)
        report.update(
            pins_before=before, pins_after=after, state_path=str(state_path),
            state_sha256=digest(state_path), baseline_sha256=baseline_pin,
            regenerated_manifest_equal=rebuilt == expected,
            all_referenced_hashes_valid=all((
                digest(item["manifest"]) == item["manifest_sha256"],
                digest(item["cluster_evidence"]["job"]) == item["cluster_evidence"]["job_sha256"],
                digest(item["cluster_evidence"]["pods"]) == item["cluster_evidence"]["pods_sha256"],
            )),
        )
        config, plan = tool.load_config(config_path, after["config"])
        tool.stage_members(config, plan, state)
        try:
            tool.verify_accepted(config, plan, tool.shard.read(state_path, digest(state_path)))
        except tool.shard.GateError as error:
            report.update(actual="REJECTED", rejection_reason=str(error))
        else:
            report.update(actual="ACCEPTED", rejection_reason=None)
        report["verdict"] = "PASS" if report["actual"] == report["expected"] else "FAIL"
    except Exception as error:
        report.update(verdict="ERROR", first_failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(output / "case.json", report)
        print(f"RECEIPT_REVIEW case={case} verdict={report['verdict']} "
              f"actual={report.get('actual', 'ERROR')} report={output / 'case.json'}", flush=True)
    return report


@pytest.mark.parametrize("case", CASES)
@pytest.mark.skipif(
    os.environ.get("OPENOPT_RUN_CLUSTER_RECEIPT_EVIDENCE_TESTS") != "1",
    reason="explicit frozen-subject evidence review opt-in; not default qualification",
)
def test_cluster_prepare_receipt_rejects_coordinated_substitution(tmp_path, case):
    commit = os.environ.get("OPENOPT_CLUSTER_RECEIPT_REVIEW_COMMIT", "HEAD")
    pin = subject_pin(ROOT, commit)
    report = replay_case(ROOT, pin, tmp_path / case, case)
    assert report["all_referenced_hashes_valid"]
    assert report["positive_before_mutation"] == "ACCEPTED"
    assert report["verdict"] == "PASS", report


@pytest.mark.parametrize("case", ("args", "env_value", "env_membership"))
def test_coordinated_fixture_changes_all_three_documents_and_preserves_baseline(case):
    container = {"command": ["python3", "worker"], "env": [
        {"name": "PYTEST_ADDOPTS", "value": ""}]}
    expected = {"spec": {"template": {"spec": {"containers": [container]}}}}
    job = copy.deepcopy(expected)
    pods = {"items": [{"spec": copy.deepcopy(expected["spec"]["template"]["spec"])}]}
    baseline = encode({"manifest": expected, "job": job, "pods": pods})
    mutate(case, expected, job, pods)
    assert expected["spec"]["template"]["spec"] == job["spec"]["template"]["spec"]
    assert pods["items"][0]["spec"] == expected["spec"]["template"]["spec"]
    assert encode({"manifest": expected, "job": job, "pods": pods}) != baseline
    assert hashlib.sha256(baseline).digest() != hashlib.sha256(
        encode({"manifest": expected, "job": job, "pods": pods})).digest()


def test_actual_args_control_keeps_submitted_bytes():
    expected = {"spec": {"template": {"spec": {
        "containers": [{"command": ["python3", "worker"], "env": []}]}}}}
    job = copy.deepcopy(expected)
    pods = {"items": [{"spec": copy.deepcopy(expected["spec"]["template"]["spec"])}]}
    baseline = encode(expected)
    mutate("actual_args_only", expected, job, pods)
    assert encode(expected) == baseline
    assert pods["items"][0]["spec"] == job["spec"]["template"]["spec"]
    assert job != expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-repo", type=Path, default=ROOT)
    parser.add_argument("--subject-commit", required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args(argv)
    subject = args.subject_repo.resolve()
    artifact_root = args.artifact_dir.resolve()
    if artifact_root.is_relative_to(subject):
        parser.error("artifacts must be outside the subject repository")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        parser.error("explicit CUDA_VISIBLE_DEVICES=-1 is required")
    artifact_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="coordinated-", dir=artifact_root))
    started = time.monotonic()
    receipt = {
        "schema_version": 1, "subject": subject_pin(subject, args.subject_commit),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version, "executable": sys.executable,
            "env": {key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")},
        },
        "cases": [], "verdict": "RUNNING", "artifact_dir": str(output),
        "qualification": "synthetic CPU transport review only; not GPU acceptance",
    }
    try:
        with (output / "execution.log").open("x", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                for case in CASES:
                    report = replay_case(subject, receipt["subject"], output / case, case)
                    receipt["cases"].append({
                        "case": case, "path": str(output / case / "case.json"),
                        "sha256": digest(output / case / "case.json"),
                        "actual": report["actual"], "verdict": report["verdict"],
                    })
                    log.flush()
        receipt["verdict"] = "PASS" if all(
            item["verdict"] == "PASS" for item in receipt["cases"]) else "FAIL"
    except Exception as error:
        receipt.update(verdict="ERROR", first_failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            wall_s=time.monotonic() - started,
            returncode=0 if receipt["verdict"] == "PASS" else 1,
            log_path=str(output / "execution.log"),
            log_sha256=digest(output / "execution.log"),
        )
        save(output / "acceptance.json", receipt)
        print(f"CLUSTER_PREPARE_RECEIPT_REVIEW verdict={receipt['verdict']} "
              f"receipt={output / 'acceptance.json'}", flush=True)
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
