"""Plan, run and independently recheck source-pinned full CPU regression.

Pytest supplies the complete collection; files, not guessed test globs, are
assigned by historical-duration LPT. Each worker executes the existing verify
entry. Artifacts are external, timeouts fail, and no speedup is inferred.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/sharded-regression")


class GateError(RuntimeError):
    """A preserved, named admission or acceptance failure."""


def require(condition, message):
    if not condition:
        raise GateError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_bytes(encoded(value))


def read(path, pin=None):
    if pin is not None:
        require(digest(path) == pin, f"hash_mismatch: {path}")
    return json.loads(Path(path).read_bytes())


def external(directory, repo):
    directory = Path(directory).expanduser().resolve()
    require(not directory.is_relative_to(repo.resolve()), "artifacts_inside_repository")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def new_artifacts(directory, repo, prefix):
    return Path(tempfile.mkdtemp(prefix=prefix, dir=external(directory, repo)))


def verifier(repo):
    path = repo / "scripts/verify.py"
    require(path.is_file(), "missing_verify_entry")
    spec = importlib.util.spec_from_file_location("_openopt_verify_sharded_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = repo
    return module


def snapshot(repo):
    value = verifier(repo).source_snapshot(repo)
    require(value["clean"], "source_not_clean")
    return value


def consistent(repo, expected):
    actual = snapshot(repo)
    result = verifier(repo).check_source_consistency(expected, actual)
    require(result["verdict"] == "PASS", f"source_drift: {result}")
    return result


def cpu_env(repo, artifacts):
    env = os.environ.copy()
    # An explicit CPU regression never consumes a local GPU.
    env.update({
        "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": "",
        "PYTEST_PLUGINS": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "TMPDIR": str(artifacts),
        "TRITON_CACHE_DIR": str(artifacts / "triton-cache"),
        "PYTHONPATH": os.pathsep.join(
            [str(repo), str(repo / "src"), str(repo / "benchmarks")]),
    })
    return env


def deadline(value):
    require(math.isfinite(value) and 0 < value <= 300, "timeout_must_be_in_0_300")
    return value


def command_logged(command, repo, artifacts, timeout, env):
    deadline(timeout)
    started = time.monotonic()
    metadata = {
        "command": command, "started_at": now(), "returncode": None,
        "timeout": False, "timeout_s": timeout,
    }
    with (artifacts / "process.log").open("x", encoding="utf-8") as log:
        log.write(f"command={command!r}\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        metadata["pid"] = process.pid
        try:
            metadata["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            metadata["timeout"] = True
            # This process group contains only this tool's admitted worker.
            os.killpg(process.pid, signal.SIGKILL)
            metadata["returncode"] = process.wait()
        finally:
            metadata.update(finished_at=now(), wall_s=time.monotonic() - started)
            log.write(f"\nreturncode={metadata['returncode']} timeout={metadata['timeout']}\n")
            save(artifacts / "process.json", metadata)
    require(not metadata["timeout"], f"worker_timeout: {artifacts}")
    require(metadata["returncode"] == 0,
            f"process_failed_rc={metadata['returncode']}: {artifacts}")
    return metadata


def junit_key(nodeid):
    from _pytest.junitxml import mangle_test_address
    parts = mangle_test_address(nodeid)
    return ".".join(parts[:-1]), parts[-1]


def pytest_collection_finish(session):
    """Explicit collection plugin, not a substitute for pytest discovery."""
    output = os.environ.get("OPENOPT_SHARD_COLLECTION")
    if output:
        save(Path(output), {"nodeids": [item.nodeid for item in session.items]})


def collect(repo, artifacts, timeout=60):
    env = cpu_env(repo, artifacts)
    env["OPENOPT_SHARD_COLLECTION"] = str(artifacts / "collection.json")
    command_logged(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "scripts.verify_sharded", "-o", "addopts=",
         "-o", f"cache_dir={artifacts / 'pytest-cache'}",
         "--basetemp", str(artifacts / "pytest-tmp")],
        repo, artifacts, timeout, env,
    )
    nodes = read(artifacts / "collection.json")["nodeids"]
    require(isinstance(nodes, list) and nodes, "empty_collection")
    require(all(isinstance(node, str) and "::" in node for node in nodes),
            "invalid_collection_nodeid")
    require(len(nodes) == len(set(nodes)), "duplicate_collection_nodeid")
    keys = [junit_key(node) for node in nodes]
    require(len(keys) == len(set(keys)), "ambiguous_junit_identity")
    for node in nodes:
        target = Path(node.split("::", 1)[0])
        require(not target.is_absolute() and ".." not in target.parts
                and (repo / target).is_file(), f"invalid_collection_path: {node}")
    return nodes


def parse_junit(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    require(bool(suites), "missing_junit_testsuite")
    records = []
    for suite in suites:
        cases = suite.findall("testcase")
        require(int(suite.get("tests", "-1")) == len(cases), "junit_count_mismatch")
        for field, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            require(int(suite.get(field, "0")) ==
                    sum(case.find(tag) is not None for case in cases),
                    f"junit_{field}_mismatch")
        for case in cases:
            seconds = float(case.get("time", "nan"))
            require(math.isfinite(seconds) and seconds >= 0, "invalid_junit_time")
            key = (case.get("classname"), case.get("name"))
            require(all(isinstance(part, str) and part for part in key), "missing_junit_identity")
            records.append({
                "key": key, "time_s": seconds,
                "failures": int(case.find("failure") is not None),
                "errors": int(case.find("error") is not None),
                "skipped": int(case.find("skipped") is not None),
            })
    keys = [record["key"] for record in records]
    require(len(keys) == len(set(keys)), "duplicate_junit_identity")
    require(bool(records), "empty_junit")
    counts = {"tests": len(records)}
    counts.update({field: sum(record[field] for record in records)
                   for field in ("failures", "errors", "skipped")})
    return records, counts


def lpt(nodes, histories, shards=3, unknown_s=1.0):
    require(type(shards) is int and 1 <= shards <= 3, "shards_must_be_1_3")
    require(math.isfinite(unknown_s) and unknown_s > 0, "invalid_unknown_prediction")
    costs, provenance = {}, []
    for path in histories:
        records, counts = parse_junit(path)
        provenance.append({"path": str(Path(path).resolve()), "sha256": digest(path),
                           "tests": counts, "precedence": len(provenance)})
        for record in records:
            if not record["failures"] and not record["errors"]:
                costs[record["key"]] = record["time_s"]
    groups = {}
    unknown = []
    for node in nodes:
        key = junit_key(node)
        if key not in costs:
            unknown.append(node)
        group = groups.setdefault(node.split("::", 1)[0], {
            "nodeids": [], "predicted_test_s": 0.0, "unknown_nodeids": [],
        })
        group["nodeids"].append(node)
        group["predicted_test_s"] += costs.get(key, unknown_s)
        if key not in costs:
            group["unknown_nodeids"].append(node)
    require(len(groups) >= shards, "too_few_files_for_nonempty_shards")
    assignments = [{"id": index, "targets": [], "nodeids": [], "predicted_test_s": 0.0,
                    "unknown_nodeids": []} for index in range(shards)]
    for target, group in sorted(groups.items(),
                                key=lambda item: (-item[1]["predicted_test_s"], item[0])):
        chosen = min(assignments, key=lambda item: (item["predicted_test_s"], item["id"]))
        chosen["targets"].append(target)
        chosen["nodeids"].extend(group["nodeids"])
        chosen["unknown_nodeids"].extend(group["unknown_nodeids"])
        chosen["predicted_test_s"] += group["predicted_test_s"]
    return assignments, provenance, unknown


def validate_plan(plan):
    require(plan.get("schema_version") == 1 and plan.get("verdict") == "PASS",
            "invalid_plan_status")
    require(plan.get("coverage") == "tests", "invalid_plan_coverage")
    require(plan.get("source", {}).get("clean") is True, "unclean_plan_source")
    nodes = plan.get("collection", {}).get("nodeids")
    require(isinstance(nodes, list) and nodes and len(nodes) == len(set(nodes)),
            "empty_or_duplicate_plan_collection")
    assignments = plan.get("shards")
    require(isinstance(assignments, list) and 1 <= len(assignments) <= 3, "invalid_shards")
    require([item["id"] for item in assignments] == list(range(len(assignments))),
            "invalid_shard_ids")
    assigned_nodes, assigned_files = [], []
    for item in assignments:
        require(item["targets"] and item["nodeids"], "empty_shard_selection")
        require(set(item["targets"]) ==
                {node.split("::", 1)[0] for node in item["nodeids"]}, "target_nodeid_conflict")
        assigned_nodes.extend(item["nodeids"])
        assigned_files.extend(item["targets"])
    require(Counter(assigned_nodes) == Counter(nodes), "shard_collection_mismatch")
    require(len(assigned_files) == len(set(assigned_files)), "duplicate_shard_file")
    deadline(plan["shard_timeout_s"])


def plan_run(repo, artifact_root, histories, shards=3, unknown_s=1.0,
             shard_timeout=300, collection_timeout=60):
    artifacts = new_artifacts(artifact_root, repo, "plan-")
    report = {"schema_version": 1, "verdict": "RUNNING", "started_at": now()}
    try:
        deadline(shard_timeout)
        report["source"] = snapshot(repo)
        directory = artifacts / "collection"
        directory.mkdir()
        nodes = collect(repo, directory, collection_timeout)
        assignments, historical, unknown = lpt(nodes, histories, shards, unknown_s)
        report.update({
            "repo": str(repo), "coverage": "tests", "collection": {"nodeids": nodes,
                "path": str(directory / "collection.json"),
                "sha256": digest(directory / "collection.json")},
            "shards": assignments, "history": historical, "unknown_nodeids": unknown,
            "prediction": {"method": "file-LPT", "unknown_test_s": unknown_s,
                "precedence": "later JUnit overrides earlier for the same identity",
                "excludes": "collection, imports, provenance, examples, contention",
                "speedup": None},
            "shard_timeout_s": shard_timeout,
            "collection_timeout_s": deadline(collection_timeout),
            "source_consistency": consistent(repo, report["source"]),
            "verdict": "PASS",
        })
        validate_plan(report)
    except Exception as error:
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at"] = now()
        save(artifacts / "plan.json", report)
        print(f"OPENOPT_SHARD_PLAN verdict={report['verdict']} plan={artifacts / 'plan.json'} "
              f"sha256={digest(artifacts / 'plan.json')}", flush=True)
    return artifacts


def accept_shard(plan, item, artifacts):
    process = read(artifacts / "process.json")
    require(process["returncode"] == 0 and process["timeout"] is False,
            "shard_process_not_pass")
    require(process["command"] == shard_command(Path(plan["repo"]), item, artifacts),
            "shard_process_command_mismatch")
    require(process["timeout_s"] == plan["shard_timeout_s"]
            and type(process["pid"]) is int and process["pid"] > 0
            and math.isfinite(process["wall_s"]) and process["wall_s"] > 0,
            "invalid_shard_process_measurement")
    reports = list(artifacts.glob("tests-*/verification.json"))
    require(len(reports) == 1, "missing_or_multiple_verification_reports")
    path = reports[0]
    report = read(path)
    require(report["verdict"] == "PASS", "verification_not_pass")
    require(report["mode"] == "tests" and report["targets"] == item["targets"]
            and report["coverage"] == "explicit_targets", "verification_target_mismatch")
    require(report["source"] == plan["source"], "shard_source_mismatch")
    expected = verifier(Path(plan["repo"])).check_source_consistency(plan["source"], plan["source"])
    require(report["source_consistency"] == expected, "shard_source_consistency_not_pass")
    xml_path = path.parent / "pytest.xml"
    records, counts = parse_junit(xml_path)
    require(counts == report["tests"], "verification_junit_count_mismatch")
    require(not counts["errors"] and not counts["failures"], "shard_junit_not_pass")
    mapping = {junit_key(node): node for node in plan["collection"]["nodeids"]}
    require(all(record["key"] in mapping for record in records), "unexpected_junit_identity")
    actual = [mapping[record["key"]] for record in records]
    require(Counter(actual) == Counter(item["nodeids"]), "shard_junit_nodeid_mismatch")
    require(report["cuda_available"] is False, "cpu_regression_observed_cuda")
    return {
        "id": item["id"], "verdict": "PASS", "nodeids": actual, "tests": counts,
        "artifact_dir": str(artifacts), "verification": str(path),
        "verification_sha256": digest(path), "junit": str(xml_path),
        "junit_sha256": digest(xml_path), "process_sha256": digest(artifacts / "process.json"),
        "predicted_test_s": item["predicted_test_s"],
        "actual_test_s": sum(record["time_s"] for record in records),
        "actual_worker_wall_s": process["wall_s"], "source": report["source"],
    }


def shard_command(repo, item, artifacts):
    command = [sys.executable, "-B", str(repo / "scripts/verify.py"),
               "tests", "--artifact-dir", str(artifacts)]
    for target in item["targets"]:
        command.extend(["--target", target])
    return command


def launch_shard(repo, plan, item, artifacts):
    consistent(repo, plan["source"])
    command = shard_command(repo, item, artifacts)
    command_logged(command, repo, artifacts, plan["shard_timeout_s"], cpu_env(repo, artifacts))
    consistent(repo, plan["source"])
    return accept_shard(plan, item, artifacts)


def combine(plan, results):
    require(len(results) == len(plan["shards"]), "partial_shard_results")
    require(sorted(item["id"] for item in results) == list(range(len(plan["shards"]))),
            "duplicate_or_missing_shard_results")
    nodes = [node for item in results for node in item["nodeids"]]
    require(Counter(nodes) == Counter(plan["collection"]["nodeids"]), "full_collection_mismatch")
    require(len(nodes) == len(set(nodes)), "duplicate_executed_nodeid")
    counts = {key: sum(item["tests"][key] for item in results)
              for key in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] == len(nodes) and not counts["failures"] and not counts["errors"],
            "combined_junit_not_pass")
    return counts


def execute(repo, plan_path, plan_pin, artifact_root, workers=3):
    require(type(workers) is int and 1 <= workers <= 3, "workers_must_be_1_3")
    artifacts = new_artifacts(artifact_root, repo, "run-")
    started = time.monotonic()
    summary = {
        "schema_version": 1, "mode": "tests", "coverage": "tests", "targets": ["tests"],
        "qualification": "full CPU correctness only; not CUDA, commercial or performance qualification",
        "started_at": now(), "verdict": "RUNNING", "workers": workers,
        "plan": str(plan_path.resolve()), "plan_sha256": plan_pin, "shards": [],
        "failures": [], "not_admitted": [], "speedup": None,
    }
    try:
        plan = read(plan_path, plan_pin)
        validate_plan(plan)
        require(Path(plan["repo"]).resolve() == repo, "plan_repo_mismatch")
        summary["source"] = plan["source"]
        consistent(repo, plan["source"])
        directory = artifacts / "collection"
        directory.mkdir()
        require(collect(repo, directory, plan["collection_timeout_s"]) ==
                plan["collection"]["nodeids"], "collection_drift")
        pending = iter(plan["shards"])
        admitted = set()
        stop = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            flying = {}

            def admit():
                item = next(pending, None)
                if item is None:
                    return False
                folder = artifacts / f"shard-{item['id']}"
                folder.mkdir()
                admitted.add(item["id"])
                flying[pool.submit(launch_shard, repo, plan, item, folder)] = item
                return True

            for _ in range(min(workers, len(plan["shards"]))):
                admit()
            while flying:
                finished, _ = wait(flying, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = flying.pop(future)
                    try:
                        summary["shards"].append(future.result())
                    except Exception as error:
                        stop = True
                        summary["failures"].append({
                            "id": item["id"], "error": f"{type(error).__name__}: {error}",
                            "artifact_dir": str(artifacts / f"shard-{item['id']}"),
                        })
                # Inspect all completions before admitting any queued work.
                if not stop:
                    for _ in finished:
                        if not admit():
                            break
        summary["not_admitted"] = [item["id"] for item in plan["shards"]
                                   if item["id"] not in admitted]
        if summary["failures"]:
            raise GateError(f"shard_failure: {summary['failures'][0]}")
        summary["tests"] = combine(plan, summary["shards"])
        summary["source_consistency"] = consistent(repo, plan["source"])
        summary["verdict"] = "PASS"
        combined = ET.Element("testsuites", name="openopt full CPU sharded tests")
        for item in sorted(summary["shards"], key=lambda value: value["id"]):
            root = ET.parse(item["junit"]).getroot()
            for suite in ([root] if root.tag == "testsuite" else root.findall("testsuite")):
                combined.append(suite)
        ET.ElementTree(combined).write(artifacts / "pytest.xml", encoding="utf-8",
                                      xml_declaration=True)
        summary["junit_sha256"] = digest(artifacts / "pytest.xml")
    except Exception as error:
        summary.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary.update(finished_at=now(), actual_total_wall_s=time.monotonic() - started)
        save(artifacts / "verification.json", summary)
        print(f"OPENOPT_SHARD_RUN verdict={summary['verdict']} "
              f"report={artifacts / 'verification.json'} "
              f"sha256={digest(artifacts / 'verification.json')}", flush=True)
    return artifacts


def check(repo, plan_path, plan_pin, report_path, report_pin, artifact_root):
    artifacts = new_artifacts(artifact_root, repo, "check-")
    audit = {"schema_version": 1, "started_at": now(), "verdict": "RUNNING"}
    try:
        plan, report = read(plan_path, plan_pin), read(report_path, report_pin)
        validate_plan(plan)
        require(Path(plan["repo"]).resolve() == repo, "plan_repo_mismatch")
        require(report["verdict"] == "PASS" and report["plan_sha256"] == plan_pin,
                "incomplete_or_unbound_run")
        require(Path(report["plan"]).resolve() == plan_path.resolve(), "report_plan_mismatch")
        require(report["coverage"] == "tests" and report["source"] == plan["source"],
                "report_source_or_coverage_mismatch")
        require(not report["failures"] and not report["not_admitted"], "hidden_partial_failure")
        require(len(report["shards"]) == len(plan["shards"]), "partial_shard_results")
        consistent(repo, plan["source"])
        require(read(plan["collection"]["path"], plan["collection"]["sha256"])["nodeids"]
                == plan["collection"]["nodeids"], "plan_collection_sidecar_mismatch")
        fresh = artifacts / "collection"
        fresh.mkdir()
        require(collect(repo, fresh, plan["collection_timeout_s"]) ==
                plan["collection"]["nodeids"], "collection_drift")
        replayed = []
        for item in report["shards"]:
            folder = report_path.parent / f"shard-{item['id']}"
            require(Path(item["artifact_dir"]).resolve() == folder.resolve(),
                    "shard_artifact_location_mismatch")
            require(digest(folder / "process.json") == item["process_sha256"],
                    "process_pin_mismatch")
            require(digest(item["verification"]) == item["verification_sha256"]
                    and digest(item["junit"]) == item["junit_sha256"], "shard_pin_mismatch")
            actual = accept_shard(plan, plan["shards"][item["id"]], folder)
            require(actual == item, "shard_replay_mismatch")
            replayed.append(actual)
        counts = combine(plan, replayed)
        require(counts == report["tests"], "combined_report_count_mismatch")
        require(digest(report_path.parent / "pytest.xml") == report["junit_sha256"],
                "combined_junit_pin_mismatch")
        records, xml_counts = parse_junit(report_path.parent / "pytest.xml")
        require(xml_counts == counts and Counter(record["key"] for record in records)
                == Counter(junit_key(node) for node in plan["collection"]["nodeids"]),
                "combined_junit_membership_mismatch")
        audit.update(tests=counts, coverage="tests", source=plan["source"],
                     source_consistency=consistent(repo, plan["source"]), verdict="PASS")
        require(report["source_consistency"] == audit["source_consistency"],
                "report_source_consistency_mismatch")
    except Exception as error:
        audit.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        audit["finished_at"] = now()
        save(artifacts / "acceptance.json", audit)
        print(f"OPENOPT_SHARD_CHECK verdict={audit['verdict']} "
              f"report={artifacts / 'acceptance.json'}", flush=True)
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "check"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--artifact-dir", "--log-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--history-junit", type=Path, action="append", default=[])
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--unknown-test-seconds", type=float, default=1.0)
    parser.add_argument("--shard-timeout", type=float, default=300)
    parser.add_argument("--collection-timeout", type=float, default=60)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-sha256")
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    try:
        if args.action == "plan":
            plan_run(repo, args.artifact_dir, args.history_junit, args.shards,
                     args.unknown_test_seconds, args.shard_timeout, args.collection_timeout)
        else:
            require(args.plan is not None and args.plan_sha256 is not None,
                    "caller_pinned_plan_required")
            if args.action == "run":
                execute(repo, args.plan, args.plan_sha256, args.artifact_dir, args.workers)
            else:
                require(args.report is not None and args.report_sha256 is not None,
                        "caller_pinned_report_required")
                check(repo, args.plan, args.plan_sha256, args.report, args.report_sha256,
                      args.artifact_dir)
    except Exception as error:
        print(f"OPENOPT_SHARDED error={type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
