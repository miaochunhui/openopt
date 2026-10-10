"""Two-worker crash fixture and sequential, disk-checked diagnosis acceptance.

Journal time is controlled, not wall time: slow FULL/WAL commits must not
accidentally expire a lease while the fixture prepares the crash.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Event, Lock, local
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from openpo import durable_shadow as durable_module
from openpo import shadow_journal as journal_module
from openpo.durable_shadow import DurableShadowCertificationService
from tests.test_facade import _request

LEASE_NOW = 1_000_000_000.0


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def crash_worker(
    journal_path: Path, tickets_path: Path, entered_path: Path, mode: str,
) -> None:
    clock = SimpleNamespace(now=LEASE_NOW)
    journal_module.time = SimpleNamespace(time=lambda: clock.now)
    entered = []
    entry_lock = Lock()
    worker_context = local()
    ready = Event()
    blocked = Event()
    original_execute = DurableShadowCertificationService._execute_shadow

    def tracked_execute(self, job, ticket, request, primary_exact):
        worker_context.job = job
        return original_execute(self, job, ticket, request, primary_exact)

    def block_forever(request):
        job = worker_context.job
        with entry_lock:
            entered.append({
                "job_id": job.job_id,
                "ticket_sha256": job.ticket_sha256,
                "submission_id": job.submission_id,
                "execution_attempt": job.execution_attempt,
                "fence_token": job.fence_token,
                "lease_owner": job.lease_owner,
            })
            if len(entered) == 2:
                ready.set()
        blocked.wait()
        raise AssertionError("crash worker must be killed, never released")

    DurableShadowCertificationService._execute_shadow = tracked_execute
    durable_module.certify_exact = block_forever
    if mode == "legacy-expired-first":
        original_claim = journal_module.SQLiteShadowJournal.claim_next

        def expire_first_claim(self, **kwargs):
            job = original_claim(self, **kwargs)
            if job is not None and job.execution_attempt == 1:
                clock.now = job.lease_deadline
            return job

        # Model a missed renewal before a free worker slot is filled. The real
        # journal still selects, increments and fences every claim.
        journal_module.SQLiteShadowJournal.claim_next = expire_first_claim
        journal_module.SQLiteShadowJournal.heartbeat = lambda self, **kwargs: False

    service = DurableShadowCertificationService(
        journal_path,
        max_pending=2,
        worker_count=2,
        lease_seconds=0.5,
        heartbeat_seconds=0.1,
    )
    submissions = [
        service.submit(_request(), submission_id=f"crash-order-{index}")
        for index in range(2)
    ]
    if not ready.wait(timeout=20.0):
        raise RuntimeError("two certification callbacks did not enter")
    jobs = service._journal.outstanding_jobs()
    evidence = {
        "schema": "durable-crash-entry-v1",
        "mode": mode,
        "lease_now": clock.now,
        "entered": entered,
        "jobs": [
            {key: getattr(job, key) for key in (
                "job_id", "ticket_sha256", "submission_id", "state",
                "execution_attempt", "fence_token", "lease_owner", "lease_deadline",
            )}
            for job in jobs
        ],
    }
    write_json(tickets_path, [asdict(submission.ticket) for submission in submissions])
    write_json(entered_path, evidence)
    print(json.dumps({"ready": str(entered_path)}), flush=True)
    blocked.wait()


def source_hashes() -> dict[str, str]:
    paths = [
        "src/openpo/durable_shadow.py", "src/openpo/shadow_journal.py",
        "tests/test_durable_shadow.py", "tests/helpers/durable_shadow_multi_crash_worker.py",
    ]
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def acceptance(artifact_dir: Path) -> None:
    """Keep the intentional red, then run three independent pytest processes."""
    from scripts.verify import run
    from tests.test_durable_shadow import (
        _assert_replay_once, _assert_two_distinct_first_executions,
    )

    artifact_dir = artifact_dir.resolve()
    if artifact_dir.is_relative_to(ROOT):
        raise ValueError("diagnosis evidence must be outside the worktree")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "durable-diagnosis-v1", "verdict": "RUNNING",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "source_sha256": source_hashes(), "runs": []}
    try:
        original = Path(
            "/hf3fs-jd/data/tmp/openopt-release-20261010/precondition/"
            "full-v1/tests-_9xpb4am"
        )
        with sqlite3.connect(
            (original / "pytest-tmp/test_durable_shadow_replays_tw0/shadow.sqlite3")
            .as_uri() + "?mode=ro", uri=True,
        ) as connection:
            connection.row_factory = sqlite3.Row
            original_jobs = [dict(row) for row in connection.execute(
                "SELECT job_id, state, execution_attempt, fence_token FROM jobs ORDER BY job_id"
            )]
            original_events = [dict(row) for row in connection.execute(
                "SELECT event_id, job_id, event_type, detail, created_at "
                "FROM journal_events ORDER BY event_id"
            )]
        if [job["execution_attempt"] for job in original_jobs] != [3, 1]:
            raise AssertionError("original failed journal no longer has [3,1]")
        original_log = (original / "pytest.log").read_bytes()
        if b"assert [3, 1] == [2, 2]" not in original_log or b"returncode=1" not in original_log:
            raise AssertionError("original failure log is missing first cause or exit code")
        report["original_failure"] = {
            "artifact_dir": str(original),
            "log_sha256": hashlib.sha256(original_log).hexdigest(),
            "jobs": original_jobs, "events": original_events,
        }
        red_dir = artifact_dir / "controlled-red"
        red_dir.mkdir()
        with (red_dir / "pytest.log").open("x", encoding="utf-8") as log:
            red = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()),
                 "legacy-repro", "--artifact-dir", str(red_dir)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        red_evidence = json.loads((red_dir / "acceptance.json").read_text())
        if (red.returncode != 1 or red_evidence["attempts"] != [3, 1]
                or red_evidence["fences"] != [3, 1]
                or red_evidence["killed_returncode"] != -9):
            raise AssertionError(f"controlled red did not reproduce [3,1]: {red_evidence}")
        report["controlled_red"] = {"returncode": red.returncode, **red_evidence}
        for index in range(1, 4):
            path = run("tests", artifact_dir / f"green-{index}", False, [
                "tests/test_durable_shadow.py::test_durable_shadow_replays_two_expired_workers_after_process_kill",
                "tests/test_durable_shadow.py::test_durable_shadow_duplicate_job_entry_is_rejected",
                "tests/test_durable_shadow.py::test_durable_shadow_replay_count_negative_control",
            ])
            counts = json.loads((path / "verification.json").read_text())["tests"]
            # pytest's *current symlinks alias the same real evidence directory.
            evidence_paths = sorted({
                item.resolve() for item in (path / "pytest-tmp").glob("*/acceptance.json")
            })
            evidence = [json.loads(item.read_text()) for item in evidence_paths]
            if counts != {"tests": 3, "failures": 0, "errors": 0, "skipped": 0}:
                raise AssertionError(f"unexpected JUnit counts: {counts}")
            if len(evidence) != 3 or any(item["verdict"] != "PASS" for item in evidence):
                raise AssertionError(f"missing/failed semantic acceptance: {evidence_paths}")
            by_case = {item["case"]: item for item in evidence}
            if set(by_case) != {
                "two-distinct-workers-replay-once", "duplicate-job-entry-rejected",
                "legacy-synchronization-replay-count-rejected",
            }:
                raise AssertionError(f"unexpected semantic cases: {sorted(by_case)}")
            if any(item["source_sha256"] != report["source_sha256"] for item in evidence):
                raise AssertionError("acceptance evidence source hash mismatch")
            green = by_case["two-distinct-workers-replay-once"]
            _assert_two_distinct_first_executions(green["entry"])
            _assert_replay_once(green)
            if green["killed_returncode"] != -9:
                raise AssertionError("positive case did not use actual SIGKILL")
            duplicate = by_case["duplicate-job-entry-rejected"]
            if not duplicate["rejected_before_recovery"]:
                raise AssertionError("duplicate entry was not rejected before recovery")
            negative = by_case["legacy-synchronization-replay-count-rejected"]
            if negative["attempts"] != [3, 1] or negative["fences"] != [3, 1]:
                raise AssertionError("count negative control lost the original failure")
            report["runs"].append({"artifact_dir": str(path), "tests": counts,
                                   "acceptance": evidence})
        if source_hashes() != report["source_sha256"]:
            raise AssertionError("source changed during sequential acceptance")
        report["verdict"] = "PASS"
    except BaseException as error:
        report["verdict"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(artifact_dir / "acceptance.json", report)
        print(f"DURABLE_DIAGNOSIS verdict={report['verdict']} "
              f"report={artifact_dir / 'acceptance.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("worker", "legacy-repro", "acceptance"))
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--tickets", type=Path)
    parser.add_argument("--entered", type=Path)
    parser.add_argument("--mode", choices=("synchronized", "legacy-expired-first"),
                        default="synchronized")
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    if args.action == "worker":
        if not all((args.journal, args.tickets, args.entered)):
            parser.error("worker requires --journal, --tickets and --entered")
        crash_worker(args.journal, args.tickets, args.entered, args.mode)
    elif args.action == "acceptance":
        if args.artifact_dir is None:
            parser.error("acceptance requires --artifact-dir")
        acceptance(args.artifact_dir)
    else:
        if args.artifact_dir is None:
            parser.error("legacy-repro requires --artifact-dir")
        from tests.test_durable_shadow import _run_two_worker_crash, _assert_replay_once

        evidence = _run_two_worker_crash(
            args.artifact_dir, mode="legacy-expired-first", validate_entry=False,
        )
        write_json(args.artifact_dir / "acceptance.json", {
            "verdict": "FAIL", "checked_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hashes(), **evidence,
        })
        _assert_replay_once(evidence)


if __name__ == "__main__":
    main()
