from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from threading import Barrier
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openpo import durable_shadow as durable_module
from openpo.durable_shadow import DurableShadowCertificationService
from tests.test_facade import _request


def main() -> None:
    journal_path = Path(sys.argv[1])
    tickets_path = Path(sys.argv[2])
    entered_path = Path(sys.argv[3])
    barrier = Barrier(2, timeout=20.0)

    def block_forever(request):
        barrier.wait()
        entered_path.write_text("entered", encoding="utf-8")
        while True:
            time.sleep(1.0)

    durable_module.certify_exact = block_forever
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
    tickets_path.write_text(
        json.dumps([asdict(submission.ticket) for submission in submissions], sort_keys=True),
        encoding="utf-8",
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
