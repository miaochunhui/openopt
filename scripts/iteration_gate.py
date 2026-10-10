"""Small correctness preflight; deliberately not the old performance gate."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify import main  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(main(["smoke"]))
    raise SystemExit(main(sys.argv[1:]))
