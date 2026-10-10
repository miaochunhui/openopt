"""Current disk acceptance for evidence, report content and secret-safe source.

No browser-rendering or online-publication claim is made by this structural
check. Includes negative report-generation tests on isolated external fixtures.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile

from render_qp_report import generate

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag not in {"meta", "link", "br", "hr", "img", "input", "source", "wbr"}:
            self.stack.append(tag)
        if tag == "a":
            self.targets.append(attributes.get("href", ""))
        if tag == "option":
            self.targets.append(attributes.get("value", ""))

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            raise ValueError(f"HTML tag mismatch: {tag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    pages = generate(args.bundle)
    if pages != generate(args.bundle):
        raise ValueError("report generation is not deterministic")
    for path, expected in pages.items():
        if not path.is_file() or path.read_text() != expected:
            raise ValueError(f"generated document differs: {path}")
        if path.suffix == ".html":
            inspector = Links()
            inspector.feed(expected)
            if inspector.stack:
                raise ValueError(f"HTML tags left open: {path}")
            for target in inspector.targets:
                if not target or ":" in target or target.startswith("#"):
                    continue
                relative = target.split("#")[0]
                if not (path.parent / relative).is_file():
                    raise ValueError(f"broken document link: {path}: {target}")
    sensitive = re.compile(
        r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|"
        r"""(?i:password|token|secret)\s*[\"']?\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']"""
    )
    for path in args.source:
        if sensitive.search(path.read_text()):
            raise ValueError(f"potential literal credential: {path}")
    root = args.bundle.parent
    original = json.loads(args.bundle.read_text())
    negatives = 0
    with tempfile.TemporaryDirectory(prefix="report-negatives-", dir=root) as tmp:
        tmp = Path(tmp)
        for name in ("failed_regression", "missing_pairs", "mismatched_reference"):
            bundle = json.loads(json.dumps(original))
            fixture = tmp / f"{name}.json"
            if name == "failed_regression":
                bad = tmp / "failed-verification.json"
                bad.write_text(json.dumps({"verdict": "FAIL"}))
                bundle["verification"] = str(bad)
            if name == "missing_pairs":
                bundle["paired_results"] = []
            if name == "mismatched_reference":
                reference = json.loads(Path(bundle["reference_results"][0]).read_text())
                reference["problem_hash"] = "intentionally-invalid-negative-control"
                bad = tmp / "mismatched-reference.json"
                bad.write_text(json.dumps(reference))
                bundle["reference_results"][0] = str(bad)
            fixture.write_text(json.dumps(bundle))
            try:
                generate(fixture)
            except ValueError:
                negatives += 1
            else:
                raise ValueError(f"negative control falsely passed: {name}")
    report = {
        "verdict": "PASS", "read_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(pages), "negative_controls": negatives,
        "sources_scanned": len(args.source), "browser_render": "UNVERIFIED",
        "online_publication": "UNVERIFIED",
    }
    (root / "delivery-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"QP_DELIVERY verdict=PASS read_at={report['read_at']} documents={len(pages)} "
        f"negative_controls={negatives} sources_scanned={len(args.source)} "
        "browser_render=UNVERIFIED online_publication=UNVERIFIED"
    )


if __name__ == "__main__":
    main()
