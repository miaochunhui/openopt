"""Discover SDK metadata without importing solvers or checking out licenses.

The default mode is offline and does not inspect environment variables,
credentials, SDK locations or host identity. UNVERIFIED is not a licensed run.
--source-url is a separate, explicit public-document reading mode for research;
it never runs an SDK and stores only provenance, not the reference document.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
SDKS = {
    "gurobipy": ("gurobipy",),
    "cplex": ("cplex",),
    "mosek": ("Mosek",),
    "cuopt": ("cuopt-cu12", "cuopt-cu13", "cuopt"),
}
SOURCE_HOSTS = {
    "docs.gurobi.com", "support.gurobi.com", "www.ibm.com",
    "docs.mosek.com", "www.mosek.com", "docs.nvidia.com",
    "raw.githubusercontent.com", "api.github.com", "arxiv.org",
    "docs.python.org",
}


def _safe_error(error: Exception, stage: str) -> dict[str, object]:
    # Exception messages can contain a license value, SDK path or server name.
    result: dict[str, object] = {
        "stage": stage, "type": type(error).__name__, "details_redacted": True,
    }
    if isinstance(error, HTTPError):
        result["http_status"] = error.code
    if isinstance(error, OSError) and error.errno is not None:
        result["errno"] = error.errno
    return result


def probe_sdk(module: str) -> dict[str, object]:
    if module not in SDKS:
        raise ValueError("unknown SDK module")
    result: dict[str, object] = {
        "module": module, "module_available": None, "versions": {},
        "sdk_status": "UNVERIFIED", "licensed_run": "UNVERIFIED",
        "solve_attempted": False, "license_connection_attempted": False,
        "errors": [],
    }
    errors = result["errors"]
    assert isinstance(errors, list)
    try:
        result["module_available"] = importlib.util.find_spec(module) is not None
    except Exception as error:
        errors.append(_safe_error(error, "find_spec"))
    versions = result["versions"]
    assert isinstance(versions, dict)
    for distribution in SDKS[module]:
        try:
            version = importlib.metadata.version(distribution)
            if not isinstance(version, str) or not re.fullmatch(
                r"[0-9][0-9A-Za-z.!+_-]{0,127}", version,
            ):
                raise ValueError("invalid distribution version")
            versions[distribution] = version
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception as error:
            errors.append(_safe_error(error, "distribution_version"))
    if errors:
        result.update(sdk_status="ERROR", reason="DISCOVERY_ERROR")
    elif not result["module_available"]:
        result["reason"] = "SDK_NOT_DISCOVERED"
    elif not versions:
        result["reason"] = "VERSION_METADATA_NOT_DISCOVERED"
    else:
        result.update(sdk_status="AVAILABLE", reason="DISCOVERY_ONLY_NOT_LICENSED")
    return result


def probe(modules: tuple[str, ...] = tuple(SDKS)) -> dict[str, object]:
    if not modules or len(set(modules)) != len(modules):
        raise ValueError("SDK selection must be nonempty and unique")
    results = {module: probe_sdk(module) for module in modules}
    failed = any(result["sdk_status"] == "ERROR" for result in results.values())
    return {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "scope": "sdk_discovery_only",
        "collection_status": "FAIL" if failed else "COMPLETE",
        "verdict": "FAIL" if failed else "UNVERIFIED",
        "qualification": "UNVERIFIED",
        "license_checked": False,
        "network_requested": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sdks": results,
    }


class _DocumentText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "svg"):
            self.hidden += 1
        if tag == "a" and not self.hidden:
            self.links.extend((key, value) for key, value in attrs if key == "href" and value)
        if not self.hidden and tag in ("p", "div", "section", "li", "h1", "h2", "h3", "tr", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "svg") and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _validate_source(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https" or parsed.hostname not in SOURCE_HOSTS
        or parsed.username or parsed.password or parsed.port not in (None, 443)
    ):
        raise ValueError("only allowlisted public HTTPS sources are permitted")


class _PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_source(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen(url: str, *, timeout: int):
    return build_opener(_PublicRedirect()).open(url, timeout=timeout)


def read_document(url: str, terms: list[str]) -> dict[str, object]:
    _validate_source(url)
    with urlopen(url, timeout=45) as response:
        _validate_source(response.url)
        payload = response.read(8_000_001)
        if not payload or len(payload) > 8_000_000:
            raise ValueError("document was empty or exceeded the size limit")
        content_type = response.headers.get("Content-Type", "")
    text = payload.decode("utf-8")
    if "html" in content_type:
        parser = _DocumentText()
        parser.feed(text)
        text = "".join(parser.parts)
        for _, link in parser.links:
            if any(term.casefold() in link.casefold() for term in terms):
                print(f"SOURCE_LINK {link[:1200]}")
    elif "json" in content_type:
        text = json.dumps(json.loads(text), indent=2)
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    selected = [
        index for index, line in enumerate(lines)
        if any(term.casefold() in line.casefold() for term in terms)
    ]
    # Display bounded context for inspection; do not persist the source text.
    indices = sorted({i for index in selected for i in range(max(0, index - 1), min(len(lines), index + 3))})
    for index in indices[:90]:
        print(f"SOURCE_TEXT line={index + 1} {lines[index][:1600]}")
    if not selected:
        raise ValueError("no requested term found in document")
    return {
        "scope": "public_document_provenance", "url": url,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "matched_lines": len(selected), "terms": terms, "verdict": "READ",
        "reference_document_persisted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path(
        "/hf3fs-jd/data/tmp/openopt-commercial-20261010/probe",
    ))
    parser.add_argument("--source-url", help="explicit public-document mode, not an SDK probe")
    parser.add_argument("--term", action="append", default=[])
    args = parser.parse_args(argv)
    if args.term and not args.source_url:
        parser.error("--term requires --source-url")
    root = args.artifact_dir.expanduser().resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("artifacts must be outside the worktree")
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="source-" if args.source_url else "sdk-", dir=root))
    try:
        report = read_document(args.source_url, args.term or ["GPU"]) if args.source_url else probe()
    except Exception as error:
        report = {
            "verdict": "FAIL", "collection_status": "FAIL",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "error": _safe_error(error, "public_document" if args.source_url else "sdk_probe"),
        }
        if args.source_url:
            try:
                _validate_source(args.source_url)
            except ValueError:
                pass  # Do not persist rejected URLs, which can contain credentials.
            else:
                report["url"] = args.source_url
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with (output / "result.json").open("x", encoding="utf-8") as result_file:
        result_file.write(serialized)
    if json.loads((output / "result.json").read_text(encoding="utf-8")) != report:
        raise RuntimeError("persisted report mismatch")
    line = (
        f"COMMERCIAL_PROBE verdict={report['verdict']} "
        f"collected_at={report['collected_at']} result={output / 'result.json'}"
    )
    with (output / "execution.log").open("x", encoding="utf-8") as log:
        log.write(line + "\n")
    print(serialized if not args.source_url else json.dumps(report), end="\n")
    print(line, flush=True)
    return 1 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
