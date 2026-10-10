"""Install a small, pinned validation wheel set into an explicit external venv.

Downloads from a domestic public mirror using a single curl capped at 4 MiB/s.
Only wheels compatible with the running interpreter are selected. Wheel hashes
from the index are checked before installation; shared environments are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urljoin, urlsplit

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

PINS = {
    "numpy": "2.2.6", "scipy": "1.15.3", "pytest": "8.4.2",
    "osqp": "1.0.4", "clarabel": "0.11.1",
    "iniconfig": "2.1.0", "pluggy": "1.6.0", "pygments": "2.19.2",
    "exceptiongroup": "1.3.0", "tomli": "2.2.1",
    "joblib": "1.5.2", "cffi": "1.17.1", "pycparser": "2.22",
}
TORCH_RUNTIME = {
    "typing-extensions": "4.15.0", "filelock": "3.19.1",
    "sympy": "1.14.0", "mpmath": "1.3.0", "networkx": "3.4.2",
    "jinja2": "3.1.6", "markupsafe": "3.0.2", "fsspec": "2025.9.0",
}
INDEX = "https://mirrors.aliyun.com/pypi/simple/"
ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attributes = dict(attrs)
            if "href" in attributes:
                self.links.append(attributes["href"])


def execute(command, log):
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    log.flush()
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed rc={result.returncode}; see {log.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--torch-runtime", action="store_true")
    parser.add_argument("--highspy", action="store_true")
    args = parser.parse_args()
    if Path(sys.prefix).resolve().is_relative_to(ROOT) or sys.prefix == sys.base_prefix:
        raise ValueError("run with an isolated external venv, not a shared interpreter")
    folder = args.wheelhouse.resolve()
    if folder.is_relative_to(ROOT):
        raise ValueError("wheelhouse must be outside the repository")
    folder.mkdir(parents=True, exist_ok=False)
    supported = list(sys_tags())
    rank = {tag: index for index, tag in enumerate(supported)}
    wheels = []
    pins = {"highspy": None} if args.highspy else TORCH_RUNTIME if args.torch_runtime else PINS
    with (folder / "bootstrap.log").open("x") as log:
        for name, version in pins.items():
            index = folder / f"{name}-index.html"
            execute(["curl", "--fail", "--location", "--limit-rate", "4M",
                     "--output", str(index), INDEX + name + "/"], log)
            links = Links()
            links.feed(index.read_text())
            candidates = []
            for link in links.links:
                filename = unquote(Path(urlsplit(link).path).name)
                if not filename.endswith(".whl"):
                    continue
                package, candidate_version, _, tags = parse_wheel_filename(filename)
                compatible = tags.intersection(rank)
                wanted = (
                    Version("1.15") <= candidate_version < Version("2") and not candidate_version.is_prerelease
                    if version is None else str(candidate_version) == version
                )
                if wanted and compatible:
                    candidates.append((candidate_version, -min(rank[tag] for tag in compatible), filename, link))
            if not candidates:
                raise ValueError(f"no compatible wheel for {name}=={version}")
            selected_version, _, filename, link = max(candidates)
            pins[name] = str(selected_version)
            url = urljoin(INDEX + name + "/", link)
            fragment = urlsplit(url).fragment
            if not fragment.startswith("sha256="):
                raise ValueError(f"index provides no SHA256: {filename}")
            wheel = folder / filename
            execute(["curl", "--fail", "--location", "--limit-rate", "4M",
                     "--output", str(wheel), url], log)
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != fragment[7:]:
                raise ValueError(f"wheel SHA256 mismatch: {filename}")
            wheels.append(str(wheel))
            print(f"VALIDATION_WHEEL verified={filename}", flush=True)
        execute([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *wheels], log)
        execute([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                 "--no-build-isolation", "--editable", str(ROOT)], log)
    (folder / "manifest.json").write_text(json.dumps(pins, indent=2) + "\n")
    print(f"VALIDATION_BOOTSTRAP verdict=PASS wheels={len(wheels)} log={folder / 'bootstrap.log'}")


if __name__ == "__main__":
    main()
