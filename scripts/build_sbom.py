"""Offline SPDX 2.3 package/selected-metadata SBOM with create-once evidence.

No SDK import, dependency installation, environment discovery or network occurs.
Dependency METADATA hashes are not hashes of dependency binaries or whole runtime.
"""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

from packaging.markers import Marker, default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SHA256 = "4126dc29f15e92feec92e9622f4939131ad58b125d6f578aa93e980a6e6f3212"
SCHEMA_URL = "https://raw.githubusercontent.com/spdx/spdx-spec/v2.3.1/schemas/spdx-schema.json"
OWNED = ("scripts/build_sbom.py", "tests/test_build_sbom.py",
         "docs/research/sbom-delivery.md")
KNOWN_LICENSES = {
    "0BSD", "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
    "ISC", "Zlib", "MPL-2.0", "PSF-2.0", "Python-2.0",
    "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later",
    "LGPL-2.1-only", "LGPL-2.1-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later",
    "CC0-1.0", "Unlicense", "BSL-1.0",
}


class SbomError(ValueError):
    """Fixed non-content first causes; metadata/child text is never echoed."""


def require(ok, code):
    if not ok:
        raise SbomError(code)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_bytes(path, limit=4 * 1024 * 1024):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "input_file_unavailable")
    require(path.stat().st_size <= limit, "input_size_budget_exceeded")
    return path.read_bytes()


def json_read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate_json_key")
            out[key] = value
        return out

    try:
        return json.loads(read_bytes(path, 32 * 1024 * 1024),
                          object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError):
        raise SbomError("json_invalid") from None


def git_env():
    return {
        "PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1", "GIT_NO_REPLACE_OBJECTS": "1",
    }


def git(root, *args, input_bytes=None):
    try:
        result = subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", "-C", str(root),
             "-c", "safe.directory=" + str(Path(root).resolve()),
             "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
             *args], env=git_env(), input=input_bytes,
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise SbomError("source_git_unavailable") from None
    require(result.returncode == 0, "source_git_failed_" + args[0].replace("-", "_"))
    require(len(result.stdout) <= 32 * 1024 * 1024, "source_size_budget_exceeded")
    return result.stdout


def source_snapshot(root, expected):
    require(type(expected) is str and re.fullmatch(r"[0-9a-f]{40}", expected),
            "source_pin_required")
    root = Path(root).resolve(strict=True)
    require(git(root, "rev-parse", "HEAD").strip().decode("ascii") == expected,
            "source_head_mismatch")
    records = git(root, "ls-tree", "-rz", "--full-tree", expected).split(b"\0")
    rows = []
    for record in records:
        if not record:
            continue
        fields, raw_name = record.split(b"\t", 1)
        mode, kind, oid = fields.split()
        require(mode in {b"100644", b"100755"} and kind == b"blob",
                "source_nonordinary_file")
        name = raw_name.decode("utf-8")
        require(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                "source_path_invalid")
        rows.append((name, oid))
    require(rows and len(rows) <= 2048, "source_membership_budget_exceeded")
    payload = git(root, "cat-file", "--batch",
                  input_bytes=b"\n".join(oid for _, oid in rows) + b"\n")
    offset = 0
    hashes = {}
    for name, oid in rows:
        end = payload.index(b"\n", offset)
        header = payload[offset:end].split()
        require(len(header) == 3 and header[:2] == [oid, b"blob"],
                "source_object_invalid")
        size = int(header[2])
        committed = payload[end + 1:end + 1 + size]
        offset = end + size + 2
        path = root / name
        require(path.resolve().is_relative_to(root), "source_path_invalid")
        actual = read_bytes(path)
        require(actual == committed, "source_bytes_changed")
        hashes[name] = digest(actual)
    require(offset == len(payload), "source_object_invalid")
    require(not git(root, "status", "--porcelain=v1", "--untracked-files=all").strip(),
            "source_dirty")
    return {
        "commit": expected, "tree": git(root, "rev-parse", "HEAD^{tree}").strip().decode(),
        "clean": True, "sha256": hashes, "membership_sha256": digest(encoded(hashes)),
    }


def requirement(text):
    try:
        row = Requirement(text)
    except InvalidRequirement:
        raise SbomError("metadata_requirement_invalid") from None
    require(row.url is None, "direct_url_dependency_unsupported")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", row.name),
            "metadata_requirement_invalid")
    return row


def license_value(value):
    if not value:
        return "NOASSERTION"
    require(len(value) <= 512, "metadata_license_invalid")
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*|\(|\)", value)
    require("".join(tokens) == re.sub(r"\s+", "", value), "metadata_license_invalid")
    depth, operand = 0, True
    identifiers = []
    for token in tokens:
        if token == "(":
            require(operand, "metadata_license_invalid")
            depth += 1
        elif token == ")":
            require(not operand and depth > 0, "metadata_license_invalid")
            depth -= 1
        elif token in {"AND", "OR"}:
            require(not operand, "metadata_license_invalid")
            operand = True
        else:
            require(operand and token != "WITH", "metadata_license_invalid")
            identifiers.append(token)
            operand = False
    require(not operand and depth == 0, "metadata_license_invalid")
    # Unrecognized identifiers are not silently promoted to SPDX license IDs.
    return value if set(identifiers) <= KNOWN_LICENSES else "NOASSERTION"


def metadata_payload(payload):
    require(payload and len(payload) <= 4 * 1024 * 1024, "metadata_invalid")
    try:
        payload.decode("utf-8")
        message = BytesParser(policy=policy.default).parsebytes(payload)
    except (UnicodeError, ValueError):
        raise SbomError("metadata_invalid") from None
    require(not message.defects, "metadata_invalid")
    names, versions = message.get_all("Name", []), message.get_all("Version", [])
    require(len(names) == len(versions) == 1, "metadata_identity_invalid")
    name, version = str(names[0]), str(versions[0])
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)),
            "metadata_identity_invalid")
    try:
        Version(version)
    except InvalidVersion:
        raise SbomError("metadata_identity_invalid") from None
    require(len(version) <= 128 and "\n" not in version, "metadata_identity_invalid")
    expressions = message.get_all("License-Expression", [])
    require(len(expressions) <= 1, "metadata_license_invalid")
    expression = str(expressions[0]) if expressions else None
    declared = license_value(expression)
    requirements = [str(item) for item in message.get_all("Requires-Dist", [])]
    for item in requirements:
        requirement(item)
    extras = [canonicalize_name(str(item)) for item in message.get_all("Provides-Extra", [])]
    legacy = [str(item) for item in message.get_all("License", [])]
    return {
        "name": canonicalize_name(name), "version": version,
        "requires": requirements, "extras": extras,
        "metadata_sha256": digest(payload), "metadata_bytes": len(payload),
        "license_declared": declared,
        "license_metadata": {
            "expression_sha256": digest(expression.encode()) if expression else None,
            "legacy_sha256": digest(encoded(legacy)) if legacy else None,
            "state": "EXPRESSION_RECORDED" if declared != "NOASSERTION" else "UNRESOLVED",
        },
    }


def parse_metadata(path):
    return metadata_payload(read_bytes(path))


def project_model(root):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    try:
        data = tomllib.loads(read_bytes(root / "pyproject.toml").decode("utf-8"))
        project = data["project"]
        require(not project.get("dynamic"), "dynamic_project_metadata_unsupported")
        require(project["name"] == "openopt" and project["license"] == "0BSD",
                "project_identity_or_license_invalid")
        Version(project["version"])
        deps = project["dependencies"]
        require(isinstance(deps, list) and deps, "empty_project_dependencies")
        optional = project.get("optional-dependencies", {})
        require(isinstance(optional, dict), "project_metadata_invalid")
        declarations = [(None, item) for item in deps]
        for group, items in sorted(optional.items()):
            require(isinstance(items, list), "project_metadata_invalid")
            declarations.extend((canonicalize_name(group), item) for item in items)
        for _, item in declarations:
            requirement(item)
        build_requirements = data.get("build-system", {}).get("requires", [])
        require(isinstance(build_requirements, list), "project_metadata_invalid")
        for item in build_requirements:
            requirement(item)
        return (project["version"], declarations, set(map(canonicalize_name, optional)),
                list(build_requirements))
    except SbomError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError):
        raise SbomError("project_metadata_invalid") from None


def resolve_metadata(declarations, roots, extras, python_version):
    require(re.fullmatch(r"3\.[0-9]+", python_version), "python_target_invalid")
    environment = default_environment()
    environment.update(python_version=python_version,
                       python_full_version=python_version + ".0",
                       implementation_version=python_version + ".0", extra="")
    roots = sorted(set(Path(root).resolve(strict=True) for root in roots))
    candidates, pins = {}, {}
    for root in roots:
        require(root.is_dir(), "metadata_root_invalid")
        for path in sorted(root.glob("*.dist-info/METADATA")):
            directory = path.parent.name
            prefix = directory[:-10].rsplit("-", 1)[0]
            candidates.setdefault(canonicalize_name(prefix), []).append(path)
    found, pending, declaration_rows, unresolved, edges = {}, [], [], set(), set()
    pending.extend(("openopt", group, text, group is None or group in extras)
                   for group, text in declarations)
    visited = set()
    while pending:
        require(len(visited) < 512, "dependency_budget_exceeded")
        parent, group, text, selected = pending.pop(0)
        item = requirement(text)
        name = canonicalize_name(item.name)
        env = dict(environment, extra=group or "")
        active = selected and (item.marker is None or item.marker.evaluate(env))
        key = (parent, group, str(item), active)
        if key in visited:
            continue
        visited.add(key)
        row = {"parent": parent, "group": group, "name": name,
               "requirement": str(item), "active": active}
        paths = candidates.get(name, [])
        if not active and (len(paths) != 1 or parent != "openopt"):
            row["resolution"] = (
                "INACTIVE_AMBIGUOUS" if len(paths) > 1 else
                "OPTIONAL_PRESENT" if paths else "OPTIONAL_ABSENT")
            declaration_rows.append(row)
            continue
        require(len(paths) <= 1, "ambiguous_dependency_metadata:" + name)
        metadata = found.get(name)
        if paths and metadata is None:
            path = paths[0]
            require(path.resolve().is_relative_to(path.parent.parent),
                    "input_file_unavailable")
            metadata = parse_metadata(path)
            require(metadata["name"] == name, "metadata_identity_invalid")
            found[name] = metadata
            pins[str(path)] = metadata["metadata_sha256"]
        if not active:
            row["resolution"] = "OPTIONAL_PRESENT" if paths else "OPTIONAL_ABSENT"
        elif metadata is None:
            row["resolution"] = "ABSENT"
            unresolved.add(name)
        elif not item.specifier.contains(metadata["version"], prereleases=True):
            row["resolution"] = "VERSION_MISMATCH"
            unresolved.add(name)
        else:
            row["resolution"] = "RESOLVED"
            edges.add((parent, name))
            requested_extras = set(map(canonicalize_name, item.extras))
            require(requested_extras <= set(metadata["extras"]), "unknown_dependency_extra")
            contexts = requested_extras or {""}
            for dependency in metadata["requires"]:
                for context in contexts:
                    pending.append((name, context or None, dependency, True))
        declaration_rows.append(row)
    require(len(pins) <= 256, "dependency_budget_exceeded")
    return (found, sorted(declaration_rows, key=lambda row: encoded(row)),
            sorted(unresolved), edges, pins, environment)


def wheel_content(wheel, pin, root, source, version, declarations):
    payload = read_bytes(wheel, 16 * 1024 * 1024)
    require(type(pin) is str and digest(payload) == pin, "wheel_pin_mismatch")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
        members = {}
        for info in archive.infolist():
            name = info.filename
            require(not info.is_dir() and name not in members
                    and not PurePosixPath(name).is_absolute()
                    and ".." not in PurePosixPath(name).parts
                    and "\\" not in name and not info.flag_bits & 1
                    and (info.external_attr >> 16) & 0o170000 != 0o120000,
                    "wheel_member_invalid")
            require(info.file_size <= 4 * 1024 * 1024, "wheel_size_budget_exceeded")
            members[name] = archive.read(info)
            require(sum(map(len, members.values())) <= 32 * 1024 * 1024,
                    "wheel_size_budget_exceeded")
    except (zipfile.BadZipFile, RuntimeError):
        raise SbomError("wheel_invalid") from None
    metadata_names = [name for name in members if name.endswith(".dist-info/METADATA")]
    require(len(metadata_names) == 1, "wheel_metadata_invalid")
    prefix = metadata_names[0].rsplit("/", 1)[0]
    require(prefix.startswith("openopt-"), "wheel_metadata_invalid")
    metadata = metadata_payload(members[metadata_names[0]])
    require(metadata["name"] == "openopt" and metadata["version"] == version
            and metadata["license_declared"] == "0BSD", "wheel_metadata_mismatch")
    expected = set()
    for group, text in declarations:
        item = requirement(text)
        if group is not None:
            guard = f'extra == "{group}"'
            item.marker = Marker(
                f"({item.marker}) and {guard}" if item.marker else guard)
        expected.add(str(item))
    require(set(str(requirement(text)) for text in metadata["requires"]) == expected,
            "wheel_dependency_mismatch")
    record_name = prefix + "/RECORD"
    require(record_name in members, "wheel_record_missing")
    records = {}
    try:
        for record in csv.reader(io.StringIO(members[record_name].decode("utf-8"))):
            require(len(record) == 3 and record[0] not in records, "wheel_record_mismatch")
            records[record[0]] = record[1:]
    except (UnicodeError, csv.Error):
        raise SbomError("wheel_record_mismatch") from None
    require(set(records) == set(members), "wheel_record_mismatch")
    for name, value in members.items():
        if name == record_name:
            require(records[name] == ["", ""], "wheel_record_mismatch")
        else:
            token = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
            require(records[name] == ["sha256=" + token, str(len(value))],
                    "wheel_record_mismatch")
    package_files = {name: value for name, value in members.items()
                     if name.startswith(("openopt/", "openpo/"))}
    expected_modules = {name[4:]: value for name, value in source["sha256"].items()
                        if name.startswith(("src/openopt/", "src/openpo/"))}
    require({name: digest(value) for name, value in package_files.items()} == expected_modules,
            "wheel_source_content_mismatch")
    require(members.get(prefix + "/licenses/LICENSE") == read_bytes(root / "LICENSE"),
            "wheel_license_mismatch")
    require(all(name in package_files or name.startswith(prefix + "/") for name in members),
            "wheel_member_invalid")
    return {
        "sha256": pin, "bytes": len(payload), "name": Path(wheel).name,
        "version": version, "record_verified": True,
        "source_binding": "PACKAGE_BYTES_MATCH_ONLY",
        "source_commit_provenance": "UNVERIFIED_WITHOUT_BUILD_ATTESTATION",
        "members": {name: digest(value) for name, value in sorted(members.items())},
        "metadata_sha256": metadata["metadata_sha256"],
    }, members, prefix


def installed_content(root, members, prefix):
    root = Path(root).resolve(strict=True)
    actual = {}
    expected = {name: value for name, value in members.items()
                if name.startswith(("openopt/", "openpo/"))}
    for package in {name.split("/")[0] for name in expected}:
        directory = root / package
        require(directory.is_dir() and not directory.is_symlink(), "installed_content_mismatch")
        for path in directory.rglob("*"):
            require(not path.is_symlink(), "installed_content_mismatch")
            if path.is_file():
                actual[path.relative_to(root).as_posix()] = read_bytes(path)
    require(set(actual) == set(expected), "installed_membership_mismatch")
    require(actual == expected, "installed_content_mismatch")
    selected = {name: value for name, value in members.items()
                if name.startswith(prefix + "/") and name != prefix + "/RECORD"}
    for name, value in selected.items():
        require(read_bytes(root / name) == value, "installed_content_mismatch")
    return {"membership_missing": [], "membership_extra": [],
            "members": {name: digest(value) for name, value in sorted({**actual, **selected}.items())},
            "scope": "wheel_package_and_metadata_bytes_not_whole_environment",
            "installer_generated_record_or_entrypoints": "NOT_ATTESTED"}


def package_id(name):
    return "SPDXRef-" + name.replace("_", "-")


def package(name, version, license_declared, checksum, comment):
    return {
        "SPDXID": package_id(name), "name": name, "versionInfo": version,
        "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
        "licenseDeclared": license_declared, "licenseConcluded": "NOASSERTION",
        "copyrightText": "NOASSERTION",
        "checksums": [{"algorithm": "SHA256", "checksumValue": checksum}],
        "comment": comment,
    }


def validate_standard(document, path, pin):
    require(pin == SCHEMA_SHA256, "schema_pin_mismatch")
    payload = read_bytes(path)
    require(digest(payload) == pin, "schema_pin_mismatch")
    schema = json_read(path)
    require("$ref" not in json.dumps(schema), "external_schema_reference_forbidden")
    try:
        from jsonschema import Draft201909Validator
    except ImportError:
        raise SbomError("offline_schema_validator_unavailable") from None
    require(not list(Draft201909Validator(schema).iter_errors(document)),
            "spdx_schema_rejected")
    ids = [item["SPDXID"] for item in document["packages"]]
    require(ids and len(ids) == len(set(ids)), "spdx_package_identity_invalid")
    universe = set(ids) | {"SPDXRef-DOCUMENT"}
    for row in document["relationships"]:
        require(row["spdxElementId"] in universe and row["relatedSpdxElement"] in universe,
                "spdx_relationship_invalid")


def build(root, expected, metadata_roots, extras, python_version, schema, schema_pin,
          *, wheel=None, wheel_sha256=None, installed_root=None):
    root = Path(root).resolve(strict=True)
    before = source_snapshot(root, expected)
    version, declarations, known_extras, build_requirements = project_model(root)
    extras = set(map(canonicalize_name, extras))
    require(extras <= known_extras, "unknown_project_extra")
    resolved, rows, unresolved, edges, pins, marker_environment = resolve_metadata(
        declarations, metadata_roots, extras, python_version)
    evidence = {
        "verdict": "INCOMPLETE" if unresolved else "PASS",
        "scope": "selected_python_metadata_closure",
        "first_cause": "active_dependency_absent" if unresolved else None,
        "source": before, "declarations": rows, "unresolved": unresolved,
        "source_root": str(root),
        "selected_extras": sorted(extras), "target_python": python_version,
        "marker_environment": marker_environment,
        "build_requirements": [{"requirement": str(requirement(item)),
                                "resolution": "DECLARED_NOT_RESOLVED"}
                               for item in build_requirements],
        "target_platform": default_environment()["sys_platform"],
        "resolved": [resolved[name] for name in sorted(resolved)],
        "unknown_licenses": sorted(name for name, row in resolved.items()
                                   if row["license_declared"] == "NOASSERTION"),
        "unlisted_environment_distributions": "NOT_SAMPLED",
        "dependency_binaries_and_vendored_native_libraries": "NOT_ANALYZED",
        "legal_qualification": "UNVERIFIED", "runtime_qualification": "NOT_TESTED",
        "network_attempts": 0, "credential_files_read": 0,
        "sdk_imports": 0, "solver_attempts": 0,
        "schema": {"url": SCHEMA_URL, "sha256": schema_pin},
    }
    packages = [package("openopt-source", version, "0BSD", before["membership_sha256"],
                        "Checksum covers source membership manifest, not an archive.")]
    for name, metadata in sorted(resolved.items()):
        packages.append(package(name, metadata["version"], metadata["license_declared"],
                                metadata["metadata_sha256"],
                                "Checksum covers METADATA only; binaries NOT_ANALYZED."))
    require((wheel is None) == (wheel_sha256 is None), "wheel_pin_required")
    if wheel is not None:
        wheel_row, members, prefix = wheel_content(
            wheel, wheel_sha256, root, before, version, declarations)
        evidence["wheel"] = wheel_row
        evidence["wheel_input"] = str(Path(wheel).resolve())
        packages.append(package("openopt-wheel", version, "0BSD", wheel_sha256,
                                "Actual wheel archive checksum; package bytes match source."))
        if installed_root is not None:
            evidence["installed"] = installed_content(installed_root, members, prefix)
            evidence["installed_input"] = str(Path(installed_root).resolve())
        else:
            evidence["installed"] = {"state": "NOT_SAMPLED"}
    else:
        require(installed_root is None, "installed_requires_wheel")
        evidence["wheel"] = {"state": "NOT_SAMPLED"}
        evidence["installed"] = {"state": "NOT_SAMPLED"}
    for path, pin in pins.items():
        require(digest(read_bytes(Path(path))) == pin, "metadata_changed")
    if wheel is not None:
        require(digest(read_bytes(wheel, 16 * 1024 * 1024)) == wheel_sha256,
                "wheel_changed")
    if installed_root is not None:
        require(installed_content(installed_root, members, prefix) == evidence["installed"],
                "installed_changed")
    after = source_snapshot(root, expected)
    require(before == after, "source_changed")
    evidence["source_after"] = after
    evidence["source_consistency"] = "PASS"
    evidence["metadata_inputs"] = [
        {"path": path, "sha256": pin} for path, pin in sorted(pins.items())]
    relationships = [{"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES",
                      "relatedSpdxElement": item["SPDXID"]} for item in packages]
    relationships.extend({
        "spdxElementId": package_id("openopt-source" if parent == "openopt" else parent),
        "relationshipType": "DEPENDS_ON", "relatedSpdxElement": package_id(child),
    } for parent, child in sorted(edges))
    document = {
        "spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT", "name": "openopt-bounded-package-sbom",
        "documentNamespace": "https://spdx.org/spdxdocs/openopt-" + digest(encoded(evidence)),
        "creationInfo": {"creators": ["Tool: openopt-build-sbom-1"],
                         "created": utc()},
        "packages": packages, "relationships": relationships,
        "comment": "Selected metadata scope only; unknown licenses NOASSERTION; no legal/runtime qualification.",
    }
    validate_standard(document, schema, schema_pin)
    return document, evidence


def output_root(path):
    path = Path(path).resolve()
    shared = Path("/hf3fs-jd/data/tmp").resolve(strict=True)
    require(path.is_relative_to(shared) and path != shared,
            "external_shared_output_required")
    require(not path.is_relative_to(ROOT), "output_inside_source")
    return path


def save_delivery(path, document, evidence, *, require_shared=True):
    path = output_root(path) if require_shared else Path(path)
    require(not path.exists(), "output_already_exists")
    path.mkdir(parents=True, exist_ok=False)
    files = {
        "sbom.spdx.json": encoded(document), "evidence.json": encoded(evidence),
        "execution.log": (
            f"SBOM verdict={evidence['verdict']} source={evidence['source']['commit']} "
            f"resolved={len(evidence['resolved'])} legal=UNVERIFIED runtime=NOT_TESTED\n"
        ).encode("ascii"),
    }
    for name, payload in files.items():
        with (path / name).open("xb") as stream:
            stream.write(payload)
    receipt = {
        "verdict": evidence["verdict"], "created": utc(),
        "artifacts_sha256": {name: digest(payload) for name, payload in files.items()},
        "source_commit": evidence["source"]["commit"],
        "schema_sha256": evidence["schema"]["sha256"],
        "runtime_qualification": "NOT_TESTED", "legal_qualification": "UNVERIFIED",
    }
    target = path / "receipt.json"
    with target.open("xb") as stream:
        stream.write(encoded(receipt))
    return target


def read_delivery(path, pin, schema, schema_pin):
    require(re.fullmatch(r"[0-9a-f]{64}", pin or ""), "receipt_pin_required")
    require(digest(read_bytes(path)) == pin, "receipt_pin_mismatch")
    receipt = json_read(path)
    require(receipt.get("schema_sha256") == schema_pin,
            "receipt_schema_pin_mismatch")
    require(set(receipt["artifacts_sha256"])
            == {"sbom.spdx.json", "evidence.json", "execution.log"},
            "artifact_membership_invalid")
    for name, checksum in receipt["artifacts_sha256"].items():
        require(digest(read_bytes(Path(path).parent / name, 32 * 1024 * 1024)) == checksum,
                "artifact_pin_mismatch")
    document = json_read(Path(path).parent / "sbom.spdx.json")
    evidence = json_read(Path(path).parent / "evidence.json")
    validate_standard(document, schema, schema_pin)
    require(evidence.get("schema", {}).get("sha256") == schema_pin,
            "evidence_schema_pin_mismatch")
    require(receipt["verdict"] == evidence["verdict"], "receipt_verdict_mismatch")
    require(receipt.get("source_commit") == evidence["source"]["commit"],
            "receipt_source_mismatch")
    for field, expected in (("legal_qualification", "UNVERIFIED"),
                            ("runtime_qualification", "NOT_TESTED")):
        require(receipt.get(field) == evidence.get(field) == expected,
                "receipt_qualification_mismatch")
    require(source_snapshot(Path(evidence["source_root"]), receipt["source_commit"])
            == evidence["source"], "reader_source_changed")
    if "producer" in evidence:
        require(source_snapshot(Path(evidence["producer_root"]),
                                evidence["producer"]["commit"]) == evidence["producer"],
                "reader_producer_changed")
    for row in evidence["metadata_inputs"]:
        require(digest(read_bytes(Path(row["path"]))) == row["sha256"],
                "reader_metadata_changed")
    if "wheel_input" in evidence:
        require(digest(read_bytes(Path(evidence["wheel_input"]), 16 * 1024 * 1024))
                == evidence["wheel"]["sha256"], "reader_wheel_changed")
    if "installed_input" in evidence:
        target = Path(evidence["installed_input"])
        expected = evidence["installed"]["members"]
        package_names = {name.split("/")[0] for name in expected
                         if name.startswith(("openopt/", "openpo/"))}
        actual = set()
        for name in package_names:
            for candidate in (target / name).rglob("*"):
                require(not candidate.is_symlink(), "reader_installed_changed")
                if candidate.is_file():
                    actual.add(candidate.relative_to(target).as_posix())
        require(actual == {name for name in expected
                           if name.startswith(("openopt/", "openpo/"))},
                "reader_installed_membership_changed")
        for name, checksum in expected.items():
            require(digest(read_bytes(target / name)) == checksum,
                    "reader_installed_changed")
    return receipt


def run_tests(test_path, argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = output_root(args.artifact_dir)
    require(not directory.exists(), "output_already_exists")
    directory.mkdir(parents=True)
    env = {
        **git_env(), "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": "-1",
    }
    from scripts import verify, verify_sharded as observer
    before = verify.source_snapshot(ROOT)
    state = "FAIL"
    try:
        collection = directory / "collect"
        collection.mkdir()
        env["OPENOPT_SHARD_COLLECTION"] = str(collection / "collection.json")
        command = [sys.executable, "-B", "-m", "pytest", "-q", "-o", "addopts=",
                   "-o", f"cache_dir={directory / 'cache'}",
                   "--basetemp", str(directory / "tmp"), str(test_path)]
        observer.command_logged(command[:-1] + ["--collect-only", "-p", "scripts.verify_sharded",
                                              str(test_path)],
                                ROOT, collection, 45, env)
        nodes = observer.read(collection / "collection.json")["nodeids"]
        env.pop("OPENOPT_SHARD_COLLECTION")
        observer.command_logged(command + [f"--junitxml={directory / 'pytest.xml'}"],
                                ROOT, directory, 170, env)
        cases, counts = observer.parse_junit(directory / "pytest.xml")
        require(len(nodes) == len(set(nodes)) and nodes, "empty_or_duplicate_collection")
        require({observer.junit_key(node) for node in nodes}
                == {tuple(row["key"]) for row in cases},
                "junit_membership_mismatch")
        require(counts == {"tests": len(nodes), "failures": 0, "errors": 0, "skipped": 0},
                "test_counts_invalid")
        after = verify.source_snapshot(ROOT)
        require(before == after, "test_source_changed")
        state = "PASS"
        report = {"verdict": state, "counts": counts, "source": before,
                  "source_after": after, "collection": nodes,
                  "sdk_imports": 0, "solver_attempts": 0, "network_attempts": 0}
    except Exception as error:
        report = {"verdict": "FAIL", "first_cause": (
            str(error) if isinstance(error, SbomError) else type(error).__name__),
                  "source": before}
    target = directory / "verification.json"
    with target.open("xb") as stream:
        stream.write(encoded(report))
    print(f"SBOM_TESTS verdict={state} receipt={target} sha256={digest(target.read_bytes())}")
    return 0 if state == "PASS" else 1


def scan_owned():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.audit_git_delivery import SHAPES

    rows = {}
    for name in OWNED:
        payload = read_bytes(ROOT / name)
        rows[name] = {"shape": sum(len(pattern.findall(payload)) for pattern in SHAPES),
                      "sha256": digest(payload)}
    require(all(row["shape"] == 0 for row in rows.values()), "owned_shape_scan_failed")
    print("SBOM_SCAN shape=0 files=3 literal=NOT_RUN credential_reads=0")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--metadata-root", type=Path, action="append", default=[])
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--target-python", default="3.10")
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--schema-sha256")
    parser.add_argument("--scan-owned", action="store_true")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--wheel-sha256")
    parser.add_argument("--installed-root", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--read-receipt", type=Path)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    try:
        if args.scan_owned:
            scan_owned()
            return 0
        require(args.schema is not None and args.schema_sha256 is not None,
                "official_schema_required")
        if args.read_receipt:
            result = read_delivery(args.read_receipt, args.receipt_sha256,
                                   args.schema, args.schema_sha256)
            print(f"SBOM_READER integrity=PASS verdict={result['verdict']} legal=UNVERIFIED runtime=NOT_TESTED")
            return 0 if result["verdict"] == "PASS" else 1
        require(args.artifact_dir is not None, "artifact_directory_required")
        path = output_root(args.artifact_dir)
        require(not path.exists(), "output_already_exists")
        producer = source_snapshot(
            ROOT, git(ROOT, "rev-parse", "HEAD").strip().decode("ascii"))
        document, evidence = build(
            args.source_root, args.source_commit, args.metadata_root, args.extra,
            args.target_python, args.schema, args.schema_sha256, wheel=args.wheel,
            wheel_sha256=args.wheel_sha256, installed_root=args.installed_root)
        evidence["source_root"] = str(args.source_root.resolve())
        evidence["producer_sha256"] = digest(read_bytes(Path(__file__)))
        evidence["producer_root"] = str(ROOT)
        evidence["producer"] = producer
        require(source_snapshot(ROOT, producer["commit"]) == producer,
                "producer_changed")
        target = save_delivery(path, document, evidence)
        print(f"SBOM verdict={evidence['verdict']} resolved={len(evidence['resolved'])} "
              f"unknown_licenses={len(evidence['unknown_licenses'])} receipt={target} "
              f"sha256={digest(target.read_bytes())} legal=UNVERIFIED runtime=NOT_TESTED")
        return 0 if evidence["verdict"] == "PASS" else 1
    except Exception as error:
        code = str(error) if isinstance(error, SbomError) else type(error).__name__
        if args.artifact_dir and not args.artifact_dir.exists():
            path = output_root(args.artifact_dir)
            path.mkdir(parents=True)
            failure = {"verdict": "FAIL", "first_cause": code, "finished_at": utc(),
                       "network_attempts": 0, "credential_files_read": 0}
            with (path / "failure.json").open("xb") as stream:
                stream.write(encoded(failure))
        print(f"SBOM verdict=FAIL first_cause={code}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
