"""Offline insufficient-evidence precheck, NEVER a release certification.

Caller SHA256 pins bind bytes, not execution, legal authority or capability.
Even complete-looking inputs return REVIEW_REQUIRED, never PASS/completed.
No solving, input code execution, network, SDK, publication or GPU submission.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 16 * 1024 * 1024
HASH = re.compile(r"[0-9a-f]{64}\Z")
SCOPE = "insufficient-release-evidence-v1"
CAPABILITIES = (
    "lp", "convex_qp", "sparse_matrix_free", "scaling", "infeasibility_recession",
    "signed_gap_status", "soc_rsoc", "sdp", "exponential_power", "general_milp",
    "general_convex_miqp", "nonlinear_global", "presolve_postsolve", "refinement",
    "held_out_public_robustness", "perturbation_long_run", "failure_tail_controls",
    "cold_prepared_changing_data", "release_runtime_compatibility",
)
SOLVERS = ("gurobi", "cplex", "mosek")


class Rejected(ValueError):
    """Safe first-cause codes; never echo untrusted payloads/exception text."""

    def __init__(self, code, at):
        self.code, self.at = code, at
        super().__init__(f"{code} at {at}")


def require(condition, code, at):
    if not condition:
        raise Rejected(code, at)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def keys(value, names, at):
    require(type(value) is dict and set(value) == set(names), "schema_keys", at)


def integer(value, at, minimum=0):
    require(type(value) is int and value >= minimum, "invalid_integer", at)
    return value


def parse(payload, at, *, historical=False):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key", at)
            result[key] = value
        return result

    def nonfinite(_):
        raise Rejected("nonfinite_json", at)

    def walk(value):
        require(type(value) is not bool or historical, "boolean_not_evidence", at)
        if type(value) in (int, float):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            require(finite, "nonfinite_json", at)
        elif type(value) is dict:
            for child in value.values():
                walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
        walk(value)
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise Rejected("invalid_json", at) from error


def absolute(value, at):
    require(isinstance(value, (str, Path)), "invalid_path", at)
    raw = str(value)
    require(raw.startswith("/") and ".." not in raw.split("/") and "\\" not in raw
            and not any(ord(c) < 32 for c in raw), "invalid_path", at)
    if raw == "/hf3fs-jd" or raw.startswith("/hf3fs-jd/"):
        raw = "/3fs/research3fs" + raw[len("/hf3fs-jd"):]
    return Path(raw)


def read(path, at):
    """Reject symlink parents/files and special files; check read-time identity."""
    path = absolute(path, at)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode), "not_regular_file", at)
            require(0 < before.st_size <= MAX_BYTES, "empty_or_large_input", at)
            payload = stream.read(MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after) and len(payload) == after.st_size,
                "input_changed_during_read", at)
        return payload
    except OSError as error:
        raise Rejected("unsafe_or_missing_file", at) from error
    finally:
        os.close(fd)


def source_map(value, at):
    require(type(value) is dict and value, "empty_source_map", at)
    for name, pin in value.items():
        require(type(name) is str and not PurePosixPath(name).is_absolute()
                and ":" not in name and "\\" not in name
                and all(p not in ("", ".", "..") for p in name.split("/")), "source_path", at)
        require(type(pin) is str and HASH.fullmatch(pin), "invalid_sha256", at)
    return value


def current_sources(root):
    root = absolute(root, "source")

    def git(*args):
        process = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
        require(process.returncode == 0, "git_query_failed", "source")
        return process.stdout

    require(not git("status", "--porcelain", "--untracked-files=all"), "source_not_clean", "source")
    commit = git("rev-parse", "HEAD").decode().strip()
    names = git("ls-files", "-z", "src", "scripts", "LICENSE", "pyproject.toml").decode().split("\0")
    sources = {}
    for name in sorted(set(names) - {""}):
        payload = read(root / name, "source")
        require(payload == git("cat-file", "blob", f"{commit}:{name}"), "source_not_committed", "source")
        sources[name] = sha(payload)
    source_map(sources, "source")
    return commit, sources


class Precheck:
    def __init__(self, manifest, pin, current_root):
        self.root = absolute(current_root, "source")
        self.inputs, self.incomplete, self.checked, self.observations = {}, [], [], []
        self.pin = pin
        self.manifest = parse(self.input(manifest, pin, "manifest"), "manifest")
        m = self.manifest
        keys(m, ("schema_version", "scope", "source_commit", "source_sha256", "capability_scope",
                 "capabilities", "commercial", "gpu_scaling", "publication", "licensing",
                 "historical_observations"), "manifest")
        require(type(m["schema_version"]) is int and m["schema_version"] == 1 and m["scope"] == SCOPE,
                "schema_version_or_scope", "manifest")
        require(type(m["source_commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", m["source_commit"]),
                "source_commit", "manifest")
        source_map(m["source_sha256"], "manifest")
        require(m["capability_scope"] == list(CAPABILITIES), "incomplete_capability_scope", "manifest")
        for field in ("capabilities", "commercial", "gpu_scaling"):
            require(type(m[field]) is dict, "schema_keys", field)
        require(set(m["capabilities"]) <= set(CAPABILITIES) and set(m["commercial"]) <= set(SOLVERS)
                and set(m["gpu_scaling"]) <= {"1", "2", "3"}, "unknown_selection", "manifest")

    def input(self, path, pin, at, size=None):
        path = absolute(path, at)
        require(not path.is_relative_to(self.root) and not path.is_relative_to(ROOT),
                "receipt_must_be_external", at)
        require(type(pin) is str and HASH.fullmatch(pin), "invalid_sha256", at)
        payload = read(path, at)
        require(sha(payload) == pin, "byte_pin_mismatch", at)
        if size is not None:
            require(integer(size, at, 1) == len(payload), "byte_size_mismatch", at)
        require(path not in self.inputs or self.inputs[path] == payload, "input_changed", at)
        self.inputs[path] = payload
        return payload

    def blob(self, desc, at):
        keys(desc, ("path", "sha256", "bytes"), at)
        integer(desc["bytes"], at, 1)
        return self.input(desc["path"], desc["sha256"], at, desc["bytes"])

    def document(self, desc, at):
        return parse(self.blob(desc, at), at)

    def attempt(self, at, operation):
        try:
            operation()
        except Rejected as error:
            self.incomplete.append({"section": at, "code": error.code, "coordinate": error.at})
        else:
            self.checked.append(at)

    def receipt(self, desc, at):
        require(desc is not None, "missing_actual_receipt", at)
        receipt = self.document(desc, at)
        keys(receipt, ("schema_version", "section", "source_sha256", "producer", "auditor",
                       "actual", "acceptance"), at)
        require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
                and receipt["section"] == at, "unsupported_receipt_schema", at)
        require(receipt["source_sha256"] == self.manifest["source_sha256"], "source_mismatch", at)
        producer, auditor = self.blob(receipt["producer"], at), self.blob(receipt["auditor"], at)
        require(sha(producer) in self.manifest["source_sha256"].values(), "producer_source_mismatch", at)
        require(producer != auditor, "nonindependent_source", at)
        actual = self.document(receipt["actual"], at)
        keys(actual, ("source_sha256", "section", "data"), at)
        require(actual["source_sha256"] == receipt["source_sha256"] and actual["section"] == at,
                "actual_source_binding", at)
        acceptance = self.document(receipt["acceptance"], at)
        keys(acceptance, ("source_sha256", "section", "actual_sha256", "auditor_sha256", "verdict"), at)
        require(acceptance["source_sha256"] == receipt["source_sha256"] and acceptance["section"] == at
                and acceptance["actual_sha256"] == receipt["actual"]["sha256"]
                and acceptance["auditor_sha256"] == receipt["auditor"]["sha256"]
                and acceptance["verdict"] == "PASS", "independent_acceptance_binding", at)
        return actual["data"]

    def samples(self, data, at, solver=None):
        counts, samples = data["counts"], data["samples"]
        keys(counts, ("attempted", "accepted", "failed", "unsupported", "skipped"), at)
        for value in counts.values():
            integer(value, at)
        require(type(samples) is list and samples, "no_actual_solves", at)
        require(counts["attempted"] == counts["accepted"] == len(samples)
                and counts["failed"] == counts["unsupported"] == counts["skipped"] == 0,
                "incomplete_or_contradictory_counts", at)
        pairs = []
        for sample in samples:
            keys(sample, ("id", "model", "result", "check", "mode", "resource_sha256"), at)
            require(type(sample["id"]) is str and re.fullmatch(r"[a-z0-9_-]{1,80}", sample["id"])
                    and sample["mode"] in ("cold", "prepared"), "sample_identity", at)
            require(type(sample["resource_sha256"]) is str and HASH.fullmatch(sample["resource_sha256"]),
                    "invalid_sha256", at)
            self.document(sample["model"], at)
            result = self.document(sample["result"], at)
            keys(result, ("source_sha256", "model_sha256", "sample_id", "mode", "resource_sha256",
                          "solver", "execution", "status", "wall_s"), at)
            require(result["source_sha256"] == self.manifest["source_sha256"]
                    and result["model_sha256"] == sample["model"]["sha256"]
                    and result["sample_id"] == sample["id"] and result["mode"] == sample["mode"]
                    and result["resource_sha256"] == sample["resource_sha256"]
                    and (solver is None or result["solver"] == solver)
                    and result["execution"] == "ACTUAL_NATIVE_SOLVE" and result["status"] == "solved"
                    and type(result["wall_s"]) in (int, float) and result["wall_s"] > 0,
                    "actual_result_binding", at)
            check = self.document(sample["check"], at)
            keys(check, ("source_sha256", "model_sha256", "result_sha256", "verdict",
                         "original_coordinates", "signed_gap", "status_certificates"), at)
            require(check["source_sha256"] == self.manifest["source_sha256"]
                    and check["model_sha256"] == sample["model"]["sha256"]
                    and check["result_sha256"] == sample["result"]["sha256"]
                    and all(check[k] == "PASS" for k in
                            ("verdict", "original_coordinates", "signed_gap", "status_certificates")),
                    "original_check_binding", at)
            pairs.append((sample["id"], sample["model"]["sha256"], sample["mode"], sample["resource_sha256"]))
        require(len(set(pairs)) == len(pairs), "duplicate_samples", at)
        return pairs

    def capability(self, cap):
        at = "capability:" + cap
        data = self.receipt(self.manifest["capabilities"].get(cap), at)
        keys(data, ("capability", "status", "negative_controls", "counts", "samples"), at)
        require(data["capability"] == cap and data["status"] == "PASS", "capability_not_qualified", at)
        integer(data["negative_controls"], at, 1)
        self.samples(data, at, "openopt")

    def proof(self, value, at):
        keys(value, ("status", "pin"), at)
        require(value["status"] == "AUTHORIZED" and type(value["pin"]) is str
                and HASH.fullmatch(value["pin"]), "authorization_not_established", at)

    def commercial(self, solver):
        at = "commercial:" + solver
        data = self.receipt(self.manifest["commercial"].get(solver), at)
        keys(data, ("authorization", "actual_native_calls", "completed", "counts", "samples", "openopt"), at)
        self.proof(data["authorization"], at)
        require(type(data["samples"]) is list, "sample_schema", at)
        require(integer(data["actual_native_calls"], at, 1) == integer(data["completed"], at, 1)
                == len(data["samples"]), "no_actual_commercial_solves", at)
        pairs = self.samples(data, at, solver)
        keys(data["openopt"], ("counts", "samples"), at)
        require(pairs == self.samples(data["openopt"], at, "openopt")
                and {p[2] for p in pairs} == {"cold", "prepared"}, "unpaired_commercial_models", at)

    def gpu(self, count):
        at = f"gpu:{count}"
        data = self.receipt(self.manifest["gpu_scaling"].get(str(count)), at)
        keys(data, ("gpu_count", "devices", "timing_scope", "counts", "samples", "trace", "coordinator"), at)
        require(integer(data["gpu_count"], at, 1) == count
                and data["timing_scope"] == "transfer_inclusive_end_to_end"
                and type(data["devices"]) is list and len(data["devices"]) == count, "gpu_count_or_timing", at)
        ids = []
        for device in data["devices"]:
            keys(device, ("id", "model"), at)
            require(type(device["id"]) is str and re.fullmatch(r"GPU-[A-Za-z0-9_-]{1,80}", device["id"])
                    and device["model"] == "RTX PRO 4500", "gpu_device_evidence", at)
            ids.append(device["id"])
        require(len(set(ids)) == count, "duplicate_gpu_devices", at)
        pairs = self.samples(data, at, "openopt")
        require({p[2] for p in pairs} == {"cold", "prepared"}, "gpu_timing_modes", at)
        results = [s["result"]["sha256"] for s in data["samples"]]
        trace = self.document(data["trace"], at)
        coordinator = self.document(data["coordinator"], at)
        require(trace == {"gpu_count": count, "result_sha256": results,
                          "kernel_events": trace.get("kernel_events")}
                and type(trace["kernel_events"]) is dict and set(trace["kernel_events"]) == set(ids),
                "gpu_trace_binding", at)
        for events in trace["kernel_events"].values():
            integer(events, at, 1)
        require(coordinator == {"source_sha256": self.manifest["source_sha256"], "gpu_count": count,
                                "result_sha256": results, "execution_verdict": "PASS"},
                "gpu_coordinator_binding", at)
        self.gpu_pairs[count] = pairs

    def publication(self):
        at = "publication"
        data = self.receipt(self.manifest[at], at)
        keys(data, ("status", "source_commit", "published", "readback", "remote_source", "transport"), at)
        require(data["status"] == "PUBLISHED" and data["source_commit"] == self.manifest["source_commit"],
                "not_published_current_source", at)
        published = self.blob(data["published"], at)
        require(self.blob(data["readback"], at) == published, "online_readback_bytes", at)
        remote = self.document(data["remote_source"], at)
        require(remote == {"commit": self.manifest["source_commit"], "source_sha256": self.manifest["source_sha256"]},
                "remote_source_mismatch", at)
        transport = self.document(data["transport"], at)
        require(transport == {"published_sha256": sha(published), "readback_sha256": sha(published),
                              "remote_source_sha256": data["remote_source"]["sha256"], "http_status": 200,
                              "status": "PUBLISHED", "method": "AUTHORIZED_MANAGED_READBACK"},
                "publication_transport_binding", at)

    def licensing(self):
        at = "licensing"
        data = self.receipt(self.manifest[at], at)
        keys(data, ("project_license", "license", "origins", "sbom", "inventory", "dependency_scope"), at)
        require(data["project_license"] == "0BSD" and data["dependency_scope"] == "runtime_build_optional_sdk",
                "license_scope_missing", at)
        license_bytes = self.blob(data["license"], at)
        require(sha(license_bytes) == self.manifest["source_sha256"].get("LICENSE")
                and b"BSD Zero Clause" in license_bytes, "license_source_bytes", at)
        require(type(data["origins"]) is list and data["origins"], "origin_rights_missing", at)
        paths = []
        for origin in data["origins"]:
            keys(origin, ("paths", "status", "proof"), at)
            require(type(origin["paths"]) is list and origin["paths"]
                    and all(type(p) is str for p in origin["paths"])
                    and origin["status"] == "REDISTRIBUTION_AUTHORIZED", "origin_rights_unverified", at)
            self.proof(origin["proof"], at)
            paths.extend(origin["paths"])
        require(len(set(paths)) == len(paths) and set(paths) == set(self.manifest["source_sha256"]),
                "origin_source_membership", at)
        sbom, inventory = self.document(data["sbom"], at), self.document(data["inventory"], at)
        keys(sbom, ("source_sha256", "scope", "dependencies"), at)
        keys(inventory, ("source_sha256", "scope", "dependencies"), at)
        require(sbom["source_sha256"] == inventory["source_sha256"] == self.manifest["source_sha256"]
                and sbom["scope"] == inventory["scope"] == data["dependency_scope"]
                and type(sbom["dependencies"]) is list and sbom["dependencies"], "sbom_scope_missing", at)
        members = []
        for dep in sbom["dependencies"]:
            keys(dep, ("name", "version", "license", "distribution", "proof"), at)
            require(all(type(dep[k]) is str and re.fullmatch(r"[A-Za-z0-9.+_-]{1,100}", dep[k])
                        for k in ("name", "version", "license"))
                    and dep["license"].upper() not in ("UNKNOWN", "UNVERIFIED")
                    and dep["distribution"] in ("BUNDLED", "NOT_BUNDLED"), "dependency_rights_unknown", at)
            self.proof(dep["proof"], at)
            members.append({"name": dep["name"], "version": dep["version"]})
        require(inventory["dependencies"] == members and len({m["name"] for m in members}) == len(members),
                "sbom_inventory_mismatch", at)

    def historical(self):
        rows = self.manifest["historical_observations"]
        require(type(rows) is list, "schema_keys", "historical")
        for row in rows:
            keys(row, ("kind", "receipt"), "historical")
            actual = parse(self.blob(row["receipt"], "historical"), "historical", historical=True)
            require(row["kind"] == "commercial_runner_discovery_v1" and type(actual) is dict
                    and type(actual.get("schema_version")) is int and actual["schema_version"] == 1
                    and actual.get("qualification") == "UNVERIFIED" and type(actual.get("solvers")) is dict
                    and set(actual["solvers"]) == set(SOLVERS), "unsupported_historical_receipt", "historical")
            for solver in SOLVERS:
                modes = actual["solvers"][solver].get("modes")
                require(type(modes) is dict and set(modes) == {"cold", "prepared"}
                        and all(type(v) is dict and type(v.get("samples")) is list and not v["samples"]
                                for v in modes.values()), "historical_zero_not_established", "historical")
            counts = actual.get("counts", {})
            require(type(counts) is dict and all(type(counts.get(k)) is int and counts[k] == 0
                    for k in ("attempted", "solve_attempted", "passed", "failed")),
                    "historical_zero_not_established", "historical")
            self.observations.append({"kind": row["kind"], "actual_commercial_solves": 0,
                                      "receipt_sha256": row["receipt"]["sha256"],
                                      "collected_at": actual.get("finished_at")})
        require(not rows, "pinned_historical_zero_commercial_solves", "historical")

    def evaluate(self):
        self.gpu_pairs = {}

        def source():
            commit, sources = current_sources(self.root)
            require(commit == self.manifest["source_commit"] and sources == self.manifest["source_sha256"],
                    "current_source_mismatch", "source")

        self.attempt("source", source)
        self.attempt("historical", self.historical)
        for cap in CAPABILITIES:
            self.attempt("capability:" + cap, lambda cap=cap: self.capability(cap))
        for solver in SOLVERS:
            self.attempt("commercial:" + solver, lambda solver=solver: self.commercial(solver))
        for count in (1, 2, 3):
            self.attempt(f"gpu:{count}", lambda count=count: self.gpu(count))
        self.attempt("gpu_pairing", lambda: require(set(self.gpu_pairs) == {1, 2, 3}
                     and self.gpu_pairs[1] == self.gpu_pairs[2] == self.gpu_pairs[3],
                     "missing_or_different_scaling_problem", "gpu_pairing"))
        self.attempt("publication", self.publication)
        self.attempt("licensing", self.licensing)
        self.attempt("source_recheck", source)
        for path, payload in list(self.inputs.items()):
            self.attempt("input_recheck", lambda path=path, payload=payload: require(
                read(path, "input_recheck") == payload, "input_changed", "input_recheck"))
        return {"schema_version": 1, "scope": SCOPE,
                "verdict": "INCOMPLETE" if self.incomplete else "REVIEW_REQUIRED",
                "release_complete": False, "execution_attestation": "UNVERIFIED",
                "legal_authority": "UNVERIFIED", "general_capability": "UNVERIFIED",
                "qualification": "insufficiency precheck only; no automatic acceptance",
                "manifest_sha256": self.pin, "source_commit": self.manifest["source_commit"],
                "source_sha256": self.manifest["source_sha256"], "incomplete": self.incomplete,
                "checked": self.checked, "historical_observations": self.observations,
                "input_files": len(self.inputs), "required_capabilities": list(CAPABILITIES)}


def precheck(manifest, manifest_sha256, *, current_root=ROOT):
    return Precheck(manifest, manifest_sha256, current_root).evaluate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--current-root", type=Path, default=ROOT)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = absolute(args.current_root, "source")
    output_root = absolute(args.artifact_dir, "output")
    require(not output_root.is_relative_to(root) and not output_root.is_relative_to(ROOT),
            "output_must_be_external", "output")
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="completion-", dir=output_root))
    started = datetime.now(timezone.utc).isoformat()
    try:
        report = precheck(args.manifest, args.manifest_sha256, current_root=root)
    except Rejected as error:
        report = {"verdict": "INCOMPLETE", "release_complete": False,
                  "execution_attestation": "UNVERIFIED", "incomplete": [
                      {"section": "input", "code": error.code, "coordinate": error.at}]}
    except Exception as error:
        report = {"verdict": "INCOMPLETE", "release_complete": False,
                  "execution_attestation": "UNVERIFIED", "incomplete": [
                      {"section": "internal", "code": type(error).__name__, "coordinate": "gate"}]}
    rc = 1 if report["verdict"] == "INCOMPLETE" else 2
    report.update(started_at=started, finished_at=datetime.now(timezone.utc).isoformat(), returncode=rc,
                  gate_source_sha256=sha(read(Path(__file__), "gate_source")))
    with (output / "completion.json").open("xb") as stream:
        stream.write(encoded(report))
    verdict, count, receipt = report["verdict"], len(report["incomplete"]), output / "completion.json"
    line = (f"RELEASE_COMPLETION verdict={verdict} release_complete=false "
            f"execution=UNVERIFIED incomplete={count} receipt={receipt}")
    with (output / "execution.log").open("x") as stream:
        stream.write(line + f"\nreturncode={rc}\n")
    print(line, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
