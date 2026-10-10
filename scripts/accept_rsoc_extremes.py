"""Replay the separate 76-case RSOC CUDA regression without running a solver.

Caller pins bind collected artifacts. Original-coordinate Decimal/Fraction
checks do not import a producer, test factory, Torch or a production solver.
This bounded retrospective audit is not remote attestation or qualification
of commercial performance, arbitrary models or a complete optimizer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from pathlib import Path
import re
import struct
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts import accept_release_cuda as execution

TEST = "tests/test_rotated_conic_cuda_extremes.py"
ORACLE = "tests/test_rotated_conic_independent.py"
FUNCTION = "test_cuda_rotated_extreme_coordinates"
SOURCE_PATHS = ("src/openopt/conic.py", "src/openopt/native_conic.py", ORACLE, TEST)
SIDECAR_FIELDS = {
    "schema_version", "case", "nodeid", "verdict", "qualification",
    "requested_device", "observed_device", "dtype", "gpu_name", "input_hex",
    "positive_hex", "negative_hex", "expected_decimal", "negative_expected_decimal",
    "exact_excess", "dual", "source_sha256", "trace", "original_coordinate_check",
    "scaled_moreau_check",
}


def require(value, coordinate, cause):
    execution.require(value, coordinate, cause)


def f32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def cases():
    """Independent, versioned input domain; never inferred from supplied tests."""
    result = {}
    for label, power in (("f64", 600), ("f32", 100)):
        head = 2.0 ** power
        middle, tiny = (2.0 ** 200, 2.0 ** -450) if label == "f64" else (2.0 ** 30, 2.0 ** -60)
        for name, tail in (
            ("unit-last", (head, head, 1.)), ("unit-first", (1., head, head)),
            ("unit-middle", (head, 1., head)), ("signed", (-head, head, -1.)),
            ("middle-last", (head, head, middle)), ("middle-first", (middle, head, head)),
            ("two-levels-last", (head, head, 1., tiny)),
            ("two-levels-first", (tiny, 1., head, head)),
        ):
            result[f"{label}-remote-{name}"] = (-head, -head, *tail)
        root = math.sqrt(2.)
        previous = math.nextafter(head, 0.)
        if label == "f32":
            root_bits = struct.unpack("!I", struct.pack("!f", root))[0]
            root = struct.unpack("!f", struct.pack("!I", root_bits + 1))[0]
            head_bits = struct.unpack("!I", struct.pack("!f", head))[0]
            previous = struct.unpack("!f", struct.pack("!I", head_bits - 1))[0]
        polar = 2.0 ** (500 if label == "f64" else 100)
        extras = {
            "polar-sqrt2": (-1., -1., root),
            "polar-dyadic": (-polar, -polar, polar, polar, 1.),
            "negative-head": (-1e20, 0., 1e8),
            "small-head": (-1e300, 0., 1e210) if label == "f64" else (-1e30, 0., 1e17),
            "overflowing-norm": (0., 0., 1.3e308, 1.3e308) if label == "f64" else (0., 0., 3e38, 3e38),
            "positive-unit-last": (head, head, head, head, 1.),
            "positive-unit-first": (head, head, 1., head, head),
            "positive-unit-middle": (head, head, head, 1., head),
            "positive-boundary": (head, head, head, head, 0.),
            "positive-inside-last": (head, head, head, previous, 1.),
            "positive-inside-first": (head, head, 1., previous, head),
        }
        result.update({f"{label}-{name}": value for name, value in extras.items()})
    return {
        name: tuple(f32(x) if name.startswith("f32-") else x for x in value)
        for name, value in result.items()
    }


def expected_identities():
    return {
        f"tests.{Path(TEST).stem}::{FUNCTION}[{case}-{orientation}]": (case, orientation == "dual")
        for case in cases() for orientation in ("primal", "dual")
    }


def projection(value):
    """Euclidean RSOC projection directly in original coordinates."""
    items = [Decimal.from_float(x) for x in value]
    a, b, *tail = items
    squared = sum(x * x for x in tail)
    if a >= 0 and b >= 0 and squared <= 2 * a * b:
        return items
    if a <= 0 and b <= 0 and squared <= 2 * a * b:
        return [Decimal(0)] * len(items)
    radius = ((a - b) ** 2 + 2 * squared).sqrt()
    head = a + b
    weight = ((squared - 2 * a * b) / (radius * (radius - head))
              if head < 0 else (radius + head) / (2 * radius))
    large = weight * (radius + abs(a - b)) / 2
    small = weight * weight * squared / (2 * large)
    return [*([large, small] if a >= b else [small, large]),
            *[weight * x for x in tail]]


def coordinates(observed, reference, dtype):
    tolerance = Decimal("2e-12" if dtype == "torch.float64" else "3e-5")
    for index, (value, expected) in enumerate(zip(observed, reference)):
        represented = float(expected)
        if dtype == "torch.float32":
            represented = f32(represented)
        actual = Decimal.from_float(value)
        if expected == 0 or represented == 0:
            require(actual == 0, f"coordinate/{index}", "unrepresentable or exact zero changed")
        else:
            require(abs(actual - expected) <= tolerance * abs(expected),
                    f"coordinate/{index}", "original coordinate error or lost representable head")


def decode(values, size, dtype):
    require(isinstance(values, list) and len(values) == size,
            "coordinates", "complete original vector required")
    result = []
    for text in values:
        require(isinstance(text, str), "coordinates", "hexadecimal represented input required")
        value = float.fromhex(text)
        require(math.isfinite(value) and value.hex() == text, "coordinates", "noncanonical/nonfinite value")
        require(dtype != "torch.float32" or f32(value) == value,
                "coordinates", "value is not represented binary32")
        result.append(value)
    return result


def operation_gate(trace, value, dtype):
    require(isinstance(trace, dict) and set(trace) == {"device", "tensor_count", "dtypes", "operations"}
            and trace["device"] == "cuda:0" and type(trace["tensor_count"]) is int
            and trace["tensor_count"] > 0, "trace", "missing actual CUDA tensor trace")
    require(isinstance(trace["dtypes"], list) and len(set(trace["dtypes"])) == len(trace["dtypes"])
            and set(trace["dtypes"]) <= {"torch.float32", "torch.float64", "torch.int32", "torch.bool"}
            and dtype in trace["dtypes"] and "torch.float64" in trace["dtypes"],
            "trace", "device-local numerical intermediates absent or unknown")
    operations = trace["operations"]
    require(isinstance(operations, dict) and operations
            and all(isinstance(name, str) and name.startswith("aten.")
                    and type(count) is int and count > 0 for name, count in operations.items()),
            "trace", "empty or invalid observed operation counts")
    a, b, *tail = map(Fraction.from_float, value)
    excess = sum(x * x for x in tail) - 2 * a * b
    early = (a >= 0 and b >= 0 or a <= 0 and b <= 0) and excess <= 0
    norms = sum(count for name, count in operations.items() if name.startswith("aten.linalg_vector_norm."))
    require(norms == (1 if early else 2), "trace", "wrong original-sign early return")
    require(early or any(name.startswith("aten.frexp.") for name in operations),
            "trace", "device-local exponent representation not observed")


def replay(saved, identity, sources, gpu):
    require(isinstance(saved, dict) and set(saved) == SIDECAR_FIELDS
            and type(saved["schema_version"]) is int and saved["schema_version"] == 1,
            identity, "exact versioned sidecar schema required")
    case, dual = expected_identities()[identity]
    dtype = "torch.float64" if case.startswith("f64-") else "torch.float32"
    require(saved["case"] == case and saved["dual"] is dual
            and saved["nodeid"] == f"{TEST}::{identity.split('::', 1)[1]}",
            identity, "case/orientation identity substituted")
    require(saved["dtype"] == dtype and saved["verdict"] == "CUDA_NUMERICAL_REGRESSION_PASS"
            and saved["requested_device"] == saved["observed_device"] == "cuda:0"
            and saved["gpu_name"] == gpu, identity, "actual CUDA execution contract mismatch")
    require(saved["qualification"] == "projection regression only; no commercial, performance or remote attestation",
            identity, "unsupported qualification claim")
    require(saved["source_sha256"] == {name: sources[name] for name in SOURCE_PATHS},
            identity, "exact numerical source membership/hash mismatch")
    require(saved["original_coordinate_check"] is True and saved["scaled_moreau_check"] is True,
            identity, "original mathematical checks absent")
    original = list(cases()[case])
    size = len(original)
    supplied = decode(saved["input_hex"], size, dtype)
    require(saved["input_hex"] == [x.hex() for x in original] and supplied == original,
            identity, "fixed represented original model changed")
    positive, negative = (decode(saved[name], size, dtype) for name in ("positive_hex", "negative_hex"))
    a, b, *tail = map(Fraction.from_float, original)
    excess = sum(x * x for x in tail) - 2 * a * b
    require(saved["exact_excess"] == {"numerator": str(excess.numerator), "denominator": str(excess.denominator)},
            identity, "exact represented original inequality changed")
    with localcontext() as context:
        context.prec = 1100
        reference = projection(original)
        opposite = projection([-x for x in original])
        require(saved["expected_decimal"] == list(map(str, reference))
                and saved["negative_expected_decimal"] == list(map(str, opposite)),
                identity, "independent original-coordinate reference mismatch")
        coordinates(positive, reference, dtype)
        coordinates(negative, opposite, dtype)
        x, p, n = [[Decimal.from_float(v) for v in vector]
                   for vector in (original, positive, negative)]
        scale = max(abs(v) for v in x)
        tolerance = Decimal("3e-12" if dtype == "torch.float64" else "4e-5")
        require(max(abs(a - b + c) / scale for a, b, c in zip(x, p, n)) <= tolerance,
                identity, "Moreau reconstruction mismatch")
        products = [a / scale * (b / scale) for a, b in zip(p, n)]
        require(abs(sum(products)) <= tolerance * max(Decimal(1), sum(abs(v) for v in products)),
                identity, "Moreau orthogonality mismatch")
        for vector in (p, n):
            squared = sum((v / scale) ** 2 for v in vector[2:])
            product = 2 * (vector[0] / scale) * (vector[1] / scale)
            require(vector[0] >= 0 and vector[1] >= 0
                    and squared - product <= tolerance * max(squared, product, Decimal("1e-1000")),
                    identity, "original primal/dual cone membership mismatch")
    if a >= 0 and b >= 0 and excess <= 0:
        require(saved["positive_hex"] == saved["input_hex"], identity, "interior bits changed")
    require(isinstance(saved["trace"], dict) and set(saved["trace"]) == {"positive", "negative"},
            identity, "complete positive/negative operation trace required")
    operation_gate(saved["trace"]["positive"], original, dtype)
    operation_gate(saved["trace"]["negative"], [-x for x in original], dtype)
    return {"case": identity, "verdict": "ORIGINAL_COORDINATES_REPLAYED"}


def environment(entries):
    require(isinstance(entries, list), "environment", "explicit execution environment required")
    result = {}
    for entry in entries:
        require(isinstance(entry, dict) and isinstance(entry.get("name"), str)
                and entry["name"] not in result, "environment", "missing or duplicate name")
        if set(entry) == {"name", "value"}:
            require(isinstance(entry["value"], str), "environment", "literal value required")
            result[entry["name"]] = entry["value"]
        else:
            require(set(entry) == {"name", "valueFrom"}
                    and set(entry["valueFrom"]) == {"fieldRef"}, "environment", "unsupported field source")
            field = entry["valueFrom"]["fieldRef"]
            require(set(field) <= {"fieldPath", "apiVersion"} and "fieldPath" in field
                    and field.get("apiVersion", "v1") == "v1",
                    "environment", "unknown field reference")
            result[entry["name"]] = {"fieldPath": field["fieldPath"]}
    return result


def named_records(values, coordinate):
    require(isinstance(values, list), coordinate, "explicit record list required")
    result = {}
    for value in values:
        require(isinstance(value, dict) and isinstance(value.get("name"), str)
                and value["name"] and value["name"] not in result,
                coordinate, "missing/duplicate record name")
        result[value["name"]] = value
    return result


def storage_gate(spec, container, expected_spec, expected_container, *, observed_pod):
    """Only admit the standard read-only service-account projection as extra."""
    expected_volumes = named_records(expected_spec.get("volumes"), "submitted volumes")
    expected_mounts = named_records(expected_container.get("volumeMounts"), "submitted mounts")
    volumes = named_records(spec.get("volumes"), "actual volumes")
    mounts = named_records(container.get("volumeMounts"), "actual mounts")
    require(all(volumes.get(name) == value for name, value in expected_volumes.items())
            and all(mounts.get(name) == value for name, value in expected_mounts.items()),
            "storage", "submitted source/scratch volume or mount drift")
    extra_volumes, extra_mounts = set(volumes) - set(expected_volumes), set(mounts) - set(expected_mounts)
    if not extra_volumes and not extra_mounts:
        return
    require(observed_pod and extra_volumes == extra_mounts and len(extra_volumes) == 1,
            "storage", "undeclared execution volume/mount")
    name = next(iter(extra_volumes))
    require(re.fullmatch(r"kube-api-access-[a-z0-9]+", name) is not None,
            "storage", "unknown injected volume")
    require(mounts[name] == {
        "name": name, "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount", "readOnly": True,
    }, "storage", "service-account mount is not the declared read-only default")
    projected = volumes[name].get("projected")
    require(set(volumes[name]) == {"name", "projected"}
            and isinstance(projected, dict) and set(projected) == {"defaultMode", "sources"}
            and type(projected["defaultMode"]) is int and projected["defaultMode"] == 420,
            "storage", "unknown service-account projection")
    sources = projected["sources"]
    require(isinstance(sources, list) and len(sources) == 3,
            "storage", "complete service-account projection required")
    token = sources[0].get("serviceAccountToken", {})
    require(set(sources[0]) == {"serviceAccountToken"}
            and set(token) == {"expirationSeconds", "path"} and token["path"] == "token"
            and type(token["expirationSeconds"]) is int and 600 <= token["expirationSeconds"] <= 86400,
            "storage", "unknown token projection configuration")
    require(sources[1] == {"configMap": {
        "items": [{"key": "ca.crt", "path": "ca.crt"}], "name": "kube-root-ca.crt",
    }} and sources[2] == {"downwardAPI": {"items": [{
        "fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.namespace"}, "path": "namespace",
    }]}}, "storage", "unknown service-account projection source")


def status_gate(status, manifest, run, report, log):
    job, pods = status["job"], status["pods"]["items"]
    item = run["jobs"][0]
    require(job["metadata"]["name"] == item["job"] and job["metadata"]["namespace"] == run["namespace"]
            and len(pods) == 1, "status", "exact one submitted Job/Pod required")
    pod = pods[0]
    require(job.get("kind") == "Job" and job.get("apiVersion") == "batch/v1"
            and pod.get("kind") == "Pod" and pod.get("apiVersion") == "v1"
            and pod["metadata"].get("namespace") == run["namespace"],
            "status", "actual Job/Pod kind, version or namespace mismatch")
    require(job["metadata"]["labels"].get("repo") == pod["metadata"]["labels"].get("repo") == "openopt"
            and job["metadata"]["labels"].get("owner") == pod["metadata"]["labels"].get("owner") == "infra-openopt",
            "status", "project ownership mismatch")
    owners = [owner for owner in pod["metadata"].get("ownerReferences", []) if owner.get("controller") is True]
    require(len(owners) == 1 and owners[0].get("uid") == job["metadata"]["uid"]
            and owners[0].get("name") == item["job"] and owners[0].get("kind") == "Job"
            and owners[0].get("apiVersion") == "batch/v1", "status", "Pod actual controller identity mismatch")
    require(type(job.get("status", {}).get("succeeded")) is int and job["status"]["succeeded"] == 1
            and not job["status"].get("failed")
            and any(c.get("type") == "Complete" and c.get("status") == "True" for c in job["status"]["conditions"])
            and not any(c.get("type") == "Failed" and c.get("status") == "True" for c in job["status"]["conditions"]),
            "status", "successful Job terminal condition required")
    expected_pod = manifest["spec"]["template"]["spec"]
    expected = expected_pod["containers"][0]
    for obj in (manifest, job):
        require(type(obj["spec"]["backoffLimit"]) is int and obj["spec"]["backoffLimit"] == 0,
                "status", "non-retrying Job required")
    for index, spec in enumerate((expected_pod, job["spec"]["template"]["spec"], pod["spec"])):
        require(spec.get("restartPolicy") == "Never" and len(spec["containers"]) == 1
                and not spec.get("initContainers") and not spec.get("ephemeralContainers"),
                "status", "unexpected restart/multiple-container execution")
        actual = spec["containers"][0]
        require(all(spec.get(name) == expected_pod.get(name) for name in (
            "nodeSelector", "runtimeClassName", "serviceAccountName", "priorityClassName", "securityContext",
        )), "status", "submitted placement, runtime, execution identity or security drift")
        require(actual.get("workingDir", "") == expected.get("workingDir", "")
                and actual.get("securityContext") == expected.get("securityContext")
                and actual.get("imagePullPolicy") == expected.get("imagePullPolicy"),
                "status", "submitted working directory or container security/image policy drift")
        storage_gate(spec, actual, expected_pod, expected, observed_pod=index == 2)
        require(actual.get("name") == expected["name"] and actual.get("command") == expected["command"]
                and not actual.get("args") and not actual.get("envFrom")
                and actual.get("image") == expected["image"]
                and environment(actual.get("env")) == environment(expected.get("env")),
                "status", "submitted/observed command, args, image or complete env drift/injection")
        for category in ("requests", "limits"):
            resources = actual["resources"][category]
            require(set(resources) == set(expected["resources"][category])
                    and all(str(resources[k]) == str(v) for k, v in expected["resources"][category].items())
                    and str(resources.get("nvidia.com/gpu")) == "1",
                    "status", "submitted resource contract drift")
    states = pod["status"].get("containerStatuses", [])
    require(pod["status"]["phase"] == "Succeeded" and len(states) == 1
            and states[0]["name"] == expected["name"] and type(states[0]["restartCount"]) is int
            and states[0]["restartCount"] == 0
            and type(states[0].get("state", {}).get("terminated", {}).get("exitCode")) is int
            and states[0]["state"]["terminated"]["exitCode"] == 0
            and isinstance(states[0].get("imageID"), str)
            and re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", states[0]["imageID"]) is not None,
            "status", "successful non-restarted resolved container required")
    require(report["hardware"]["node"] == pod["spec"]["nodeName"], "status", "observed node mismatch")
    markers = [line for line in log.splitlines() if line.startswith("OPENOPT_RELEASE_CUDA verdict=")]
    require(markers == ["OPENOPT_RELEASE_CUDA verdict=PASS tests=76 evidence="
                        + str(Path(item["output"]) / Path(report["_result_path"]).parent.name)],
            "status", "actual collected completion marker mismatch")
    return {"job": item["job"], "job_uid": job["metadata"]["uid"],
            "pod": pod["metadata"]["name"], "pod_uid": pod["metadata"]["uid"],
            "image_id": states[0]["imageID"], "node": pod["spec"]["nodeName"], "restart_count": 0}


def pinned(path, pin):
    descriptor = execution.descriptor(path)
    require(descriptor["sha256"] == execution.digest(pin), path, "independent caller pin mismatch")
    return descriptor


def accept(*, result, result_sha256, run, run_sha256, junit, junit_sha256,
           status, status_sha256, current_root):
    inputs = [pinned(path, pin) for path, pin in (
        (result, result_sha256), (run, run_sha256), (junit, junit_sha256), (status, status_sha256),
    )]
    run = execution.external(run)
    result = execution.external(result, run.parent)
    junit = execution.external(junit, run.parent)
    status = execution.external(status, run.parent)
    require(junit == result.parent / "pytest.xml"
            and result.parent.parent == run.parent / "case-0"
            and result.parent.name.startswith("cuda-check-"), "paths", "immutable case output required")
    record = execution.parse(execution.read(run), run)
    require(record.get("stage") == "release-regression" and record.get("state") == "SUBMITTED"
            and len(record.get("jobs", [])) == 1, "run", "one-card submitted regression required")
    require(record["source_identity"]["clean"] is True
            and record["source_identity"]["commit"] == record["source_sha"],
            "source", "clean committed execution source required")
    sources = record["source_hashes"]
    snapshot = run.parent / "source"
    source_manifest = execution.descriptor(snapshot / "source-hashes.json")
    inputs.append(source_manifest)
    execution.equal_sources(sources, execution.parse(execution.bound(source_manifest), "sources"),
                            "run source manifest")
    execution.equal_sources(sources, execution.source_files(snapshot), "frozen source")
    require(all(record["source_identity"]["sha256"].get(name) == pin for name, pin in sources.items()),
            "source", "frozen identity does not cover actual snapshot")
    current_core = {
        name: execution.sha(path.read_bytes())
        for path in sorted((Path(current_root) / "src/openopt").rglob("*.py"))
        for name in (path.relative_to(Path(current_root)).as_posix(),)
        if not path.is_symlink()
    }
    require(not any(path.is_symlink() for path in (Path(current_root) / "src/openopt").rglob("*.py")),
            "current core", "symlink Python implementation rejected")
    execution.equal_sources(
        {name: pin for name, pin in sources.items() if name.startswith("src/openopt/")},
        current_core, "current numerical core",
    )
    for name in SOURCE_PATHS:
        require(sources[name] == execution.sha(execution.read(Path(current_root) / name)),
                name, "trusted current numerical/test source differs")
    report = execution.parse(execution.read(result), result)
    require(report.get("verdict") == "PASS" and type(report.get("returncode")) is int
            and report["returncode"] == 0, "result", "successful completed worker required")
    require(report["targets"] == [f"{TEST}::{FUNCTION}"] and report["selection"] == ""
            and report["tests"] == {"tests": 76, "failures": 0, "errors": 0, "skipped": 0},
            "selection", "separate exact 76-case CUDA contract required")
    expected_sources = {
        name: pin for name, pin in sources.items()
        if name.startswith("src/openopt/") and "/" not in name[len("src/openopt/"):]
    }
    expected_sources.update({name: sources[name] for name in (TEST, execution.RUNNER)})
    execution.equal_sources(expected_sources, report["source_sha256"], "executed source")
    mirror = result.parent.parent / "result.json"
    inputs.append(execution.descriptor(mirror))
    require(execution.read(mirror) == execution.read(result), "result", "coordinator mirror differs")
    counts, members = execution.junit(execution.read(junit), report, sources)
    require({member["id"] for member in members} == set(expected_identities()),
            "selection", "complete exact input/orientation denominator required")
    hardware = report["hardware"]
    require(all(isinstance(hardware.get(name), str) and hardware[name]
                for name in ("gpu", "node", "torch", "cuda", "cpu_quota"))
            and "RTX PRO 4500" in hardware["gpu"], "hardware", "complete actual GPU runtime required")
    for name in ("pytest.log", "execution.log", "telemetry.json"):
        inputs.append(execution.descriptor(result.parent / name))
    pytest_log = execution.read(result.parent / "pytest.log").decode("utf-8")
    require("76 passed" in pytest_log and pytest_log.rstrip().endswith("returncode=0"),
            "pytest.log", "actual completion missing")
    expected_marker = f"OPENOPT_RELEASE_CUDA verdict=PASS tests=76 evidence={result.parent}"
    require(execution.read(result.parent / "execution.log").decode("utf-8").strip() == expected_marker,
            "execution.log", "exact worker completion mismatch")
    telemetry = execution.parse(execution.read(result.parent / "telemetry.json"), "telemetry")
    require(isinstance(telemetry, list) and telemetry
            and type(report.get("telemetry_samples")) is int
            and len(telemetry) == report["telemetry_samples"]
            and all(type(sample.get("rc")) is int and sample["rc"] == 0
                    and isinstance(sample.get("gpu"), str) and sample["gpu"].startswith("GPU-")
                    and type(sample.get("cpu", {}).get("usage_usec")) is int for sample in telemetry),
            "telemetry", "actual GPU/cgroup samples incomplete")
    item = record["jobs"][0]
    require(execution.external(item["output"], run.parent) == result.parent.parent
            and execution.external(item["manifest"], run.parent) == run.parent / "job-0.json",
            "run", "exact isolated output/manifest required")
    require(status == run.parent / f"{item['job']}-status.json",
            "status", "collected status substituted")
    manifest_descriptor = execution.descriptor(item["manifest"])
    inputs.append(manifest_descriptor)
    manifest = execution.parse(execution.bound(manifest_descriptor), "manifest")
    execution.check_job(manifest, record, snapshot, report, result.parent.parent)
    expected_env = {
        "PYTHONPATH": str(snapshot / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "OPENOPT_NODE": {"fieldPath": "spec.nodeName"},
        "TMPDIR": str(result.parent.parent),
    }
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    actual_env = environment(container["env"])
    threads = container["resources"]["limits"]["cpu"]
    expected_env.update({name: str(threads) for name in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    require(actual_env == expected_env and not container.get("envFrom")
            and not container.get("args"), "manifest", "unsubmitted execution environment/arguments")
    report["_result_path"] = str(Path(result).resolve())
    status_value = execution.parse(execution.read(status), status)
    pod_name = status_value["pods"]["items"][0]["metadata"]["name"]
    log_descriptor = execution.descriptor(Path(run).parent / (pod_name + ".log"))
    inputs.append(log_descriptor)
    observed = status_gate(status_value, manifest,
                           record, report, execution.bound(log_descriptor).decode("utf-8"))
    paths, results = set(), []
    for member in members:
        properties = member["properties"]
        require(set(properties) == {"rsoc_cuda_extremes_path", "rsoc_cuda_extremes_sha256"},
                member["id"], "exact pinned sidecar properties required")
        path = execution.external(properties["rsoc_cuda_extremes_path"], Path(run).parent)
        require(str(path) not in paths and path.is_relative_to(result.parent / "pytest-tmp")
                and path.name == "rotated-conic-extreme.json",
                member["id"], "duplicated or substituted sidecar")
        paths.add(str(path))
        descriptor = pinned(path, properties["rsoc_cuda_extremes_sha256"])
        inputs.append(descriptor)
        saved = execution.parse(execution.bound(descriptor), path)
        results.append(replay(saved, member["id"], sources, report["hardware"]["gpu"]))
    require(all(execution.descriptor(item["path"]) == item for item in inputs),
            "inputs", "artifact changed during audit")
    execution.equal_sources(sources, execution.source_files(snapshot), "source after replay")
    return {"schema_version": 1, "verdict": "RSOC_EXTREMES_REPLAY_PASS",
            "checked_at": datetime.now(timezone.utc).isoformat(), "tests": counts,
            "source_commit": record["source_sha"], "execution": observed, "members": results,
            "inputs": inputs, "qualified": False, "commercial": "UNVERIFIED",
            "performance": "UNVERIFIED", "remote_attestation": "NOT_PROVIDED",
            "scope": "separate 76-case CUDA projection regression and original-coordinate replay"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "run", "junit", "status"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = execution.external(args.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="rsoc-audit-", dir=root))
    rc, report = 0, {}
    try:
        values = vars(args).copy()
        values.pop("artifact_dir")
        report = accept(**values)
    except Exception as error:
        rc = 1
        report = {"verdict": "FAIL", "error": f"{type(error).__name__}: {error}",
                  "checked_at": datetime.now(timezone.utc).isoformat()}
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
    execution.write(output / "acceptance.json", report)
    line = (f"RSOC_EXTREMES_ACCEPT verdict={report['verdict']} at={report['checked_at']} "
            f"report={output / 'acceptance.json'}")
    (output / "execution.log").write_text(line + f"\nreturncode={rc}\n", encoding="utf-8")
    print(line, flush=True)
    if rc:
        print(report["error"], flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
