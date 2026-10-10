"""Self-contained, test-only evidence mechanics, not release qualifications.

The optional historical freeze entry reads existing external receipts without
solving or modifying them. Default pytest never needs those historical files.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import check_release_completion as gate

HISTORICAL = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/commercial-readiness-review/runner-discovery/discovery-3bsk_wt3/result.json")
HISTORICAL_PIN = "c14f3a58f79a8cf9a827eb0ad8ff7f2fdda0f18d0cf1f4c465d0060443be71fe"


class Evidence:
    def __init__(self, root):
        self.root, self.index = root, 0
        root.mkdir(parents=True, exist_ok=True)
        self.sources = {"src/openopt/qp.py": gate.sha(b"test-only producer, not executable"),
                        "LICENSE": gate.sha(b"BSD Zero Clause License test-only")}
        self.commit = "a" * 40
        self.manifest = {
            "schema_version": 1, "scope": gate.SCOPE, "source_commit": self.commit,
            "source_sha256": self.sources, "capability_scope": list(gate.CAPABILITIES),
            "capabilities": {}, "commercial": {}, "gpu_scaling": {},
            "publication": None, "licensing": None, "historical_observations": [],
        }
        self.model = self.put({"test_only_original_model": {"P": [[1]], "q": [0], "A": [], "b": []}})

    def put(self, value):
        self.index += 1
        payload = value if type(value) is bytes else gate.encoded(value)
        path = self.root / f"fixture-{self.index}.json"
        path.write_bytes(payload)
        return {"path": str(path), "sha256": gate.sha(payload), "bytes": len(payload)}

    def load(self, desc):
        return json.loads(Path(desc["path"]).read_bytes())

    def counts(self, n):
        return {"attempted": n, "accepted": n, "failed": 0, "unsupported": 0, "skipped": 0}

    def sample(self, solver, mode):
        sample = {"id": mode, "model": self.model, "mode": mode,
                  "resource_sha256": gate.sha(b"test-only paired resource")}
        result = {"source_sha256": self.sources, "model_sha256": self.model["sha256"],
                  "sample_id": mode, "mode": mode, "resource_sha256": sample["resource_sha256"],
                  "solver": solver, "execution": "ACTUAL_NATIVE_SOLVE", "status": "solved", "wall_s": 1.25}
        sample["result"] = self.put(result)
        sample["check"] = self.put({"source_sha256": self.sources, "model_sha256": self.model["sha256"],
                                    "result_sha256": sample["result"]["sha256"], "verdict": "PASS",
                                    "original_coordinates": "PASS", "signed_gap": "PASS",
                                    "status_certificates": "PASS"})
        return sample

    def receipt(self, section, data):
        producer, auditor = self.put(b"test-only producer, not executable"), self.put(b"test-only independent auditor")
        actual = self.put({"source_sha256": self.sources, "section": section, "data": data})
        acceptance = self.put({"source_sha256": self.sources, "section": section,
                               "actual_sha256": actual["sha256"], "auditor_sha256": auditor["sha256"],
                               "verdict": "PASS"})
        return self.put({"schema_version": 1, "section": section, "source_sha256": self.sources,
                         "producer": producer, "auditor": auditor, "actual": actual, "acceptance": acceptance})

    def full_test_only(self):
        for cap in gate.CAPABILITIES:
            self.manifest["capabilities"][cap] = self.receipt("capability:" + cap, {
                "capability": cap, "status": "PASS", "negative_controls": 1,
                "counts": self.counts(1), "samples": [self.sample("openopt", "cold")]})
        for solver in gate.SOLVERS:
            self.manifest["commercial"][solver] = self.receipt("commercial:" + solver, {
                "authorization": {"status": "AUTHORIZED", "pin": gate.sha(b"test-only approval digest")},
                "actual_native_calls": 2, "completed": 2, "counts": self.counts(2),
                "samples": [self.sample(solver, mode) for mode in ("cold", "prepared")],
                "openopt": {"counts": self.counts(2), "samples": [
                    self.sample("openopt", mode) for mode in ("cold", "prepared")]}})
        for count in (1, 2, 3):
            samples = [self.sample("openopt", mode) for mode in ("cold", "prepared")]
            ids = [f"GPU-test-{i}" for i in range(count)]
            results = [s["result"]["sha256"] for s in samples]
            self.manifest["gpu_scaling"][str(count)] = self.receipt(f"gpu:{count}", {
                "gpu_count": count, "devices": [{"id": i, "model": "RTX PRO 4500"} for i in ids],
                "timing_scope": "transfer_inclusive_end_to_end", "counts": self.counts(2), "samples": samples,
                "trace": self.put({"gpu_count": count, "result_sha256": results,
                                   "kernel_events": dict.fromkeys(ids, 3)}),
                "coordinator": self.put({"source_sha256": self.sources, "gpu_count": count,
                                         "result_sha256": results, "execution_verdict": "PASS"})})
        payload = self.put(b"test-only published payload, NOT ONLINE")
        remote = self.put({"commit": self.commit, "source_sha256": self.sources})
        self.manifest["publication"] = self.receipt("publication", {
            "status": "PUBLISHED", "source_commit": self.commit, "published": payload,
            "readback": self.put(Path(payload["path"]).read_bytes()), "remote_source": remote,
            "transport": self.put({"published_sha256": payload["sha256"], "readback_sha256": payload["sha256"],
                                   "remote_source_sha256": remote["sha256"], "http_status": 200,
                                   "status": "PUBLISHED", "method": "AUTHORIZED_MANAGED_READBACK"})})
        self.manifest["licensing"] = self.receipt("licensing", {
            "project_license": "0BSD", "license": self.put(b"BSD Zero Clause License test-only"),
            "origins": [{"paths": list(self.sources), "status": "REDISTRIBUTION_AUTHORIZED",
                         "proof": {"status": "AUTHORIZED", "pin": gate.sha(b"test-only rights")}}],
            "dependency_scope": "runtime_build_optional_sdk",
            "sbom": self.put({"source_sha256": self.sources, "scope": "runtime_build_optional_sdk",
                              "dependencies": [{"name": "numpy", "version": "test-only", "license": "BSD-3-Clause",
                                                "distribution": "NOT_BUNDLED", "proof": {
                                                    "status": "AUTHORIZED", "pin": gate.sha(b"test-only dependency rights")}}]}),
            "inventory": self.put({"source_sha256": self.sources, "scope": "runtime_build_optional_sdk",
                                   "dependencies": [{"name": "numpy", "version": "test-only"}]})})
        return self

    def mutate(self, desc, mutation, *, refresh_acceptance=False):
        receipt = self.load(desc)
        actual = self.load(receipt["actual"])
        mutation(actual["data"])
        receipt["actual"] = self.put(actual)
        if refresh_acceptance:
            acceptance = self.load(receipt["acceptance"])
            acceptance["actual_sha256"] = receipt["actual"]["sha256"]
            receipt["acceptance"] = self.put(acceptance)
        return self.put(receipt)

    def run(self, *, current=None):
        manifest = self.put(self.manifest)
        return gate.precheck(manifest["path"], manifest["sha256"], current_root=current or self.root / "source")


@pytest.fixture(scope="module")
def immutable_evidence(tmp_path_factory):
    return Evidence(tmp_path_factory.mktemp("immutable-evidence")).full_test_only()


@pytest.fixture
def evidence(tmp_path, monkeypatch, immutable_evidence):
    fixture = Evidence(tmp_path / "fixtures")
    fixture.sources = copy.deepcopy(immutable_evidence.sources)
    fixture.manifest = copy.deepcopy(immutable_evidence.manifest)
    fixture.model = copy.deepcopy(immutable_evidence.model)
    monkeypatch.setattr(gate, "current_sources", lambda root: (fixture.commit, fixture.sources))
    return fixture


def codes(report):
    return {row["code"] for row in report["incomplete"]}


def test_full_synthetic_mechanics_never_release_pass(evidence):
    report = evidence.run()
    assert report["verdict"] == "REVIEW_REQUIRED"
    assert report["incomplete"] == [] and report["input_files"] > 200
    assert report["release_complete"] is False
    assert report["execution_attestation"] == report["legal_authority"] == report["general_capability"] == "UNVERIFIED"


def test_enumerated_scope_alone_is_incomplete(tmp_path, monkeypatch):
    fixture = Evidence(tmp_path)
    monkeypatch.setattr(gate, "current_sources", lambda root: (fixture.commit, fixture.sources))
    report = fixture.run()
    assert report["verdict"] == "INCOMPLETE"
    assert len(report["incomplete"]) == len(gate.CAPABILITIES) + len(gate.SOLVERS) + 6


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), 10 ** 400])
def test_bool_and_nonfinite_rejected(evidence, value):
    manifest = copy.deepcopy(evidence.manifest)
    manifest["schema_version"] = value
    payload = (json.dumps(manifest) + "\n").encode()
    desc = evidence.put(payload)
    with pytest.raises(gate.Rejected):
        gate.precheck(desc["path"], desc["sha256"], current_root=evidence.root / "source")


def test_duplicate_keys_fail_loud(tmp_path):
    payload = b"{\"schema_version\":1,\"schema_version\":1}"
    with pytest.raises(gate.Rejected, match="duplicate_json_key"):
        gate.parse(payload, "manifest")


@pytest.mark.parametrize("field", ["verdict", "release_complete", "synthetic", "evidence_class"])
def test_no_self_approval_or_synthetic_production_selector(evidence, field):
    evidence.manifest[field] = "PASS"
    with pytest.raises(gate.Rejected, match="schema_keys"):
        evidence.run()


def test_scope_omission_not_allowed(evidence):
    evidence.manifest["capability_scope"].remove("sdp")
    with pytest.raises(gate.Rejected, match="incomplete_capability_scope"):
        evidence.run()


def test_current_source_mismatch(evidence, monkeypatch):
    monkeypatch.setattr(gate, "current_sources", lambda root: ("b" * 40, evidence.sources))
    assert "current_source_mismatch" in codes(evidence.run())


def test_source_changes_between_reads(evidence, monkeypatch):
    calls = []

    def changing(root):
        calls.append(root)
        return (evidence.commit if len(calls) == 1 else "b" * 40, evidence.sources)

    monkeypatch.setattr(gate, "current_sources", changing)
    assert "current_source_mismatch" in codes(evidence.run()) and len(calls) == 2


def test_unpinned_receipt_changed(evidence):
    desc = evidence.manifest["capabilities"]["lp"]
    desc = evidence.put(Path(desc["path"]).read_bytes())
    evidence.manifest["capabilities"]["lp"] = desc
    Path(desc["path"]).write_bytes(b"changed bytes")
    assert "byte_pin_mismatch" in codes(evidence.run())


def test_rehashed_actual_cannot_reuse_old_acceptance(evidence):
    desc = evidence.manifest["capabilities"]["lp"]
    evidence.manifest["capabilities"]["lp"] = evidence.mutate(desc, lambda d: d.update(negative_controls=2))
    assert "independent_acceptance_binding" in codes(evidence.run())


@pytest.mark.parametrize("mutation,expected", [
    (lambda d: d["counts"].update(accepted=7), "incomplete_or_contradictory_counts"),
    (lambda d: d.update(samples=[], counts={"attempted": 0, "accepted": 0, "failed": 0, "unsupported": 0, "skipped": 0}), "no_actual_solves"),
    (lambda d: d.update(status="UNVERIFIED"), "capability_not_qualified"),
    (lambda d: d.update(negative_controls=0), "invalid_integer"),
    (lambda d: d.update(negative_controls=True), "boolean_not_evidence"),
])
def test_coordinated_capability_mutants(evidence, mutation, expected):
    desc = evidence.manifest["capabilities"]["lp"]
    evidence.manifest["capabilities"]["lp"] = evidence.mutate(desc, mutation, refresh_acceptance=True)
    assert expected in codes(evidence.run())


def test_bare_pass_receipt_is_not_actual(evidence):
    evidence.manifest["capabilities"]["lp"] = evidence.put({"verdict": "PASS"})
    assert "schema_keys" in codes(evidence.run())


def test_producer_not_current_committed_source(evidence):
    receipt = evidence.load(evidence.manifest["capabilities"]["lp"])
    receipt["producer"] = evidence.put(b"unmatched test-only producer")
    evidence.manifest["capabilities"]["lp"] = evidence.put(receipt)
    assert "producer_source_mismatch" in codes(evidence.run())


def test_signed_gap_unknown_cannot_be_hidden_by_pass(evidence):
    def mutation(data):
        check = evidence.load(data["samples"][0]["check"])
        check["signed_gap"] = "UNVERIFIED"
        data["samples"][0]["check"] = evidence.put(check)
    evidence.manifest["capabilities"]["convex_qp"] = evidence.mutate(
        evidence.manifest["capabilities"]["convex_qp"], mutation, refresh_acceptance=True)
    assert "original_check_binding" in codes(evidence.run())


@pytest.mark.parametrize("mutation,expected", [
    (lambda d: d.update(actual_native_calls=0), "invalid_integer"),
    (lambda d: d["authorization"].update(status="UNVERIFIED"), "authorization_not_established"),
    (lambda d: d["authorization"].update(extra_material="forbidden test-only field"), "schema_keys"),
    (lambda d: d["openopt"]["samples"][0].update(resource_sha256="0" * 64), "actual_result_binding"),
])
def test_commercial_negative_controls(evidence, mutation, expected):
    desc = evidence.manifest["commercial"]["gurobi"]
    evidence.manifest["commercial"]["gurobi"] = evidence.mutate(desc, mutation, refresh_acceptance=True)
    assert expected in codes(evidence.run())


def test_one_card_jobs_do_not_establish_multigpu(evidence):
    evidence.manifest["gpu_scaling"].pop("2")
    evidence.manifest["gpu_scaling"].pop("3")
    assert "missing_actual_receipt" in codes(evidence.run())
    assert "missing_or_different_scaling_problem" in codes(evidence.run())


@pytest.mark.parametrize("mutation,expected", [
    (lambda d: d.update(gpu_count=1), "gpu_count_or_timing"),
    (lambda d: d.update(timing_scope="kernel_only"), "gpu_count_or_timing"),
    (lambda d: d["devices"][1].update(id=d["devices"][0]["id"]), "duplicate_gpu_devices"),
])
def test_gpu_coordinated_negatives(evidence, mutation, expected):
    desc = evidence.manifest["gpu_scaling"]["2"]
    evidence.manifest["gpu_scaling"]["2"] = evidence.mutate(desc, mutation, refresh_acceptance=True)
    assert expected in codes(evidence.run())


def test_zero_kernel_trace_not_execution(evidence):
    def mutation(data):
        trace = evidence.load(data["trace"])
        trace["kernel_events"][next(iter(trace["kernel_events"]))] = 0
        data["trace"] = evidence.put(trace)
    evidence.manifest["gpu_scaling"]["3"] = evidence.mutate(
        evidence.manifest["gpu_scaling"]["3"], mutation, refresh_acceptance=True)
    assert "invalid_integer" in codes(evidence.run())


def test_rehashed_different_original_model_cannot_scale_same_problem(evidence):
    def mutation(data):
        sample = data["samples"][0]
        sample["model"] = evidence.put({"test_only_other_model": {"q": [2]}})
        result = evidence.load(sample["result"])
        result["model_sha256"] = sample["model"]["sha256"]
        sample["result"] = evidence.put(result)
        check = evidence.load(sample["check"])
        check.update(model_sha256=sample["model"]["sha256"], result_sha256=sample["result"]["sha256"])
        sample["check"] = evidence.put(check)
        for key in ("trace", "coordinator"):
            record = evidence.load(data[key])
            record["result_sha256"] = [s["result"]["sha256"] for s in data["samples"]]
            data[key] = evidence.put(record)
    evidence.manifest["gpu_scaling"]["2"] = evidence.mutate(
        evidence.manifest["gpu_scaling"]["2"], mutation, refresh_acceptance=True)
    assert "missing_or_different_scaling_problem" in codes(evidence.run())


def test_publication_readback_changed_even_rehashed(evidence):
    desc = evidence.manifest["publication"]
    evidence.manifest["publication"] = evidence.mutate(desc, lambda d: d.update(
        readback=evidence.put(b"different test-only readback")), refresh_acceptance=True)
    assert "online_readback_bytes" in codes(evidence.run())


def test_remote_source_mismatch(evidence):
    evidence.manifest["publication"] = evidence.mutate(evidence.manifest["publication"], lambda d: d.update(
        remote_source=evidence.put({"commit": "b" * 40, "source_sha256": evidence.sources})), refresh_acceptance=True)
    assert "remote_source_mismatch" in codes(evidence.run())


@pytest.mark.parametrize("mutation,expected", [
    (lambda d: d["origins"][0].update(status="UNKNOWN"), "origin_rights_unverified"),
    (lambda d: d["origins"][0].update(paths=["LICENSE"]), "origin_source_membership"),
    (lambda d: d.update(dependency_scope="project_0bsd_everything"), "license_scope_missing"),
])
def test_0bsd_does_not_relicense_origins_dependencies(evidence, mutation, expected):
    evidence.manifest["licensing"] = evidence.mutate(evidence.manifest["licensing"], mutation, refresh_acceptance=True)
    assert expected in codes(evidence.run())


def test_sbom_inventory_missing_member(evidence):
    def mutation(data):
        inventory = evidence.load(data["inventory"])
        inventory["dependencies"] = []
        data["inventory"] = evidence.put(inventory)
    evidence.manifest["licensing"] = evidence.mutate(evidence.manifest["licensing"], mutation, refresh_acceptance=True)
    assert "sbom_inventory_mismatch" in codes(evidence.run())


def test_wrong_pin_size_and_symlink(tmp_path):
    fixture = Evidence(tmp_path)
    desc = fixture.put({"test_only": "data"})
    checker = gate.Precheck.__new__(gate.Precheck)
    checker.root, checker.inputs = tmp_path / "source", {}
    wrong = dict(desc, bytes=desc["bytes"] + 1)
    with pytest.raises(gate.Rejected, match="byte_size_mismatch"):
        checker.blob(wrong, "test")
    wrong = dict(desc, sha256="0" * 64)
    with pytest.raises(gate.Rejected, match="byte_pin_mismatch"):
        checker.blob(wrong, "test")
    link = tmp_path / "link"
    link.symlink_to(desc["path"])
    with pytest.raises(gate.Rejected, match="unsafe_or_missing_file"):
        gate.read(link, "test")


def test_real_git_source_clean_and_exact(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "sample.py").write_text("# test-only source\n")
    (repo / "LICENSE").write_text("BSD Zero Clause test-only\n")
    for args in (("init",), ("add", "src/sample.py", "LICENSE"),
                 ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")):
        subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    commit, sources = gate.current_sources(repo)
    assert len(commit) == 40 and set(sources) == {"LICENSE", "src/sample.py"}
    (repo / "LICENSE").write_text("changed test-only bytes\n")
    with pytest.raises(gate.Rejected, match="source_not_clean"):
        gate.current_sources(repo)


def test_cli_never_zero_or_pass_and_receipt_is_readable(evidence, monkeypatch, tmp_path):
    manifest = evidence.put(evidence.manifest)
    rc = gate.main(["--manifest", manifest["path"], "--manifest-sha256", manifest["sha256"],
                    "--current-root", str(tmp_path / "source"), "--artifact-dir", str(tmp_path / "output")])
    report = json.loads(next((tmp_path / "output").glob("completion-*/completion.json")).read_bytes())
    assert rc == report["returncode"] == 2 and report["verdict"] == "REVIEW_REQUIRED"
    assert report["release_complete"] is False
    assert "returncode=2" in next((tmp_path / "output").glob("completion-*/execution.log")).read_text()


def test_cli_failure_preserves_safe_first_cause(evidence, tmp_path):
    manifest = evidence.put(evidence.manifest)
    rc = gate.main(["--manifest", manifest["path"], "--manifest-sha256", "0" * 64,
                    "--current-root", str(tmp_path / "source"), "--artifact-dir", str(tmp_path / "output")])
    report = json.loads(next((tmp_path / "output").glob("completion-*/completion.json")).read_bytes())
    assert rc == 1 and report["verdict"] == "INCOMPLETE" and "byte_pin_mismatch" in codes(report)


def freeze_historical(argv=None):
    parser = argparse.ArgumentParser(description="Optional real historical INCOMPLETE freeze, never default pytest")
    parser.add_argument("--freeze-known", action="store_true", required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = gate.read(HISTORICAL, "historical")
    gate.require(gate.sha(payload) == HISTORICAL_PIN, "historical_pin_mismatch", "historical")
    commit, sources = gate.current_sources(args.current_root)
    output = gate.absolute(args.artifact_dir, "output")
    gate.require(not output.is_relative_to(gate.ROOT) and not output.is_relative_to(
        gate.absolute(args.current_root, "source")), "output_must_be_external", "output")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "scope": gate.SCOPE, "source_commit": commit,
                "source_sha256": sources, "capability_scope": list(gate.CAPABILITIES),
                "capabilities": {}, "commercial": {}, "gpu_scaling": {},
                "publication": None, "licensing": None,
                "historical_observations": [{"kind": "commercial_runner_discovery_v1", "receipt": {
                    "path": str(HISTORICAL), "sha256": HISTORICAL_PIN, "bytes": len(payload)}}]}
    manifest_path = output / "known-incomplete-manifest.json"
    with manifest_path.open("xb") as stream:
        stream.write(gate.encoded(manifest))
    pin = gate.sha(gate.read(manifest_path, "manifest"))
    rc = gate.main(["--manifest", str(manifest_path), "--manifest-sha256", pin,
                    "--current-root", str(args.current_root), "--artifact-dir", str(output / "cli")])
    report_path = next((output / "cli").glob("completion-*/completion.json"))
    report = json.loads(report_path.read_bytes())
    gate.require(rc == 1 and report["verdict"] == "INCOMPLETE" and report["release_complete"] is False
                 and report["historical_observations"][0]["actual_commercial_solves"] == 0,
                 "historical_freeze_not_incomplete", "freeze")
    gate.require("pinned_historical_zero_commercial_solves" in codes(report), "historical_missing_cause", "freeze")
    with (output / "freeze-receipt.json").open("xb") as stream:
        stream.write(gate.encoded({"verdict": "INCOMPLETE", "manifest_sha256": pin,
                                  "report_sha256": gate.sha(report_path.read_bytes()),
                                  "report": str(report_path), "actual_commercial_solves": 0,
                                  "historical_source_only": "NOT_CURRENT_SOURCE_QUALIFICATION", "cli_returncode": rc}))
    receipt = output / "freeze-receipt.json"
    print(f"KNOWN_INCOMPLETE_FROZEN manifest_sha256={pin} receipt={receipt}")
    return 0


def scan_source(argv=None):
    parser = argparse.ArgumentParser(description="Scoped credential scan; values never printed")
    parser.add_argument("--scan-only", action="store_true", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = gate.absolute(args.artifact_dir, "output")
    gate.require(not output.is_relative_to(gate.ROOT), "output_must_be_external", "output")
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("safe_scanner", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(
        secret_hex=None, secrets_file=None, no_needle=False,
        needle_from=str(Path.home() / ".config/dsh-n600-token")))
    gate.require(bool(needles), "credential_scan_needle_unavailable", "scan")
    paths = [gate.ROOT / "scripts/check_release_completion.py", Path(__file__).resolve(),
             gate.ROOT / "docs/research/release-completion.md"]
    literal, _ = scanner.scan_literal([str(p) for p in paths], needles)
    pattern = gate.re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                              r"(?i:password|token|secret)\s*[\"\x27]?\s*[:=]\s*[\"\x27][^\"\x27\s]{6,}[\"\x27]")
    shape = sum(bool(pattern.search(p.read_text())) for p in paths)
    report = {"verdict": "PASS" if literal == shape == 0 else "FAIL", "literal_hits": literal,
              "shape_hits": shape, "files": {str(p): gate.sha(p.read_bytes()) for p in paths}}
    with (output / "source-security-scan.json").open("xb") as stream:
        stream.write(gate.encoded(report))
    print(f"RELEASE_SOURCE_SCAN literal_hits={literal} shape_hits={shape} files={len(paths)}")
    return 0 if literal == shape == 0 else 1


if __name__ == "__main__":
    raise SystemExit(scan_source() if "--scan-only" in sys.argv else freeze_historical())
