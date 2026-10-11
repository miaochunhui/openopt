"""Narrow reader controls using unchanged real Git/metadata author fixtures."""

from __future__ import annotations

import json
import argparse
import ast
from pathlib import Path
import sys

import pytest

from scripts import build_sbom as sbom
from tests.test_build_sbom import SCHEMA, build, inputs  # noqa: F401

BASE = "3d6b5da403dd7296447fefbf17c1f1f5004b8bad"
OWNED = ("scripts/build_sbom.py", "tests/test_sbom_reader_schema_binding.py",
         "docs/research/sbom-reader-schema-binding.md")


def delivery(inputs, tmp_path):
    document, evidence = build(inputs)
    target = sbom.save_delivery(tmp_path / "delivery", document, evidence,
                                require_shared=False)
    receipt = json.loads(target.read_bytes())
    assert receipt["schema_sha256"] == evidence["schema"]["sha256"] == sbom.SCHEMA_SHA256
    return target


def read(target, schema=SCHEMA, schema_pin=sbom.SCHEMA_SHA256):
    return sbom.read_delivery(target, sbom.digest(target.read_bytes()), schema, schema_pin)


def test_valid_receipt_retains_bounded_qualification(inputs, tmp_path):
    receipt = read(delivery(inputs, tmp_path))
    assert receipt["verdict"] == "PASS"
    assert receipt["legal_qualification"] == "UNVERIFIED"
    assert receipt["runtime_qualification"] == "NOT_TESTED"


def test_rehashed_wrong_receipt_schema_is_rejected(inputs, tmp_path):
    target = delivery(inputs, tmp_path)
    receipt = json.loads(target.read_bytes())
    receipt["schema_sha256"] = "0" * 64
    target.write_bytes(sbom.encoded(receipt))
    with pytest.raises(sbom.SbomError, match="^receipt_schema_pin_mismatch$"):
        read(target)


@pytest.mark.parametrize("value", [None, True, ""])
def test_invalid_receipt_schema_is_rejected(inputs, tmp_path, value):
    target = delivery(inputs, tmp_path)
    receipt = json.loads(target.read_bytes())
    receipt["schema_sha256"] = value
    target.write_bytes(sbom.encoded(receipt))
    with pytest.raises(sbom.SbomError, match="^receipt_schema_pin_mismatch$"):
        read(target)


def mutate_evidence(target, field, value):
    evidence_path = target.parent / "evidence.json"
    evidence = json.loads(evidence_path.read_bytes())
    evidence[field] = value
    evidence_path.write_bytes(sbom.encoded(evidence))
    receipt = json.loads(target.read_bytes())
    receipt["artifacts_sha256"]["evidence.json"] = sbom.digest(evidence_path.read_bytes())
    target.write_bytes(sbom.encoded(receipt))


def test_rehashed_wrong_evidence_schema_is_rejected(inputs, tmp_path):
    target = delivery(inputs, tmp_path)
    mutate_evidence(target, "schema", {"sha256": "0" * 64})
    with pytest.raises(sbom.SbomError, match="^evidence_schema_pin_mismatch$"):
        read(target)


def test_actual_schema_bytes_remain_bound(inputs, tmp_path):
    target = delivery(inputs, tmp_path)
    altered = tmp_path / "schema.json"
    altered.write_bytes(SCHEMA.read_bytes() + b"\n")
    with pytest.raises(sbom.SbomError, match="^schema_pin_mismatch$"):
        read(target, schema=altered)


def test_coordinated_false_schema_pin_is_rejected(inputs, tmp_path):
    target = delivery(inputs, tmp_path)
    mutate_evidence(target, "schema", {"sha256": "0" * 64})
    receipt = json.loads(target.read_bytes())
    receipt["schema_sha256"] = "0" * 64
    target.write_bytes(sbom.encoded(receipt))
    with pytest.raises(sbom.SbomError, match="^schema_pin_mismatch$"):
        read(target, schema_pin="0" * 64)


@pytest.mark.parametrize("field", ["legal_qualification", "runtime_qualification"])
@pytest.mark.parametrize("side", ["receipt", "evidence"])
def test_rehashed_qualification_upgrade_is_rejected(inputs, tmp_path, field, side):
    target = delivery(inputs, tmp_path)
    if side == "evidence":
        mutate_evidence(target, field, "PASS")
    else:
        receipt = json.loads(target.read_bytes())
        receipt[field] = "PASS"
        target.write_bytes(sbom.encoded(receipt))
    with pytest.raises(sbom.SbomError, match="^receipt_qualification_mismatch$"):
        read(target)


def test_rehashed_source_declaration_is_rejected(inputs, tmp_path):
    target = delivery(inputs, tmp_path)
    receipt = json.loads(target.read_bytes())
    receipt["source_commit"] = "0" * 40
    target.write_bytes(sbom.encoded(receipt))
    with pytest.raises(sbom.SbomError, match="^receipt_source_mismatch$"):
        read(target)


def test_rehashed_verdict_disagreement_is_rejected(inputs, tmp_path):
    target = delivery(inputs, tmp_path)
    receipt = json.loads(target.read_bytes())
    receipt["verdict"] = "INCOMPLETE"
    target.write_bytes(sbom.encoded(receipt))
    with pytest.raises(sbom.SbomError, match="^receipt_verdict_mismatch$"):
        read(target)


def test_real_missing_dependency_remains_incomplete(inputs, tmp_path):
    document, evidence = sbom.build(inputs[0], inputs[1], [], (), "3.10",
                                    SCHEMA, sbom.SCHEMA_SHA256)
    assert evidence["unresolved"] == ["alpha"]
    target = sbom.save_delivery(tmp_path / "delivery", document, evidence,
                                require_shared=False)
    assert read(target)["verdict"] == "INCOMPLETE"


def test_original_author_file_and_nonreader_functions_unchanged():
    original = sbom.git(sbom.ROOT, "show", BASE + ":tests/test_build_sbom.py")
    assert original == (sbom.ROOT / "tests/test_build_sbom.py").read_bytes()
    before = ast.parse(sbom.git(sbom.ROOT, "show", BASE + ":scripts/build_sbom.py"))
    after = ast.parse((sbom.ROOT / "scripts/build_sbom.py").read_bytes())
    for tree in (before, after):
        tree.body = [node for node in tree.body
                     if not isinstance(node, ast.FunctionDef) or node.name != "read_delivery"]
    assert ast.dump(before) == ast.dump(after)


def read_tests(directory, pin):
    from scripts import verify, verify_sharded as observer

    target = directory / "verification.json"
    sbom.require(sbom.digest(target.read_bytes()) == pin, "verification_pin_mismatch")
    report = sbom.json_read(target)
    process = sbom.json_read(directory / "process.json")
    collection = sbom.json_read(directory / "collect/collection.json")["nodeids"]
    cases, counts = observer.parse_junit(directory / "pytest.xml")
    sbom.require(collection and len(collection) == len(set(collection)),
                 "empty_or_duplicate_collection")
    sbom.require(len(cases) == len(collection) and
                 {observer.junit_key(node) for node in collection}
                 == {tuple(case["key"]) for case in cases},
                 "junit_membership_mismatch")
    sbom.require(report["verdict"] == "PASS" and process["returncode"] == 0
                 and process["timeout"] is False, "execution_not_pass")
    sbom.require(counts == report["counts"] ==
                 {"tests": len(collection), "failures": 0, "errors": 0, "skipped": 0},
                 "test_counts_invalid")
    sbom.require(report["collection"] == collection, "collection_report_mismatch")
    sbom.require(report["source"] == report["source_after"]
                 == verify.source_snapshot(sbom.ROOT), "reader_source_changed")
    print(f"SBOM_REPAIR_READER verdict=PASS tests={len(collection)} skip=0 "
          f"membership=PASS source=PASS rc=0 receipt_sha256={pin} "
          f"junit_sha256={sbom.digest((directory / 'pytest.xml').read_bytes())}")


def scan_owned(output=None):
    from scripts.audit_git_delivery import content_hits, load_needles

    needles = load_needles()
    rows = {name: content_hits((sbom.ROOT / name).read_bytes(), needles) for name in OWNED}
    sbom.require(not any(any(row.values()) for row in rows.values()), "owned_scan_failed")
    if output is not None:
        output = sbom.output_root(output)
        with output.open("xb") as stream:
            stream.write(sbom.encoded({
                "verdict": "PASS", "counts": rows, "values_recorded": False,
                "sha256": {name: sbom.digest((sbom.ROOT / name).read_bytes())
                           for name in OWNED},
            }))
    print("SBOM_REPAIR_SCAN literal=0 shape=0 files=3 values_recorded=false")


def main(argv):
    if "--read-tests" in argv or "--scan-owned" in argv:
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--read-tests", type=Path)
        parser.add_argument("--verification-sha256")
        parser.add_argument("--scan-owned", action="store_true")
        parser.add_argument("--scan-output", type=Path)
        args = parser.parse_args(argv)
        if args.scan_owned:
            scan_owned(args.scan_output)
        else:
            read_tests(args.read_tests, args.verification_sha256)
        return 0
    return sbom.run_tests(Path(__file__).resolve(), argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
