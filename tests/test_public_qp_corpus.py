"""Actual external MPS files, independent semantic checks and corruption gates.

These hand-authored format fixtures are synthetic, not public/industrial data.
The mandatory highspy reader is exercised, not mocked or optionally skipped.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import public_qp_corpus as corpus


MPS = """NAME OFFSET
ROWS
 N OBJ
 E EQ
 L CAP
 G FLOOR
COLUMNS
 X OBJ -3 EQ 1
 X CAP 2 FLOOR 1
 Y OBJ 4 EQ -1
 Y CAP 1 FLOOR 2
RHS
 RHS1 OBJ -7 EQ 1
 RHS1 CAP 8 FLOOR -2
BOUNDS
 LO BND X -2
 UP BND X 3
 FR BND Y
QUADOBJ
 X X 4
 X Y 1
 Y Y 2
ENDATA
"""


@pytest.fixture
def environment(tmp_path):
    assert not tmp_path.resolve().is_relative_to(corpus.ROOT)
    local = tmp_path / "local"
    local.mkdir()
    source = local / "offset.mps"
    source.write_text(MPS, encoding="ascii")
    entry = {
        "case_id": "format-offset", "path": source.name,
        "source_url": "https://example.invalid/synthetic-format-fixture.mps",
        "license": "0BSD; hand-authored format-test fixture, not a public dataset",
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "origin": "synthetic",
    }
    manifest = tmp_path / "manifest.json"
    corpus.write_json(manifest, {"schema_version": 1, "cases": [entry]})
    return tmp_path, local, source, entry, manifest


def frozen(environment, monkeypatch, text=None):
    root, local, source, entry, manifest = environment
    monkeypatch.setattr(corpus, "SHARED_TMP", root.resolve())
    if text is not None:
        source.write_text(text, encoding="ascii")
        entry["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest.write_text(json.dumps({"schema_version": 1, "cases": [entry]}))
    output = root / "frozen"
    corpus.freeze(manifest, local, output)
    report = corpus.read_json(output / "corpus.json")
    metadata = corpus.read_json(output / entry["case_id"] / "metadata.json")
    with np.load(output / entry["case_id"] / "arrays.npz", allow_pickle=False) as a:
        model = {**metadata, "arrays": {key: a[key] for key in a.files}}
    return output, report, model


@pytest.mark.parametrize("suffix", [".mps", ".qps"])
def test_real_reader_frozen_bytes_bounds_offset_and_json_replay(environment, monkeypatch, suffix):
    root, local, source, entry, manifest = environment
    if suffix == ".qps":
        source = source.rename(source.with_suffix(suffix))
        entry["path"] = source.name
        manifest.write_text(json.dumps({"schema_version": 1, "cases": [entry]}))
    output, report, model = frozen((root, local, source, entry, manifest), monkeypatch)
    a = model["arrays"]
    assert model["reader"]["device"] == "cpu"
    assert model["reader"]["status"] == "kOk"
    assert model["reader"]["raw_record_semantics"] == "exactly matched"
    assert np.array_equal(a["P"], [[4, 1], [1, 2]])
    assert np.array_equal(a["q"], [-3, 4])
    assert np.array_equal(a["A"], [[1, -1], [2, 1], [1, 2]])
    assert np.array_equal(a["l"], [1, -np.inf, -2])
    assert np.array_equal(a["u"], [1, 8, np.inf])
    assert np.array_equal(a["variable_lower"], [-2, -np.inf])
    assert np.array_equal(a["variable_upper"], [3, np.inf])
    assert float(a["offset"]) == 7
    assert corpus.original_objective(model, np.array([2, 1])) == 16
    assert (output / entry["case_id"] / "raw.mps").read_bytes() == source.read_bytes()
    assert model["raw_sha256"] == entry["input_sha256"]
    assert report["case_count"] == 1 and report["public_qualified_count"] == 0
    assert model["qualification"] == "SYNTHETIC_NOT_COVERAGE"
    result = corpus.verify_frozen(output, corpus.sha256((output / "corpus.json").read_bytes()))
    assert result["verdict"] == "VERIFIED" and result["case_count"] == 1
    json.dumps(result, allow_nan=False)
    solver = corpus.solver_arrays(model)
    assert solver["offset"] == 7 and solver["objective_sign"] == 1
    assert np.array_equal(solver["A"][-2:], np.eye(2))
    assert np.array_equal(solver["l"][-2:], a["variable_lower"])
    assert np.array_equal(solver["u"][-2:], a["variable_upper"])
    assert solver["row_mapping"][-2:] == [("variable_bound", 0), ("variable_bound", 1)]


@pytest.mark.parametrize("kind", ["lp", "fixed", "max-concave", "qmatrix"])
def test_actual_reader_supported_variants(environment, monkeypatch, kind):
    text = MPS
    if kind == "lp":
        text = text[:text.index("QUADOBJ")] + "ENDATA\n"
    elif kind == "fixed":
        text = text.replace(" LO BND X -2\n UP BND X 3", " FX BND X -1")
    elif kind == "max-concave":
        text = text.replace("ROWS", "OBJSENSE\n MAX\nROWS", 1)
        text = text.replace("X X 4", "X X -4").replace("X Y 1", "X Y -1").replace("Y Y 2", "Y Y -2")
    else:
        text = text.replace("QUADOBJ", "QMATRIX").replace(" X Y 1\n", " X Y 1\n Y X 1\n")
    _, _, model = frozen(environment, monkeypatch, text)
    if kind == "lp":
        assert np.count_nonzero(model["arrays"]["P"]) == 0
    elif kind == "fixed":
        assert model["arrays"]["variable_lower"][0] == model["arrays"]["variable_upper"][0] == -1
    elif kind == "max-concave":
        solver = corpus.solver_arrays(model)
        assert solver["objective_sign"] == -1 and solver["offset"] == -7
        x = np.array([2, 1])
        internal = 0.5 * x @ solver["P"] @ x + solver["q"] @ x + solver["offset"]
        assert -internal == corpus.original_objective(model, x)
    else:
        assert np.array_equal(model["arrays"]["P"], [[4, 1], [1, 2]])


@pytest.mark.parametrize("mutation,reason", [
    ("integer-marker", "integer"),
    ("integer-bound", "integer"),
    ("semicontinuous", "integer/semicontinuous"),
    ("ranges", "unsupported construct RANGES"),
    ("sos", "unsupported construct SOS"),
    ("qsection", "unsupported construct QSECTION"),
    ("nonconvex", "nonconvex"),
    ("max-convex", "nonconvex"),
    ("duplicate-coefficient", "duplicate record"),
    ("multiple-rhs", "multiple RHS"),
    ("multiple-bounds", "multiple BOUNDS"),
    ("unknown-row", "unknown COLUMNS row"),
    ("nonfinite", "nonfinite"),
    ("missing-end", "missing required"),
    ("free-row", "exactly one"),
    ("bounds-order", "invalid bounds"),
    ("asymmetric-qmatrix", "exactly symmetric"),
    ("double-triangle", "only one triangle"),
    ("trailing-data", "unsupported construct"),
])
def test_semantic_negative_controls_keep_first_cause(environment, monkeypatch, mutation, reason):
    root, local, source, entry, manifest = environment
    monkeypatch.setattr(corpus, "SHARED_TMP", root.resolve())
    edits = {
        "integer-marker": ("COLUMNS\n", "COLUMNS\n MARK 'MARKER' 'INTORG'\n"),
        "integer-bound": (" FR BND Y", " BV BND Y"),
        "semicontinuous": (" UP BND X 3", " SC BND X 3"),
        "ranges": ("BOUNDS", "RANGES\n R EQ 1\nBOUNDS"),
        "sos": ("ENDATA", "SOS\nENDATA"),
        "qsection": ("QUADOBJ", "QSECTION OBJ"),
        "nonconvex": ("X X 4", "X X -4"),
        "max-convex": ("ROWS", "OBJSENSE\n MAX\nROWS"),
        "duplicate-coefficient": (" X OBJ -3 EQ 1", " X OBJ -3 EQ 1\n X OBJ -3"),
        "multiple-rhs": (" RHS1 CAP 8", " RHS2 CAP 8"),
        "multiple-bounds": (" FR BND Y", " FR OTHER Y"),
        "unknown-row": ("X CAP 2", "X UNKNOWN 2"),
        "nonfinite": ("X X 4", "X X NaN"),
        "missing-end": ("ENDATA", ""),
        "free-row": (" N OBJ", " N OBJ\n N FREE"),
        "bounds-order": (" LO BND X -2", " LO BND X 5"),
        "asymmetric-qmatrix": ("QUADOBJ", "QMATRIX"),
        "double-triangle": (" X Y 1\n", " X Y 1\n Y X 1\n"),
        "trailing-data": ("ENDATA\n", "ENDATA\nJUNK\n"),
    }
    source.write_text(MPS.replace(*edits[mutation]), encoding="ascii")
    entry["input_sha256"] = corpus.sha256(source.read_bytes())
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [entry]}))
    output = root / "rejected"
    with pytest.raises(ValueError, match=reason):
        corpus.freeze(manifest, local, output)
    report = corpus.read_json(output / "corpus.json")
    assert report["verdict"] == "ERROR" and report["current_case"] == entry["case_id"]
    assert reason.split()[0].lower() in report["error"]["message"].lower()
    assert report["cases"] == []
    assert "ValueError" in (output / "failure.log").read_text()


@pytest.mark.parametrize("field", ["source_url", "license", "input_sha256", "origin"])
def test_missing_provenance_is_rejected(environment, field):
    entry = dict(environment[3])
    del entry[field]
    with pytest.raises(ValueError, match="requires exactly"):
        corpus.declaration(entry)


@pytest.mark.parametrize("url", ["", "file:///tmp/test", "https://user:secret@example.invalid/x",
                                "https://example.invalid/x?token=redacted", "https://example.invalid/x#fragment"])
def test_unsafe_url_is_rejected(environment, url):
    entry = dict(environment[3], source_url=url)
    with pytest.raises(ValueError, match="source_url"):
        corpus.declaration(entry)


def test_public_declaration_does_not_establish_coverage(environment, monkeypatch):
    root, local, source, entry, manifest = environment
    entry["origin"] = "public"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [entry]}))
    _, report, model = frozen(environment, monkeypatch)
    assert model["qualification"] == "UNVERIFIED"
    assert report["public_declared_count"] == 1
    assert report["public_qualified_count"] == report["industrial_qualified_count"] == 0


@pytest.mark.parametrize("mutation", ["empty", "bad-sha", "escape", "duplicate", "budget"])
def test_input_negative_controls_and_no_overwrite(environment, monkeypatch, mutation):
    root, local, source, entry, manifest = environment
    monkeypatch.setattr(corpus, "SHARED_TMP", root.resolve())
    cases = [entry]
    kwargs = {}
    if mutation == "empty":
        cases = []
    elif mutation == "bad-sha":
        entry["input_sha256"] = "0" * 64
    elif mutation == "escape":
        entry["path"] = "../manifest.json"
    elif mutation == "duplicate":
        cases.append(dict(entry))
    else:
        kwargs["max_dense_bytes"] = 1
    manifest.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    output = root / "reject"
    with pytest.raises(ValueError):
        corpus.freeze(manifest, local, output, **kwargs)
    before = (output / "corpus.json").read_bytes()
    with pytest.raises(FileExistsError):
        corpus.freeze(manifest, local, output)
    assert (output / "corpus.json").read_bytes() == before
    assert corpus.read_json(output / "corpus.json")["verdict"] == "ERROR"


@pytest.mark.parametrize("target", ["raw", "npz", "metadata", "manifest", "input-manifest", "pinned-hash"])
def test_byte_hash_negative_controls(environment, monkeypatch, target):
    output, report, model = frozen(environment, monkeypatch)
    pinned = corpus.sha256((output / "corpus.json").read_bytes())
    case_dir = output / report["cases"][0]["case_id"]
    paths = {"raw": case_dir / "raw.mps", "npz": case_dir / "arrays.npz",
             "metadata": case_dir / "metadata.json", "manifest": output / "corpus.json",
             "input-manifest": output / "input-manifest.json"}
    if target == "pinned-hash":
        pinned = "0" * 64
    else:
        path = paths[target]
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256|byte hash"):
        corpus.verify_frozen(output, pinned)


def reseal(output, model):
    case_dir = output / model["provenance"]["case_id"]
    archive = case_dir / "arrays.npz"
    with archive.open("wb") as stream:
        np.savez(stream, **model["arrays"])
    metadata = {k: v for k, v in model.items() if k != "arrays"}
    metadata["npz_sha256"] = corpus.sha256(archive.read_bytes())
    metadata["array_hashes"] = corpus.array_hashes(model["arrays"])
    metadata["model_sha256"] = corpus.model_hash(model)
    (case_dir / "metadata.json").write_text(json.dumps(metadata, allow_nan=False))
    report = corpus.read_json(output / "corpus.json")
    report["cases"][0]["metadata_sha256"] = corpus.sha256((case_dir / "metadata.json").read_bytes())
    report["cases"][0]["model_sha256"] = metadata["model_sha256"]
    (output / "corpus.json").write_text(json.dumps(report, allow_nan=False))
    return corpus.sha256((output / "corpus.json").read_bytes())


@pytest.mark.parametrize("mutation", ["offset", "bounds", "hessian", "names", "objective-sense"])
def test_self_resealed_wrong_semantics_still_rejected(environment, monkeypatch, mutation):
    output, _, model = frozen(environment, monkeypatch)
    if mutation == "offset":
        model["arrays"]["offset"] = np.asarray(0.0)
    elif mutation == "bounds":
        model["arrays"]["variable_lower"][0] = 0
    elif mutation == "hessian":
        model["arrays"]["P"] *= 2
    elif mutation == "names":
        model["variable_names"].reverse()
    else:
        model["objective_sense"] = "maximize"
    pinned = reseal(output, model)
    with pytest.raises(ValueError, match="semantics|nonconvex"):
        corpus.verify_frozen(output, pinned)


def test_canonical_hash_binds_shape_dtype_and_scalar_offset():
    a = corpus.audit_records(MPS.encode())["arrays"]
    hashes = corpus.array_hashes(a)
    b = {key: value.copy() for key, value in a.items()}
    b["offset"] = np.asarray(8.0)
    assert corpus.array_hashes(b)["offset"] != hashes["offset"]
    b["A"] = b["A"].reshape(2, 3)
    assert corpus.array_hashes(b)["A"] != hashes["A"]
    assert all(len(h) == 64 for h in hashes.values())


def test_reader_silent_coefficient_dropping_is_not_accepted(environment, monkeypatch):
    # HiGHS' default small-matrix threshold may discard this nonzero coefficient.
    text = MPS.replace("X CAP 2", "X CAP 1e-14")
    with pytest.raises(ValueError, match="reader semantics|readModel failed/warned"):
        frozen(environment, monkeypatch, text)


def test_cli_actual_process_import_and_pinned_verify(environment, monkeypatch):
    root, local, source, entry, manifest = environment
    # Actual processes must use the real authorized external output root.
    assert root.resolve().is_relative_to(corpus.SHARED_TMP)
    output = root / "cli"
    command = [sys.executable, "-B", str(Path(corpus.__file__).resolve())]
    imported = subprocess.run(command + ["import", "--manifest", str(manifest),
                              "--local-dir", str(local), "--artifact-dir", str(output)],
                              capture_output=True, text=True, check=False)
    (root / "cli-import.log").write_text(imported.stdout + imported.stderr)
    assert imported.returncode == 0, imported.stderr
    assert '"verdict": "FROZEN"' in imported.stdout
    pinned = corpus.sha256((output / "corpus.json").read_bytes())
    verified = subprocess.run(command + ["verify", "--corpus", str(output),
                              "--manifest-sha256", pinned],
                              capture_output=True, text=True, check=False)
    (root / "cli-verify.log").write_text(verified.stdout + verified.stderr)
    assert verified.returncode == 0, verified.stderr
    assert '"case_count": 1' in verified.stdout and '"verdict": "VERIFIED"' in verified.stdout
    rejected = subprocess.run(command + ["verify", "--corpus", str(output),
                             "--manifest-sha256", "0" * 64],
                             capture_output=True, text=True, check=False)
    (root / "cli-negative.log").write_text(rejected.stdout + rejected.stderr)
    assert rejected.returncode == 2 and "SHA256 mismatch" in rejected.stderr


def test_external_output_gate_and_duplicate_json(environment):
    root = environment[0]
    with pytest.raises(ValueError, match="outside worktree"):
        corpus.external(corpus.ROOT / "forbidden")
    path = root / "duplicate.json"
    path.write_text('{"cases": [], "cases": []}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        corpus.read_json(path)


@pytest.mark.parametrize("section", ["OBJSENSE", "OBJNAME", "QUADOBJ", "RHS", "BOUNDS"])
def test_empty_declaration_cannot_silently_degrade_to_lp(section):
    if section in ("OBJSENSE", "OBJNAME"):
        text = MPS.replace("ROWS", section + "\nROWS", 1)
    else:
        start = MPS.index(section + "\n") + len(section) + 1
        next_header = next(i for i in range(start, len(MPS))
                           if (i == 0 or MPS[i - 1] == "\n") and not MPS[i].isspace())
        text = MPS[:start] + MPS[next_header:]
    with pytest.raises(ValueError, match="empty"):
        corpus.audit_records(text.encode("ascii"))


def test_multi_case_membership_and_repeat_import_hashes(environment, monkeypatch):
    root, local, source, entry, manifest = environment
    monkeypatch.setattr(corpus, "SHARED_TMP", root.resolve())
    second = local / "second.qps"
    second.write_text(MPS.replace("OBJ -7", "OBJ -9"), encoding="ascii")
    other = dict(entry, case_id="second", path=second.name,
                 input_sha256=corpus.sha256(second.read_bytes()))
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [entry, other]}))
    first_output, second_output = root / "first", root / "second"
    corpus.freeze(manifest, local, first_output)
    corpus.freeze(manifest, local, second_output)
    first_report = corpus.read_json(first_output / "corpus.json")
    second_report = corpus.read_json(second_output / "corpus.json")
    assert first_report["case_count"] == 2 and len(first_report["cases"]) == 2
    assert first_report["cases"] == second_report["cases"]
    assert first_report["cases"][0]["model_sha256"] != first_report["cases"][1]["model_sha256"]
    assert first_report["source_hashes"] == {
        "scripts/public_qp_corpus.py": corpus.sha256(Path(corpus.__file__).read_bytes()),
    }
    pinned = corpus.sha256((first_output / "corpus.json").read_bytes())
    assert corpus.verify_frozen(first_output, pinned)["case_count"] == 2


def test_npz_corruption_during_freeze_is_error_not_frozen(environment, monkeypatch):
    root, local, source, entry, manifest = environment
    monkeypatch.setattr(corpus, "SHARED_TMP", root.resolve())
    original = np.savez

    def corrupt(stream, **arrays):
        arrays["offset"] = np.asarray(0.0)
        original(stream, **arrays)

    monkeypatch.setattr(np, "savez", corrupt)
    output = root / "corrupt-write"
    with pytest.raises(ValueError, match="NPZ roundtrip"):
        corpus.freeze(manifest, local, output)
    report = corpus.read_json(output / "corpus.json")
    assert report["verdict"] == "ERROR" and report["cases"] == []
    assert "NPZ roundtrip" in (output / "failure.log").read_text()
