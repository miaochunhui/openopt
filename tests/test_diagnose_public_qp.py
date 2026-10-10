"""Offline synthetic diagnostic controls; no solver or public coverage claims."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import diagnose_public_qp as gate
from scripts import public_qp_corpus as importer
from scripts import qualify_public_qp as runner

LP = b"""NAME DIAGNOSTIC_FIXTURE
ROWS
 N COST
 L CAP
COLUMNS
 X COST -1 CAP 1
RHS
 RHS1 CAP 2
ENDATA
"""


def model(q, A, l, u):
    q = np.asarray(q, dtype=float)
    return {"arrays": {"P": np.zeros((len(q), len(q))), "q": q, "A": np.asarray(A, dtype=float),
                       "l": np.asarray(l, dtype=float), "u": np.asarray(u, dtype=float),
                       "variable_lower": np.full(len(q), -np.inf),
                       "variable_upper": np.full(len(q), np.inf), "offset": np.asarray(0.0)},
            "objective_sense": "minimize", "variable_names": ["x" + str(i) for i in range(len(q))],
            "row_names": ["r" + str(i) for i in range(len(l))]}


@pytest.fixture
def checker():
    return gate.load_code(gate.ROOT / "src/openopt/certificates.py", "_diagnosis_unit_checker")


def compare(m, x, y, checker, atol=1e-7, rtol=1e-7):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    measured = gate.independent_lp(m, x, y, atol, rtol)
    expected = checker.check_qp_candidate(runner.problem(importer.solver_arrays(m)), x, y,
                                          atol=atol, rtol=rtol).to_dict()
    gate.assert_candidate_equal(measured["candidate"], expected)
    return measured


def test_tiny_wrong_sign_infinite_support_remains_failed(checker):
    m = model([0], [[1]], [0], [np.inf])
    measured = compare(m, [0], [1e-17, 0], checker)
    assert measured["failed_components"]["bound_sign"] == [0]
    assert measured["constraint_and_bound_rows"][0]["y"] == 1e-17
    assert measured["candidate"]["metrics"]["bound_sign"] == 1e-17
    assert measured["candidate"]["thresholds"]["bound_sign"] == 0
    assert measured["candidate"]["metrics"]["dual_objective"] is None
    assert not measured["candidate"]["passed"]


def test_component_complementarity_not_unrelated_maximum(checker):
    m = model([-1, -1], np.eye(2), [-np.inf, -np.inf], [0, 1e9])
    measured = compare(m, [-5e-7, 1e9], [1, 1, 0, 0], checker)
    assert measured["candidate"]["metrics"]["complementarity"] < (
        measured["candidate"]["thresholds"]["complementarity_max"])
    assert measured["maxima_can_mask_failure"]["complementarity"]
    assert measured["failed_components"]["complementarity"] == [0]
    assert measured["constraint_and_bound_rows"][0]["complementarity"]["ratio"] > 1


def test_lp_range_is_necessary_not_stationarity_scaled_proxy(checker):
    m = model([-1000], [[1]], [-np.inf], [1])
    measured = compare(m, [1], [1000 + 1e-5, 0], checker)
    assert measured["candidate"]["checks"]["stationarity"]
    assert not measured["candidate"]["checks"]["dual_objective_available"]
    assert measured["variables"][0]["lp_range_ratio"] > 1
    assert measured["candidate"]["unavailable"]["dual_objective"].endswith("range(P)")


def test_global_stationarity_threshold_does_not_mask_zero_cost_variable(checker):
    m = model([-1000, 0], np.eye(2), [-np.inf, -np.inf], [1, 1])
    measured = compare(m, [1, 1], [1000, 2e-6, 0, 0], checker)
    assert measured["maxima_can_mask_failure"]["stationarity"]
    assert measured["failed_components"]["stationarity"] == [1]
    assert not measured["candidate"]["passed"]


def test_good_lp_and_signed_negative_numerical_gap_preserved(checker):
    m = model([-1], [[1]], [-np.inf], [1])
    measured = compare(m, [1], [1 - 1e-9, 0], checker)
    assert measured["candidate"]["passed"]
    assert measured["candidate"]["metrics"]["duality_gap"] < 0


@pytest.mark.parametrize("bad", ["shape", "nan", "qp", "tolerance"])
def test_malformed_or_out_of_scope_candidate_rejected(bad):
    m = model([0], [[1]], [0], [np.inf])
    x, y, atol = np.asarray([0.0]), np.asarray([0.0, 0.0]), 1e-7
    if bad == "shape":
        y = y[:1]
    elif bad == "nan":
        y[0] = np.nan
    elif bad == "qp":
        m["arrays"]["P"][0, 0] = 1
    else:
        atol = 0
    with pytest.raises(ValueError):
        gate.independent_lp(m, x, y, atol, 1e-7)


@pytest.fixture
def captured(tmp_path, monkeypatch, checker):
    monkeypatch.setattr(importer, "SHARED_TMP", tmp_path.resolve())
    root = tmp_path / "captured"
    root.mkdir()
    local = tmp_path / "local"
    local.mkdir()
    declarations = []
    for case in ("afiro", "adlittle"):
        path = local / (case + ".mps")
        path.write_bytes(LP)
        declarations.append({"case_id": case, "path": path.name, "input_sha256": gate.sha(LP),
                             "source_url": "https://example.invalid/diagnostic-fixture.mps",
                             "license": "synthetic diagnostic fixture, not public coverage",
                             "origin": "synthetic"})
    manifest = tmp_path / "input.json"
    importer.write_json(manifest, {"schema_version": 1, "cases": declarations})
    importer.freeze(manifest, local, root / "frozen")
    (root / "code").mkdir()
    for target, source in (
        ("importer.py", Path(importer.__file__)), ("runner.py", Path(runner.__file__)),
        ("certificates.py", gate.ROOT / "src/openopt/certificates.py"),
        ("qp.py", gate.ROOT / "src/openopt/qp.py"),
    ):
        (root / "code" / target).write_bytes(source.read_bytes())
    frozen = importer.read_json(root / "frozen/corpus.json")
    config = {"eps_abs": 1e-7, "eps_rel": 1e-7, "direction_atol": 1e-8, "direction_rtol": 1e-8}
    rows = []
    for entry in frozen["cases"]:
        m = runner.load_model(root / "frozen", entry["case_id"])
        for solver in gate.SOLVERS:
            vectors = {"x": np.asarray([2.0]), "y": np.asarray([1.0, 0.0]),
                       "reported_objective": np.asarray(-2.0)}
            raw = {"status": "Solved" if solver == "clarabel" else "solved", "iterations": 1,
                   "settings": {"fixture": True}, "vectors": vectors}
            assessed = runner.assess(m, solver, raw, config, checker)
            relative = f"attempts/{entry['case_id']}/{solver}.npz"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            runner.save_vectors(path, vectors)
            row = {"case_id": entry["case_id"], "solver": solver, "model_sha256": entry["model_sha256"],
                   "attempted": True, "attempt": 1, "denominator": 1, "raw": {
                       key: value for key, value in raw.items() if key != "vectors"},
                   "assessment": assessed, "verdict": assessed["verdict"], "vectors_path": relative}
            rows.append(row)
            importer.write_json(path.with_suffix(".json"), row)
    eligible = runner.source_eligibility(frozen, root / "frozen", None,
                                         gate.sha((root / "frozen/corpus.json").read_bytes()))
    report = {"schema_version": 1, "runner_version": runner.VERSION,
              "corpus_sha256": gate.sha((root / "frozen/corpus.json").read_bytes()),
              "max_dense_bytes": importer.DEFAULT_MAX_DENSE_BYTES, "rows": rows, "config": config,
              "source_eligibility": eligible, **runner.summarize(rows, eligible, config),
              "files": runner.inventory(root)}
    importer.write_json(root / "result.json", report)
    return root / "result.json"


def test_captured_candidate_diagnosis_roundtrip_zero_solves(captured, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("diagnostic attempted solve")

    monkeypatch.setattr(runner, "run_solver", forbidden)
    output = tmp_path / "diagnosed"
    status = gate.produce(captured, gate.sha(captured.read_bytes()), output)
    diagnosis = importer.read_json(output / "diagnosis.json")
    assert diagnosis["candidate_count"] == 6
    summary = importer.read_json(output / "summary.json")
    assert summary["attempts"] == diagnosis["compact_summary"]
    assert all(row["original_checker_agrees"] for row in diagnosis["diagnoses"])
    assert diagnosis["original_verdict"] == "UNQUALIFIED"
    assert not diagnosis["qualification_upgraded"]
    assert importer.read_json(output / "status.json") == status
    assert gate.verify(output / "diagnosis.json", status["diagnosis_sha256"])["candidate_count"] == 6
    with pytest.raises(FileExistsError):
        gate.produce(captured, gate.sha(captured.read_bytes()), output)


@pytest.mark.parametrize("mutation", ["caller-pin", "npz", "npz-resealed", "rules",
                                     "membership", "row", "status", "aggregate"])
def test_saved_evidence_tamper_negative_controls(captured, mutation):
    pin = gate.sha(captured.read_bytes())
    root = captured.parent
    if mutation == "caller-pin":
        pin = "0" * 64
    elif mutation == "npz-resealed":
        path = root / "attempts/afiro/osqp.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["x"][0] = 3.0
        np.savez(path, **arrays)
        report = importer.read_json(captured)
        report["files"]["attempts/afiro/osqp.npz"] = {
            "bytes": path.stat().st_size, "sha256": gate.sha(path.read_bytes())}
        captured.write_text(json.dumps(report, sort_keys=True))
        pin = gate.sha(captured.read_bytes())
    elif mutation in ("npz", "rules"):
        path = root / ("attempts/afiro/osqp.npz" if mutation == "npz" else "code/certificates.py")
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "membership":
        (root / "unlisted.json").write_bytes(b"{}")
    else:
        report = importer.read_json(captured)
        if mutation == "row":
            report["rows"].reverse()
        elif mutation == "aggregate":
            report["attempted_count"] += 1
        else:
            report["rows"][0]["assessment"]["strict_status"] = False
        captured.write_text(json.dumps(report, sort_keys=True))
        pin = gate.sha(captured.read_bytes())
    with pytest.raises(ValueError):
        gate.build_diagnosis(captured, pin)


def test_diagnosis_resealed_claim_does_not_upgrade_qualification(captured, tmp_path):
    status = gate.produce(captured, gate.sha(captured.read_bytes()), tmp_path / "out")
    path = tmp_path / "out/diagnosis.json"
    report = importer.read_json(path)
    report["qualification_upgraded"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differs"):
        gate.verify(path, gate.sha(path.read_bytes()))
    assert status["original_verdict"] == "UNQUALIFIED"


def test_compact_summary_tamper_is_not_an_independent_claim(captured, tmp_path):
    status = gate.produce(captured, gate.sha(captured.read_bytes()), tmp_path / "summary-out")
    path = tmp_path / "summary-out/summary.json"
    report = importer.read_json(path)
    report["attempts"][0]["original_verdict"] = "FALSE_UPGRADE"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="compact summary"):
        gate.verify(path.parent / "diagnosis.json", status["diagnosis_sha256"])


def test_failure_retains_pin_first_cause(captured, tmp_path):
    with pytest.raises(ValueError, match="caller result"):
        gate.produce(captured, "0" * 64, tmp_path / "failed")
    status = importer.read_json(tmp_path / "failed/status.json")
    assert status["verdict"] == "ERROR" and status["exit_code"] == 2
    assert (tmp_path / "failed/failure.log").is_file()


def test_duplicate_json_and_symlink_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        gate.object_json(b'{"a":1,"a":2}')
    source = tmp_path / "source"
    source.write_bytes(b"x")
    (tmp_path / "link").symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        gate.read_bytes(tmp_path.resolve(), "link")
