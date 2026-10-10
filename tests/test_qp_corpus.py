"""Independent oracle tests and decoys; solver limitations must stay visible."""

from dataclasses import replace
from fractions import Fraction
import json

import numpy as np
import pytest
import torch

from openopt.corpus import (
    FAMILIES, build_corpus, check_infeasibility, check_unboundedness,
    generate_case, quality, validate_oracle,
)
from openopt.qp import QPConfig
from scripts.benchmark_general_qp import generate as original_generate
from scripts import qualify_qp_corpus as qualification


@pytest.mark.parametrize("n", [3, 7, 16])
@pytest.mark.parametrize("seed", [0, 83])
@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_dimensions_seed_repeatability(n, seed, family):
    case = generate_case(family, n, seed)
    assert validate_oracle(case)["verdict"] == "PASS"
    assert case.problem_hash == generate_case(family, n, seed).problem_hash
    assert case.problem_hash != generate_case(family, n, seed + 1).problem_hash
    metadata = case.metadata()
    assert metadata["class"] == family
    assert metadata["seed"] == seed
    assert len(metadata["problem_hash"]) == 64
    assert metadata["oracle"]["scope"]
    assert case.q.shape == (n,)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("n", [3, 8, 24])
def test_original_dense_is_byte_identical(n):
    p, x, y = original_generate(n, 20261010)
    case = generate_case("original_dense", n, 20261010)
    for name in ("P", "q", "A", "l", "u"):
        assert np.array_equal(getattr(case, name), getattr(p, name).numpy())
    assert np.array_equal(x, case.oracle.x)
    assert np.array_equal(y, case.oracle.y)


def _dot_exact(left, right):
    return sum((Fraction(float(a)) * Fraction(float(b)) for a, b in zip(left, right)), Fraction(0))


@pytest.mark.parametrize("family", [
    "dense_spd", "illconditioned", "psd_singular", "equality_redundant", "lp",
])
@pytest.mark.parametrize("n", [3, 7, 16])
def test_dyadic_witness_is_exact_without_solver_or_numpy_residual(family, n):
    case = generate_case(family, n, 83)
    x, y = case.oracle.x, case.oracle.y
    for index in range(n):
        stationarity = _dot_exact(case.P[index], x) + Fraction(float(case.q[index])) \
            + _dot_exact(case.A[:, index], y)
        assert stationarity == 0
    for row, lower, upper, multiplier in zip(case.A, case.l, case.u, y):
        value = _dot_exact(row, x)
        assert Fraction(float(lower)) <= value <= Fraction(float(upper))
        if multiplier > 0:
            assert value == Fraction(float(upper))
        if multiplier < 0:
            assert value == Fraction(float(lower))


@pytest.mark.parametrize("n", [3, 7, 16])
@pytest.mark.parametrize("seed", [0, 83])
def test_property_witnesses_are_exact_rational_not_reference_consensus(n, seed):
    infeasible = generate_case("infeasible", n, seed)
    y = infeasible.oracle.certificate
    assert all(_dot_exact(column, y) == 0 for column in infeasible.A.T)
    assert _dot_exact(y[y > 0], infeasible.u[y > 0]) \
        + _dot_exact(y[y < 0], infeasible.l[y < 0]) == -1
    unbounded = generate_case("unbounded", n, seed)
    d = unbounded.oracle.ray
    assert all(_dot_exact(row, d) == 0 for row in unbounded.P)
    assert all(_dot_exact(row, d) == 0 for row in unbounded.A)
    assert _dot_exact(unbounded.q, d) == -1


@pytest.mark.parametrize("n", [3, 7, 16])
def test_required_mathematical_structures(n):
    spd = generate_case("dense_spd", n)
    assert np.linalg.eigvalsh(spd.P).min() >= 1 - 1e-12
    assert np.count_nonzero(spd.P - np.diag(np.diag(spd.P))) > 0
    ill = generate_case("illconditioned", n)
    assert np.linalg.cond(ill.P) > 1e5
    singular = generate_case("psd_singular", n)
    assert np.linalg.matrix_rank(singular.P) == n // 2
    redundant = generate_case("equality_redundant", n)
    E = redundant.A[redundant.l == redundant.u]
    assert len(E) > np.linalg.matrix_rank(E)
    lp = generate_case("lp", n)
    assert np.count_nonzero(lp.P) == 0
    infeasible = generate_case("infeasible", n)
    assert np.all(infeasible.l <= infeasible.u)
    assert np.isneginf(infeasible.l[0]) and np.isposinf(infeasible.u[1])


@pytest.mark.parametrize("family", ["psd_singular", "lp"])
def test_nonunique_solution_is_not_rejected_for_witness_distance(family):
    case = generate_case(family, 8)
    x = case.oracle.x.copy()
    x[-1] += 0.5
    measured = quality(case, x, case.oracle.y)
    assert not case.oracle.unique
    assert measured["witness_distance"] == 0.5
    assert measured["relative_objective_gap"] == 0
    assert measured["verdict"] == "PASS"


@pytest.mark.parametrize("mutation", ["primal", "dual", "normal", "objective"])
def test_quality_negative_controls(mutation):
    case = generate_case("dense_spd", 8)
    x, y = case.oracle.x.copy(), case.oracle.y.copy()
    if mutation == "primal":
        x[0] += 2
        assert quality(case, x, y)["primal_residual"] > 1
    elif mutation == "dual":
        y[1] += 2
        assert quality(case, x, y)["dual_residual"] > 1
    elif mutation == "normal":
        y[0] = -1
        assert quality(case, x, y)["normal_cone_residual"] > 0.1
    else:
        # Loosen only residual gates to isolate the independent objective gate.
        x[1] += 0.25
        measured = quality(case, x, y, residual_tol=100, gap_tol=1e-12)
        assert measured["relative_objective_gap"] > 1e-12
        assert measured["verdict"] == "FAIL"
    if mutation != "objective":
        assert quality(case, x, y)["verdict"] == "FAIL"


@pytest.mark.parametrize("mutation", ["sign", "matrix", "bounds", "zero"])
def test_farkas_negative_controls(mutation):
    case = generate_case("infeasible", 8)
    y = case.oracle.certificate.copy()
    if mutation == "sign":
        y = -y
    elif mutation == "matrix":
        A = case.A.copy()
        A[1, 0] += 1
        case = replace(case, A=A)
    elif mutation == "bounds":
        l = case.l.copy()
        l[1] = -1
        case = replace(case, l=l)
    else:
        with pytest.raises(ValueError, match="zero"):
            check_infeasibility(case, y * 0)
        return
    assert check_infeasibility(case, y)["verdict"] == "FAIL"
    with pytest.raises(ValueError, match="invalid independent oracle"):
        validate_oracle(replace(case, oracle=replace(case.oracle, certificate=y)))


@pytest.mark.parametrize("mutation", ["slope", "curvature", "constraint", "base", "zero"])
def test_recession_negative_controls(mutation):
    case = generate_case("unbounded", 8, 83)
    d, x = case.oracle.ray.copy(), case.oracle.x.copy()
    if mutation == "slope":
        d = -d
    elif mutation == "curvature":
        P = case.P.copy()
        P[-1, -1] = 1
        case = replace(case, P=P)
    elif mutation == "constraint":
        A = case.A.copy()
        A[0, -1] = 1
        case = replace(case, A=A)
    elif mutation == "base":
        x[0] += 3
    else:
        with pytest.raises(ValueError, match="zero"):
            check_unboundedness(case, d * 0, x)
        return
    assert check_unboundedness(case, d, x)["verdict"] == "FAIL"


@pytest.mark.parametrize("mutation", ["nan", "empty", "wrong_shape", "bad_tolerance"])
def test_quality_invalid_inputs_fail_loud(mutation):
    case = generate_case("dense_spd", 8)
    x, y = case.oracle.x.copy(), case.oracle.y.copy()
    kwargs = {}
    if mutation == "nan":
        x[0] = np.nan
    elif mutation == "empty":
        x = np.array([])
    elif mutation == "wrong_shape":
        y = np.zeros(len(y) + 1)
    else:
        kwargs["residual_tol"] = np.inf
    with pytest.raises(ValueError):
        quality(case, x, y, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"sizes": []}, {"seeds": []}, {"families": []}, {"sizes": [3, 3]},
    {"sizes": [2]}, {"seeds": [-1]}, {"families": ["unknown"]}, {"sizes": [True]},
])
def test_corpus_invalid_or_empty_selection_fails(kwargs):
    with pytest.raises(ValueError):
        build_corpus(**kwargs)


def test_hash_includes_input_shape_bytes_and_owns_solver_copies():
    case = generate_case("dense_spd", 8)
    before = case.problem_hash
    p = case.problem()
    p.q.add_(1)
    assert case.problem_hash == before
    q = case.q.copy()
    q[0] += 1
    assert replace(case, q=q).problem_hash != before
    assert len({case.problem_hash for case in build_corpus([3, 7], [0, 83])}) == 32


@pytest.mark.parametrize("family", ["infeasible", "unbounded"])
def test_actual_dense_limit_is_unqualified_not_fake_certificate(family):
    case = generate_case(family, 3)
    raw = qualification.run_solver(case, "openopt", QPConfig(max_iterations=2, check_interval=1))
    assert raw["status"] == "max_iterations"
    assessed = qualification.assess(case, "openopt", raw, 5e-6, 1e-6)
    assert assessed["verdict"] == "UNQUALIFIED"
    assert "not a solver bug" in assessed["reason"]
    assert assessed["certificate_quality"] is None


@pytest.mark.parametrize("solver", qualification.SOLVERS)
def test_actual_original_dense_matched_quality(solver):
    case = generate_case("original_dense", 8)
    raw = qualification.run_solver(case, solver, QPConfig())
    assessed = qualification.assess(case, solver, raw, 5e-6, 1e-6)
    assert assessed["verdict"] == "QUALIFIED", assessed
    assert raw["observed_device"] == "cpu"
    if solver == "osqp":
        assert raw["settings"]["eps_abs"] == raw["settings"]["eps_rel"] == 1e-7
        assert raw["settings"]["eps_prim_inf"] == 1e-4


@pytest.mark.parametrize("family", ["infeasible", "unbounded"])
@pytest.mark.parametrize("solver", ["osqp", "clarabel"])
def test_actual_reference_certificate_is_independently_checked(family, solver):
    case = generate_case(family, 8, 83)
    raw = qualification.run_solver(case, solver, QPConfig())
    assessed = qualification.assess(case, solver, raw, 5e-6, 1e-6)
    assert assessed["status_matches_expected"], assessed
    certificate = assessed["certificate_quality"]
    assert certificate is not None
    # A genuine reference certificate failure must remain UNQUALIFIED, not
    # be hidden by widening tolerances or by skipping a known failing solver.
    expected = "QUALIFIED" if certificate["verdict"] == "PASS" else "UNQUALIFIED"
    assert assessed["verdict"] == expected, assessed
    if expected == "UNQUALIFIED":
        assert assessed["reason"] == "independent residual/objective/certificate check failed"


def test_invalid_reference_certificate_is_unqualified_even_with_correct_status():
    case = generate_case("infeasible", 8, 83)
    bad = case.oracle.certificate.copy()
    bad[2] = 0.01
    raw = {"status": "primal infeasible", "primal_certificate": bad}
    assessed = qualification.assess(case, "osqp", raw, 5e-6, 1e-6)
    assert assessed["status_matches_expected"]
    assert assessed["certificate_quality"]["annihilation_residual"] > 5e-6
    assert assessed["verdict"] == "UNQUALIFIED"


def test_false_solved_status_and_inaccurate_status_do_not_override_quality():
    case = generate_case("dense_spd", 8)
    raw = {"status": "solved", "x": case.oracle.x + 3, "y": case.oracle.y}
    assert qualification.assess(case, "openopt", raw, 5e-6, 1e-6)["verdict"] == "UNQUALIFIED"
    raw = {"status": "solved inaccurate", "x": case.oracle.x, "y": case.oracle.y}
    assert qualification.assess(case, "osqp", raw, 5e-6, 1e-6)["verdict"] == "UNQUALIFIED"


def test_clarabel_mapping_handles_infinite_bounds_and_redundant_equalities():
    for family in ("infeasible", "equality_redundant"):
        case = generate_case(family, 7)
        A, b, cones, mapping = qualification.clarabel_form(case)
        assert np.isfinite(A).all() and np.isfinite(b).all()
        assert len(b) == len(mapping) > 0 and cones
        for row, bound, (index, sign) in zip(A, b, mapping):
            assert np.array_equal(row, sign * case.A[index])
            assert bound == (case.u[index] if sign > 0 else -case.l[index])


def test_cli_single_case_has_own_seed_and_dimension():
    args = qualification.parser().parse_args([
        "--case", "lp-n7-seed83", "--artifact-dir", "/hf3fs-jd/data/tmp/unused",
    ])
    assert [case.case_id for case in qualification.select_cases(args)] == ["lp-n7-seed83"]
    args.case = ["not-a-case"]
    with pytest.raises(ValueError, match="case ID"):
        qualification.select_cases(args)


def test_cli_exception_and_empty_selection_preserve_first_cause(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "SHARED_TMP", tmp_path)
    output = tmp_path / "exception"

    def broken(*args):
        raise RuntimeError("negative-control-first-cause")

    monkeypatch.setattr(qualification, "run_solver", broken)
    assert qualification.main([
        "--case", "dense_spd-n3-seed0", "--solver", "openopt",
        "--artifact-dir", str(output),
    ]) == 2
    report = json.loads((output / "result.json").read_text())
    assert report["error"]["message"] == "negative-control-first-cause"
    assert report["error"]["case_id"] == "dense_spd-n3-seed0"
    assert report["error"]["solver"] == "openopt"
    assert "negative-control-first-cause" in (output / "failure.log").read_text()
    assert report["rows"] == []
    assert report["missing_rows"] == [{"case_id": "dense_spd-n3-seed0", "solver": "openopt"}]
    assert (output / "dense_spd-n3-seed0/problem.npz").stat().st_size > 0
    monkeypatch.setattr(qualification, "select_cases", lambda args: [])
    assert qualification.main(["--artifact-dir", str(tmp_path / "empty")]) == 2
    report = json.loads((tmp_path / "empty/result.json").read_text())
    assert report["verdict"] == "ERROR" and "empty corpus" in report["error"]["message"]


def test_cli_unqualified_is_nonzero_and_evidence_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "SHARED_TMP", tmp_path)
    output = tmp_path / "unqualified"
    argv = [
        "--case", "infeasible-n3-seed0", "--solver", "openopt",
        "--max-iterations", "2", "--artifact-dir", str(output),
    ]
    assert qualification.main(argv) == 1
    report = json.loads((output / "result.json").read_text())
    assert report["unqualified_count"] == 1 and report["qualified_count"] == 0
    assert report["missing_rows"] == []
    assert len(report["unresolved_rows"]) == 1
    before = (output / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        qualification.main(argv)
    assert (output / "result.json").read_bytes() == before


def test_explicit_cuda_unavailable_is_error_not_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "SHARED_TMP", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "cuda-denied"
    assert qualification.main([
        "--case", "dense_spd-n3-seed0", "--device", "cuda:0",
        "--artifact-dir", str(output),
    ]) == 2
    report = json.loads((output / "result.json").read_text())
    assert "explicit CUDA" in report["error"]["message"]
    assert report["rows"] == []


def test_cli_family_shard_has_matched_artifacts_and_real_independent_quality(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "SHARED_TMP", tmp_path)
    output = tmp_path / "family"
    assert qualification.main([
        "--family", "original_dense", "--size", "3", "--seed", "83",
        "--artifact-dir", str(output),
    ]) == 0
    report = json.loads((output / "result.json").read_text())
    assert len(report["rows"]) == 3 and report["qualified_count"] == 3
    assert report["missing_rows"] == report["unresolved_rows"] == []
    case = generate_case("original_dense", 3, 83)
    for row in report["rows"]:
        assert row["problem_hash"] == case.problem_hash
        assert row["verdict"] == "QUALIFIED"
        with np.load(output / case.case_id / f"{row['solver']}-vectors.npz") as vectors:
            assert quality(case, vectors["x"], vectors["y"]) == row["candidate_quality"]
    with np.load(output / case.case_id / "problem.npz") as frozen:
        for name in ("P", "q", "A", "l", "u"):
            assert np.array_equal(frozen[name], getattr(case, name))
    assert set(report["source_hashes"]) == {
        "src/openopt/corpus.py", "src/openopt/qp.py", "scripts/qualify_qp_corpus.py",
    }


def test_diagnostic_empty_failure_selection_is_error_not_zero_row_success(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "SHARED_TMP", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    qualification.write_json(source / "result.json", {"rows": []})
    output = tmp_path / "diagnosis"
    assert qualification.main([
        "--diagnose-osqp", str(source), "--artifact-dir", str(output),
    ]) == 2
    report = json.loads((output / "result.json").read_text())
    assert report["verdict"] == "ERROR"
    assert "no failed OSQP" in report["error"]["message"]
