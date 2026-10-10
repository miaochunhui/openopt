"""Independent CPU audit of committed bounded-checker bytes, without solving.

Run this file through scripts/verify.py --target. Receipts are external pytest
artifacts. Exact rational arithmetic below is a test oracle, not the checker or
an optimizer capability. Default baselines were verified against trusted Git
before being frozen here; historical source is not executed on subsequent runs.
"""

from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import openopt.certificates as checker
from openopt.refinement import RefinementConfig


ROOT = Path(__file__).resolve().parents[1]
INITIAL_REVIEW_COMMIT = "0c5524e8afbfa067889c1696b8617d17d4cc90ce"
PARENT_COMMIT = "1d7cfe4b94aa66dbe6cb02e7e1e59dee05da94c2"
CHECKER_PATH = "src/openopt/certificates.py"
PROTECTED = (
    "docs/research/reproducible-checker.md",
    "scripts/accept_extension_cuda.py",
    CHECKER_PATH,
    "src/openopt/refinement.py",
    "tests/test_accept_extension_cuda.py",
    "tests/test_qp_certificates.py",
    "tests/test_qp_refinement.py",
)
MODE = checker.REPRODUCIBLE_ARITHMETIC
DEFAULT_EVIDENCE_SHA256 = (
    "770513b4ecfe093ac6a7aabf052a278fb349ae3e6bf06af7ada02dffb81948c3",
    "94da2de6f62886630f8ba4813481563f0f7f1a0f3cd377dbf329bb45851f66bc",
    "c792f632e741f361518816df16717c93b204844a6df41b6265b81a7a65e48236",
    "14d36ba759ca6b3d5ac4ee222326029e27fb902814974aac8df94934e2bcf94c",
    "590f59f224a1937fba44c60dce4dcd7190f513fe43061ac65bcce8c883972cfd",
    "2893aee854bb6520349e272cd675384b4df64ab83f66adf9fe6fd0f7a4317e8e",
)


def git_bytes(commit, path):
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise AssertionError(f"trusted committed source missing: {commit}:{path}")
    return result.stdout


@pytest.fixture(autouse=True)
def committed_source_guard():
    """Pin current committed bytes, so a committed repair can be audited too."""
    pins = {}
    for path in PROTECTED:
        committed = git_bytes("HEAD", path)
        assert (ROOT / path).read_bytes() == committed, f"uncommitted source: {path}"
        pins[path] = hashlib.sha256(committed).hexdigest()
    assert Path(checker.__file__).resolve() == (ROOT / CHECKER_PATH).resolve()
    yield pins
    for path, pin in pins.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == pin


def model(P, q, A=(), l=(), u=()):
    n = len(q)
    return SimpleNamespace(
        P=np.asarray(P, dtype=np.float64), q=np.asarray(q, dtype=np.float64),
        A=np.asarray(A, dtype=np.float64).reshape(-1, n),
        l=np.asarray(l, dtype=np.float64), u=np.asarray(u, dtype=np.float64),
    )


def exact_objective(p, x):
    values = [Fraction(float(v)) for v in x]
    return sum(
        Fraction(float(p.P[i, j])) * values[i] * values[j] / 2
        for i in range(len(values)) for j in range(len(values))
    ) + sum(Fraction(float(p.q[i])) * values[i] for i in range(len(values)))


def exact_dot(left, right):
    return sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(left, right))


def receipt(tmp_path, request, name, payload):
    path = tmp_path / f"{name}.json"
    identity = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert identity.returncode == 0, "current committed source identity unavailable"
    data = {
        "reviewed_commit": identity.stdout.strip(),
        "initial_review_commit": INITIAL_REVIEW_COMMIT,
        "checker_sha256": hashlib.sha256((ROOT / CHECKER_PATH).read_bytes()).hexdigest(),
        "scope": "independent CPU numerical audit; no solve, GPU or exact-proof claim",
        **payload,
    }
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    request.node.user_properties.extend([
        ("independent_checker_receipt", str(path)),
        ("independent_checker_receipt_sha256", hashlib.sha256(path.read_bytes()).hexdigest()),
    ])
    return path


def test_rounded_zero_kernel_cannot_accept_exactly_indefinite_model(tmp_path, request):
    epsilon = 2. ** -52
    p = model([[1 + 4 * epsilon, 1 + 5 * epsilon],
               [1 + 5 * epsilon, 1 + 6 * epsilon]], [0., 0.])
    determinant = Fraction(float(p.P[0, 0])) * Fraction(float(p.P[1, 1])) - (
        Fraction(float(p.P[0, 1])) ** 2
    )
    descent = [float(2**52 + 5), -float(2**52 + 4)]
    objective = exact_objective(p, descent)
    assert determinant == -Fraction(1, 2**104)
    assert objective == -Fraction(float(p.P[0, 0])) / 2
    assert all(Fraction(v).denominator == 1 for v in descent)
    try:
        evidence = checker.check_qp_candidate(p, [0., 0.], [], arithmetic=MODE)
    except ValueError as error:
        receipt(tmp_path, request, "rounded-zero-kernel", {
            "model": {"P": p.P.tolist(), "q": p.q.tolist(), "A": [], "l": [], "u": []},
            "exact_determinant": str(determinant), "descent": descent,
            "exact_descent_objective": str(objective), "rejected": str(error),
        })
        return
    eigenvalues, vectors = checker._jacobi(p.P.tolist())
    columns = checker._bt(vectors, 2)
    fixed_residuals = [checker._bmv(p.P.tolist(), column) for column in columns]
    exact_residuals = [[str(exact_dot(row, column)) for row in p.P] for column in columns]
    path = receipt(tmp_path, request, "rounded-zero-kernel", {
        "model": {"P": p.P.tolist(), "q": p.q.tolist(), "A": [], "l": [], "u": []},
        "candidate": {"x": [0., 0.], "y": []},
        "exact_determinant": str(determinant), "descent": descent,
        "exact_descent_objective": str(objective), "eigenvalues": eigenvalues,
        "vectors_hex": [[v.hex() for v in row] for row in vectors],
        "fixed_original_P_residuals": fixed_residuals,
        "exact_original_P_residuals": exact_residuals,
        "evidence": evidence.to_dict(),
    })
    assert not evidence.passed, f"nonconvex model was accepted; exact-byte receipt={path}"


@pytest.mark.parametrize("kind", ["early_stop", "rounded_spd", "tiny_diagonal"])
def test_previous_spectral_counterexamples_are_rejected(kind, tmp_path, request):
    epsilon = 2. ** -52
    if kind == "early_stop":
        p = model([[0., epsilon], [epsilon, 1.]], [0., 0.])
    elif kind == "rounded_spd":
        p = model([[1., 1.], [1., 1. + epsilon]], [2. ** -28, -2. ** -28])
    else:
        p = model([[1., 0.], [0., 1e-16]], [0., 0.])
    evidence = checker.check_qp_candidate(p, [0., 0.], [], arithmetic=MODE)
    receipt(tmp_path, request, kind, {"evidence": evidence.to_dict()})
    assert not evidence.passed and not evidence.checks["dual_objective_available"]


@pytest.mark.parametrize("kind", ["zero", "singular", "one_dim", "equality_positive", "equality_negative"])
def test_known_psd_primal_dual_and_gap(kind, tmp_path, request):
    if kind == "zero":
        p, x, y = model([[0.]], [0.], [[0.]], [-np.inf], [np.inf]), [3.], [0.]
    elif kind == "singular":
        p, x, y = model([[1., 1.], [1., 1.]], [-2., -2.]), [3., -1.], []
    elif kind == "one_dim":
        p, x, y = model([[2.]], [-4.]), [2.], []
    elif kind == "equality_positive":
        p, x, y = model([[2.]], [-3.], [[1.]], [1.], [1.]), [1.], [1.]
    else:
        p, x, y = model([[2.]], [-1.], [[1.]], [1.], [1.]), [1.], [-1.]
    evidence = checker.check_qp_candidate(p, x, y, arithmetic=MODE)
    receipt(tmp_path, request, kind, {"evidence": evidence.to_dict(),
                                   "exact_objective": str(exact_objective(p, x))})
    assert evidence.passed and all(evidence.checks.values())
    assert evidence.metrics["primal_objective"] == float(exact_objective(p, x))
    assert evidence.metrics["dual_objective"] == pytest.approx(
        float(exact_objective(p, x)), rel=0, abs=1e-14,
    )
    assert abs(evidence.metrics["duality_gap"]) <= 1e-14


@pytest.mark.parametrize("kind", ["free_positive", "free_negative", "lower_wrong", "upper_wrong"])
def test_every_nonzero_wrong_infinite_support_sign_rejects(kind, tmp_path, request):
    tiny = 2. ** -100
    if kind == "free_positive":
        low, high, multiplier = -np.inf, np.inf, tiny
    elif kind == "free_negative":
        low, high, multiplier = -np.inf, np.inf, -tiny
    elif kind == "lower_wrong":
        low, high, multiplier = 0., np.inf, tiny
    else:
        low, high, multiplier = -np.inf, 0., -tiny
    p = model([[2.]], [0.], [[1.]], [low], [high])
    evidence = checker.check_qp_candidate(p, [0.], [multiplier], arithmetic=MODE)
    receipt(tmp_path, request, kind, {"evidence": evidence.to_dict()})
    assert not evidence.passed and not evidence.checks["bound_sign"]
    assert evidence.metrics["bound_sign"] == tiny
    assert evidence.metrics["dual_objective"] is None
    assert evidence.metrics["duality_gap"] is None


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_correct_one_sided_lp_support(side):
    if side == "lower":
        p, x, y, objective = model([[0.]], [1.], [[1.]], [1.], [np.inf]), [1.], [-1.], 1.
    else:
        p, x, y, objective = model([[0.]], [-1.], [[1.]], [-np.inf], [1.]), [1.], [1.], -1.
    evidence = checker.check_qp_candidate(p, x, y, arithmetic=MODE)
    assert evidence.passed
    assert evidence.metrics["primal_objective"] == evidence.metrics["dual_objective"] == objective
    assert evidence.metrics["duality_gap"] == 0.


@pytest.mark.parametrize("sign", [-1, 1])
def test_gap_keeps_sign_and_rejects_both_directions(sign, tmp_path, request):
    p = model([[0.]], [-1.], [[1.]], [-np.inf], [1.])
    delta = 2. ** -20
    x = [1. - sign * delta]
    evidence = checker.check_qp_candidate(p, x, [1.], arithmetic=MODE)
    receipt(tmp_path, request, f"gap-{sign}", {"evidence": evidence.to_dict()})
    assert evidence.checks["stationarity"]
    assert evidence.metrics["duality_gap"] == sign * delta
    assert not evidence.passed and not evidence.checks["duality_gap"]


def test_original_range_residual_cannot_be_replaced_with_stationarity():
    p = model([[2., 0.], [0., 0.]], [-4., 1.])
    evidence = checker.check_qp_candidate(p, [2., 0.], [], arithmetic=MODE)
    assert not evidence.passed and evidence.metrics["range_residual"] == 1.
    assert evidence.metrics["dual_objective"] is None
    assert "outside numerical range" in evidence.unavailable["dual_objective"]


def test_unambiguously_negative_curvature_fails_loud():
    p = model([[1., 0.], [0., -.125]], [0., 0.])
    with pytest.raises(ValueError, match="positive semidefinite"):
        checker.check_qp_candidate(p, [0., 0.], [], arithmetic=MODE)


def test_default_evidence_and_config_match_verified_preimplementation_baseline(tmp_path, request):
    cases = [
        (model([[2.]], [-4.]), [2.], []),
        (model([[0.]], [-1.], [[1.]], [-np.inf], [1.]), [1.], [1.]),
        (model([[0.]], [1.], [[1.]], [1.], [np.inf]), [1.], [-1.]),
        (model([[1., 1.], [1., 1.]], [-2., -2.]), [3., -1.], []),
        (model([[2.]], [0.], [[1.]], [-np.inf], [np.inf]), [0.], [2. ** -100]),
        (model([[1., 0.], [0., 1e-16]], [0., 0.]), [0., 0.], []),
    ]
    rows = []
    for (p, x, y), expected in zip(cases, DEFAULT_EVIDENCE_SHA256):
        snapshots = {name: getattr(p, name).tobytes() for name in ("P", "q", "A", "l", "u")}
        before = json.dumps(checker.check_qp_candidate(p, x, y).to_dict(), sort_keys=True)
        checker.check_qp_candidate(p, x, y, arithmetic=MODE)
        after = json.dumps(checker.check_qp_candidate(p, x, y).to_dict(), sort_keys=True)
        explicit = json.dumps(checker.check_qp_candidate(p, x, y, arithmetic="numpy").to_dict(), sort_keys=True)
        assert before == after == explicit
        assert hashlib.sha256(before.encode()).hexdigest() == expected
        assert snapshots == {name: getattr(p, name).tobytes() for name in snapshots}
        rows.append(expected)
    config = asdict(RefinementConfig())
    assert config.pop("check_arithmetic") == "numpy"
    assert config == {
        "device": "cpu", "dtype": checker.torch.float64, "factor_dtype": checker.torch.float64,
        "max_refinement_steps": 3, "active_atol": 1e-5, "active_rtol": 1e-7,
        "check_atol": 1e-8, "check_rtol": 1e-8,
    }
    receipt(tmp_path, request, "legacy-default", {
        "legacy_commit": PARENT_COMMIT,
        "legacy_checker_sha256": "e7ece6f726bb7613d504ff6d84eedc20fb1a5879213a653680a26f8a64eb4c9c",
        "exact_json_matches": len(rows), "evidence_sha256": rows,
        "default_config_preserved_except_new_opt_in_field": True,
    })
