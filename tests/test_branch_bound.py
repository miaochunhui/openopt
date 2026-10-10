"""Self-authored tiny MIQP exact oracles; no historical artifact dependency."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import itertools
import json
import math
from types import SimpleNamespace

import pytest
import torch

from openopt import branch_bound as bb


def problem(P, q, lower, upper, integer, A=None, l=None, u=None, dtype=torch.float64):
    n = len(q)
    def tensor(value):
        return torch.tensor(value, dtype=dtype)
    return bb.BoundedMIQPProblem(tensor(P), tensor(q),
        torch.empty((0, n), dtype=dtype) if A is None else tensor(A),
        tensor([] if l is None else l), tensor([] if u is None else u),
        tensor(lower), tensor(upper), torch.tensor(integer, dtype=torch.int64))


def config(**kwargs):
    return bb.BranchBoundConfig(enabled=True, **kwargs)


def replay(p, c, report):
    return bb.check_bounded_miqp_result(p, c, report, trusted_source=bb.source_sha256())


def pure():
    return problem([[2.]], [-1.], [0.], [2.], [1])


def test_opt_in_and_default_exports_unchanged(monkeypatch):
    import openopt
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("disabled must not solve"))
    with pytest.raises(ValueError, match="disabled"):
        bb.solve_bounded_miqp(pure())
    assert "solve_bounded_miqp" not in openopt.__all__


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pure_analytic_branch_and_exact_gap(dtype):
    p = pure()
    c = config(dtype=dtype, abs_gap=0.)
    r = bb.solve_bounded_miqp(p, c)
    assert r.status == "solved_bounded"
    assert r.report["summary"]["objective"] == "0/1"
    assert r.report["summary"]["gap_abs"] == "0/1"
    assert any(v["disposition"] == "branch" for v in r.report["nodes"])
    assert r.report["actual_qp_calls"] > 0
    assert r.x.device == torch.device("cpu") and r.x.dtype == torch.float64
    assert replay(p, c, r.report)["coverage_verdict"] == "PASS"


@pytest.mark.parametrize("kind", ["mixed", "singular", "lp"])
def test_mixed_singular_lp_analytic(kind):
    if kind == "mixed":
        p = problem([[4., -2.], [-2., 2.]], [-1., 0.], [0., 0.], [1., 1.], [1, 0])
        expected = Fraction(0)
    elif kind == "singular":
        p = problem([[1., 1.], [1., 1.]], [-1., -1.], [0., 0.], [1., 1.], [1, 0])
        expected = Fraction(-1, 2)
    else:
        p = problem([[0.]], [1.], [0.], [2.], [1], [[1.]], [1.], [math.inf])
        expected = Fraction(1)
    c = config(abs_gap=0.)
    result = bb.solve_bounded_miqp(p, c)
    assert result.status == "solved_bounded"
    assert Fraction(result.report["summary"]["objective"]) == expected
    assert Fraction(result.report["summary"]["gap_abs"]) == 0
    assert replay(p, c, result.report)["numerical_verdict"] == "PASS"


def test_exact_bound_not_approximate_qp_objective():
    m = bb._Model(pure())
    bound = bb._bound(m, m.box, [.5], [])
    assert bound == Fraction(-1, 4)
    assert bb._enclose(bound) == [-.25, -.25]
    assert bb._bound(m, m.box, [0.], []) == -2


def test_qp_iteration_limit_retained_not_promoted():
    c = config(max_nodes=1, qp_max_iterations=1, abs_gap=0.)
    r = bb.solve_bounded_miqp(pure(), c)
    assert r.status == "feasible"
    assert r.report["qp_iteration_limit_count"] == r.report["actual_qp_calls"] == 1
    assert r.report["nodes"][0]["proposal"]["status"] == "max_iterations"
    assert replay(pure(), c, r.report)["numerical_verdict"] == "PASS"


def test_infinite_support_sign_even_subnormal_rejected():
    p = problem([[0.]], [1.], [0.], [2.], [1], [[1.]], [1.], [math.inf])
    m = bb._Model(p)
    assert bb._bound(m, m.box, [0.], [-1.]) == 1
    assert bb._bound(m, m.box, [0.], [0.]) == 0
    with pytest.raises(ArithmeticError, match="infinite signed support"):
        bb._bound(m, m.box, [0.], [math.nextafter(0., 1.)])


@pytest.mark.parametrize("kind", ["empty_integer", "zero_row", "equality"])
def test_exact_infeasible_empty_zero_equality(kind, monkeypatch):
    if kind == "empty_integer":
        p = problem([[0.]], [0.], [.1], [.9], [1])
    elif kind == "zero_row":
        p = problem([[0.]], [0.], [0.], [1.], [1], [[0.]], [1.], [1.])
    else:
        p = problem([[0.]], [0.], [0.], [1.], [1], [[1.]], [.5], [.5])
    r = bb.solve_bounded_miqp(p, config())
    assert r.status == "infeasible_bounded" and r.x is None
    assert r.report["summary"]["gap_abs"] is None
    assert replay(p, config(), r.report)["coverage_verdict"] == "PASS"
    if kind != "equality":
        assert r.report["actual_qp_calls"] == 0


@pytest.mark.parametrize("limit", [0, 1])
def test_node_limits_keep_coverage_and_no_false_solved(limit):
    c = config(max_nodes=limit, abs_gap=0.)
    r = bb.solve_bounded_miqp(pure(), c)
    assert r.report["stop_reason"] == "node_limit"
    assert r.status == ("unknown" if limit == 0 else "feasible")
    assert r.report["visited_nodes"] == limit
    if limit == 0:
        assert r.report["actual_qp_calls"] == 0 and r.x is None
    assert replay(pure(), c, r.report)["coverage_verdict"] == "PASS"


def test_zero_wall_budget_no_qp(monkeypatch):
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("zero wall budget must not solve"))
    c = config(wall_seconds=0.)
    r = bb.solve_bounded_miqp(pure(), c)
    assert r.status == "unknown" and r.report["stop_reason"] == "wall_budget"
    assert r.report["actual_qp_calls"] == 0
    assert replay(pure(), c, r.report)["numerical_verdict"] == "PASS"


def test_mixed_leaf_approximate_primal_cannot_close():
    # Integer fixed, continuous minimizer 1/3 is not binary64 representable.
    p = problem([[0., 0.], [0., 6.]], [0., -2.], [0., 0.], [0., 1.], [1, 0])
    c = config(abs_gap=1.)
    r = bb.solve_bounded_miqp(p, c)
    assert r.status == "unknown"
    assert r.x is not None and r.report["nodes"][0]["disposition"] == "unresolved"
    assert Fraction(r.report["summary"]["gap_abs"]) > 0
    assert replay(p, c, r.report)["coverage_verdict"] == "PASS"


def test_unvisited_mixed_fixed_integer_leaf_large_gap_stays_unknown():
    p = problem([[4., -2.], [-2., 2.]], [-1., 0.], [0., 0.], [1., 1.], [1, 0])
    c = config(max_nodes=1, abs_gap=1.)
    result = bb.solve_bounded_miqp(p, c)
    assert result.report["stop_reason"] == "node_limit"
    assert any(node["disposition"] == "frontier" for node in result.report["nodes"])
    assert result.status == "unknown"
    assert replay(p, c, result.report)["coverage_verdict"] == "PASS"
    altered = deepcopy(result.report)
    altered["summary"]["status"] = "solved_bounded"
    with pytest.raises(ValueError, match="summary mismatch"):
        replay(p, c, altered)


def test_mixed_nonrepresentable_equality_no_incumbent():
    p = problem([[0., 0.], [0., 2.]], [0., 0.], [0., 0.], [0., 1.], [1, 0],
                [[0., 3.]], [1.], [1.])
    c = config()
    r = bb.solve_bounded_miqp(p, c)
    assert r.status == "unknown" and r.x is None
    assert r.report["nodes"][0]["disposition"] == "unresolved"
    assert r.report["summary"]["gap_abs"] is None
    assert replay(p, c, r.report)["numerical_verdict"] == "PASS"


@pytest.mark.parametrize("P", [[[1., 2.], [2., 4.]], [[1., 1.], [1., math.nextafter(1., 0.)]]])
def test_psd_scope_exact_not_spectral_tolerance(P):
    p = problem(P, [0., 0.], [0., 0.], [1., 1.], [1, 0])
    with pytest.raises(NotImplementedError, match="psd_scope_unproved"):
        bb.solve_bounded_miqp(p, config())


@pytest.mark.parametrize("kind", ["nonfinite", "asymmetric", "reversed", "bad_sign", "bad_integer", "dtype", "shape"])
def test_invalid_models_fail_loud(kind):
    p = pure()
    if kind == "nonfinite":
        p = replace(p, q=torch.tensor([math.nan], dtype=torch.float64))
    elif kind == "asymmetric":
        p = problem([[1., 1e-20], [0., 1.]], [0., 0.], [0., 0.], [1., 1.], [1, 0])
    elif kind == "reversed":
        p = replace(p, lower=torch.tensor([3.], dtype=torch.float64))
    elif kind == "bad_sign":
        p = problem([[0.]], [0.], [0.], [1.], [1], [[0.]], [math.inf], [math.inf])
    elif kind == "bad_integer":
        p = replace(p, integrality=torch.tensor([2]))
    elif kind == "dtype":
        p = replace(p, q=p.q.half())
    else:
        p = replace(p, upper=torch.empty(2, dtype=torch.float64))
    with pytest.raises((ValueError, TypeError)):
        bb.solve_bounded_miqp(p, config())


@pytest.mark.parametrize("kind", ["dimension", "infinite_box", "integer_range", "all_continuous"])
def test_valid_out_of_scope_explicit_unsupported(kind):
    p = pure()
    if kind == "dimension":
        p = problem(torch.eye(9).tolist(), [0.] * 9, [0.] * 9, [1.] * 9, [1] * 9)
    elif kind == "infinite_box":
        p = replace(p, upper=torch.tensor([math.inf], dtype=torch.float64))
    elif kind == "integer_range":
        p = replace(p, upper=torch.tensor([2.**24], dtype=torch.float64))
    else:
        p = replace(p, integrality=torch.tensor([0]))
    with pytest.raises(NotImplementedError, match="unsupported"):
        bb.solve_bounded_miqp(p, config())


def test_prepared_original_mutation_and_source_missing_fail(monkeypatch):
    p = pure()
    prepared = bb.prepare_bounded_miqp(p, config())
    p.q[0] += 1.
    with pytest.raises(ValueError, match="model changed"):
        prepared.solve()
    prepared = bb.prepare_bounded_miqp(pure(), config())
    monkeypatch.setattr(bb, "source_sha256", lambda: {})
    with pytest.raises(ValueError, match="source changed"):
        prepared.solve()


def test_loaded_dependency_root_mismatch_rejected(monkeypatch):
    monkeypatch.setattr(bb._qp_module, "__file__", "/missing/qp.py")
    with pytest.raises(ValueError, match="source root"):
        bb.prepare_bounded_miqp(pure(), config())


@pytest.mark.parametrize("mutation", ["bool_shape", "embedded_gpu_pass", "unvisited_candidate", "lost_witness", "lost_node_limit"])
def test_typed_binding_and_missing_evidence_fail(mutation):
    p, c = pure(), config(max_nodes=1)
    report = deepcopy(bb.solve_bounded_miqp(p, c).report)
    if mutation == "bool_shape":
        report["model"]["arrays"]["q"]["shape"] = [True]
    elif mutation == "embedded_gpu_pass":
        report["acceptance"]["execution_verdict"] = "PASS"
    elif mutation == "unvisited_candidate":
        report["nodes"][1]["candidates"] = deepcopy(report["nodes"][0]["candidates"][:1])
    elif mutation == "lost_witness":
        report["nodes"][1]["witnesses"] = []
    else:
        report["stop_reason"] = "tree_exhausted"
    with pytest.raises(ValueError):
        replay(p, c, report)


def test_negative_signed_gap_is_arithmetic_error():
    m = bb._Model(pure())
    with pytest.raises(ArithmeticError, match="negative signed global gap"):
        bb._summary(m, [{"bound": "1/1", "disposition": "singleton"}], [0.], config())


@pytest.mark.parametrize("mutation", ["bound", "candidate", "deleted_branch", "infinite_dual_sign", "gap", "source", "model", "device", "mixed_singleton"])
def test_coordinated_outer_rehash_cannot_replace_proof(mutation):
    p, c = pure(), config(abs_gap=0.)
    report = deepcopy(bb.solve_bounded_miqp(p, c).report)
    if mutation == "bound":
        report["nodes"][0]["bound"] = "100/1"
    elif mutation == "candidate":
        report["incumbent"] = [.5]
        report["summary"]["objective"] = "-1/4"
    elif mutation == "deleted_branch":
        report["nodes"].pop()
    elif mutation == "infinite_dual_sign":
        p = problem([[0.]], [1.], [0.], [2.], [1], [[1.]], [1.], [math.inf])
        report = deepcopy(bb.solve_bounded_miqp(p, c).report)
        report["nodes"][0]["witnesses"][0]["y"] = [math.nextafter(0., 1.)]
    elif mutation == "gap":
        report["summary"]["gap_abs"] = "-1/1"
    elif mutation == "source":
        report["source_sha256"] = {}
    elif mutation == "model":
        report["model"]["sha256"] = "0" * 64
    elif mutation == "device":
        report["nodes"][0]["proposal"]["actual_device"] = "cuda:0"
    else:
        p = problem([[0., 0.], [0., 6.]], [0., -2.], [0., 0.], [0., 1.], [1, 0])
        report = deepcopy(bb.solve_bounded_miqp(p, c).report)
        report["nodes"][0]["disposition"] = "singleton"
    # The attacker can regenerate a valid outer hash, not mathematical evidence.
    payload = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    assert len(hashlib.sha256(payload).hexdigest()) == 64
    with pytest.raises((ValueError, ArithmeticError)):
        replay(p, c, report)


def test_replay_no_solve_factory(monkeypatch):
    p, c = pure(), config()
    report = bb.solve_bounded_miqp(p, c).report
    def forbidden(*args, **kwargs):
        pytest.fail("replay must never execute solve/factory")
    monkeypatch.setattr(bb, "solve_qp", forbidden)
    monkeypatch.setattr(bb, "prepare_bounded_miqp", forbidden)
    monkeypatch.setattr(bb, "solve_bounded_miqp", forbidden)
    assert replay(p, c, report)["execution_verdict"] == "UNVERIFIED"


def test_exact_fraction_extremes_and_overflow_fail():
    tiny = math.nextafter(0., 1.)
    value = Fraction(tiny) / 2
    assert bb._enclose(value) == [0., tiny]
    assert bb._enclose(-value) == [-tiny, -0.]
    with pytest.raises(ArithmeticError, match="8192"):
        bb._checked(Fraction(1 << 8192))
    with pytest.raises(ArithmeticError, match="overflow"):
        bb._enclose(Fraction(1 << 1024))
    p = problem([[1e308]], [0.], [2.], [2.], [1])
    with pytest.raises(ArithmeticError, match="overflow"):
        bb.solve_bounded_miqp(p, config())


def test_finite_original_row_bound_cast_overflow_rejected(monkeypatch):
    p = problem([[2.]], [-1.], [0.], [2.], [1], [[1.]], [-math.inf], [1e300])
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("cast overflow must precede QP"))
    with pytest.raises(ArithmeticError, match="row-bound cast overflow"):
        bb.solve_bounded_miqp(p, config(dtype=torch.float32))


def test_exact_enumeration_only_test_oracle():
    for q in (-3., -1., 0., 2.):
        p = problem([[2., -1.], [-1., 2.]], [q, 1.], [-1., -1.], [2., 2.], [1, 1])
        values = []
        for x in itertools.product(range(-1, 3), repeat=2):
            # Independent two-dimensional polynomial, not production helper.
            a, b = map(Fraction, x)
            values.append(a*a - a*b + b*b + Fraction(q)*a + b)
        r = bb.solve_bounded_miqp(p, config(abs_gap=0.))
        assert r.status == "solved_bounded"
        assert Fraction(r.report["summary"]["objective"]) == min(values)
        assert Fraction(r.report["summary"]["gap_abs"]) == 0


def test_same_original_fixed_integer_continuous_clarabel_reference():
    import numpy as np
    from openopt.conic import ConicConfig, ConicProblem, NonnegativeCone, ZeroCone, solve_conic

    p = problem([[4., -2.], [-2., 2.]], [-1., 0.], [0., 0.], [1., 1.], [1, 0])
    r = bb.solve_bounded_miqp(p, config(abs_gap=0.))
    # Original objective, original box, integer face fixed at the actual incumbent.
    original = {name: getattr(p, name).clone() for name in bb.MODEL_FIELDS}
    A = np.vstack((np.array([[1., 0.]]), np.eye(2), -np.eye(2)))
    b = np.array([float(r.x[0]), 1., 1., 0., 0.])
    model = ConicProblem(p.P.numpy().copy(), p.q.numpy().copy(), A, b,
                         (ZeroCone(1), NonnegativeCone(4)))
    ref = solve_conic(model, ConicConfig(backend="clarabel_reference", device="cpu"))
    assert ref.status == "reference_solved"
    assert abs(ref.objective - float(Fraction(r.report["summary"]["objective"]))) < 1e-7
    assert np.max(np.abs(ref.x - r.x.numpy())) < 5e-4
    assert all(torch.equal(getattr(p, name), value) for name, value in original.items())
    # Continuous-face numerical control is not a licensed/global MIQP certificate.


@pytest.mark.parametrize("kind", ["nonfinite", "dtype", "device", "status"])
def test_bad_native_proposals_raise_no_retry(monkeypatch, kind):
    calls = []
    def broken(p, c):
        calls.append(c.device)
        x = torch.tensor([.5], dtype=c.dtype)
        y = torch.zeros(1, dtype=c.dtype)
        if kind == "nonfinite":
            x[:] = math.nan
        elif kind == "dtype":
            x = x.float()
        elif kind == "device":
            x = torch.empty(1, device="meta", dtype=c.dtype)
        return SimpleNamespace(x=x, y=y, status="infeasible" if kind == "status" else "solved", iterations=1)
    monkeypatch.setattr(bb, "solve_qp", broken)
    with pytest.raises(RuntimeError, match="QP proposal"):
        bb.solve_bounded_miqp(pure(), config())
    assert calls == ["cpu"]


def test_explicit_missing_cuda_no_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("must not call CPU QP"))
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        bb.solve_bounded_miqp(pure(), config(device="cuda:0"))


@pytest.mark.parametrize("kind", ["pure", "mixed", "singular", "lp", "node_limit", "unresolved_equality"])
def test_real_cpu_receipt(tmp_path, request, kind):
    p, c = pure(), config(abs_gap=0.)
    if kind == "mixed":
        p = problem([[4., -2.], [-2., 2.]], [-1., 0.], [0., 0.], [1., 1.], [1, 0])
    elif kind == "singular":
        p = problem([[1., 1.], [1., 1.]], [-1., -1.], [0., 0.], [1., 1.], [1, 0])
    elif kind == "lp":
        p = problem([[0.]], [1.], [0.], [2.], [1], [[1.]], [1.], [math.inf])
    elif kind == "node_limit":
        c = config(max_nodes=1, abs_gap=0.)
    elif kind == "unresolved_equality":
        p = problem([[0., 0.], [0., 2.]], [0., 0.], [0., 0.], [0., 1.], [1, 0], [[0., 3.]], [1.], [1.])
    original = {name: getattr(p, name).clone() for name in bb.MODEL_FIELDS}
    descriptor = bb._descriptor(p)
    r = bb.solve_bounded_miqp(p, c)
    assert all(torch.equal(getattr(p, name), v) for name, v in original.items())
    assert bb._descriptor(p) == descriptor == r.report["model"]
    path = tmp_path / "bounded-miqp-cpu.json"
    path.write_text(json.dumps(r.report, indent=2, allow_nan=False) + "\n")
    request.node.user_properties.extend([("receipt", str(path)),
        ("receipt_sha256", hashlib.sha256(path.read_bytes()).hexdigest())])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; MIQP proposal execution UNVERIFIED")
def test_cuda_frozen_bound_tree(tmp_path, request):
    p, c = pure(), config(device="cuda:0", abs_gap=0.)
    descriptor = bb._descriptor(p)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as trace:
        r = bb.solve_bounded_miqp(p, c)
        torch.cuda.synchronize(0)
    assert r.x.device == torch.device("cuda:0")
    assert descriptor == bb._descriptor(p) == r.report["model"]
    assert any(event.device_type == torch.autograd.DeviceType.CUDA for event in trace.events())
    assert r.report["actual_qp_calls"] > 0
    assert all(n["proposal"]["actual_device"] == "cuda:0" for n in r.report["nodes"] if n["proposal"])
    assert replay(p, c, r.report)["execution_verdict"] == "UNVERIFIED"
    path, trace_path = tmp_path / "bounded-miqp-cuda.json", tmp_path / "cuda-trace.json"
    path.write_text(json.dumps(r.report, indent=2, allow_nan=False) + "\n")
    trace.export_chrome_trace(str(trace_path))
    request.node.user_properties.extend([("receipt", str(path)), ("trace", str(trace_path)),
        ("receipt_sha256", hashlib.sha256(path.read_bytes()).hexdigest())])
