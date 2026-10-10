"""Bounded contract-review unit tests; actual nine-case evidence is external."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contract_review", ROOT / "scripts/review_gpu_proof_contract.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_ambiguous_json_fails(data):
    with pytest.raises(review.ReviewError):
        review.parse(data)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "", "a/../b", "a//b"])
def test_paths_cannot_escape(tmp_path, name):
    with pytest.raises(review.ReviewError):
        review.at(tmp_path, name)


@pytest.mark.parametrize("mutation", ["bytes", "bytes-count", "hash", "symlink"])
def test_pinned_descriptor_rejects_tamper(tmp_path, mutation):
    path = tmp_path / "result.json"
    review.write(path, {"value": 1})
    item = review.descriptor(tmp_path, path)
    if mutation == "bytes":
        path.write_bytes(b"changed")
    elif mutation == "bytes-count":
        item["bytes"] = True
    elif mutation == "hash":
        item["sha256"] = "0" * 64
    else:
        target = tmp_path / "target"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    with pytest.raises(review.ReviewError):
        review.bound(tmp_path, item)


def test_rejected_requires_exact_exception_and_first_cause():
    def bad():
        raise ValueError("actual first cause")
    assert review.rejected(bad, ValueError, "actual first")["verdict"] == "REJECTED"
    with pytest.raises(review.ReviewError, match="wrong first cause"):
        review.rejected(bad, ValueError, "different")
    with pytest.raises(review.ReviewError, match="accepted forbidden"):
        review.rejected(lambda: "PASS", ValueError, "expected")
    with pytest.raises(ValueError):
        review.rejected(bad, TypeError, "actual")


def test_dispatch_load_does_not_execute_module_or_cluster(tmp_path):
    functions = "\n".join(f"def {name}(*args, **kwargs):\n    return 1"
                          for name in review.SELECTED_FUNCTIONS)
    source = ("raise AssertionError('module execution forbidden')\n"
              "def submit():\n    raise AssertionError('cluster forbidden')\n" + functions)
    dispatch = review.dispatch_functions(source, SimpleNamespace(ROOT=tmp_path, CASES=review.CASES))
    assert dispatch.collected_result({}, {}) == 1
    assert not hasattr(dispatch, "submit")


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_dispatch_missing_or_duplicate_function_fails(kind, tmp_path):
    names = list(review.SELECTED_FUNCTIONS)
    if kind == "missing":
        names.pop()
    else:
        names.append(names[0])
    source = "\n".join(f"def {name}(*args):\n    return None" for name in names)
    with pytest.raises(review.ReviewError, match="contract functions"):
        review.dispatch_functions(source, SimpleNamespace(ROOT=tmp_path, CASES=review.CASES))


def test_commit_requires_full_caller_pin(tmp_path):
    with pytest.raises(review.ReviewError, match="full commit"):
        review.commit_id(tmp_path, "275cef7")


def test_empty_inventory_is_not_success(tmp_path):
    with pytest.raises(review.ReviewError, match="empty inventory"):
        review.inventory(tmp_path)


def test_mutation_denominator_exact():
    with pytest.raises(review.ReviewError, match="exact four"):
        review.evaluate_mutations(None, None, {}, {}, create=False)


def test_replay_rejects_wrong_pin_before_import(tmp_path, monkeypatch):
    path = tmp_path / "bundle.json"
    review.write(path, {"schema_version": 1})
    monkeypatch.setattr(review, "external", lambda path: Path(path))
    monkeypatch.setattr(review, "load_module",
                        lambda *args: pytest.fail("bad pin must fail before module import"))
    with pytest.raises(review.ReviewError, match="caller pin mismatch"):
        review.replay(path, "0" * 64, tmp_path, "1" * 40, tmp_path, "2" * 40)


@pytest.mark.parametrize("mutation", ["add", "remove", "change"])
def test_replay_inventory_tamper_fails_before_import(tmp_path, monkeypatch, mutation):
    root = tmp_path / "evidence"
    root.mkdir()
    review.write(root / "review.json", {"verdict": "CONTRACT_REVIEW_PASS"})
    review.write(root / "original.json", {"original": True})
    bundle = {"schema_version": 1, "evidence_root": str(root),
              "files": review.inventory(root),
              "report": review.descriptor(root, root / "review.json")}
    path = tmp_path / "bundle.json"
    review.write(path, bundle)
    if mutation == "add":
        review.write(root / "added.json", {})
    elif mutation == "remove":
        (root / "review.json").unlink()
    else:
        (root / "review.json").write_bytes(b"modified")
    monkeypatch.setattr(review, "external", lambda path: Path(path))
    monkeypatch.setattr(review, "load_module",
                        lambda *args: pytest.fail("tamper must fail before module import"))
    with pytest.raises(review.ReviewError, match="added, removed or changed"):
        review.replay(path, review.sha(path.read_bytes()), tmp_path, "1" * 40, tmp_path, "2" * 40)


@pytest.mark.parametrize("reject_foreign", [False, True])
def test_loaded_module_probe_has_real_calls_and_restores_modules(tmp_path, reject_foreign):
    import sys
    import openopt
    import openopt.qp as qp
    import openopt.certificates as certificates

    class ProofError(ValueError):
        pass

    def validate(*args, **kwargs):
        from openopt.qp import QPConfig
        from openopt.certificates import check_primal_infeasibility, check_unbounded_direction
        if reject_foreign:
            assert sys.modules["openopt.qp"].__file__.endswith("foreign-loaded-qp.py")
            raise ProofError("loaded source outside reviewed root")
        QPConfig()
        # Genuine certificate functions on a small host model, not solver calls.
        import numpy as np
        model = SimpleNamespace(P=np.eye(1), q=np.zeros(1), A=np.eye(1),
                                l=np.zeros(1), u=np.ones(1))
        check_primal_infeasibility(model, np.ones(1))
        check_unbounded_direction(model, np.ones(1), feasible_x=np.zeros(1))
        return {"case_count": 9, "verdict": "CPU_SMOKE_REPLAY_PASS",
                "execution_verdict": "UNVERIFIED"}

    proof = SimpleNamespace(ProofError=ProofError, ROOT=tmp_path, validate_result=validate)
    result = tmp_path / "result.json"
    review.write(result, {"fixture": True})
    outcome = review.loaded_module_probe(proof, result, {})
    assert outcome["verdict"] == ("REJECTED" if reject_foreign else "ACCEPTED_FOREIGN_MODULES")
    assert outcome["calls"] == {
        "foreign_policy": 0 if reject_foreign else 1,
        "foreign_infeasible": 0 if reject_foreign else 1,
        "foreign_unbounded": 0 if reject_foreign else 1,
    }
    assert sys.modules["openopt.qp"] is qp and openopt.qp is qp
    assert sys.modules["openopt.certificates"] is certificates
    assert openopt.certificates is certificates
