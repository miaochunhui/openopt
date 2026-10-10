"""Regression for identity-preserving mutations of the retained HSD guard."""

import numpy as np
import pytest
import torch

import openopt.hsd_lp as hsd
from openopt.conic import ConicProblem, ZeroCone


def disabled_binding(buffers=None):
    return buffers


def equality():
    return ConicProblem(
        np.zeros((1, 1)), np.ones(1), np.ones((1, 1)),
        np.array([2.]), (ZeroCone(1),),
    )


@pytest.mark.parametrize("mutation", ["code", "defaults", "replacement"])
@pytest.mark.parametrize("stage", ["after_prepare", "during_solve"])
def test_retained_guard_executable_checked_independently(monkeypatch, mutation, stage):
    prepared = hsd.prepare_hsd_lp(
        equality(), hsd.HSDLPConfig(enabled=True, max_iterations=1),
    )
    entry = hsd._loaded_binding
    original_code = entry.__code__
    original_linear = torch.linalg.lu_solve
    injections = []

    def alter():
        if mutation == "code":
            monkeypatch.setattr(entry, "__code__", disabled_binding.__code__)
            assert hsd._loaded_binding is entry and entry.__code__ is not original_code
        elif mutation == "defaults":
            monkeypatch.setattr(entry, "__defaults__", ({},))
            assert hsd._loaded_binding is entry and entry.__code__ is original_code
        else:
            monkeypatch.setattr(hsd, "_loaded_binding", disabled_binding)
            assert hsd._loaded_binding is not entry
        injections.append(stage)

    if stage == "after_prepare":
        alter()
    else:
        def during_linear(*args, **kwargs):
            value = original_linear(*args, **kwargs)
            if not injections:
                alter()
            return value
        monkeypatch.setattr(torch.linalg, "lu_solve", during_linear)
    expected = "before solve" if stage == "after_prepare" else "after solve"
    with pytest.raises(RuntimeError, match="binding entry code changed " + expected):
        prepared.solve()
    assert injections == [stage], "control must alter the actually invoked guard"


def test_unchanged_retained_guard_and_candidate_still_work():
    prepared = hsd.prepare_hsd_lp(
        equality(), hsd.HSDLPConfig(enabled=True, max_iterations=200),
    )
    expected_code = prepared._binding_entry_code
    result = prepared.solve()
    assert result.status == "solved_candidate"
    assert result.candidate_check.passed and abs(float(result.x[0]) - 2.) <= 1e-6
    assert hsd._loaded_binding.__code__ is expected_code
