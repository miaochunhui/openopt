"""Negative controls for strict CUDA evidence acceptance without GPU access."""

from pathlib import Path

import pytest

from scripts import gpu_release_check as check


def xml(tmp_path, counts, cases):
    path = tmp_path / "pytest.xml"
    path.write_text(f"<testsuites><testsuite {counts}>{cases}</testsuite></testsuites>")
    return path


def test_junit_complete_nonempty(tmp_path):
    path = xml(tmp_path, 'tests="1" failures="0" errors="0" skipped="0"',
               '<testcase name="cuda_case"/>')
    assert check.junit_counts(path) == {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}


@pytest.mark.parametrize("counts,cases", [
    ('tests="0"', ""),
    ('tests="2"', '<testcase name="one"/>'),
    ('tests="1" skipped="1"', '<testcase name="one"><skipped/></testcase>'),
    ('tests="1" failures="1"', '<testcase name="one"><failure/></testcase>'),
    ('tests="1" errors="1"', '<testcase name="one"><error/></testcase>'),
    ('tests="1"', '<testcase name="one"><failure/></testcase>'),
])
def test_junit_rejects_empty_partial_skipped_or_failed(tmp_path, counts, cases):
    with pytest.raises(ValueError):
        check.junit_counts(xml(tmp_path, counts, cases))


def test_missing_junit_is_not_success(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        check.junit_counts(tmp_path / "absent.xml")


@pytest.mark.parametrize("targets", [
    [], ["../test.py"], ["/tmp/test.py"], ["tests/../../test_x.py"],
    ["tests/test_absent.py"], ["scripts/gpu_release_check.py"],
])
def test_rejects_uncontrolled_test_selection(targets):
    with pytest.raises(ValueError):
        check.validate_targets(targets)


def test_valid_target_exists_and_node_selection_preserved():
    check.validate_targets(["tests/test_general_qp_cuda.py::test_cuda"])
    assert Path(check.ROOT / "tests/test_general_qp_cuda.py").is_file()
