"""Guard object identity and the independent package boundary."""

import importlib.metadata
from pathlib import Path

import pytest
import openopt
import openpo


def test_openopt_reexports_the_entire_existing_api() -> None:
    assert set(openpo.__all__).issubset(openopt.__all__)
    for name in openpo.__all__:
        assert getattr(openopt, name) is getattr(openpo, name)


def test_implementation_is_loaded_from_this_checkout() -> None:
    root = Path(__file__).resolve().parents[1]
    assert Path(openpo.__file__).resolve() == root / "src/openpo/__init__.py"
    assert Path(openopt.__file__).resolve() == root / "src/openopt/__init__.py"


def test_no_general_purpose_api_is_falsely_advertised() -> None:
    assert not hasattr(openopt, "minimize")


def test_version_matches_package_metadata_when_installed() -> None:
    try:
        version = importlib.metadata.version("openopt")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("distribution metadata unavailable in source-only execution")
    assert version == openopt.__version__
