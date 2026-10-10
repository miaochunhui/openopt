"""An installed wheel must contain the exact accepted source, not a subset."""

import io
import zipfile

import pytest

from scripts.check_distribution import validate_wheel_sources


def archive(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as wheel:
        for name, value in entries.items():
            wheel.writestr(name, value)
    return zipfile.ZipFile(buffer)


def test_wheel_source_bytes_and_bidirectional_membership(tmp_path):
    package = tmp_path / "src" / "openopt"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"VALUE = 1\n")
    with archive({"openopt/__init__.py": b"VALUE = 1\n"}) as wheel:
        assert validate_wheel_sources(wheel, tmp_path / "src") == 1


@pytest.mark.parametrize("entries", [
    {},
    {"openopt/__init__.py": b"VALUE = 2\n"},
    {"openopt/__init__.py": b"VALUE = 1\n", "openopt/extra.py": b"VALUE = 1\n"},
])
def test_wheel_source_mutations_rejected(tmp_path, entries):
    package = tmp_path / "src" / "openopt"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"VALUE = 1\n")
    with archive(entries) as wheel, pytest.raises(ValueError):
        validate_wheel_sources(wheel, tmp_path / "src")
