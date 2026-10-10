"""Ensure release artifacts carry the selected license and detect bad wheels."""

from io import BytesIO
from pathlib import Path
import zipfile

import pytest

from scripts.check_distribution import validate_wheel_license

LICENSE = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes()


@pytest.mark.parametrize(
    "expression,license_file,payload,valid",
    [
        ("0BSD", "LICENSE", LICENSE, True),
        ("MIT", "LICENSE", LICENSE, False),
        ("", "LICENSE", LICENSE, False),
        ("0BSD", "OTHER", LICENSE, False),
        ("0BSD", "LICENSE", None, False),
        ("0BSD", "LICENSE", b"modified license", False),
    ],
)
def test_wheel_license_contract(expression, license_file, payload, valid):
    buffer = BytesIO()
    prefix = "openopt-0.1.0.dist-info"
    with zipfile.ZipFile(buffer, "w") as wheel:
        wheel.writestr(
            prefix + "/METADATA",
            "Metadata-Version: 2.4\nName: openopt\nVersion: 0.1.0\n"
            f"License-Expression: {expression}\nLicense-File: {license_file}\n\n",
        )
        if payload is not None:
            wheel.writestr(prefix + "/licenses/LICENSE", payload)
    with zipfile.ZipFile(buffer) as wheel:
        if valid:
            validate_wheel_license(wheel, LICENSE)
        else:
            with pytest.raises(ValueError):
                validate_wheel_license(wheel, LICENSE)


def test_source_license_is_zero_clause_without_attribution_condition():
    text = LICENSE.decode("ascii")
    assert text.startswith("BSD Zero Clause License\n")
    assert "for any\npurpose with or without fee is hereby granted." in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in text
    assert "provided that" not in text.lower()
    assert "must retain" not in text.lower()
