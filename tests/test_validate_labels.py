"""Unit tests for the YOLO label validator."""

import pytest

from src.data.validate_labels import validate_label_line


def test_valid_line():
    cls, err = validate_label_line("3 0.5 0.5 0.2 0.3")
    assert err is None
    assert cls == 3


def test_wrong_field_count():
    _, err = validate_label_line("3 0.5 0.5 0.2")
    assert err is not None and "5 fields" in err


def test_class_out_of_range():
    _, err = validate_label_line("10 0.5 0.5 0.2 0.3")
    assert err is not None and "out of range" in err


def test_negative_class():
    _, err = validate_label_line("-1 0.5 0.5 0.2 0.3")
    assert err is not None


@pytest.mark.parametrize(
    "line",
    [
        "0 1.5 0.5 0.2 0.3",  # xc > 1
        "0 0.5 -0.1 0.2 0.3",  # yc < 0
        "0 0.5 0.5 1.2 0.3",  # w > 1
    ],
)
def test_coords_out_of_unit_range(line):
    _, err = validate_label_line(line)
    assert err is not None and "[0,1]" in err


def test_non_positive_box():
    _, err = validate_label_line("0 0.5 0.5 0.0 0.3")
    assert err is not None and "non-positive" in err


def test_non_numeric():
    _, err = validate_label_line("0 a b c d")
    assert err is not None and "non-numeric" in err
