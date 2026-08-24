import numpy as np
import pytest
import torch

from mldft.utils.conversions import as_scalar_bool, as_scalar_float


def test_as_scalar_float_converts_scalar_values() -> None:
    assert as_scalar_float(1.25) == 1.25
    assert as_scalar_float(2) == 2.0
    assert as_scalar_float(True) == 1.0
    assert as_scalar_float(np.float64(3.5)) == 3.5
    assert as_scalar_float(np.array(4.5)) == 4.5
    assert as_scalar_float(torch.tensor([5.5])) == 5.5


def test_as_scalar_bool_converts_scalar_values() -> None:
    assert as_scalar_bool(True) is True
    assert as_scalar_bool(1) is True
    assert as_scalar_bool(0) is False
    assert as_scalar_bool(np.bool_(True)) is True
    assert as_scalar_bool(np.array(False)) is False
    assert as_scalar_bool(torch.tensor([1])) is True


def test_as_scalar_float_rejects_none() -> None:
    with pytest.raises(TypeError, match="bad must be a scalar float, got None"):
        as_scalar_float(None, name="bad")


def test_as_scalar_bool_rejects_none() -> None:
    with pytest.raises(TypeError, match="flag must be a scalar bool, got None"):
        as_scalar_bool(None, name="flag")


@pytest.mark.parametrize(
    "value",
    [torch.tensor([1.0, 2.0]), [1.0, 2.0], np.array([1.0, 2.0])],
)
def test_as_scalar_float_rejects_non_scalar_values(value) -> None:
    with pytest.raises(ValueError, match="scalar-like|exactly one value"):
        as_scalar_float(value, name="bad")


@pytest.mark.parametrize(
    "value",
    [torch.tensor([True, False]), [True, False], np.array([True, False])],
)
def test_as_scalar_bool_rejects_non_scalar_values(value) -> None:
    with pytest.raises(ValueError, match="scalar-like|exactly one value"):
        as_scalar_bool(value, name="flag")


def test_as_scalar_float_rejects_non_numeric_string() -> None:
    with pytest.raises(ValueError):
        as_scalar_float("not-a-number", name="bad")
