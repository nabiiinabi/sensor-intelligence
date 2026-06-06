import numpy as np

from src.features.binned_distribution import binned_distribution


def test_binned_distribution_sums_to_one() -> None:
    values = np.linspace(0.0, 9.0, 200)
    result = binned_distribution(values, prefix="X")

    assert set(result) == {f"X{i}" for i in range(10)}
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_binned_distribution_constant_values_edge_case() -> None:
    values = [3.5] * 200
    result = binned_distribution(values, prefix="X")

    assert result["X0"] == 1.0
    assert sum(result[f"X{i}"] for i in range(1, 10)) == 0.0


def test_binned_distribution_empty_input() -> None:
    result = binned_distribution([], prefix="Y")

    assert all(value == 0.0 for value in result.values())
