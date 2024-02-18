import pytest
import numpy as np


from metastable.neighbour_average import neighbour_average


@pytest.fixture
def example_array():
    return np.array([[np.nan, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.mark.parametrize(
    "i, j, expected",
    [
        (0, 0, (2 + 4 + 5) / 3),  # Rop-right corner
        (0, 1, (2 + 3 + 4 + 5 + 6) / 5),  # Top center
        (0, 2, (2 + 3 + 5 + 6) / 4),  # Top-right corner
        (1, 0, (2 + 4 + 5 + 7 + 8) / 5),  # Edge, middle-left
        (1, 1, (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 8),  # Center
        (2, 0, (4 + 5 + 7 + 8) / 4),  # Bottom-left corner
        (2, 2, (5 + 6 + 8 + 9) / 4),  # Bottom-right corner
    ],
)
def test_neighbor_averages(example_array, i, j, expected):
    assert neighbour_average(example_array, i, j) == expected


@pytest.fixture
def example_array_2():
    return np.array(
        [
            [[[1, 2]], [[3, 4]]],
            [[[5, 6]], [[7, 8]]],
        ]
    )


def test_neighbor_averages_2(example_array_2):
    assert np.all(
        (
            neighbour_average(example_array_2, 0, 0)
            == np.array([[1 + 3 + 5 + 7, 2 + 4 + 6 + 8]]) / 4
        )
    )


@pytest.fixture
def example_array_3():
    return np.array(
        [
            [[[1, 2]], [[3, np.nan]]],
            [[[5, 6]], [[7, 8]]],
        ]
    )


def test_neighbor_averages_3(example_array_3):
    assert np.all(
        (
            neighbour_average(example_array_3, 0, 0)
            == np.array([[(1 + 3 + 5 + 7) / 4, (2 + 6 + 8) / 3]])
        )
    )
