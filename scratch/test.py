import pytest
import numpy as np
from numpy.typing import NDArray


def neighbor_averages(
    array: NDArray[np.float64], index1: int, index2: int
) -> NDArray[np.float64]:

    if index1 < 0 or index1 >= array.shape[0] or index2 < 0 or index2 >= array.shape[1]:
        raise ValueError("Index out of bounds.")

    neighbors = [
        (i, j)
        for i in range(max(0, index1 - 1), min(array.shape[0], index1 + 2))
        for j in range(max(0, index2 - 1), min(array.shape[1], index2 + 2))
    ]

    sub_arrays = np.array([array[i, j] for i, j in neighbors])
    average_sub_array = np.nanmean(sub_arrays, axis=0)

    return average_sub_array


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
    assert neighbor_averages(example_array, i, j) == expected


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
            neighbor_averages(example_array_2, 0, 0)
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
            neighbor_averages(example_array_3, 0, 0)
            == np.array([[(1 + 3 + 5 + 7) / 4, (2 + 6 + 8) / 3]])
        )
    )
