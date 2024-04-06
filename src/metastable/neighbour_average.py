import numpy as np
from numpy.typing import NDArray
from itertools import product


def neighbour_average(array: NDArray[np.float64], *indexes) -> NDArray:
    """Find the mean values of all elements of an N-dimensional array including and neighboring the given indexes.
    NaN values are ignored."""

    if any(index < 0 or index >= array.shape[dim] for dim, index in enumerate(indexes)):
        raise IndexError("Index out of bounds.")

    # Generate ranges for each dimension
    ranges = [
        range(max(0, index - 1), min(array.shape[dim], index + 2))
        for dim, index in enumerate(indexes)
    ]

    # Generate all combinations of neighbor indexes
    neighbors_indexes = product(*ranges)

    # Extract values from these indexes, ignoring the original point if included
    sub_arrays = np.array([array[indexes] for indexes in neighbors_indexes])

    # Calculate and return the mean, ignoring NaNs
    return np.nanmean(sub_arrays, axis=0)
