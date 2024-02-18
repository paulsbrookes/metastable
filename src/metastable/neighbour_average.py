import numpy as np
from numpy.typing import NDArray


def neighbour_average(
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
