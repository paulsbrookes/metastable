import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from scipy.signal import convolve2d


def find_boundary(boolean_mask: NDArray[np.bool_]) -> List[Tuple[int, int]]:
    """Find indexes of the False cells on the boundary of the True region of a 2D boolean mask."""

    kernel = np.ones((3, 3), dtype=int)

    # Count of neighborhood cells which are True
    count_result = convolve2d(
        boolean_mask, kernel, mode="same", boundary="fill", fillvalue=0
    )

    false_cells_with_true_neighbors = (count_result > 0) & (boolean_mask == False)

    # Extract the row and column indices where the condition is met
    rows, cols = np.where(false_cells_with_true_neighbors)

    # Convert to list of tuples
    boundary_coordinates = list(zip(rows, cols))

    return boundary_coordinates
