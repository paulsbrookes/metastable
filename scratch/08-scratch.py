import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.signal import convolve2d
from scipy import optimize


from metastable.eom import EOM
from metastable.zero_damping import solve_zero_damping
from metastable.neighbour_average import neighbour_average


class StateTracker:

    def __init__(self, epsilon_linspace: NDArray, kappa_linspace: NDArray):

        # Store the parameter grids
        self.epsilon_linspace = epsilon_linspace
        self.kappa_linspace = kappa_linspace

        # Initialize the container of calculated fixed points
        self.fixed_points: NDArray = np.full(
            shape=(len(epsilon_linspace), len(kappa_linspace), 3, 2),
            fill_value=np.nan,
            dtype=float,
        )

        # Initialize boolean mask tracking which cells have been checked
        self.checked_points: NDArray = np.full(
            shape=(len(epsilon_linspace), len(kappa_linspace)),
            fill_value=False,
            dtype=bool,
        )

    def update_state(self, epsilon_idx: int, kappa_idx: int, new_fixed_points: NDArray):
        """
        Update the fixed_points and checked_points arrays with new data.
        """
        self.fixed_points[epsilon_idx, kappa_idx] = new_fixed_points
        self.checked_points[epsilon_idx, kappa_idx] = True



def find_boundary(boolean_mask: NDArray[np.bool_]) -> List[Tuple[int, int]]:
    """Find indxes of the False cells on the boundary of the True region of a 2D boolean mask."""

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



# Set the fixed parameters
delta = 7.8
chi = -0.1

# Define the grid of parameters
epsilon_linspace = np.linspace(start=0.0, stop=25.0, num=21)
kappa_linspace = np.linspace(start=0.0, stop=10.0, num=21)

state = StateTracker(
    epsilon_linspace=epsilon_linspace,
    kappa_linspace=kappa_linspace
)

# Initialize the container of calculated fixed points
fixed_points = np.full(
    shape=(len(epsilon_linspace), len(kappa_linspace), 3, 2),
    fill_value=np.nan,
    dtype=float,
)

# Initialize the array to keep track of which points in the grid have been checked
checked_points = np.full(
    shape=(len(epsilon_linspace), len(kappa_linspace)), fill_value=False, dtype=bool
)

# Choose a point in parameter space for the seed solution
epsilon_idx = 5
kappa_idx = 0

# Double check that we are at zero damping
assert kappa_linspace[kappa_idx] == 0.0

# Generate the seed solution analytically
seed_points = solve_zero_damping(
    epsilon=epsilon_linspace[epsilon_idx], delta=delta, chi=chi
)

# We need to start with seeds for all three types of fixed point
assert len([point for point in seed_points if point is not None]) == 3

# Update the state of the arrays
state.update_state(epsilon_idx=epsilon_idx, kappa_idx=kappa_idx, new_fixed_points=seed_points)



tol = 1e-8
method = "hybr"


for idx in range(21):

    boundary_indexes = find_boundary(state.checked_points)
    boundary_guesses = [neighbour_average(state.fixed_points, *bi) for bi in boundary_indexes]

    for (epsilon_idx, kappa_idx), guesses in zip(boundary_indexes, boundary_guesses):
        eom = EOM(
            epsilon=state.epsilon_linspace[epsilon_idx],
            kappa=state.kappa_linspace[kappa_idx],
            delta=delta,
            chi=chi
        )
        results = np.array([
            optimize.root(eom.y_dot_classical_func, guess, tol=tol, method=method).x for guess in guesses
        ])
        state.update_state(epsilon_idx=epsilon_idx, kappa_idx=kappa_idx, new_fixed_points=results)

np.save("fixed_points.npy", state.fixed_points)