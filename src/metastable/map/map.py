from __future__ import annotations
from typing import Optional, Final, Union, Tuple
from numpy.typing import NDArray
from scipy.integrate._bvp import BVPResult
from enum import Enum
from pathlib import Path
import numpy as np


class FixedPointType(Enum):
    """Enumeration of fixed point types in the system.

    Warning:
        The validation that saddle, bright, and dim states are stored at the correct indexes
        is not implemented. The index ordering depends on the seed solution used to initialize
        the mapping of the fixed points.
    """

    SADDLE = 0
    BRIGHT = 1
    DIM = 2


class PathType(Enum):
    """Enumeration of path types between fixed points."""

    BRIGHT_TO_SADDLE = 0
    DIM_TO_SADDLE = 1


class FixedPointMap:
    """A map of the fixed points over a grid of epsilon and kappa values.

    This class stores and manages information about fixed points and paths for a nonlinear
    oscillator system across a range of drive amplitudes (epsilon) and damping rates (kappa).

    The map tracks three types of fixed points (saddle, bright state, and dim state) and the paths
    between them for each parameter combination in the epsilon-kappa space.

    Attributes:
        epsilon_linspace (NDArray[np.float_]): Drive amplitude values
        kappa_linspace (NDArray[np.float_]): Damping rate values
        delta (float): Detuning parameter (drive vs. natural frequency difference)
        chi (float): Nonlinearity parameter (frequency-amplitude coupling strength)
        fixed_points (NDArray[np.float_]): Fixed points array with shape (n_epsilon, n_kappa, 3, 2)
            where the last two dimensions represent fixed point type and (x, p) coordinates
        checked_points (NDArray[np.bool_]): Boolean mask of analyzed parameter combinations
        path_results (NDArray[object]): Array of path calculation results between fixed points
    """

    def __init__(
        self,
        epsilon_linspace: NDArray[np.float_],
        kappa_linspace: NDArray[np.float_],
        delta: float,
        chi: float,
        fixed_points: Optional[NDArray[np.float_]] = None,
        checked_points: Optional[NDArray[np.bool_]] = None,
        path_results: Optional[
            NDArray[object]
        ] = None,  # Actually NDArray[Optional[BVPResult]]
    ):
        self._epsilon_linspace: Final[NDArray] = epsilon_linspace
        self._kappa_linspace: Final[NDArray] = kappa_linspace
        self._delta: Final[float] = delta
        self._chi: Final[float] = chi

        self._fixed_points = (
            fixed_points
            if fixed_points is not None
            else np.full(
                shape=(len(epsilon_linspace), len(kappa_linspace), 3, 2),
                fill_value=np.nan,
                dtype=float,
            )
        )

        self._checked_points = (
            checked_points
            if checked_points is not None
            else np.full(
                shape=(len(epsilon_linspace), len(kappa_linspace)),
                fill_value=False,
                dtype=bool,
            )
        )

        self.path_results = (
            path_results
            if path_results is not None
            else np.empty(
                shape=(len(epsilon_linspace), len(kappa_linspace), 2),
                dtype=object,
            )
        )

    @property
    def epsilon_linspace(self) -> NDArray:
        """
        The linearly spaced array of epsilon values representing the drive amplitude.

        This array defines the first dimension of the phase space map, ranging from 0
        to some maximum value. Each point represents a different drive amplitude in the system.

        Returns:
            NDArray: 1D array of epsilon values
        """
        return self._epsilon_linspace

    @property
    def kappa_linspace(self) -> NDArray:
        """
        The linearly spaced array of kappa values representing the damping rate.

        This array defines the second dimension of the phase space map, ranging from 0
        to some maximum value. Each point represents a different damping rate in the system.

        Returns:
            NDArray: 1D array of kappa values
        """
        return self._kappa_linspace

    @property
    def delta(self) -> float:
        """
        The detuning parameter of the system.

        This value represents the difference between the drive frequency and the natural
        frequency of the oscillator.

        Returns:
            float: The detuning parameter
        """
        return self._delta

    @property
    def chi(self) -> float:
        """
        The nonlinearity parameter of the system.

        This value characterizes the strength of the nonlinear interaction in the
        oscillator. It determines how much the oscillation frequency changes with amplitude.

        Returns:
            float: The nonlinearity parameter
        """
        return self._chi

    @property
    def fixed_points(self) -> NDArray:
        """
        Array containing the fixed points of the system for each parameter combination.

        The array has shape (epsilon_points, kappa_points, 3, 2) where:
        - First dimension: different epsilon (drive amplitude) values
        - Second dimension: different kappa (damping rate) values
        - Third dimension: three types of fixed points (saddle, bright, dim) as defined in FixedPointType
        - Fourth dimension: (x, p) coordinates of the fixed point in phase space

        NaN values indicate parameter combinations where fixed points haven't been found or don't exist.

        Returns:
            NDArray: 4D array of fixed points
        """
        return self._fixed_points

    @fixed_points.setter
    def fixed_points(self, value: NDArray):
        """
        Set the fixed points array.

        Args:
            value (NDArray): 4D array with shape (epsilon_points, kappa_points, 3, 2)
                           containing the fixed points for each parameter combination
        """
        self._fixed_points = value

    @property
    def checked_points(self) -> NDArray[np.bool_]:
        """
        Boolean array indicating which parameter combinations have been checked for fixed points.

        The array has shape (epsilon_points, kappa_points) where:
        - First dimension: different epsilon (drive amplitude) values
        - Second dimension: different kappa (damping rate) values

        True values indicate parameter combinations that have been checked for fixed points,
        regardless of whether fixed points were found.

        Returns:
            NDArray[np.bool_]: 2D boolean array of checked parameter combinations
        """
        return self._checked_points

    @checked_points.setter
    def checked_points(self, value: NDArray[np.bool_]):
        """
        Set the checked points array.

        Args:
            value (NDArray[np.bool_]): 2D boolean array with shape (epsilon_points, kappa_points)
                                     indicating which parameter combinations have been checked
        """
        self._checked_points = value

    def update_map(
        self, epsilon_idx: int, kappa_idx: int, new_fixed_points: NDArray[np.float_]
    ) -> None:
        """Update the fixed points and checked points arrays at specified indices.

        Args:
            epsilon_idx: Index in the epsilon dimension
            kappa_idx: Index in the kappa dimension
            new_fixed_points: Array of shape (3, 2) containing the (x, p) coordinates
                for each fixed point type
        """
        self._fixed_points[epsilon_idx, kappa_idx] = new_fixed_points
        self._checked_points[epsilon_idx, kappa_idx] = True

    def add_seed(
        self,
        epsilon_idx: int,
        kappa_idx: int,
        saddle_fixed_point: Tuple[float, float],
        bright_fixed_point: Tuple[float, float],
        dim_fixed_point: Tuple[float, float],
    ) -> None:
        """Add a seed solution to the map for a bistable parameter combination.

        Args:
            epsilon_idx: Index in the epsilon dimension
            kappa_idx: Index in the kappa dimension
            seed_fixed_points: Array of shape (3, 2) containing the (x, p) coordinates
                for all three fixed points
                
        """
        seed_fixed_points = np.zeros((3, 2))
        seed_fixed_points[FixedPointType.SADDLE.value] = saddle_fixed_point
        seed_fixed_points[FixedPointType.BRIGHT.value] = bright_fixed_point
        seed_fixed_points[FixedPointType.DIM.value] = dim_fixed_point
        self.update_map(epsilon_idx, kappa_idx, seed_fixed_points)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save the current state of the map to a NumPy .npz file.

        Args:
            file_path: Path where the state should be saved
        """
        np.savez(
            file_path,
            epsilon_linspace=self.epsilon_linspace,
            kappa_linspace=self.kappa_linspace,
            delta=self.delta,
            chi=self.chi,
            fixed_points=self.fixed_points,
            checked_points=self.checked_points,
            path_results=self.path_results,
        )

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> FixedPointMap:
        """Load a previously saved map state from a file.

        Args:
            file_path: Path to the saved state file

        Returns:
            A new PhaseSpaceMap instance with the loaded state
        """
        loaded_data = np.load(file_path, allow_pickle=True)
        tracker = cls(**loaded_data)
        return tracker

    def copy(self) -> FixedPointMap:
        return FixedPointMap(
            epsilon_linspace=self.epsilon_linspace.copy(),
            kappa_linspace=self.kappa_linspace.copy(),
            delta=self.delta,
            chi=self.chi,
            fixed_points=self.fixed_points.copy(),
            checked_points=self.checked_points.copy(),
            path_results=self.path_results.copy(),
        )
