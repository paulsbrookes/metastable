from __future__ import annotations
from typing import Optional, Final
from numpy.typing import NDArray
import numpy as np


class FixedPointMap:
    def __init__(
        self,
        epsilon_linspace: NDArray,
        kappa_linspace: NDArray,
        delta: float,
        chi: float,
        fixed_points: Optional[NDArray] = None,
        checked_points: Optional[NDArray] = None,
        path_results: Optional[NDArray] = None,
    ):
        self.epsilon_linspace: Final[NDArray] = epsilon_linspace
        self.kappa_linspace: Final[NDArray] = kappa_linspace
        self.delta: Final[float] = delta
        self.chi: Final[float] = chi

        if fixed_points is not None:
            self.fixed_points = fixed_points
        else:
            self.fixed_points: NDArray = np.full(
                shape=(len(epsilon_linspace), len(kappa_linspace), 3, 2),
                fill_value=np.nan,
                dtype=float,
            )

        if checked_points is not None:
            self.checked_points = checked_points
        else:
            self.checked_points: NDArray = np.full(
                shape=(len(epsilon_linspace), len(kappa_linspace)),
                fill_value=False,
                dtype=bool,
            )

        if path_results is not None:
            self.path_results = path_results
        else:
            self.path_results: NDArray = np.empty(
                shape=(len(epsilon_linspace), len(kappa_linspace)), dtype=object
            )

    def update_map(self, epsilon_idx: int, kappa_idx: int, new_fixed_points: NDArray):
        """
        Update the fixed_points and checked_points arrays with new data.
        """
        self.fixed_points[epsilon_idx, kappa_idx] = new_fixed_points
        self.checked_points[epsilon_idx, kappa_idx] = True

    def save_state(self, file_path: str):
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
    def load(cls, file_path: str) -> FixedPointMap:
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
