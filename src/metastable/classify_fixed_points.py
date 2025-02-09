import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


from metastable.eom import EOM


@dataclass
class FixedPointsClassification:
    saddle: Tuple[float, float]  # Unstable fixed point (saddle)
    dim: Tuple[float, float]  # Low-amplitude stable fixed point
    bright: Tuple[float, float]  # High-amplitude stable fixed point


def is_stable(fixed_point: Tuple[float, float], eom: EOM) -> bool:
    """Determine if a fixed point is stable based on its Jacobian eigenvalues.

    Args:
        fixed_point: The fixed point coordinates (xc, pc)
        eom: Equations of motion instance

    Returns:
        bool: True if the fixed point is stable (no positive real eigenvalues)
    """
    jacobian = np.array(eom.jacobian_classical_func(fixed_point))
    eigenvalues = np.linalg.eigvals(jacobian)
    return not any(np.real(ev) > 0 for ev in eigenvalues)


def classify_fixed_points(
    fixed_points: List[Tuple[float, float]], eom: EOM
) -> FixedPointsClassification:
    """
    Classify a list of fixed points into saddle (unstable), dim (low amplitude stable)
    and bright (high amplitude stable) states in the bistable regime.

    Args:
        fixed_points: List of exactly three tuples representing fixed points to be classified (xc, pc).
        eom: An EOM instance.

    Returns:
        FixedPointsClassification: A dataclass containing the classified fixed points
            with fields 'saddle', 'dim', and 'bright'.

    Notes:
        - Requires exactly three fixed points (bistable regime)
        - None of the fixed points can be None
        - Stability is determined by the eigenvalues of the classical Jacobian.
          A fixed point is stable if none of its eigenvalues have a positive real part.
        - Among stable fixed points, the lower |p| amplitude point is classified as dim,
          while the higher |p| amplitude point is classified as bright.
        - This function is sensitive to failure at epsilon = kappa = 0.0
    """

    if not fixed_points or len(fixed_points) != 3:
        raise ValueError(
            "Exactly three fixed points must be provided (bistable regime)"
        )

    if any(fp is None for fp in fixed_points):
        raise ValueError("None of the fixed points can be None")

    if eom.params.epsilon == 0.0 and eom.params.kappa == 0.0:
        raise ValueError(
            "Cannot classify fixed points at epsilon = kappa = 0.0 (bifurcation point)"
        )

    saddle = None
    stable_points = []

    for fp in fixed_points:
        if is_stable(fp, eom):
            stable_points.append(fp)
        else:
            if saddle is not None:
                raise ValueError("Found multiple unstable (saddle) points")
            saddle = fp

    if len(stable_points) != 2:
        raise ValueError(
            f"Found {len(stable_points)} stable fixed points, but expected exactly 2 in bistable regime"
        )

    # Sort stable points by amplitude |p|
    stable_points.sort(key=lambda point: abs(point[1]))

    return FixedPointsClassification(
        saddle=saddle, dim=stable_points[0], bright=stable_points[1]
    )
