import numpy as np
from typing import List, Tuple


def solve_zero_damping(
    epsilon: float, delta: float, chi: float
) -> List[Tuple[float, float]]:

    # Calculate the roots of steady state eom for x_2
    x_2_candidates = np.roots([chi / 2.0, 0.0, delta, -2.0 * epsilon])

    # Check the number of real roots is as expected based on the discriminant
    discriminant = (
        -4 * (chi / 2.0) * delta**3 - 27 * ((chi / 2.0) ** 2) * (-2.0 * epsilon) ** 2
    )
    real_roots = [root.real for root in x_2_candidates if np.isreal(root)]
    expected_real_roots_count = 3 if discriminant > 0 else 1
    actual_real_roots_count = len(real_roots)
    if actual_real_roots_count != expected_real_roots_count:
        raise ValueError(
            f"The number of real roots was found to be {actual_real_roots_count}, but we expected "
            f"{expected_real_roots_count}."
        )

    # Filter out the complex solutions
    x_1_x_2_pairs = [
        (0.0, x_2.real) if np.isreal(x_2) else None for x_2 in x_2_candidates
    ]

    return x_1_x_2_pairs
