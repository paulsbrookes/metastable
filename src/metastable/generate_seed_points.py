from typing import Tuple


from metastable.zero_damping import solve_zero_damping
from metastable.classify_fixed_points import classify_fixed_points
from metastable.eom import EOM, Params
from metastable.classify_fixed_points import FixedPointsClassification


def generate_seed_points(
    epsilon: float,
    delta: float,
    chi: float,
) -> FixedPointsClassification:
    
    kappa = 0.0

    # Generate seed solutions
    fixed_points = solve_zero_damping(
        epsilon=epsilon,
        delta=delta,
        chi=chi,
    )

    # Classify the seed points
    params = Params(
        epsilon=epsilon,
        delta=delta,
        chi=chi,
        kappa=kappa,
    )
    seed_points = classify_fixed_points(
        fixed_points=fixed_points,
        eom=EOM(params=params),
    )

    return seed_points