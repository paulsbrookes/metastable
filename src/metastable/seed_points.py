from metastable.zero_damping import solve_zero_damping
from metastable.classify_fixed_points import classify_fixed_points
from metastable.eom import EOM, Params
from dataclasses import dataclass


@dataclass
class ClassifiedSeedPoints:
    """Container for classified seed points."""
    dim: complex
    bright: complex
    saddle: complex


def generate_seed_points(
    epsilon: float,
    delta: float,
    chi: float,
    kappa: float = 0.0,
) -> ClassifiedSeedPoints:
    """
    Generate and classify seed points for the fixed point map.

    Args:
        epsilon: Drive strength
        delta: Detuning parameter
        chi: Nonlinearity parameter
        kappa: Damping parameter (defaults to 0.0 for zero damping case)

    Returns:
        ClassifiedSeedPoints: Container with classified dim, bright, and saddle fixed points
    """
    # Generate seed solutions at zero damping
    seed_points = solve_zero_damping(
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
    classified_points = classify_fixed_points(
        fixed_points=seed_points,
        eom=EOM(params=params),
    )

    return ClassifiedSeedPoints(
        dim=classified_points.dim,
        bright=classified_points.bright,
        saddle=classified_points.saddle,
    ) 