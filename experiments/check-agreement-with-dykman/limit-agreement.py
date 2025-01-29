from metastable.zero_damping import solve_zero_damping
from metastable.rescaled import (
    calculate_beta_limits,
    map_beta_to_epsilon,
    calculate_kappa_rescaled,
)

delta = 7.8
chi = -0.1
epsilon = 18.748
kappa = 0.0

seed_points = solve_zero_damping(
    epsilon=epsilon,
    delta=delta,
    chi=chi,
)

print(seed_points)


kappa_rescaled = calculate_kappa_rescaled(kappa=kappa, delta=delta)
beta_limits = calculate_beta_limits(kappa_rescaled=kappa_rescaled)
epsilon_limits = (
    map_beta_to_epsilon(beta_limits[0], delta=delta, chi=chi),
    map_beta_to_epsilon(beta_limits[1], delta=delta, chi=chi),
)
print(epsilon_limits)
