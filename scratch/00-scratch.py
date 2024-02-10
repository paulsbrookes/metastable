import sympy
import random
import autograd.numpy as np
from scipy.optimize import root
from metastable.dykman import *
from tqdm import tqdm
from typing import NamedTuple, List
from time import perf_counter


x1, x2, p1, p2 = sympy.symbols("x1 x2 p1 p2")
sym = (x1, x2, p1, p2)


class EscapeModel:

    def __init__(self, epsilon: float, delta: float, chi: float, kappa: float = 1):
        self.epsilon = epsilon
        self.delta = delta
        self.chi = chi
        self.kappa = kappa

        self.H = (
            kappa * (p1**2 + p2**2)
            - kappa * (x1 * p1 + x2 * p2)
            - delta * (x1 * p2 - x2 * p1)
            - (chi / 2) * (x1**2 + x2**2 - p1**2 - p2**2) * (x1 * p2 - x2 * p1)
            - 2 * epsilon * p1
        )
        dx1_over_dt = sympy.diff(self.H, p1)
        dx2_over_dt = sympy.diff(self.H, p2)
        dp1_over_dt = -sympy.diff(self.H, x1)
        dp2_over_dt = -sympy.diff(self.H, x2)

        self.y_dot_expr = sympy.Matrix(
            [dx1_over_dt, dx2_over_dt, dp1_over_dt, dp2_over_dt]
        )
        self.jacobian_expr = self.y_dot_expr.jacobian(sym)
        self.y_dot_func = lambda y: sympy.lambdify(sym, self.y_dot_expr)(*y)[:, 0]
        self.jacobian_func = lambda y: sympy.lambdify(sym, self.jacobian_expr)(*y)

        self.y_dot_classical_expr = sympy.Matrix(
            self.y_dot_expr.subs([(p1, 0), (p2, 0)])[0:2]
        )
        self.jacobian_classical_expr = self.y_dot_classical_expr.jacobian(sym[:2])
        self.y_dot_classical_func = lambda y: sympy.lambdify(
            sym[:2], self.y_dot_classical_expr
        )(*y)[:, 0]
        self.jacobian_classical_func = lambda y: sympy.lambdify(
            sym[:2], self.jacobian_classical_expr
        )(*y)

        self.fixed_points = None


def estimate_fixed_points(
        model: EscapeModel,
        max_iter: int = 50,
        magnitude: float = 10,
        dp: int = 5,
        tol: float = 1e-10,
        method: str = "hybr",
        n_seeking: int = 3,
        initial_guesses: list = None,  # New parameter for initial guesses
):
    if initial_guesses is None:
        initial_guesses = [np.zeros(2)]  # Default to zero vector if none provided

    fixed_points = set()
    idx = 0
    n_found = 0
    while idx < max_iter and n_found < n_seeking:
        idx += 1
        # Choose a random initial guess from the list and add noise
        base_guess = random.choice(initial_guesses)
        y0 = base_guess + magnitude * np.random.randn(2)
        res = root(model.y_dot_classical_func, y0, tol=tol, method=method, jac=model.jacobian_classical_func)
        if res.success:
            # Round the solution to specified decimal places and update the set of fixed points
            fixed_points = fixed_points.union(set([tuple(np.round(res.x, dp))]))
            n_found = len(fixed_points)

    # Convert the set of fixed points into a list before returning
    return list(fixed_points)


class OptimizationResult(NamedTuple):
    estimate: np.ndarray
    success: bool


def optimize_fixed_point(
    y0: np.ndarray,
    model: EscapeModel,
    tol: float = 1e-15,
    method: str = "krylov",
    maxiter: int = 10,
) -> OptimizationResult:
    """
    Attempts to refine a single fixed point estimate.

    Parameters:
    - y0: The initial estimate to refine.
    - model: An instance of EscapeModel containing the function to optimize.
    - tol: Tolerance for the optimization.
    - method: The method used by scipy.optimize.root.
    - maxiter: Maximum number of iterations.

    Returns:
    - An OptimizationResult namedtuple containing the refined estimate (or the original estimate if optimization failed) and a boolean indicating success.
    """
    success = False
    idx = 0
    while idx < maxiter and not success:
        res = root(model.y_dot_classical_func, y0, tol=tol, method=method)
        success = res.success
        if success:
            return OptimizationResult(estimate=res.x, success=True)
        idx += 1
    return OptimizationResult(estimate=np.full(y0.shape, np.nan), success=False)


def refine_fixed_points(
    estimates: np.ndarray,
    model: EscapeModel,
    tol: float = 1e-15,
    method: str = "krylov",
    maxiter: int = 10,
):
    refined: List[np.array] = []
    for point_idx in tqdm(range(estimates.shape[0])):
        y0 = estimates[point_idx]
        result = optimize_fixed_point(y0, model, tol, method, maxiter)
        if not result.success:
            print(f"Failed to refine point {point_idx}.")
        else:
            refined.append(result.estimate)
    precision = -int(np.log10(tol) + 2)
    n_unique = len(set(tuple(np.round(arr, decimals=precision)) for arr in refined))
    return refined, n_unique


def find_epsilon_transition(
    epsilon_min: float,
    epsilon_max: float,
    delta: float,
    chi: float,
    kappa: float,
    tol=0.001,
    maxiter=100,
):

    observed_points = []

    fixed_points_min = estimate_fixed_points(
        EscapeModel(epsilon_min, delta, chi, kappa),
        magnitude=10.0,
        method="hybr"
    )
    fixed_points_max = estimate_fixed_points(
        EscapeModel(epsilon_max, delta, chi, kappa),
        magnitude=10.0,
        method="hybr"
    )

    observed_points += fixed_points_min
    observed_points += fixed_points_max

    if not (len(fixed_points_min), len(fixed_points_max)) in [(1, 3), (3, 1)]:
        raise ValueError(
            f"Number of fixed points at epsilon_min and epsilon_max is found to be {len(fixed_points_min)} and "
            f"{len(fixed_points_max)}."
        )

    iter_count = 0
    while epsilon_max - epsilon_min > tol and iter_count < maxiter:
        epsilon_mid = (epsilon_min + epsilon_max) / 2
        fixed_points_mid = estimate_fixed_points(
            EscapeModel(epsilon_mid, delta, chi, kappa),
            magnitude=2.0,
            initial_guesses=observed_points,
            method="lm",
            tol=1e-7
        )
        observed_points += fixed_points_mid
        print(f"Iteration: {iter_count}, N fixed points: {len(fixed_points_mid)}, epsilon: {epsilon_mid}, fixed_points: {fixed_points_mid}")

        if len(fixed_points_mid) == len(fixed_points_min):
            epsilon_min = epsilon_mid
            fixed_points_min = fixed_points_mid
        elif len(fixed_points_mid) == len(fixed_points_max):
            epsilon_max = epsilon_mid
            fixed_points_max = fixed_points_mid
        else:
            raise ValueError(
                f"The number of fixed points at epsilon_mid = {len(fixed_points_mid)}."
            )

        iter_count += 1

    return (epsilon_min + epsilon_max) / 2


epsilon = 1e1
kappa = 0.3
delta = 7.8
chi = -0.1
t_end = 8

n_occupation_estimate = epsilon**2 / (delta**2 + kappa**2)

print(-(epsilon**2) * chi / (2 * delta**3))


epsilon_min = 0.0
epsilon_max = 3.0

start = perf_counter()
epsilon_transition = find_epsilon_transition(
    epsilon_min, epsilon_max, delta, chi, kappa
)
print("Time taken:", perf_counter()-start)
print(f"Transition occurs approximately at epsilon = {epsilon_transition}.")
