import sympy
import random
import autograd.numpy as np
from src.metastable.dykman import *
from typing import NamedTuple, List, Tuple
from numpy.typing import NDArray


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
        res = root(
            model.y_dot_classical_func,
            y0,
            tol=tol,
            method=method,
            jac=model.jacobian_classical_func,
        )
        if res.success:
            # Round the solution to specified decimal places and update the set of fixed points
            fixed_points = fixed_points.union(set([tuple(np.round(res.x, dp))]))
            n_found = len(fixed_points)

    # Convert the set of fixed points into a list before returning
    return list(fixed_points)


class OptimizationResult(NamedTuple):
    estimate: np.ndarray
    success: bool


def calculate_beta_12(kappa_rescaled: float) -> float:
    """Calculates the rescaled power parameter at the bifurcation points from the rescaled decay rate."""
    beta_1 = (2 / 27) * (
        1 + 9 * kappa_rescaled**2 - (1 - 3 * kappa_rescaled**2) ** (3 / 2)
    )
    beta_2 = (2 / 27) * (
        1 + 9 * kappa_rescaled**2 + (1 - 3 * kappa_rescaled**2) ** (3 / 2)
    )
    return beta_1, beta_2


def calculate_kappa_rescaled(kappa: float, delta: float) -> float:
    """Calculates the rescaled decay rate from the original parameters."""
    kappa_rescaled = kappa / (2 * np.abs(delta))
    return kappa_rescaled


def map_beta_to_epsilon(beta: float, delta: float, chi: float) -> float:
    lambda_ = abs(chi / delta)
    epsilon = delta * np.sqrt(beta / (2 * lambda_))
    return epsilon


def solve_weak_nonlinearity(epsilon, delta, kappa) -> Tuple[float, float]:
    """Solves for x1 and x2 in the limit of weak effective nonlinearity i.e. the occupation multiplied by the nonlinearity
    is small relative to other terms.
    """
    x1 = (-2 * epsilon * kappa) / (delta**2 + kappa**2)
    x2 = (2 * epsilon * delta) / (delta**2 + kappa**2)
    return x1, x2


def solve_no_damping(
    epsilon: float, delta: float, chi: float
) -> List[Tuple[float, float]]:
    # Calculate the roots of the cubic equation
    x_2_candidates = np.roots([chi / 2.0, 0.0, delta, -2.0 * epsilon])

    # Calculate the discriminant of the cubic equation
    discriminant = (
        -4 * (chi / 2.0) * delta**3 - 27 * ((chi / 2.0) ** 2) * (-2.0 * epsilon) ** 2
    )

    # Filter out the real roots
    real_roots = [root.real for root in x_2_candidates if np.isreal(root)]

    # Check the number of real roots is as expected based on the discriminant
    expected_real_roots_count = 3 if discriminant > 0 else 1
    actual_real_roots_count = len(real_roots)

    if actual_real_roots_count != expected_real_roots_count:
        raise ValueError(
            f"The number of real roots was found to be {actual_real_roots_count}, but we expected {expected_real_roots_count}."
        )

    # Calculate x_1 for each real root x_2 and return tuples of (x_1, x_2)
    x_1_x_2_pairs = [(0.0, x_2) for x_2 in real_roots]

    return x_1_x_2_pairs


epsilon = 17.0
kappa = 0.0
delta = 7.8
chi = -0.1


kappa_rescaled = calculate_kappa_rescaled(kappa, delta)
beta_1, beta_2 = calculate_beta_12(kappa_rescaled)
epsilon_1 = map_beta_to_epsilon(beta_1, delta, chi)
epsilon_2 = map_beta_to_epsilon(beta_2, delta, chi)


print(f"epsilon_1: {epsilon_1}, epsilon_2: {epsilon_2}, epsilon: {epsilon}")


model = EscapeModel(epsilon=epsilon, delta=delta, chi=chi, kappa=kappa)

fixed_points_no_damping = solve_no_damping(epsilon=epsilon, delta=delta, chi=chi)


kappa_array = np.linspace(start=0.0, stop=3.0, num=51)

fixed_points = np.empty(shape=(len(kappa_array), 2))

y0 = fixed_points_no_damping[0]

from tqdm import tqdm

# for idx, kappa in tqdm(enumerate(kappa_array)):
#     tol = 1e-8
#     method = "hybr"
#     model = EscapeModel(epsilon=epsilon, delta=delta, chi=chi, kappa=kappa)
#     result = root(model.y_dot_classical_func, y0, tol=tol, method=method)
#     if not result.success:
#         raise Exception(f"Unsuccessful optimization.")
#     fixed_points[idx] = result.x
#     y0 = result.x


def trace_fixed_point(
    initial_guess: Tuple[float, float],
    kappa_linspace: NDArray[np.float64],
    epsilon: float,
    delta: float,
    chi: float,
) -> NDArray[np.float64]:

    tol = 1e-8
    method = "hybr"
    y0 = initial_guess

    fixed_points = np.full(shape=(len(kappa_linspace), 2), fill_value=np.nan)

    for idx, kappa in tqdm(enumerate(kappa_linspace)):
        model = EscapeModel(epsilon=epsilon, delta=delta, chi=chi, kappa=kappa)
        result = root(model.y_dot_classical_func, y0, tol=tol, method=method)
        if not result.success:
            return fixed_points
        fixed_points[idx] = result.x
        y0 = result.x

    return fixed_points


n_kappa = 11
n_epsilon = 11
kappa_linspace = np.linspace(start=0.0, stop=3.0, num=n_kappa)
epsilon_linspace = np.linspace(start=0.0, stop=20.0, num=n_epsilon)


fixed_points = np.full(shape=(n_epsilon, n_kappa, 3, 2), fill_value=np.nan)


for epsilon_idx, epsilon in enumerate(epsilon_linspace):
    fixed_points_no_damping = solve_no_damping(epsilon=epsilon, delta=delta, chi=chi)
    for type_idx, fixed_point in enumerate(fixed_points_no_damping):
        fixed_points[epsilon_idx, :, type_idx, :] = trace_fixed_point(
            fixed_point, kappa_linspace, epsilon, delta, chi
        )

...
