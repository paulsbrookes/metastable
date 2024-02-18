import sympy
import autograd.numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import root
from metastable.dykman import *
from typing import List, Tuple
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class EscapeModel:
    x1, x2, p1, p2 = sympy.symbols("x1 x2 p1 p2")
    sym = (x1, x2, p1, p2)

    def __init__(self, epsilon: float, delta: float, chi: float, kappa: float = 1):
        self.epsilon = epsilon
        self.delta = delta
        self.chi = chi
        self.kappa = kappa

        self.H = (
            kappa * (self.p1**2 + self.p2**2)
            - kappa * (self.x1 * self.p1 + self.x2 * self.p2)
            - delta * (self.x1 * self.p2 - self.x2 * self.p1)
            - (chi / 2) * (self.x1**2 + self.x2**2 - self.p1**2 - self.p2**2) * (self.x1 * self.p2 - self.x2 * self.p1)
            - 2 * epsilon * self.p1
        )
        dx1_over_dt = sympy.diff(self.H, self.p1)
        dx2_over_dt = sympy.diff(self.H, self.p2)
        dp1_over_dt = -sympy.diff(self.H, self.x1)
        dp2_over_dt = -sympy.diff(self.H, self.x2)

        self.y_dot_expr = sympy.Matrix(
            [dx1_over_dt, dx2_over_dt, dp1_over_dt, dp2_over_dt]
        )
        self.jacobian_expr = self.y_dot_expr.jacobian(self.sym)
        self.y_dot_func = lambda y: sympy.lambdify(self.sym, self.y_dot_expr)(*y)[:, 0]
        self.jacobian_func = lambda y: sympy.lambdify(self.sym, self.jacobian_expr)(*y)

        self.y_dot_classical_expr = sympy.Matrix(
            self.y_dot_expr.subs([(self.p1, 0), (self.p2, 0)])[0:2]
        )
        self.jacobian_classical_expr = self.y_dot_classical_expr.jacobian(self.sym[:2])
        self.y_dot_classical_func = lambda y: sympy.lambdify(
            self.sym[:2], self.y_dot_classical_expr
        )(*y)[:, 0]
        self.jacobian_classical_func = lambda y: sympy.lambdify(
            self.sym[:2], self.jacobian_classical_expr
        )(*y)


def calculate_beta_12(kappa_rescaled: float) -> float:
    """Calculates the rescaled power parameter at the bifurcation points from the rescaled decay rate."""
    beta_1 = (2 / 27) * (1 + 9 * kappa_rescaled ** 2 - (1 - 3 * kappa_rescaled ** 2) ** (3 / 2))
    beta_2 = (2 / 27) * (1 + 9 * kappa_rescaled ** 2 + (1 - 3 * kappa_rescaled ** 2) ** (3 / 2))
    return beta_1, beta_2


def calculate_kappa_rescaled(kappa: float, delta: float) -> float:
    """Calculates the rescaled decay rate from the original parameters."""
    kappa_rescaled = kappa / (2*np.abs(delta))
    return kappa_rescaled


def map_beta_to_epsilon(beta: float, delta: float, chi: float) -> float:
    lambda_ = abs(chi / delta)
    epsilon = delta * np.sqrt(beta / (2 * lambda_))
    return epsilon


def solve_weak_nonlinearity(epsilon, delta, kappa) -> Tuple[float, float]:
    """Solves for self.x1 and self.x2 in the limit of weak effective nonlinearity i.e. the occupation multiplied by the nonlinearity
    is small relative to other terms.
    """
    x1 = (-2 * epsilon * kappa) / (delta**2 + kappa**2)
    x2 = (2 * epsilon * delta) / (delta**2 + kappa**2)
    return x1, x2


def solve_no_damping(epsilon: float, delta: float, chi: float) -> List[Tuple[float, float]]:
    # Calculate the roots of the cubic equation
    x_2_candidates = np.roots([chi / 2.0, 0.0, delta, -2.0 * epsilon])

    # Calculate the discriminant of the cubic equation
    discriminant = -4 * (chi / 2.0) * delta ** 3 - 27 * ((chi / 2.0) ** 2) * (-2.0 * epsilon) ** 2

    # Filter out the real roots
    real_roots = [root.real for root in x_2_candidates if np.isreal(root)]

    # Check the number of real roots is as expected based on the discriminant
    expected_real_roots_count = 3 if discriminant > 0 else 1
    actual_real_roots_count = len(real_roots)

    if actual_real_roots_count != expected_real_roots_count:
        raise ValueError(
            f"The number of real roots was found to be {actual_real_roots_count}, but we expected {expected_real_roots_count}.")

    # Replace complex solutions with (np.nan, np.nan) tuples
    x_1_x_2_pairs = [(0.0, x_2.real) if np.isreal(x_2) else None for x_2 in x_2_candidates]

    return x_1_x_2_pairs










def trace_single_fixed_point(
        initial_guess: Tuple[float, float],
        kappa_linspace: NDArray[np.float64],
        epsilon: float,
        delta: float,
        chi: float
) -> NDArray[np.float64]:

    tol = 1e-8
    method = "hybr"
    y0 = initial_guess

    fixed_points = np.full(shape=(len(kappa_linspace), 2), fill_value=np.nan)

    for idx, kappa in enumerate(kappa_linspace):
        model = EscapeModel(epsilon=epsilon, delta=delta, chi=chi, kappa=kappa)
        result = root(model.y_dot_classical_func, y0, tol=tol, method=method)
        if not result.success:
            return fixed_points
        fixed_points[idx] = result.x
        y0 = result.x

    return fixed_points


def trace_fixed_points(epsilon: float, delta: float, chi: float, kappa_max: float, n_kappa: int) -> NDArray[np.float64]:
    kappa_linspace = np.linspace(start=0.0, stop=kappa_max, num=n_kappa)
    fixed_points_no_damping = solve_no_damping(epsilon=epsilon, delta=delta, chi=chi)
    results = np.full((n_kappa, 3, 2), fill_value=np.nan)
    for type_idx, fixed_point in enumerate(fixed_points_no_damping):
        if fixed_point is not None:
            results[:, type_idx, :] = trace_single_fixed_point(fixed_point, kappa_linspace, epsilon, delta, chi)
    return results




kappa = 2.0
delta = 7.8
chi = -0.1


kappa_rescaled = calculate_kappa_rescaled(kappa, delta)
beta_1, beta_2 = calculate_beta_12(kappa_rescaled)
epsilon_1 = map_beta_to_epsilon(beta_1, delta, chi)
epsilon_2 = map_beta_to_epsilon(beta_2, delta, chi)


print(f"epsilon_1: {epsilon_1}, epsilon_2: {epsilon_2}")


n_epsilon = 100
n_kappa = 100
epsilon_linspace = np.linspace(start=18.0, stop=20.0, num=n_epsilon)
kappa_max = 3.5


fixed_points = np.full(shape=(n_epsilon, n_kappa, 3, 2), fill_value=np.nan)
with ProcessPoolExecutor(max_workers=1) as executor:
    futures = [
        executor.submit(trace_fixed_points, epsilon, delta, chi, kappa_max, n_kappa) for epsilon in epsilon_linspace
    ]
    for epsilon_idx, future in tqdm(enumerate(futures)):
        result = future.result()
        fixed_points[epsilon_idx, :, :, :] = result


count = 3 - 0.5*np.sum(np.isnan(fixed_points), axis=(2, 3))



# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(count, origin='lower', aspect='auto', extent=[0.0, kappa_max, epsilon_linspace[0], epsilon_linspace[-1]], cmap='viridis')

plt.colorbar(label='Count of non-NaN values')
plt.xlabel('Kappa')
plt.ylabel('Epsilon')
plt.title('Count of Non-NaN Values for Each (Epsilon, Kappa) Combination')

plt.show()