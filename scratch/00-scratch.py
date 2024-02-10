import scipy
import sympy
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from metastable.dykman import *
from tqdm import tqdm
from typing import Type
import scipy.integrate as integrate


x1, x2, p1, p2 = sympy.symbols("x1 x2 p1 p2")
sym = (x1, x2, p1, p2)


class EscapeModel:

    def __init__(self, epsilon:float, delta:float, chi:float, kappa:float=1):
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
        self, maxiter=50, magnitude=10, dp=5, tol=1e-10, method="hybr", n_seeking=3
    ):
        fixed_points = set()
        idx = 0
        n_found = 0
        while idx < maxiter and n_found < n_seeking:
            idx += 1
            y0 = magnitude * np.random.randn(2)
            res = scipy.optimize.root(
                self.y_dot_classical_func, y0, tol=tol, method=method
            )
            if res.success:
                fixed_points = fixed_points.union(set([tuple(np.round(res.x, dp))]))
                n_found = len(fixed_points)
        self.fixed_points = np.array(list(fixed_points))

    def refine_fixed_points(self, tol=1e-15, method="krylov", maxiter=10):
        for point_idx in tqdm(range(self.fixed_points.shape[0])):
            idx = 0
            success = False
            while idx < maxiter and not success:
                y0 = self.fixed_points[point_idx]
                res = scipy.optimize.root(
                    self.y_dot_classical_func, y0, tol=tol, method=method
                )
                success = res.success
                idx += 1
                if success:
                    self.fixed_points[point_idx] = res.x
            if not success:
                print("Failed to refine point " + str(point_idx) + ".")

    def classify_fixed_points(self):
        self.stability_classification = np.full(
            self.fixed_points.shape[0], True, dtype=bool
        )
        for idx, point in enumerate(self.fixed_points):
            self.stability_classification[idx] = np.all(
                np.linalg.eigvals(self.jacobian_classical_func(point)).real < 0
            )
        stable_point_indices = np.where(self.stability_classification)[0]

        if self.fixed_points.shape[0] == 3:
            if not np.sum(self.stability_classification) == 2:
                raise Exception(
                    "Found "
                    + str(np.sum(self.stability_classification))
                    + " fixed points."
                )
            stable_point_order = np.argsort(
                np.linalg.norm(self.fixed_points[self.stability_classification], axis=1)
            )
            point_order = np.array(
                [
                    stable_point_indices[stable_point_order[0]],
                    np.where(~self.stability_classification)[0][0],
                    stable_point_indices[stable_point_order[1]],
                ]
            )
            self.stability_classification = self.stability_classification[point_order]
            self.fixed_points = self.fixed_points[point_order]

    def find_fixed_points(self):
        self.estimate_fixed_points()
        print("Refining fixed points.")
        self.refine_fixed_points()

    def obtain_incoming_quantum_vector(self):
        unstable_point = np.hstack([self.fixed_points[1], [0, 0]])
        J = self.jacobian_func(unstable_point)
        eigenvalues, eigenvectors = np.linalg.eig(J)
        quantum_vector_indices = np.where(
            ~np.isclose(np.linalg.norm(eigenvectors[2:, :], axis=0), 0)
        )[0]
        incoming_quantum_vector_indices = quantum_vector_indices[
            np.where(eigenvalues[quantum_vector_indices] < 0)[0]
        ]

        if incoming_quantum_vector_indices.shape[0] != 1:
            raise Exception("Number of incoming quantum vectors != 1.")

        incoming_quantum_vector_idx = incoming_quantum_vector_indices[0]
        self.incoming_quantum_vector = eigenvectors[:, incoming_quantum_vector_idx]


def remove_incoming_component(outgoing_vector, incoming_vectors):
    plane_vectors = np.copy(incoming_vectors)

    plane_vectors[:, 1] -= (
        np.dot(plane_vectors[:, 0], plane_vectors[:, 1]) * plane_vectors[:, 0]
    )
    plane_vectors[:, 1] /= np.linalg.norm(plane_vectors[:, 1], axis=0)

    out_of_plane_vector = np.copy(outgoing_vector)
    out_of_plane_vector = (
        out_of_plane_vector
        - np.dot(out_of_plane_vector, plane_vectors[:, 0]) * plane_vectors[:, 0]
    )
    out_of_plane_vector = (
        out_of_plane_vector
        - np.dot(out_of_plane_vector, plane_vectors[:, 1]) * plane_vectors[:, 1]
    )

    return out_of_plane_vector


def remove_plane_component(vector, plane_vectors):
    plane_vectors = np.copy(plane_vectors)
    plane_vectors /= np.linalg.norm(plane_vectors, axis=0)
    plane_vectors[:, 1] -= (
        np.dot(plane_vectors[:, 0], plane_vectors[:, 1]) * plane_vectors[:, 0]
    )
    plane_vectors /= np.linalg.norm(plane_vectors, axis=0)

    out_of_plane_vector = np.copy(vector)
    out_of_plane_vector = (
        out_of_plane_vector
        - np.dot(out_of_plane_vector, plane_vectors[:, 0]) * plane_vectors[:, 0]
    )
    out_of_plane_vector = (
        out_of_plane_vector
        - np.dot(out_of_plane_vector, plane_vectors[:, 1]) * plane_vectors[:, 1]
    )

    return out_of_plane_vector


def func(epsilon, kappa, delta, chi, t_end, guess=None):
    model = EscapeModel(epsilon, delta, chi)
    model.find_fixed_points()
    model.classify_fixed_points()
    model.obtain_incoming_quantum_vector()

    unstable_point = np.hstack([model.fixed_points[1], [0, 0]])
    unstable_jac = model.jacobian_func(unstable_point)
    unstable_eigenvalues, unstable_eigenvectors = np.linalg.eig(unstable_jac)
    unstable_incoming_vectors = unstable_eigenvectors[:, [1, 2]]
    unstable_outgoing_vectors = unstable_eigenvectors[:, [0, 3]]
    plane_vectors = unstable_incoming_vectors
    out_of_plane_vectors = unstable_outgoing_vectors.T
    unstable_out_of_plane_vectors = np.array(
        [remove_plane_component(v, plane_vectors) for v in out_of_plane_vectors]
    )

    stable_point = np.hstack([model.fixed_points[2], [0, 0]])
    stable_jac = model.jacobian_func(stable_point)
    stable_eigenvalues, stable_eigenvectors = np.linalg.eig(stable_jac)
    stable_incoming_vectors = np.array(
        [
            (stable_eigenvectors[:, 0] + stable_eigenvectors[:, 1]).real,
            (1j * stable_eigenvectors[:, 0] - 1j * stable_eigenvectors[:, 1]).real,
        ]
    ).T
    stable_outgoing_vectors = np.array(
        [
            (stable_eigenvectors[:, 2] + stable_eigenvectors[:, 3]).real,
            (1j * stable_eigenvectors[:, 2] - 1j * stable_eigenvectors[:, 3]).real,
        ]
    ).T
    plane_vectors = stable_outgoing_vectors
    out_of_plane_vectors = stable_incoming_vectors.T
    stable_out_of_plane_vectors = np.array(
        [remove_plane_component(v, plane_vectors) for v in out_of_plane_vectors]
    )

    def bc(ya, yb):
        return np.hstack(
            [
                np.dot(ya - stable_point, stable_out_of_plane_vectors.T),
                np.dot(yb - unstable_point, unstable_out_of_plane_vectors.T),
            ]
        )

    t_guess = np.linspace(0, t_end, 10001)
    if guess is None:
        y_guess = (
            stable_point[:, np.newaxis]
            + t_guess[np.newaxis, :]
            * (unstable_point - stable_point)[:, np.newaxis]
            / t_end
        )
    else:
        y_guess = guess(t_guess)

    wrapper = lambda x, y: model.y_dot_func(y)
    res = scipy.integrate.solve_bvp(
        wrapper, bc, t_guess, y_guess, tol=1e-14, max_nodes=100000
    )

    return res


def estimate_fixed_points(
    model: EscapeModel,
    maxiter: int = 50,
    magnitude: float = 10,
    dp: int = 5,
    tol: float = 1e-10,
    method: str = "hybr",
    n_seeking: int = 3,
):
    fixed_points = set()
    idx = 0
    n_found = 0
    while idx < maxiter and n_found < n_seeking:
        idx += 1
        y0 = magnitude * np.random.randn(2)
        res = scipy.optimize.root(
            model.y_dot_classical_func, y0, tol=tol, method=method
        )
        if res.success:
            fixed_points = fixed_points.union(set([tuple(np.round(res.x, dp))]))
            n_found = len(fixed_points)
    fixed_points = np.array(list(fixed_points))
    return fixed_points


def refine_fixed_points(
    estimates: np.ndarray,
    model: EscapeModel,
    tol: float = 1e-15,
    method: str = "krylov",
    maxiter: int = 10,
):
    refined = np.empty(estimates.shape)
    for point_idx in tqdm(range(estimates.shape[0])):
        idx = 0
        success = False
        while idx < maxiter and not success:
            y0 = estimates[point_idx]
            res = scipy.optimize.root(
                model.y_dot_classical_func, y0, tol=tol, method=method
            )
            success = res.success
            idx += 1
            if success:
                refined[point_idx] = res.x
        if not success:
            print("Failed to refine point " + str(point_idx) + ".")
            refined[point_idx] = estimates[point_idx]
    return refined


epsilon = 1e1
kappa = 0.01
delta = 7.8
chi = -0.1
t_end = 8
print(-(epsilon**2) * chi / (2 * delta**3))


model = EscapeModel(epsilon=epsilon, delta=delta, chi=chi, kappa=kappa)
fixed_point_estimates = estimate_fixed_points(model=model, n_seeking=3)
fixed_points = refine_fixed_points(estimates=fixed_point_estimates, model=model)


epsilon_min = 0.0
epsilon_max = 3.0


def initial_fixed_point_check(model_class, epsilon, delta, chi, kappa):
    model = model_class(epsilon=epsilon, delta=delta, chi=chi, kappa=kappa)
    fixed_point_estimates = estimate_fixed_points(model=model, n_seeking=3)
    fixed_points = refine_fixed_points(estimates=fixed_point_estimates, model=model)
    return len(fixed_points), fixed_points


def find_epsilon_transition(model_class, epsilon_min, epsilon_max, delta, chi, kappa, tol=0.01, maxiter=100):
    n_fixed_points_min, fixed_points_min = initial_fixed_point_check(model_class, epsilon_min, delta, chi, kappa)
    n_fixed_points_max, fixed_points_max = initial_fixed_point_check(model_class, epsilon_max, delta, chi, kappa)

    if not (((n_fixed_points_min, n_fixed_points_max) == (1, 3)) or (
            (n_fixed_points_min, n_fixed_points_max) == (3, 1))):
        raise ValueError(
            "Initial bounds do not meet the strict condition (1, 3) or (3, 1) for transitioning fixed points.")

    iter_count = 0
    while epsilon_max - epsilon_min > tol and iter_count < maxiter:
        epsilon_mid = (epsilon_min + epsilon_max) / 2
        n_fixed_points_mid, fixed_points_mid = initial_fixed_point_check(model_class, epsilon_mid, delta, chi, kappa)

        print(f"Iteration {iter_count}: Epsilon = {epsilon_mid}, Fixed Points = {n_fixed_points_mid}")

        if n_fixed_points_mid == n_fixed_points_min:
            epsilon_min = epsilon_mid
            n_fixed_points_min = n_fixed_points_mid
        elif n_fixed_points_mid == n_fixed_points_max:
            epsilon_max = epsilon_mid
            n_fixed_points_max = n_fixed_points_mid
        else:
            raise ValueError(f"The number of fixed points at epsilon_mid = {n_fixed_points_mid}.")

        iter_count += 1

    return (epsilon_min + epsilon_max) / 2


def find_epsilon_transition_2(
        model_class: Type[EscapeModel],
        epsilon_min,
        epsilon_max,
        delta,
        chi,
        kappa,
        tol=0.01,
        maxiter=100
):
    n_fixed_points_min, fixed_points_min = initial_fixed_point_check(model_class, epsilon_min, delta, chi, kappa)
    n_fixed_points_max, fixed_points_max = initial_fixed_point_check(model_class, epsilon_max, delta, chi, kappa)

    if not (((n_fixed_points_min, n_fixed_points_max) == (1, 3)) or (
            (n_fixed_points_min, n_fixed_points_max) == (3, 1))):
        raise ValueError(
            "Initial bounds do not meet the strict condition (1, 3) or (3, 1) for transitioning fixed points.")

    if n_fixed_points_min == 3:
        fixed_points_mid_estimates = fixed_points_min
    else:
        fixed_points_mid_estimates = fixed_points_max

    iter_count = 0
    while epsilon_max - epsilon_min > tol and iter_count < maxiter:
        epsilon_mid = (epsilon_min + epsilon_max) / 2
        model = model_class(epsilon=epsilon_mid, delta=delta, chi=chi, kappa=kappa)


        fixed_points = refine_fixed_points(estimates=fixed_points_mid_estimates, model=model)
        n_fixed_points_mid, fixed_points_mid = initial_fixed_point_check(model_class, epsilon_mid, delta, chi, kappa)

        print(f"Iteration {iter_count}: Epsilon = {epsilon_mid}, Fixed Points = {n_fixed_points_mid}")

        if n_fixed_points_mid == n_fixed_points_min:
            epsilon_min = epsilon_mid
            n_fixed_points_min = n_fixed_points_mid
        elif n_fixed_points_mid == n_fixed_points_max:
            epsilon_max = epsilon_mid
            n_fixed_points_max = n_fixed_points_mid
        else:
            raise ValueError(f"The number of fixed points at epsilon_mid = {n_fixed_points_mid}.")

        iter_count += 1

    return (epsilon_min + epsilon_max) / 2


# Example call to the function
# Ensure delta, chi, kappa, etc., are defined or passed appropriately to this function
epsilon_transition = find_epsilon_transition(EscapeModel, epsilon_min, epsilon_max, delta, chi, kappa)
print(f"Transition occurs approximately at epsilon = {epsilon_transition}.")
