import sympy
import numpy as np
import scipy
from tqdm import tqdm

x1, x2, p1, p2 = sympy.symbols("x1 x2 p1 p2")
sym = (x1, x2, p1, p2)


class EscapeModel:
    def __init__(self, epsilon, delta, chi, kappa=1):
        """Initialise the model of the system by finding the equations of motion."""
        self.epsilon = epsilon
        self.delta = delta
        self.chi = chi
        self.kappa = kappa

        # The Hamiltonian is defined in symbolic form to allow us to find the equations of motion using differentiation.
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

        # We construct symbolic expressions and functions to describe the equations of motion and their Jacobian matrix.
        self.y_dot_expr = sympy.Matrix(
            [dx1_over_dt, dx2_over_dt, dp1_over_dt, dp2_over_dt]
        )
        self.jacobian_expr = self.y_dot_expr.jacobian(sym)
        self.y_dot_func = lambda y: sympy.lambdify(sym, self.y_dot_expr)(*y)[:, 0]
        self.jacobian_func = lambda y: sympy.lambdify(sym, self.jacobian_expr)(*y)

        # The classical equations of motion describe motion which is restricted to the classical plane.
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
        self,
        maxiter: int = 50,
        magnitude: float = 10.0,
        dp: int = 5,
        tol: float = 1e-10,
        method: str = "hybr",
        n_seeking: int = 3,
    ):
        """Get quick estimates of the fixed points using random guesses to start root finding. Can fail if the magnitude
        is set too large or too small.

        Args:
            maxiter: The maximum number of attempts to find roots.
            magnitude: Magnitude of random guesses.
            dp: How many decimal places to use when checking if two points are identical.
            tol: Error tolerance of the root finder.
            method: Method of root finding.
            n_seeking: How many fixed points to seek.

        """
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
        """Get more accurate values of the fixed points which have already been found.

        Args:
            tol: Tolerance of the root finder.
            method: Root finding method.
            maxiter: How many times to attempt refining a point.

        """
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
        """Places the fixed points in the order [dim, unstable, bright]."""
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
        """Finds the incoming vector at the unstable point which has components in the quantum dimensions."""
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
