import sympy
from pydantic import BaseModel


class Params(BaseModel):
    epsilon: float
    delta: float
    chi: float
    kappa: float = 1.0


class EOM:
    x1, x2, p1, p2 = sympy.symbols("x1 x2 p1 p2")
    sym = (x1, x2, p1, p2)

    def __init__(self, params: Params):
        self.params: Params = params

        self.H = (
            self.params.kappa * (self.p1**2 + self.p2**2)
            - self.params.kappa * (self.x1 * self.p1 + self.x2 * self.p2)
            - self.params.delta * (self.x1 * self.p2 - self.x2 * self.p1)
            - (self.params.chi / 2)
            * (self.x1**2 + self.x2**2 - self.p1**2 - self.p2**2)
            * (self.x1 * self.p2 - self.x2 * self.p1)
            - 2 * self.params.epsilon * self.p1
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
