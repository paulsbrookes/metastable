import sympy
from pydantic import BaseModel


class Params(BaseModel):
    epsilon: float
    delta: float
    chi: float
    kappa: float = 1.0


class EOM:
    xc, pc, xq, pq = sympy.symbols("xc pc xq pq")
    sym = (xc, pc, xq, pq)

    def __init__(self, params: Params):
        self.params: Params = params

        self.H = (
            self.params.kappa * (self.xq**2 + self.pq**2)
            - self.params.kappa * (self.xc * self.xq + self.pc * self.pq)
            - self.params.delta * (self.xc * self.pq - self.pc * self.xq)
            - (self.params.chi / 2)
            * (self.xc**2 + self.pc**2 - self.xq**2 - self.pq**2)
            * (self.xc * self.pq - self.pc * self.xq)
            - 2 * self.params.epsilon * self.xq
        )

        dxc_over_dt = sympy.diff(self.H, self.xq)
        dpc_over_dt = sympy.diff(self.H, self.pq)
        dxq_over_dt = -sympy.diff(self.H, self.xc)
        dpq_over_dt = -sympy.diff(self.H, self.pc)

        self.y_dot_expr = sympy.Matrix(
            [dxc_over_dt, dpc_over_dt, dxq_over_dt, dpq_over_dt]
        )
        self.jacobian_expr = self.y_dot_expr.jacobian(self.sym)
        self.y_dot_func = lambda y: sympy.lambdify(self.sym, self.y_dot_expr)(*y)[:, 0]
        self.jacobian_func = lambda y: sympy.lambdify(self.sym, self.jacobian_expr)(*y)

        self.y_dot_classical_expr = sympy.Matrix(
            self.y_dot_expr.subs([(self.xq, 0), (self.pq, 0)])[0:2]
        )
        self.jacobian_classical_expr = self.y_dot_classical_expr.jacobian(self.sym[:2])
        self.y_dot_classical_func = lambda y: sympy.lambdify(
            self.sym[:2], self.y_dot_classical_expr
        )(*y)[:, 0]
        self.jacobian_classical_func = lambda y: sympy.lambdify(
            self.sym[:2], self.jacobian_classical_expr
        )(*y)
