import sympy
from pydantic import BaseModel


class Params(BaseModel):
    epsilon: float
    delta: float
    chi: float
    kappa: float = 1.0


class EOM:
    # Dynamic variables
    xc, pc, xq, pq = sympy.symbols("xc pc xq pq")
    sym = (xc, pc, xq, pq)
    
    # Parameters as symbols
    epsilon_sym, delta_sym, chi_sym, kappa_sym = sympy.symbols("epsilon delta chi kappa")
    param_sym = (epsilon_sym, delta_sym, chi_sym, kappa_sym)

    # Hamiltonian as class variable with symbolic parameters
    H_sym = (
        kappa_sym * (xq**2 + pq**2)
        - kappa_sym * (xc * xq + pc * pq)
        - delta_sym * (xc * pq - pc * xq)
        - (chi_sym / 2)
        * (xc**2 + pc**2 - xq**2 - pq**2)
        * (xc * pq - pc * xq)
        + 2 * epsilon_sym * xq
    )

    # Pre-compute symbolic expressions
    y_dot_expr = sympy.Matrix([
        sympy.diff(H_sym, xq),
        sympy.diff(H_sym, pq),
        -sympy.diff(H_sym, xc),
        -sympy.diff(H_sym, pc)
    ])
    jacobian_expr = y_dot_expr.jacobian(sym)
    
    y_dot_classical_expr = sympy.Matrix(
        y_dot_expr.subs([(xq, 0), (pq, 0)])[0:2]
    )
    jacobian_classical_expr = y_dot_classical_expr.jacobian(sym[:2])

    def __init__(self, params: Params):
        self.params = params
        
        # Create parameter substitution list
        param_subs = [
            (self.epsilon_sym, params.epsilon),
            (self.delta_sym, params.delta),
            (self.chi_sym, params.chi),
            (self.kappa_sym, params.kappa)
        ]
        
        # Substitute parameters into expressions
        self.H = self.H_sym.subs(param_subs)
        self.y_dot_expr_sub = self.y_dot_expr.subs(param_subs)
        self.jacobian_expr_sub = self.jacobian_expr.subs(param_subs)
        self.y_dot_classical_expr_sub = self.y_dot_classical_expr.subs(param_subs)
        self.jacobian_classical_expr_sub = self.jacobian_classical_expr.subs(param_subs)

        # Create lambda functions with substituted expressions
        self.y_dot_func = lambda y: sympy.lambdify(self.sym, self.y_dot_expr_sub)(*y)[:, 0]
        self.jacobian_func = lambda y: sympy.lambdify(self.sym, self.jacobian_expr_sub)(*y)
        self.y_dot_classical_func = lambda y: sympy.lambdify(
            self.sym[:2], self.y_dot_classical_expr_sub
        )(*y)[:, 0]
        self.jacobian_classical_func = lambda y: sympy.lambdify(
            self.sym[:2], self.jacobian_classical_expr_sub
        )(*y)
