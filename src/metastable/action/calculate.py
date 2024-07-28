from typing import Tuple


import scipy.integrate
from scipy.integrate._bvp import BVPResult


def calculate_action(bvp_result: BVPResult) -> Tuple[float, float]:
    def integrand_func(t):
        integrand = -bvp_result.sol(t, nu=1)[0] * bvp_result.sol(t)[2]
        integrand -= bvp_result.sol(t, nu=1)[1] * bvp_result.sol(t)[3]
        return integrand

    action, action_error = scipy.integrate.quad(
        integrand_func, 0, bvp_result.x[-1], limit=2000, epsabs=1e-2
    )

    return action, action_error
