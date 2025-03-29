from typing import Tuple


import scipy.integrate
from scipy.integrate._bvp import BVPResult


def integrand_func(t, bvp_result: BVPResult):
    integrand = -bvp_result.sol(t, nu=1)[0] * bvp_result.sol(t)[2]
    integrand -= bvp_result.sol(t, nu=1)[1] * bvp_result.sol(t)[3]
    return integrand


def calculate_action(bvp_result: BVPResult) -> Tuple[float, float]:
    action, action_error = scipy.integrate.quad(
        lambda t: integrand_func(t, bvp_result), 0, bvp_result.x[-1], limit=2000, epsabs=1e-2
    )

    return action, action_error
