import scipy
from qutip import (
    parallel_map,
    wigner,
    tracedist,
    qload,
    fock_dm,
    tensor,
    destroy,
    qeye,
    expect,
)
import numpy as np


def metastable_calc(
    rho_ss, rho_ad, x_limits=[-2, 2], n_x_points=21, offset=1e-6, return_coeffs=False
):
    rho_ad /= np.sqrt((rho_ad**2).tr())
    x_array = np.linspace(x_limits[0], x_limits[1], n_x_points)
    lowest_occupations = np.zeros(x_array.shape)
    for idx, x in enumerate(x_array):
        lowest_occupations[idx] = lowest_occupation_calc(x, rho_ss, rho_ad, offset)
    x_metastable_estimates = x_array[
        np.where((lowest_occupations[1:] * lowest_occupations[:-1]) < 0)
    ]
    assert x_metastable_estimates.shape[0] == 2
    x_divider = np.mean(x_metastable_estimates)

    res_1 = scipy.optimize.root_scalar(
        lowest_occupation_calc,
        bracket=(x_limits[0], x_divider),
        method="bisect",
        args=(rho_ss, rho_ad, offset),
    )

    res_2 = scipy.optimize.root_scalar(
        lowest_occupation_calc,
        bracket=(x_divider, x_limits[1]),
        method="bisect",
        args=(rho_ss, rho_ad, offset),
    )

    rho_1 = rho_ss + res_1.root * rho_ad
    rho_2 = rho_ss + res_2.root * rho_ad
    # rho_list = np.array([rho_1, rho_2], dtype=object)
    rho_list = [rho_1, rho_2]
    coeff_list = np.array([res_1.root, res_2.root])
    a = destroy(rho_ss.dims[0][0])
    a_list = [np.abs(expect(a, rho)) for rho in rho_list]
    # rho_list = rho_list[np.argsort(a_list)]
    rho_list = [rho_list[idx] for idx in np.argsort(a_list)]
    coeff_list = coeff_list[np.argsort(a_list)]
    if return_coeffs:
        return rho_list, coeff_list
    else:
        return rho_list


def metastable_calc_task(states, kwargs={}):
    rho_ss = states[0]
    rho_ad = states[1]
    out = metastable_calc(rho_ss, rho_ad, **kwargs)
    # metastable_states = [rho_d, rho_b]
    return out


def lowest_occupation_calc(x, rho_ss, rho_ad, offset):
    rho = rho_ss + x * rho_ad
    lowest_occupation = rho.eigenenergies(eigvals=1)[0]
    lowest_occupation += offset
    return lowest_occupation


def overlap_objective(p_b, rho_ss, rho_b, rho_d):
    rho = p_b * rho_b + (1 - p_b) * rho_d
    objective = tracedist(rho_ss, rho)
    return objective


def p_b_calc(rho_ss, rho_b, rho_d):
    res = scipy.optimize.minimize_scalar(
        overlap_objective, args=(rho_ss, rho_b, rho_d), method="brent"
    )
    return res.x
