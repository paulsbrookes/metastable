import pandas as pd
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


def metastable_calc_task_duffing(states):
    rho_ss = states[0]
    rho_ad = states[1]
    try:
        rho_d, rho_b = metastable_calc_optimization_duffing(rho_ss, rho_ad)
        metastable_states = [rho_d, rho_b]
    except Exception as e:
        print(e)
        metastable_states = [None, None]
    return metastable_states


def metastable_calc_optimization_duffing(
    rho_ss, rho_adr, mode_idx=0, min_dist=10, ranges=None
):
    kwargs = {"min_dist": min_dist, "ranges": ranges}

    rho_ss /= rho_ss.tr()
    rho_adr /= (rho_adr.dag() * rho_adr).tr()

    rho_c_ss = rho_ss.ptrace(mode_idx)
    rho_c_adr = rho_adr.ptrace(mode_idx)

    res = scipy.optimize.minimize(
        objective_calc, 0.0, method="Nelder-Mead", args=(rho_c_adr, rho_c_ss, kwargs)
    )
    rho_d = rho_adr + res.x[0] * rho_ss
    rho_d /= rho_d.tr()

    rho_2 = rho_d
    rho_c_2 = rho_2.ptrace(mode_idx)

    rho_1 = rho_adr
    rho_1 -= rho_2 * (rho_1 * rho_2).tr() / (rho_2 * rho_2).tr()
    rho_c_1 = rho_1.ptrace(mode_idx)
    res = scipy.optimize.minimize(
        objective_calc, 0.0, method="Nelder-Mead", args=(rho_c_1, rho_c_2, kwargs)
    )
    rho_b_adr = rho_1 + res.x[0] * rho_2
    rho_b_adr /= rho_b_adr.tr()

    rho_1 = rho_ss
    rho_1 -= rho_2 * (rho_1 * rho_2).tr() / (rho_2 * rho_2).tr()
    rho_c_1 = rho_1.ptrace(mode_idx)
    res = scipy.optimize.minimize(
        objective_calc, 0.0, method="Nelder-Mead", args=(rho_c_1, rho_c_2, kwargs)
    )
    rho_b_ss = rho_1 + res.x[0] * rho_2
    rho_b_ss /= rho_b_ss.tr()

    states_b = [rho_b_ss, rho_b_adr]
    distances = [tracedist(rho_b, rho_d) for rho_b in states_b]
    rho_b = states_b[np.argmax(distances)]

    dims = rho_b.dims
    components = (
        [qeye(levels) for levels in dims[0][:mode_idx]]
        + [destroy(dims[0][mode_idx])]
        + [qeye(levels) for levels in dims[0][mode_idx + 1 :]]
    )
    a_op = tensor(components)
    a_exp = [np.abs(expect(a_op, rho_d)), np.abs(expect(a_op, rho_b))]
    states = np.array([rho_d, rho_b], dtype=object)
    states = states[np.argsort(a_exp)]

    return states[0], states[1]


def objective_calc(x, rho1, rho2, kwargs={}):
    rho = rho1 + x[0] * rho2
    ratio = ratio_calc(rho, **kwargs)
    return ratio


def ratio_calc(
    state, n_bins=100, return_peaks=False, min_dist=10, ranges=[[40, 60], [45, 75]]
):
    xvec = np.linspace(-15, 15, n_bins)
    W = np.abs(wigner(state, xvec, xvec, g=2))
    if ranges is not None:
        W[ranges[0][0] : ranges[0][1], ranges[1][0] : ranges[1][1]] = 0
    peak_indices = maximum_finder(W)
    peak_heights = np.array(
        [
            W[peak_indices[0][idx], peak_indices[1][idx]]
            for idx in range(len(peak_indices[0]))
        ]
    )
    peaks = pd.DataFrame(
        np.array([peak_indices[0], peak_indices[1], peak_heights]).T,
        columns=["i", "j", "height"],
    )
    dtypes = {"i": int, "j": int, "height": float}
    peaks = peaks.astype(dtypes)
    peaks.sort_values(by="height", axis=0, ascending=False, inplace=True)

    i_diff = peaks.iloc[0].i - peaks.iloc[1].i
    j_diff = peaks.iloc[0].j - peaks.iloc[1].j
    dist = np.sqrt(i_diff**2 + j_diff**2)
    if dist < min_dist:
        index = peaks.index[1]
        peaks.drop(index=index, inplace=True)

    ratio = peaks["height"].iloc[1] / peaks["height"].iloc[0]
    if return_peaks:
        return ratio, peaks
    else:
        return ratio


def maximum_finder(data):
    n_rows = data.shape[0]
    n_columns = data.shape[1]
    row_indices = []
    column_indices = []
    for i in range(1, n_rows - 1):
        for j in range(1, n_columns - 1):
            if maximum_yn(i, j, data):
                row_indices.append(i)
                column_indices.append(j)
    row_indices = np.array(row_indices)
    column_indices = np.array(column_indices)
    return row_indices, column_indices


def maximum_yn(i, j, data):
    if data[i, j] > data[i + 1, j] and data[i, j] > data[i - 1, j]:
        if data[i, j] > data[i, j + 1] and data[i, j] > data[i, j - 1]:
            if data[i, j] > data[i + 1, j + 1] and data[i, j] > data[i - 1, j - 1]:
                if data[i, j] > data[i - 1, j + 1] and data[i, j] > data[i + 1, j - 1]:
                    return True
    return False
