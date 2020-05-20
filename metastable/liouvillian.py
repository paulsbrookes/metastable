from copy import deepcopy
from qutip import *
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.optimize import root

try:
    import quimb as qu
except:
    print('Quimb not available.')

import scipy
from .calc_metastable_occupation import lowest_occupation_calc


def fixed_point_tracker_duffing(params_frame, alpha0=0, fill_value=None, threshold=1e-4, crosscheck_frame=None,
                                columns=['a']):
    trip = False
    index = params_frame.index
    amplitude_array = np.zeros([len(index), 1], dtype=complex)
    for i, idx in tqdm(enumerate(index)):
        params = params_frame.loc[idx]
        if not trip:
            alpha_fixed = locate_fixed_point_mf_duffing(params, alpha0=[alpha0.real, alpha0.imag])
            if alpha_fixed is None:
                amplitude_array[i, :] = [fill_value]
            else:
                amplitude_array[i, :] = [alpha_fixed]
                alpha0 = alpha_fixed

    amplitude_frame = pd.DataFrame(amplitude_array, index=params_frame.index, columns=columns)
    amplitude_frame.sort_index(inplace=True)
    return amplitude_frame


def mf_characterise_duffing(params_frame):
    alpha0 = 0
    reversed_params_frame = params_frame.iloc[::-1]
    mf_amplitude_frame_bright = fixed_point_tracker_duffing(reversed_params_frame, alpha0=alpha0)
    mf_amplitude_frame_dim = fixed_point_tracker_duffing(params_frame, alpha0=alpha0, columns=['a_dim'],
                                                         crosscheck_frame=mf_amplitude_frame_bright)
    mf_amplitude_frame_bright.columns = ['a_bright']
    mf_amplitude_frame = pd.concat([mf_amplitude_frame_bright, mf_amplitude_frame_dim], axis=1)

    overlap_find = ~np.isclose(mf_amplitude_frame['a_bright'].values, mf_amplitude_frame['a_dim'].values)

    if not np.any(overlap_find):
        mf_amplitude_frame = pd.DataFrame(mf_amplitude_frame.values[:, 0], index=mf_amplitude_frame.index,
                                          columns=['a'])
    else:
        start_idx = np.where(overlap_find)[0][0]
        end_idx = np.where(overlap_find)[0][-1]

        mf_amplitude_frame['a_bright'].iloc[0:start_idx] = None
        mf_amplitude_frame['a_dim'].iloc[end_idx + 1:] = None

    return mf_amplitude_frame


def dalpha_calc_mf_duffing(alpha, params):
    dalpha = params.eps - 1j * (params.fc - params.fd) * alpha - 2 * 1j * params.chi * alpha * np.abs(
        alpha) ** 2 - 0.5 * params.kappa * alpha
    return dalpha


def classical_eom_mf_duffing(x, params):
    alpha = x[0] + 1j * x[1]
    dalpha = dalpha_calc_mf_duffing(alpha, params)
    dx = np.array([dalpha.real, dalpha.imag])
    return dx


def locate_fixed_point_mf_duffing(params, alpha0=(0, 0)):
    x0 = np.array([alpha0[0], alpha0[1]])
    res = root(classical_eom_mf_duffing, x0, args=(params,), method='hybr')
    if res.success:
        alpha = res.x[0] + 1j * res.x[1]
    else:
        alpha = None
    return alpha


class Parameters:
    def __init__(self, fc=None, Ej=None, g=None, Ec=None, eps=None, fd=None, kappa=None, gamma=None, t_levels=None,
                 c_levels=None, gamma_phi=None, kappa_phi=None, n_t=None, n_c=None):
        self.fc = fc
        self.Ej = Ej
        self.eps = eps
        self.g = g
        self.Ec = Ec
        self.gamma = gamma
        self.kappa = kappa
        self.t_levels = t_levels
        self.c_levels = c_levels
        self.fd = fd
        self.gamma_phi = gamma_phi
        self.kappa_phi = kappa_phi
        self.n_t = n_t
        self.n_c = n_c
        self.labels = ['f_d', 'eps', 'E_j', 'f_c', 'g', 'kappa', 'kappa_phi', 'gamma', 'gamma_phi', 'E_c', 'n_t', 'n_c']

    def copy(self):
        params = Parameters(self.fc, self.Ej, self.g, self.Ec, self.eps, self.fd, self.kappa, self.gamma, self.t_levels,
                            self.c_levels, self.gamma_phi, self.kappa_phi, self.n_t, self.n_c)
        return params


def ham_duffing_gen(params):
    a = destroy(params.c_levels)
    H = (params.fc - params.fd) * a.dag() * a + params.chi * a.dag() * a.dag() * a * a + 1j * params.eps * (a.dag() - a)
    return H


def c_ops_duffing_gen(params):
    a = destroy(params.c_levels)
    c_ops = [np.sqrt((1 + params.n_c) * params.kappa) * a]
    if params.n_c > 0.0:
        c_ops.append(np.sqrt(params.n_c * params.kappa) * a.dag())
    return c_ops


def state_cut_gen(params, results_all=None, threshold=0.1, extend=0.001,
                  iterations=2, prune_threshold=0.23, fd_mf_limits=[10.4, 10.5],
                  backend='slepc', sigma=-0.001, k=2, n_frequencies=21, fd_array=None):
    if fd_array is None:
        if results_all is None:
            fd_array = np.linspace(fd_mf_limits[0], fd_mf_limits[1], 2001)
            mf_amplitude_frame = mf_characterise_duffing(params, fd_array)
            fd_lower = mf_amplitude_frame.dropna().index[0] - extend
            fd_upper = mf_amplitude_frame.dropna().index[-1] + extend
            fd_array = np.linspace(fd_lower, fd_upper, n_frequencies)
        else:
            fd_array = _new_frequencies_gen(results_all.index, np.abs(results_all['rate_ad'].values), threshold=threshold)

    for i in range(iterations):
        print('epsilon = %f, iteration = %i, number of frequencies = %i' % (params.eps, i, fd_array.shape[0]))
        for fd_idx, fd in tqdm(enumerate(fd_array)):
            params.fd = fd
            ham = ham_duffing_gen(params)
            c_ops = c_ops_duffing_gen(params)
            L = -liouvillian(ham, c_ops)
            try:
                rates, states = _eigensolver_wrapper(L.data, backend=backend, sigma=sigma, k=k)
                # rates, states = L.eigenstates(eigvals=2, sort='high', sparse=True, tol=0.1, maxiter=1e5)
                states = [vector_to_operator(state) for state in states]
                states = [state for state in states]
                states = [state + state.dag() for state in states]
                states[0] /= states[0].tr()
                results = [[rates[0], rates[1], states[0], states[1]]]
                results = pd.DataFrame(results, columns=['rate_ss', 'rate_ad', 'state_ss', 'state_ad'],
                                       index=[params.fd])
                if results_all is None:
                    results_all = results
                else:
                    results_all = pd.concat([results_all, results])
            except Exception as e:
                print(e, params.fd, params.eps)
        results_all.sort_index(inplace=True)
        if prune_threshold is not None:
            mask = -1 / results_all.rate_ad.real > prune_threshold
            results_all = results_all[mask]
        if i < iterations - 1:
            fd_array = _new_frequencies_gen(results_all.index, np.abs(results_all['rate_ad'].values),
                                            threshold=threshold)

    results_all.index.name = 'fd'

    return results_all


def _derivative(x, y, n_derivative=1):
    derivatives = np.zeros(y.size - 1)
    positions = np.zeros(x.size - 1)
    for index in np.arange(y.size - 1):
        grad = (y[index + 1] - y[index]) / (x[index + 1] - x[index])
        position = np.mean([x[index], x[index + 1]])
        derivatives[index] = grad
        positions[index] = position

    if n_derivative > 1:
        positions, derivatives = _derivative(positions, derivatives, n_derivative - 1)

    return positions, derivatives


def _moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    averages = np.convolve(interval, window, 'same')
    return averages[window_size - 1: averages.size]


def _new_frequencies_gen(x, y, threshold=10.0):
    n_points = len(y)
    curvature_positions, curvatures = _derivative(x, y, 2)
    abs_curvatures = np.abs(curvatures)
    mean_curvatures = _moving_average(abs_curvatures, 2)
    midpoint_curvatures = np.concatenate((np.array([abs_curvatures[0]]), mean_curvatures))
    midpoint_curvatures = np.concatenate((midpoint_curvatures, np.array([abs_curvatures[n_points - 3]])))
    midpoint_transmissions = _moving_average(y, 2)
    midpoint_curvatures_normed = midpoint_curvatures / midpoint_transmissions
    midpoints = _moving_average(x, 2)
    intervals = np.diff(x)
    num_of_sections_required = np.ceil(intervals * np.sqrt(midpoint_curvatures_normed / threshold)).astype(int)
    new_fd_points = np.array([])
    for index in np.arange(n_points - 1):
        multi_section = np.linspace(x[index], x[index + 1], num_of_sections_required[index] + 1)
        new_fd_points = np.concatenate((new_fd_points, multi_section))
    unique_set = set(new_fd_points) - set(x)
    new_fd_points_unique = np.array(list(unique_set))
    return new_fd_points_unique


def _eigensolver_wrapper(L, backend='slepc', sigma=-0.001, k=2):
    rates, states = qu.eig(L, k=k, backend=backend, sigma=sigma)
    rates = np.flip(rates)
    states = np.flip(states, axis=1)
    states_Qobj = [Qobj(states[:, i]) for i in range(states.shape[1])]
    c_levels = int(np.sqrt(states_Qobj[0].shape[0]))
    dims = [[[c_levels], [c_levels]], [1, 1]]
    for state in states_Qobj:
        state.dims = dims
    return rates, states_Qobj


def ham_gen(params):
    a = destroy(params['c_levels'])
    H = (params['fc'] - params['fd']) * a.dag() * a + params['chi'] * a.dag() * a.dag() * a * a + 1j * params['eps'] * (
                a.dag() - a)
    return H


def c_ops_gen(params):
    a = destroy(params['c_levels'])
    c_ops = [np.sqrt((1 + params['n_c']) * params['kappa']) * a]
    if params['n_c'] > 0.0:
        c_ops.append(np.sqrt(params['n_c'] * params['kappa']) * a.dag())
    return c_ops


def calc_liouvillian_eigenstates(L, backend='slepc', sigma=-0.001, k=2):
    rates, states_raw = qu.eig(L.data, k=k, backend=backend, sigma=sigma)
    #rates = np.flip(rates)
    #states_raw = np.flip(states_raw, axis=1)
    states_vec = [Qobj(states_raw[:, i]) for i in range(states_raw.shape[1])]

    c_levels = int(np.sqrt(states_vec[0].shape[0]))
    dims = [[[c_levels], [c_levels]], [1, 1]]
    for state in states_vec:
        state.dims = dims

    states = [vector_to_operator(state) for state in states_vec]
    states = [state + state.dag() for state in states]
    states[0] /= states[0].tr()

    return rates, states


def task(params, k=2, backend='scipy', sigma=-0.001):
    ham = ham_gen(params)
    c_ops = c_ops_gen(params)
    L = -liouvillian(ham, c_ops)
    rates, states = calc_liouvillian_eigenstates(L, k=k, backend=backend, sigma=sigma)
    return rates, states


def liouvillian_eigenstate_sweep(params, sweep_param_name, sweep_param_values, k=2, backend='scipy', sigma=-0.001):
    columns = ['rate_' + str(n) for n in range(k)] + ['state_' + str(n) for n in range(k)]
    content = []
    for idx, value in tqdm(enumerate(sweep_param_values)):
        params_instance = params.copy()
        params_instance[sweep_param_name] = value
        try:
            rates, states = task(params_instance, k=k, backend=backend, sigma=sigma)
        except Exception as e:
            print(e, params['fd'], params['eps'])
        content += [list(rates) + states]
    results = pd.DataFrame(content, columns=columns, index=sweep_param_values)
    results.index.name = sweep_param_name
    return results


def calc_metastable_states(rho_ss, rho_ad, x_limits=[-2, 2], n_x_points=21, offset=1e-10):
    rho_ad /= np.sqrt((rho_ad ** 2).tr())
    x_array = np.linspace(x_limits[0], x_limits[1], n_x_points)
    lowest_occupations = np.zeros(x_array.shape)
    for idx, x in enumerate(x_array):
        lowest_occupations[idx] = lowest_occupation_calc(x, rho_ss, rho_ad, offset)
    x_metastable_estimates = x_array[np.where((lowest_occupations[1:] * lowest_occupations[:-1]) < 0)]
    if x_metastable_estimates.shape[0] != 2:
        print('Could not find two coefficient estimates.')
        packaged_results = pd.DataFrame([[None, None, None, None]], columns=['rho_d', 'rho_b', 'p_d', 'p_b'])
        return packaged_results

    x_divider = np.mean(x_metastable_estimates)

    res_1 = scipy.optimize.root_scalar(lowest_occupation_calc, bracket=(x_limits[0], x_divider),
                                       method='bisect', args=(rho_ss, rho_ad, offset))

    res_2 = scipy.optimize.root_scalar(lowest_occupation_calc, bracket=(x_divider, x_limits[1]),
                                       method='bisect', args=(rho_ss, rho_ad, offset))

    rho_1 = rho_ss + res_1.root * rho_ad
    rho_2 = rho_ss + res_2.root * rho_ad
    rho_array = np.array([rho_1, rho_2], dtype=object)
    coeffs = np.array([res_1.root, res_2.root])
    a = destroy(rho_ss.dims[0][0])
    a_list = [np.abs(expect(a, rho)) for rho in rho_array]
    rho_array = rho_array[np.argsort(a_list)]
    coeffs = coeffs[np.argsort(a_list)]

    p_d = -coeffs[0] / (coeffs[1] - coeffs[0])
    p_b = coeffs[1] / (coeffs[1] - coeffs[0])

    occupations = np.array([p_d, p_b])

    packaged_results = pd.DataFrame([np.hstack((rho_array, occupations))], columns=['rho_d', 'rho_b', 'p_d', 'p_b'])

    return packaged_results


def calc_metastable_task(liouvillian_eigenstates, **kwargs):
    rho_ss = liouvillian_eigenstates['state_ss'].iloc[0]
    rho_ad = liouvillian_eigenstates['state_ad'].iloc[0]
    metastable_states = calc_metastable_states(rho_ss, rho_ad, **kwargs)
    metastable_states.index = liouvillian_eigenstates.index
    return metastable_states


def calc_overlap_column(frame):
    overlaps = []
    for rho_d, rho_b in zip(frame['rho_d'], frame['rho_b']):
        if (rho_d is not None) and (rho_b is not None):
            overlaps.append((rho_d * rho_b).tr())
        else:
            overlaps.append(None)
    return overlaps