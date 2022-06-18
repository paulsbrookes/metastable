import numpy as np


def calc_beta_limits_from_duffing(delta=0, chi=0, kappa=0):
    delta_omega, lam, beta, Gamma = calc_dykman_params(delta, chi, kappa=kappa)
    beta_1, beta_2 = calc_beta_limits(delta_omega, Gamma)
    return np.array([beta_1, beta_2])


def calc_b_from_duffing(delta=0, chi=0, kappa=0):
    delta_omega, lam, beta, Gamma = calc_dykman_params(delta, chi, kappa=kappa)
    Omega = delta_omega / Gamma
    Y_B_1 = (1.0 / 3.0) * (2 + (1 - 3 * Omega ** (-2)) ** (0.5))
    Y_B_2 = (1.0 / 3.0) * (2 - (1 - 3 * Omega ** (-2)) ** (0.5))
    beta_1, beta_2 = calc_beta_limits(delta_omega, Gamma)
    b_1 = (
        -(beta_1**0.5)
        * (2 * Y_B_1) ** (-1)
        * (1 - 2 * (Omega**2) * Y_B_1 + Omega**2)
    )
    b_2 = (
        -(beta_2**0.5)
        * (2 * Y_B_2) ** (-1)
        * (1 - 2 * (Omega**2) * Y_B_2 + Omega**2)
    )
    return np.array([b_1, b_2])


def calc_eta_from_duffing(delta=0, chi=0, kappa=0, eps=0):
    delta_omega, lam, beta, Gamma = calc_dykman_params(delta, chi, eps=eps, kappa=kappa)
    beta_1, beta_2 = calc_beta_limits_from_duffing(delta=delta, chi=chi, kappa=kappa)
    eta_1 = beta - beta_1
    eta_2 = beta - beta_2
    return np.array([eta_1, eta_2])


def dykman_calc(delta=0, chi=0, eps=0, kappa=0, n_c=0, kappa_phi=0, components=False):
    delta_omega, lam, beta, Gamma = calc_dykman_params(delta, chi, eps, kappa)
    Omega = delta_omega / Gamma

    Gamma_ph = kappa_phi / 2
    n = n_c
    chi_ph = Gamma_ph / (lam * Gamma)

    beta_1, beta_2 = calc_beta_limits(delta_omega, Gamma)
    Y_B_1 = (1.0 / 3.0) * (2 + (1 - 3 * Omega ** (-2)) ** (0.5))
    Y_B_2 = (1.0 / 3.0) * (2 - (1 - 3 * Omega ** (-2)) ** (0.5))
    D_B_1 = Omega ** (-1) * ((n + 0.5) + 0.5 * chi_ph * (1 - Y_B_1))
    D_B_2 = Omega ** (-1) * ((n + 0.5) + 0.5 * chi_ph * (1 - Y_B_2))

    b_1 = (
        -(beta_1**0.5)
        * (2 * Y_B_1) ** (-1)
        * (1 - 2 * (Omega**2) * Y_B_1 + Omega**2)
    )
    b_2 = (
        -(beta_2**0.5)
        * (2 * Y_B_2) ** (-1)
        * (1 - 2 * (Omega**2) * Y_B_2 + Omega**2)
    )

    eta_1 = beta - beta_1
    eta_2 = beta - beta_2

    C_1 = np.abs(delta_omega) * (b_1 * eta_1 / 2) ** 0.5 / (np.pi * beta_1**0.25)
    C_2 = np.abs(delta_omega) * (b_2 * eta_2 / 2) ** 0.5 / (np.pi * beta_2**0.25)

    R_A_1 = (
        np.sqrt(2)
        * (np.abs(eta_1) ** 1.5)
        / (3 * D_B_1 * (np.abs(b_1) ** 0.5) * (beta_1**0.75))
    )
    R_A_2 = (
        np.sqrt(2)
        * (np.abs(eta_2) ** 1.5)
        / (3 * D_B_2 * (np.abs(b_2) ** 0.5) * (beta_2**0.75))
    )

    if components:
        return np.array([[C_1, R_A_1 / lam], [C_2, R_A_2 / lam]])
    else:
        W_sw_1 = C_1 * np.exp(R_A_1 / lam)
        W_sw_2 = C_2 * np.exp(R_A_2 / lam)
        return np.array([W_sw_1, W_sw_2])


def calc_eps(delta_omega, lam, beta):
    eps = -delta_omega * np.sqrt(beta / (2 * lam))
    return eps


def calc_eps_limits(chi, delta, kappa):
    delta_omega, lam, beta, Gamma = calc_dykman_params(delta, chi, kappa=kappa)
    beta_limits = calc_beta_limits(delta_omega, Gamma)
    delta, chi, eps_limits, kappa = calc_duffing_params(delta_omega, lam, beta_limits)
    return eps_limits


def calc_duffing_params(delta_omega, lam, beta, Gamma=None):
    eps = -delta_omega * np.sqrt(beta / (2 * lam))
    chi = lam * delta_omega
    delta = (2 * lam - 1) * delta_omega
    if Gamma is not None:
        kappa = 2 * Gamma
    else:
        kappa = None
    return delta, chi, eps, kappa


def calc_dykman_params(delta, chi, eps=None, kappa=None):
    delta_omega = 2 * chi - delta
    lam = chi / delta_omega
    if eps is not None:
        beta = 2 * lam * eps**2 / delta_omega**2
    else:
        beta = None
    if kappa is not None:
        Gamma = kappa / 2
    else:
        Gamma = None
    return delta_omega, lam, beta, Gamma


def calc_beta_limits(delta_omega, Gamma):
    Omega = np.abs(delta_omega) / Gamma
    beta_1 = (2.0 / 27) * (1 + 9 * Omega ** (-2) - (1 - 3 * Omega ** (-2)) ** (1.5))
    beta_2 = (2.0 / 27) * (1 + 9 * Omega ** (-2) + (1 - 3 * Omega ** (-2)) ** (1.5))
    return np.array([beta_1, beta_2])


def potential_func(delta_P, b, beta_limit, eta):
    U = b * delta_P**3 / 3 - eta * delta_P / (np.sqrt(beta_limit * 2))
    return U
