import numpy as np


from metastable.rescaled import calculate_kappa_rescaled, calculate_beta_limits


def calc_dykman_params(delta, chi, eps):
    delta_omega = -delta
    lam = chi / delta_omega
    beta = 2 * lam * eps**2 / delta_omega**2
    return delta_omega, lam, beta


def dykman_actions_calc(delta, chi, eps, kappa):

    delta_omega, lam, beta = calc_dykman_params(delta, chi, eps)

    n = 0.0
    chi_ph = 0.0

    kappa_rescaled = calculate_kappa_rescaled(kappa, delta)

    Omega = 1 / kappa_rescaled

    beta_1, beta_2 = calculate_beta_limits(kappa_rescaled)
    Y_B_1 = (1.0 / 3.0) * (2 + (1 - 3 * Omega ** (-2)) ** (0.5))
    Y_B_2 = (1.0 / 3.0) * (2 - (1 - 3 * Omega ** (-2)) ** (0.5))
    D_B_1 = Omega ** (-1) * ((n + 0.5) + 0.5 * chi_ph * (1 - Y_B_1))
    D_B_2 = Omega ** (-1) * ((n + 0.5) + 0.5 * chi_ph * (1 - Y_B_2))

    b_1 = -(beta_1**0.5) * (2 * Y_B_1) ** (-1) * (1 - 2 * (Omega**2) * Y_B_1 + Omega**2)
    b_2 = -(beta_2**0.5) * (2 * Y_B_2) ** (-1) * (1 - 2 * (Omega**2) * Y_B_2 + Omega**2)

    eta_1 = beta - beta_1
    eta_2 = beta - beta_2

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

    return -R_A_1 / lam, -R_A_2 / lam
