import numpy as np


from numpy.typing import NDArray


def generate_guess_from_sol(bvp_result, t_end: float):
    t_guess = np.linspace(0.0, t_end, 10001)
    y_guess = bvp_result.sol(t_guess)
    return t_guess, y_guess


def generate_linear_guess(y0: NDArray, y1: NDArray, t_end: float):
    t_guess = np.linspace(0.0, t_end, 10001)
    y_guess = (
        y0[:, np.newaxis] + t_guess[np.newaxis, :] * (y1 - y0)[:, np.newaxis] / t_end
    )
    return t_guess, y_guess
