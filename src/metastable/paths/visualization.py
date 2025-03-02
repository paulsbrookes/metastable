import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate._bvp import BVPResult
from typing import Tuple

def plot_solution(res: BVPResult, t_guess: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    t_plot = np.linspace(0, t_guess[-1], 1001)
    y0_plot = res.sol(t_plot)[0]
    y1_plot = res.sol(t_plot)[1]
    axes.plot(y0_plot, y1_plot)
    return fig, axes 