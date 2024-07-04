import numpy as np
import scipy
import contextlib
import io
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from tqdm import tqdm
from pathlib import Path


from metastable.state import FixedPointMap
from metastable.eom import EOM, Params
from metastable.incoming_quantum_vector import extend_to_keldysh_state
from metastable.generate_guess import generate_linear_guess, generate_guess_from_sol
from metastable.generate_boundary_conditions import generate_boundary_condition_func


@dataclass
class IndexPair:
    epsilon_idx: int
    kappa_idx: int


map_path = "/home/paul/Projects/keldysh/metastable/00-attempt/map.npz"

output_path = Path("/home/paul/Projects/keldysh/metastable/00-attempt/output")
# Check if the directory exists
if output_path.exists():
    raise FileExistsError(f"The directory {output_path} already exists.")
else:
    # Create the directory
    output_path.mkdir(parents=True)



fixed_point_map = FixedPointMap.load(map_path)


def extract_params(fixed_point_map: FixedPointMap, index_pair: IndexPair) -> Params:
    return Params(
        epsilon=fixed_point_map.epsilon_linspace[index_pair.epsilon_idx],
        kappa=fixed_point_map.kappa_linspace[index_pair.kappa_idx],
        delta=fixed_point_map.delta,
        chi=fixed_point_map.chi,
    )


def prepare_saddle_and_focus_points(
    fixed_point_map: FixedPointMap, index_pair: IndexPair
) -> Tuple[np.ndarray, np.ndarray]:
    classical_saddle_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, 0
    ]
    classical_focus_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, 2
    ]

    keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
    keldysh_focus_point = extend_to_keldysh_state(classical_focus_point)

    return keldysh_saddle_point, keldysh_focus_point


def solve_bvp(
    eom: EOM, boundary_condition_func, t_guess: np.ndarray, y_guess: np.ndarray
) -> scipy.integrate.OdeSolution:
    wrapper = lambda x, y: eom.y_dot_func(y)
    return scipy.integrate.solve_bvp(
        wrapper,
        boundary_condition_func,
        t_guess,
        y_guess,
        tol=1e-3,
        max_nodes=1000000,
        verbose=2,
    )

def process_index(
    fixed_point_map: FixedPointMap,
    index_pair: IndexPair,
    t_guess: np.ndarray,
    y_guess: np.ndarray,
) -> scipy.integrate.OdeSolution:
    params = extract_params(fixed_point_map, index_pair)
    print(params)

    eom = EOM(params=params)

    keldysh_saddle_point, keldysh_focus_point = prepare_saddle_and_focus_points(
        fixed_point_map, index_pair
    )

    boundary_condition_func = generate_boundary_condition_func(
        keldysh_saddle_point, keldysh_focus_point, params
    )

    path_result = solve_bvp(eom, boundary_condition_func, t_guess, y_guess)

    return path_result


def generate_linear_guess_from_map(
    fixed_point_map: FixedPointMap, index_pair: IndexPair
) -> Tuple[np.ndarray, np.ndarray]:
    classical_saddle_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, 0
    ]
    classical_focus_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, 2
    ]
    keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
    keldysh_focus_point = extend_to_keldysh_state(classical_focus_point)
    t_guess, y_guess = generate_linear_guess(
        keldysh_focus_point, keldysh_saddle_point, 8.0
    )

    return t_guess, y_guess


def plot_solution(res: scipy.integrate.OdeSolution, t_guess: np.ndarray):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    t_plot = np.linspace(0, t_guess[-1], 1001)
    y0_plot = res.sol(t_plot)[0]
    y1_plot = res.sol(t_plot)[1]
    axes.plot(y0_plot, y1_plot)
    return fig, axes


def map_switching_paths(
    fixed_point_map: FixedPointMap,
        index_list: List[IndexPair],
        output_path: Path
) -> List[scipy.integrate.OdeSolution]:
    t_guess, y_guess = generate_linear_guess_from_map(fixed_point_map, index_list[0])

    output_map_path = output_path / "output_map.npz"
    log_path = output_path / "logs"
    log_path.mkdir(parents=True)

    results = []

    for index_pair in tqdm(index_list):
        log_file_path = log_path / f"(epsilon_idx,kappa_idx)-({index_pair.epsilon_idx},{index_pair.kappa_idx}).log"
        plot_file_path = log_path / f"(epsilon_idx,kappa_idx)-({index_pair.epsilon_idx},{index_pair.kappa_idx}).png"

        with open(log_file_path, 'w') as log_file:
            log_file.write('-' * 80 + '\n')
            log_file.write(f"epsilon_idx: {index_pair.epsilon_idx}, kappa_idx: {index_pair.kappa_idx}\n")
            log_file.write('-' * 80 + '\n')

            # Create a StringIO object to capture stdout
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                path_result = process_index(fixed_point_map, index_pair, t_guess, y_guess)

            # Write the captured stdout to the log file
            log_file.write(stdout_buffer.getvalue())

            log_file.write('-' * 80 + '\n')
            log_file.write(f"Message: {path_result.message} Success: {path_result.success}\n")
            log_file.write('-' * 80 + '\n')

            fig, _ = plot_solution(path_result, t_guess)
            fig.savefig(plot_file_path, bbox_inches='tight')

            fixed_point_map.path_results[index_pair.epsilon_idx, index_pair.kappa_idx] = path_result
            fixed_point_map.save_state(output_map_path)

            t_guess, y_guess = generate_guess_from_sol(bvp_result=path_result, t_end=8.0)

            results.append(path_result)

    return results


# Example list of IndexPair for epsilon and kappa indexes
epsilon_idx = 360
seed_kappa_idx = 230
index_list = [
    IndexPair(epsilon_idx, seed_kappa_idx),
    IndexPair(epsilon_idx, seed_kappa_idx - 1),
]

# Call the function with the list of IndexPair
results = map_switching_paths(fixed_point_map, index_list, output_path)
