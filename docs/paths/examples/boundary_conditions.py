from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths import (
    get_bistable_kappa_range, 
    generate_sweep_index_pairs,
    map_switching_paths
)
from metastable.eom import Params, EOM
from metastable.extend_to_keldysh import extend_to_keldysh_state
# Import the plotting function from epsilon_sweep
from metastable.paths.visualization import plot_parameter_sweeps
import numpy as np


if __name__ == "__main__":

    # Load the fixed point map
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)

    epsilon_idx = 250
    kappa_idx = 130

    params = Params(
        epsilon=fixed_point_map.epsilon_linspace[epsilon_idx],
        kappa=fixed_point_map.kappa_linspace[kappa_idx],
        delta=fixed_point_map.delta,
        chi=fixed_point_map.chi,
    )
    eom = EOM(params)

    fixed_point_type = FixedPointType.BRIGHT.value

    fixed_point = extend_to_keldysh_state(fixed_point_map.fixed_points[epsilon_idx, kappa_idx, fixed_point_type])


    eigenvalues = fixed_point_map.eigenvalues[epsilon_idx, kappa_idx, fixed_point_type]

    # eigenvectors is a 4x4 array where the first index is the component and the second index is the eigenvalue
    eigenvectors = fixed_point_map.eigenvectors[epsilon_idx, kappa_idx, fixed_point_type]
    
    # Calculate the reciprocal set (left eigenvectors)
    # For a matrix V of right eigenvectors, the left eigenvectors W satisfy W^T·V = I
    # So W^T = V^(-1) or W = (V^(-1))^T
    reciprocal_eigenvectors = np.linalg.inv(eigenvectors).T
    
    # Now reciprocal_eigenvectors[:, i] is the left eigenvector corresponding to eigenvalues[i]
    
    # Verify biorthogonality: the dot product of right eigenvector i and left eigenvector j
    # should be 1 if i=j and 0 otherwise
    biorthogonality_check = np.zeros((eigenvectors.shape[1], eigenvectors.shape[1]), dtype=np.complex128)
    for i in range(eigenvectors.shape[1]):
        for j in range(eigenvectors.shape[1]):
            biorthogonality_check[i, j] = np.dot(reciprocal_eigenvectors[:, i], eigenvectors[:, j])
    
    print("Eigenvalues:", eigenvalues)
    print("Biorthogonality check (should be identity matrix):")
    print(np.round(biorthogonality_check, decimals=10))
