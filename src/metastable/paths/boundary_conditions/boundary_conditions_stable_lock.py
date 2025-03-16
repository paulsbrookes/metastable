import numpy as np
from typing import Callable, Tuple
from numpy.typing import NDArray


from metastable.manifold_inverses import calculate_manifold_inverses
from metastable.eom import Params, EOM


def calculate_jacobian_eigenbasis(
    point: NDArray, params: Params
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Calculate the eigenvalues and eigenvectors of the Jacobian at a given point.
    
    Args:
        point: The point at which to calculate the Jacobian
        params: System parameters
        
    Returns:
        A tuple containing:
        - eigenvalues: Array of eigenvalues of the Jacobian, sorted by real part (ascending)
        - left_eigenvectors: Matrix where each row is a left eigenvector of the Jacobian
          (also called reciprocal eigenvectors), sorted to match eigenvalues
        - right_eigenvectors: Matrix where each column is a right eigenvector of the Jacobian,
          sorted to match eigenvalues
          
    Notes:
        Right eigenvectors satisfy the equation J·v = λ·v where v is a column vector.
        Left eigenvectors satisfy the equation u·J = λ·u where u is a row vector.
        The left eigenvectors are computed as the inverse of the right eigenvector matrix.
        These eigenvectors are used for projecting deviations onto the eigenbasis.
        All return values are sorted by the real part of the eigenvalues in ascending order.
    """
    eom = EOM(params)
    jacobian = eom.jacobian_func(point)
    
    eigenvalues, right_eigenvectors = np.linalg.eig(jacobian)
    
    # Sort by real part of eigenvalues
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    right_eigenvectors = right_eigenvectors[:, idx]
    
    left_eigenvectors = np.linalg.inv(right_eigenvectors)
    
    return eigenvalues, left_eigenvectors, right_eigenvectors


def generate_boundary_condition_func(
    keldysh_saddle_point: NDArray, keldysh_stable_point: NDArray, params: Params
) -> Callable[[NDArray, NDArray], NDArray]:
    """
    Generate a boundary condition function for finding paths between stable and saddle points.
    
    Args:
        keldysh_saddle_point: The saddle point in Keldysh coordinates
        keldysh_stable_point: The stable point in Keldysh coordinates
        params: System parameters
        
    Returns:
        A function that evaluates boundary conditions for a BVP solver
    """
    # Calculate Jacobian properties at the saddle point
    saddle_eigenvalues, saddle_left_eigenvectors, saddle_eigenvectors = calculate_jacobian_eigenbasis(
        keldysh_saddle_point, params
    )
    
    # Calculate Jacobian properties at the stable point
    stable_eigenvalues, stable_left_eigenvectors, stable_eigenvectors = calculate_jacobian_eigenbasis(
        keldysh_stable_point, params
    )
    
    # Find unstable directions at saddle point (positive real part eigenvalues)
    saddle_unstable_indices = np.where(np.real(saddle_eigenvalues) > 0)[0]
    
    # Find stable directions at stable point (negative real part eigenvalues)
    stable_stable_indices = np.where(np.real(stable_eigenvalues) < 0)[0]

    def calculate_components(ya: NDArray, yb: NDArray):
        """
        Calculate all components of the deviation vectors in the eigenvector basis.
        
        Args:
            ya: Point near the stable fixed point
            yb: Point near the saddle fixed point
            
        Returns:
            Dictionary containing all component projections
        """
        # Calculate all components at the stable point
        stable_components = np.dot(stable_left_eigenvectors, ya - keldysh_stable_point)
        
        # Calculate all components at the saddle point
        saddle_components = np.dot(saddle_left_eigenvectors, yb - keldysh_saddle_point)
        
        return stable_components, saddle_components
    
    def select_boundary_conditions(stable_components, saddle_components):
        """
        Select the appropriate components for boundary conditions.
        
        Args:
            components_dict: Dictionary of component projections
            
        Returns:
            Array of boundary condition values
        """
        
        # Check if magnitude of stable_components[2:4] is less than 1e-4
        stable_comp_magnitude = np.linalg.norm(stable_components[2:4])
        # Make stable_comp_condition increase linearly above the threshold
        threshold = 1e-2
        stable_comp_condition = 0.0 if stable_comp_magnitude < threshold else (stable_comp_magnitude - threshold)
        
        # Combine the components for the boundary conditions
        return np.hstack([
            np.abs(stable_components[0]+stable_components[1]) + stable_comp_condition,
            np.abs(stable_components[0]-stable_components[1]) + stable_comp_condition,
            np.abs(saddle_components[2]),
            np.abs(saddle_components[3]),
            ])

    def boundary_condition_func(ya: NDArray, yb: NDArray):
        """
        Evaluate boundary conditions for the BVP solver.
        
        Args:
            ya: Point near the stable fixed point
            yb: Point near the saddle fixed point
            
        Returns:
            Array of boundary condition values
        """
        stable_components, saddle_components = calculate_components(ya, yb)
        return select_boundary_conditions(stable_components, saddle_components)

    return boundary_condition_func
