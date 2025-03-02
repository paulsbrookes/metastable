import numpy as np
from typing import Callable, Tuple
from numpy.typing import NDArray


from metastable.manifold_inverses import calculate_manifold_inverses
from metastable.eom import Params, EOM


def calculate_jacobian_properties(
    point: NDArray, params: Params
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Calculate eigenvalues, eigenvectors, and reciprocal eigenvectors (left eigenvectors)
    of the Jacobian at a given point.
    
    Args:
        point: The state point at which to calculate the Jacobian
        params: System parameters
        
    Returns:
        Tuple containing:
            - eigenvalues: Array of eigenvalues
            - eigenvectors: Matrix where columns are the right eigenvectors
            - reciprocal_eigenvectors: Matrix where columns are the left eigenvectors
    """
    eom = EOM(params)
    jacobian = eom.jacobian_func(point)
    
    # Calculate eigenvalues and right eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    
    # Calculate left eigenvectors (reciprocal set)
    # For a matrix V of right eigenvectors, the left eigenvectors W satisfy W^T·V = I
    # So W^T = V^(-1) or W = (V^(-1))^T
    reciprocal_eigenvectors = np.linalg.inv(eigenvectors)
    
    return eigenvalues, eigenvectors, reciprocal_eigenvectors


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
    saddle_eigenvalues, saddle_eigenvectors, saddle_recip_eigenvectors = calculate_jacobian_properties(
        keldysh_saddle_point, params
    )
    
    # Calculate Jacobian properties at the stable point
    stable_eigenvalues, stable_eigenvectors, stable_recip_eigenvectors = calculate_jacobian_properties(
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
        stable_components = np.dot(stable_recip_eigenvectors, ya - keldysh_stable_point)
        
        # Calculate all components at the saddle point
        saddle_components = np.dot(saddle_recip_eigenvectors, yb - keldysh_saddle_point)
        
        return {
            'stable_components': stable_components,
            'saddle_components': saddle_components,
            'stable_eigenvalues': stable_eigenvalues,
            'saddle_eigenvalues': saddle_eigenvalues,
            'stable_stable_indices': stable_stable_indices,
            'saddle_unstable_indices': saddle_unstable_indices,
            'stable_eigenvectors': stable_eigenvectors,
            'stable_recip_eigenvectors': stable_recip_eigenvectors,
            'saddle_eigenvectors': saddle_eigenvectors,
            'saddle_recip_eigenvectors': saddle_recip_eigenvectors
        }
    
    def select_boundary_conditions(components_dict):
        """
        Select the appropriate components for boundary conditions.
        
        Args:
            components_dict: Dictionary of component projections
            
        Returns:
            Array of boundary condition values
        """
        # Extract components from dictionary
        stable_components = components_dict['stable_components']
        saddle_components = components_dict['saddle_components']
        stable_stable_indices = components_dict['stable_stable_indices']
        saddle_unstable_indices = components_dict['saddle_unstable_indices']
        
        # Select stable components at stable point
        stable_point_bc = np.abs(stable_components[stable_stable_indices])
        
        # Select unstable components at saddle point
        saddle_point_bc = np.abs(saddle_components[saddle_unstable_indices])
        
        # Combine the components for the boundary conditions
        return np.hstack([stable_point_bc, saddle_point_bc])

    def boundary_condition_func(ya: NDArray, yb: NDArray):
        """
        Evaluate boundary conditions for the BVP solver.
        
        Args:
            ya: Point near the stable fixed point
            yb: Point near the saddle fixed point
            
        Returns:
            Array of boundary condition values
        """
        components = calculate_components(ya, yb)
        return select_boundary_conditions(components)

    return boundary_condition_func
