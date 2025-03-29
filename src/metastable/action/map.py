from metastable.action.calculate import calculate_action
from metastable.map.map import FixedPointMap


def map_actions(fixed_point_map: FixedPointMap) -> FixedPointMap:
    """
    Calculate actions for all paths in a FixedPointMap.
    
    This function creates a copy of the input map and calculates the action values
    and their associated errors for all bright-to-saddle and dim-to-saddle paths
    where BVP results exist.
    
    Args:
        fixed_point_map: The original FixedPointMap containing path results
        
    Returns:
        A new FixedPointMap with calculated action values and errors
    """
    # Create a deep copy of the map to avoid modifying the original
    new_map = fixed_point_map.copy()
    
    # Get dimensions of the parameter space
    n_epsilon = len(fixed_point_map.epsilon_linspace)
    n_kappa = len(fixed_point_map.kappa_linspace)
    
    # Iterate through all parameter combinations
    for eps_idx in range(n_epsilon):
        for kap_idx in range(n_kappa):
            # Check both path types
            for path_type in range(2):  # 0: BRIGHT_TO_SADDLE, 1: DIM_TO_SADDLE
                # Get the BVP result for this parameter combination and path type
                bvp_result = fixed_point_map.path_results[eps_idx, kap_idx, path_type]
                
                # Calculate action if a path result exists
                if bvp_result is not None:
                    try:
                        action, error = calculate_action(bvp_result)
                        new_map._path_actions[eps_idx, kap_idx, path_type, 0] = action
                        new_map._path_actions[eps_idx, kap_idx, path_type, 1] = error
                    except Exception as e:
                        # Keep NaN values in case of calculation errors
                        print(f"Error calculating action at ε={fixed_point_map.epsilon_linspace[eps_idx]}, "
                              f"κ={fixed_point_map.kappa_linspace[kap_idx]}, path_type={path_type}: {e}")
    
    return new_map
