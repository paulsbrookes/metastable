from typing import Optional
from metastable.paths.data_structures import IndexPair, BistableBoundaries, Sweeps

def generate_sweep_index_pairs(
    boundary: BistableBoundaries, 
    max_points: Optional[int] = None,
    sweep_fraction: float = 1.0,
    dim_sweep_fraction: Optional[float] = None,
    bright_sweep_fraction: Optional[float] = None
) -> Sweeps:
    """
    Generate paths of IndexPair objects that traverse the bistable region.
    
    Args:
        boundary: BistableBoundary object containing the dim_saddle and bright_saddle points
        max_points: Optional maximum number of points to include in each path. If None, uses the maximum
                 distance between start and target indices (either epsilon or kappa) as the number of points.
        sweep_fraction: Float between 0 and 1 indicating what fraction of the path to traverse.
                 1.0 means go all the way from one endpoint to the other.
                 0.5 means go halfway from each endpoint toward the other.
                 This is used as the default for both paths if specific fractions are not provided.
        dim_sweep_fraction: Optional float between 0 and 1 for the dim_saddle path specifically.
                 If provided, overrides sweep_fraction for the dim_saddle path.
        bright_sweep_fraction: Optional float between 0 and 1 for the bright_saddle path specifically.
                 If provided, overrides sweep_fraction for the bright_saddle path.
        
    Returns:
        Sweeps object containing:
            - dim_saddle: List of IndexPair objects from dim_saddle toward bright_saddle
            - bright_saddle: List of IndexPair objects from bright_saddle toward dim_saddle
            
    Raises:
        ValueError: If boundary points are None or invalid or if any sweep_fraction is outside [0,1]
    """
    if boundary.dim_saddle is None or boundary.bright_saddle is None:
        raise ValueError("Both boundary points must be defined")
    
    if not 0 <= sweep_fraction <= 1:
        raise ValueError("sweep_fraction must be between 0 and 1")
    
    # Use specific sweep fractions if provided, otherwise use the default
    dim_fraction = dim_sweep_fraction if dim_sweep_fraction is not None else sweep_fraction
    bright_fraction = bright_sweep_fraction if bright_sweep_fraction is not None else sweep_fraction
    
    # Validate the specific sweep fractions
    if not 0 <= dim_fraction <= 1:
        raise ValueError("dim_sweep_fraction must be between 0 and 1")
    if not 0 <= bright_fraction <= 1:
        raise ValueError("bright_sweep_fraction must be between 0 and 1")
    
    # Extract coordinates
    dim_epsilon = boundary.dim_saddle.epsilon_idx
    dim_kappa = boundary.dim_saddle.kappa_idx
    bright_epsilon = boundary.bright_saddle.epsilon_idx
    bright_kappa = boundary.bright_saddle.kappa_idx
    
    # Calculate target points based on sweep_fractions
    target_dim_epsilon = dim_epsilon + dim_fraction * (bright_epsilon - dim_epsilon)
    target_dim_kappa = dim_kappa + dim_fraction * (bright_kappa - dim_kappa)
    target_bright_epsilon = bright_epsilon + bright_fraction * (dim_epsilon - bright_epsilon)
    target_bright_kappa = bright_kappa + bright_fraction * (dim_kappa - bright_kappa)
    
    # Determine number of points if not specified, based on distance to target
    if max_points is None:
        dim_epsilon_distance = abs(dim_epsilon - target_dim_epsilon)
        dim_kappa_distance = abs(dim_kappa - target_dim_kappa)
        bright_epsilon_distance = abs(bright_epsilon - target_bright_epsilon)
        bright_kappa_distance = abs(bright_kappa - target_bright_kappa)
        
        dim_max_distance = max(dim_epsilon_distance, dim_kappa_distance)
        bright_max_distance = max(bright_epsilon_distance, bright_kappa_distance)
        
        dim_points = int(dim_max_distance) + 1  # +1 to include both endpoints
        bright_points = int(bright_max_distance) + 1  # +1 to include both endpoints
    else:
        dim_points = max_points
        bright_points = max_points
    
    # Ensure we have at least 2 points (start and end)
    dim_points = max(2, dim_points)
    bright_points = max(2, bright_points)
    
    # Create evenly spaced points for dim to bright path
    dim_to_bright = []
    for i in range(dim_points):
        # Linear interpolation between endpoints
        t = i / (dim_points - 1)  # t goes from 0 to 1
        epsilon_idx = round(dim_epsilon + t * (target_dim_epsilon - dim_epsilon))
        kappa_idx = round(dim_kappa + t * (target_dim_kappa - dim_kappa))
        dim_to_bright.append(IndexPair(epsilon_idx=epsilon_idx, kappa_idx=kappa_idx))
    
    # Create evenly spaced points for bright to dim path
    bright_to_dim = []
    for i in range(bright_points):
        # Linear interpolation between endpoints
        t = i / (bright_points - 1)  # t goes from 0 to 1
        epsilon_idx = round(bright_epsilon + t * (target_bright_epsilon - bright_epsilon))
        kappa_idx = round(bright_kappa + t * (target_bright_kappa - bright_kappa))
        bright_to_dim.append(IndexPair(epsilon_idx=epsilon_idx, kappa_idx=kappa_idx))
    
    return Sweeps(dim_saddle=dim_to_bright, bright_saddle=bright_to_dim) 