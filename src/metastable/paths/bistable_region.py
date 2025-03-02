import numpy as np
from metastable.paths.data_structures import IndexPair, BistableBoundaries

def get_bistable_epsilon_range(
    bistable_region: np.ndarray, kappa_idx: int
) -> BistableBoundaries:
    """
    Get the epsilon indices at the bistable boundaries for a given kappa index.
    
    This function identifies the lower and upper epsilon indices that bound the bistable
    region at a specific kappa value.
    
    Args:
        bistable_region: 2D boolean array indicating where the system is bistable
        kappa_idx: Index in the kappa dimension
        
    Returns:
        BistableBoundary: A dataclass containing:
            - bright_saddle: IndexPair at the bright-saddle bifurcation (lower epsilon boundary)
            - dim_saddle: IndexPair at the dim-saddle bifurcation (upper epsilon boundary)
            Both values are None if no bistable region exists at the specified kappa.
    """
    bistable_epsilon = np.where(bistable_region[:, kappa_idx])[0]
    
    if len(bistable_epsilon) == 0:
        return BistableBoundaries(bright_saddle=None, dim_saddle=None)
        
    bright_saddle_idx = bistable_epsilon[0]  # Lower boundary is bright-saddle bifurcation
    dim_saddle_idx = bistable_epsilon[-1]    # Upper boundary is dim-saddle bifurcation
    
    bright_saddle_pair = IndexPair(epsilon_idx=bright_saddle_idx, kappa_idx=kappa_idx)
    dim_saddle_pair = IndexPair(epsilon_idx=dim_saddle_idx, kappa_idx=kappa_idx)
    
    return BistableBoundaries(bright_saddle=bright_saddle_pair, dim_saddle=dim_saddle_pair)


def get_bistable_kappa_range(
    bistable_region: np.ndarray, epsilon_idx: int
) -> BistableBoundaries:
    """
    Get the kappa indices at the bistable boundaries for a given epsilon index.
    
    This function identifies the lower and upper kappa indices that bound the bistable
    region at a specific epsilon value.
    
    Args:
        bistable_region: 2D boolean array indicating where the system is bistable
        epsilon_idx: Index in the epsilon dimension
        
    Returns:
        BistableBoundary: A dataclass containing:
            - dim_saddle: IndexPair at the dim-saddle bifurcation (lower kappa boundary)
            - bright_saddle: IndexPair at the bright-saddle bifurcation (upper kappa boundary)
            Both values are None if no bistable region exists at the specified epsilon.
    """
    bistable_kappa = np.where(bistable_region[epsilon_idx, :])[0]
    
    if len(bistable_kappa) == 0:
        return BistableBoundaries(dim_saddle=None, bright_saddle=None)
        
    dim_saddle_idx = bistable_kappa[0]       # Lower boundary is dim-saddle bifurcation
    bright_saddle_idx = bistable_kappa[-1]   # Upper boundary is bright-saddle bifurcation
    
    dim_saddle_pair = IndexPair(epsilon_idx=epsilon_idx, kappa_idx=dim_saddle_idx)
    bright_saddle_pair = IndexPair(epsilon_idx=epsilon_idx, kappa_idx=bright_saddle_idx)
    
    return BistableBoundaries(dim_saddle=dim_saddle_pair, bright_saddle=bright_saddle_pair) 