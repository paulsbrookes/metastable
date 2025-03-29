from dataclasses import dataclass
from typing import List, Optional

@dataclass
class IndexPair:
    epsilon_idx: int
    kappa_idx: int

@dataclass
class BistableBoundaries:
    dim_saddle: IndexPair
    bright_saddle: IndexPair

@dataclass
class Sweeps:
    """Sweep traversing the bistable region from each boundary point."""
    dim_saddle: List[IndexPair]  # Sweep starting at the dim-saddle bifurcation
    bright_saddle: List[IndexPair]  # Sweep starting at the bright-saddle bifurcation 