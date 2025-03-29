from pathlib import Path


from metastable.map.map import FixedPointMap
from metastable.generate_fixed_point_map import generate_fixed_point_map


# Example usage:
if __name__ == "__main__":
    map: FixedPointMap = generate_fixed_point_map(
        epsilon_max=1.0,
        kappa_max=1.0,
        epsilon_points=601,
        kappa_points=401,
        delta=7.8,
        chi=-0.1,
        max_workers=20,
    )

    map.save(Path("map.npz"))
