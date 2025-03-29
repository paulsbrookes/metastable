from metastable.map.map import FixedPointMap
from metastable.generate_stability_map import generate_stability_map

if __name__ == '__main__':
    # Load the fixed point map
    map = FixedPointMap.load("map.npz")
    
    # Generate stability map
    map_with_stability = generate_stability_map(map, n_workers=20)
    
    # Save the updated map
    map_with_stability.save("map-with-stability.npz")
