import numpy as np
from typing import List, Hashable
from numpy.typing import NDArray
from scipy import optimize


from metastable.eom import EOM, Params


def calculate_fixed_points(params: Params, guesses: NDArray) -> NDArray:
    eom = EOM(params=params)
    results = [
        optimize.root(eom.y_dot_classical_func, guess, tol=1e-10, method="hybr")
        for guess in guesses
    ]
    fixed_points = np.array(
        [res.x if res.success else np.array([np.nan, np.nan]) for res in results]
    )
    hashable_fixed_points = [
        tuple(np.round(vector, decimals=7)) for vector in fixed_points
    ]
    duplicate_groups = _find_duplicate_groups(hashable_fixed_points)
    validated_fixed_points = np.copy(fixed_points)
    for group in duplicate_groups:
        validated_fixed_points[group] = _filter_closest_vectors(
            guesses[group], fixed_points[group]
        )
    return validated_fixed_points


def _filter_closest_vectors(guesses, results):
    distances = np.linalg.norm(guesses - results, axis=1)
    index_of_closest_vector = np.argmin(distances)
    filtered_results = np.full_like(results, np.nan)
    filtered_results[index_of_closest_vector] = results[index_of_closest_vector]
    return filtered_results


def _find_duplicate_groups(items: List[Hashable]) -> List[List[int]]:
    index_groups: dict[Hashable, List[int]] = {}
    for index, obj in enumerate(items):
        if obj in index_groups:
            index_groups[obj].append(index)
        else:
            index_groups[obj] = [index]
    groups_of_indexes: List[List[int]] = list(index_groups.values())
    groups_of_duplicates: List[List[int]] = [
        group for group in groups_of_indexes if len(group) > 1
    ]
    return groups_of_duplicates
