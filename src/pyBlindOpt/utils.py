# coding: utf-8

'''
Utilities for optimization methods.
'''

__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import math
import joblib
import numpy as np
from collections.abc import Callable

def assert_bounds(solution: np.ndarray, bounds: np.ndarray) -> bool:
    """
    Verifies if the solution is contained within the defined bounds.

    Args:
        solution (np.ndarray): The solution vector(s) to check.
        bounds (np.ndarray): The bounds of valid solutions (shape: N x 2).

    Returns:
        bool: True if the solution is within bounds, False otherwise.
    """
    x = solution.T
    min_bounds = bounds[:, 0]
    max_bounds = bounds[:, 1]
    rv = ((x > min_bounds[:, np.newaxis]) & (x < max_bounds[:, np.newaxis])).any(1)
    return bool(np.all(rv))


def check_bounds(solution: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    '''
    Check if a solution is within the given bounds.
    If not, values are clipped to the nearest bound.

    Args:
        solution (np.ndarray): The solution vector to be validated.
        bounds (np.ndarray): The bounds of valid solutions (shape: N x 2).

    Returns:
        np.ndarray: A clipped version of the solution vector.
    '''
    return np.clip(solution, bounds[:, 0], bounds[:, 1])


def get_random_solution(bounds: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    '''
    Generates a random solution that is within the bounds.

    Args:
        bounds (np.ndarray): The bounds of valid solutions (shape: N x 2).
                             Column 0 is min, Column 1 is max.
        rng (np.random.Generator | None, optional): A numpy random generator instance. 
                                                    If None, a new one is created.
    
    Returns:
        np.ndarray: A random solution within the bounds.
    '''
    # 1. Handle Generator Initialization
    if rng is None:
        rng = np.random.default_rng()
    
    # 2. Optimized Generation
    # rng.uniform implements the scaling and shifting in C-level code
    return rng.uniform(low=bounds[:, 0], high=bounds[:, 1])


def scale(arr: np.ndarray, min_val: float | np.ndarray | None = None, max_val: float | np.ndarray | None = None) -> tuple[np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Scales an array to the [0, 1] range using Min-Max scaling.

    Args:
        arr (np.ndarray): The input array to scale.
        min_val (float | np.ndarray | None, optional): Minimum value for scaling. If None, computed from arr.
        max_val (float | np.ndarray | None, optional): Maximum value for scaling. If None, computed from arr.

    Returns:
        tuple[np.ndarray, float | np.ndarray, float | np.ndarray]: 
            - The scaled array.
            - The minimum value used.
            - The maximum value used.
    """
    # Use strict temporary variables to ensure type safety (guaranteed not None)
    actual_min = np.min(arr) if min_val is None else min_val
    actual_max = np.max(arr) if max_val is None else max_val

    # Avoid division by zero if max == min
    denominator = actual_max - actual_min
    
    if np.any(denominator == 0):
        scl_arr = np.zeros_like(arr)
    else:
        scl_arr = (arr - actual_min) / denominator
        
    return scl_arr, actual_min, actual_max


def inv_scale(scl_arr: np.ndarray, min_val: float | np.ndarray, max_val: float | np.ndarray) -> np.ndarray:
    """
    Inverse scales an array from [0, 1] back to the original range.

    Args:
        scl_arr (np.ndarray): The scaled array.
        min_val (float | np.ndarray): The minimum value used in the original scaling.
        max_val (float | np.ndarray): The maximum value used in the original scaling.

    Returns:
        np.ndarray: The array rescaled to the original range.
    """
    return scl_arr * (max_val - min_val) + min_val


def global_distances(samples: np.ndarray) -> np.ndarray:
    """
    Computes the sum of Euclidean distances from each sample to every other sample.

    Args:
        samples (np.ndarray): An array of samples (shape: N_samples x N_features).

    Returns:
        np.ndarray: An array where the i-th element is the sum of distances 
                    between sample i and all other samples.
    """
    distances = np.zeros(len(samples))
    for i in range(len(samples)):
        s1 = samples[i]
        dist = 0.0
        for s2 in samples:
            dist += math.dist(s1, s2)
        distances[i] = dist
    return distances


def score_2_probs(scores: np.ndarray) -> np.ndarray:
    """
    Converts a vector of scores into a probability distribution.
    
    This method normalizes the scores, inverts them (so lower scores get 
    higher probabilities), and renormalizes.

    Args:
        scores (np.ndarray): Input scores (usually objective function values).

    Returns:
        np.ndarray: A probability distribution summing to 1.0.
    """
    total = np.sum(scores)
    
    # Avoid division by zero
    if total == 0:
        norm_scores = np.ones_like(scores) / len(scores)
    else:
        norm_scores = scores / total
        
    norm_scores = (1.0 - norm_scores)
    total_norm = np.sum(norm_scores)
    
    if total_norm == 0:
        return np.ones_like(scores) / len(scores)
        
    norm_scores = norm_scores / total_norm
    return norm_scores


def compute_objective(population: list | np.ndarray, function: Callable[[object], float], n_jobs: int = -1) -> np.ndarray:
    """
    Computes the objective function for a population of solutions, optionally in parallel.

    Args:
        population (list | np.ndarray): The population of solutions to evaluate.
        function (Callable[[object], float]): The objective function to apply to each solution.
        n_jobs (int, optional): The number of parallel jobs to run. 
                                -1 uses all available processors. 
                                1 disables parallelism. Defaults to -1.

    Returns:
        np.ndarray: A NumPy array of objective values corresponding to the population.
    """
    # If n_jobs is 1 or None, skip joblib overhead entirely
    if n_jobs == 1 or n_jobs is None:
        # List comprehension is usually faster than pre-allocated numpy assignment 
        # for dynamic function calls in Python.
        obj_list = [function(c) for c in population]
        return np.array(obj_list)
    else:
        try:
            # Joblib returns a list; we convert it to an array immediately.
            # Backend 'loky' is robust for generic Python objects.
            obj_list = joblib.Parallel(backend='loky', n_jobs=n_jobs)(
                joblib.delayed(function)(c) for c in population
            )
        except Exception:
            # Fallback to threading if serialization fails
            obj_list = joblib.Parallel(backend='threading', n_jobs=n_jobs)(
                joblib.delayed(function)(c) for c in population
            )
        
        return np.array(obj_list)