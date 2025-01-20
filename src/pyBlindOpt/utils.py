# coding: utf-8


'''
Utilities for optimization methods.
'''


__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import numpy as np


def check_bounds(solution:np.ndarray, bounds:np.ndarray) -> np.ndarray:
    '''
    Check if a solution is within the given bounds

    Args:
        solution (np.ndarray): the solution vector to be validated
        bounds (np.ndarray): the bounds of valid solutions

    Returns:
        np.ndarray: a clipped version of the solution vector
    '''
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return np.clip(solution, lower, upper)


def get_random_solution(bounds:np.ndarray) -> np.ndarray:
    '''
    Generates a random solutions that is within the bounds.

    Args:
        bounds (np.ndarray): the bounds of valid solutions

    Returns:
        np.ndarray: a random solutions that is within the bounds
    '''
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    return check_bounds(solution, bounds)


def scale(arr, min_val=None, max_val=None):
    if min_val is None:
        min_val = min(arr)
    
    if max_val is None:
        max_val = max(arr)

    scl_arr = (arr - min_val) / (max_val - min_val)
    return scl_arr, min_val, max_val


def inv_scale(scl_arr, min_val, max_val):
    return scl_arr*(max_val - min_val) + min_val


def global_distances(samples:np.ndarray)->np.ndarray:
    distances = np.zeros(len(samples))
    for i in range(len(samples)):
        s1 = samples[i]
        dist = 0.0
        for s2 in samples:
            dist += math.dist(s1, s2)
        distances[i] = dist
    return distances

def score_2_probs(scores:np.ndarray)->np.ndarray:
    total = np.sum(scores)
    norm_scores = scores/total
    norm_scores = (1.0-norm_scores)
    total = np.sum(norm_scores)
    norm_scores = norm_scores/total
    return norm_scores