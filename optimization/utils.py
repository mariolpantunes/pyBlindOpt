# coding: utf-8


'''
Utilities for optimization methods.
'''


__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


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