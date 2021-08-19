# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import typing
import joblib
import logging
import tempfile
import numpy as np


logger = logging.getLogger(__name__)


# Cache from joblib
location = tempfile.gettempdir()
memory = joblib.Memory(location, verbose=0)


def hillclimbing(objective:typing.Callable, bounds:np.ndarray, n_iterations:int=200, step_size:float=.01) -> list:
    """
    Hill climbing local search algorithm.

    Args:
        objective (typing.Callable): objective fucntion
        bounds (np.ndarray): the bounds of valid solutions
        n_iterations (int): the number of iterations (default 200)
        step_size (float): the step size (default 0.01)

    Returns:
        list: [solution, solution_cost]
    """
    # cache the initial objective function
    objective_cache = memory.cache(objective)

    # min and max for each bound
    bounds_max = bounds.max(axis = 1)
    bounds_min = bounds.min(axis = 1)
    
    # generate an initial point
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # evaluate the initial point
    solution_cost = objective_cache(solution)
    # run the hill climb
    
    for _ in range(n_iterations):
		# take a step
        candidate = solution + np.random.randn(len(bounds)) * step_size
        
        # Fix out of bounds value
        candidate = np.minimum(candidate, bounds_max)
        candidate = np.maximum(candidate, bounds_min)
        
        # evaluate candidate point
        candidte_cost = objective_cache(candidate)
		# check if we should keep the new point
        
        if candidte_cost < solution_cost:
			# store the new point
            solution, solution_cost = candidate, candidte_cost
			# report progress
            #logger.info('>%d f(%s) = %.5f', i, solution, solution_cost)
    return [solution, solution_cost]
