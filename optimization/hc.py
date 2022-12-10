# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import tqdm
import typing
import joblib
import logging
import tempfile
import numpy as np
import optimization.utils as utils


logger = logging.getLogger(__name__)


# Cache from joblib
location = tempfile.gettempdir()
memory = joblib.Memory(location, verbose=0)


def hillclimbing(objective:typing.Callable, bounds:np.ndarray,
callback:typing.Callable=None, n_iter:int=200, step_size:float=.01,
cached=False, debug=False, verbose=False) -> tuple:
    """
    Hill climbing local search algorithm.

    Args:
        objective (typing.Callable): objective fucntion
        bounds (np.ndarray): the bounds of valid solutions
        n_iter (int): the number of iterations (default 200)
        step_size (float): the step size (default 0.01)

    Returns:
        list: [solution, solution_cost]
    """
    # cache the initial objective function
    if cached:
        # Cache from joblib
        location = tempfile.gettempdir()
        memory = joblib.Memory(location, verbose=0)
        objective_cache = memory.cache(objective)
    else:
        objective_cache = objective

    # generate an initial point
    solution = utils.get_random_solution(bounds)
    # evaluate the initial point
    solution_cost = objective_cache(solution)
    # run the hill climb
    
    for epoch in tqdm.tqdm(range(n_iter), disable=not verbose):
		# take a step
        candidate = solution + np.random.randn(len(bounds)) * step_size
        # Fix out of bounds value
        candidate = utils.check_bounds(candidate, bounds)
        # evaluate candidate point
        candidate_cost = objective_cache(candidate)
		# check if we should keep the new point
        
        if candidate_cost < solution_cost:
			# store the new point
            solution, solution_cost = candidate, candidate_cost
        
        ## Optional execute the callback code
        if callback is not None:
            callback(epoch, solution_cost, candidate_cost)
    
    if cached:
        memory.clear(warn=False)

    return (solution, solution_cost)
