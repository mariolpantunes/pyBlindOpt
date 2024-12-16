# coding: utf-8


'''
Hill climbing is a mathematical optimization technique that belongs to the 
family of local search. It is an iterative algorithm that starts with an 
arbitrary solution to a problem and then attempts to find a better solution 
by making an incremental change to the solution.
'''


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
import pyBlindOpt.utils as utils


from collections.abc import Sequence


logger = logging.getLogger(__name__)


def hillclimbing(objective:typing.Callable, bounds:list,
callback:"Sequence[callable] | callable"=None, n_iter:int=200, step_size:float=.01,
cached:bool=False, debug:bool=False, verbose:bool=False, seed:int=42) -> tuple:
    """
    Hill climbing local search algorithm.

    Args:
        objective (typing.Callable): objective function used to evaluate the candidate solutions (lower is better)
        bounds (list): bounds that limit the search space
        callback (typing.Callable): callback function that is called at each epoch (deafult None)
        n_iter (int): the number of iterations (default 200)
        step_size (float): the step size (default 0.01)
        cached (bool): controls if the objective function is cached by joblib (default False)
        debug (bool): controls if debug information is returned (default False)
        verbose (bool): controls the usage of tqdm as a progress bar (default False)
        seed (int): seed to init the random generator (default 42)

    Returns:
        tuple: the best solution
    """
    # define the seed of the random generation
    np.random.seed(seed)

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
    
    # arrays to store the debug information
    if debug:
        obj_cost_iter = []
    
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
        
        ## Optional store the debug information
        if debug:
            # store the best cost
            obj_cost_iter.append(solution_cost)
        
        ## Optional execute the callback code
        if callback is not None:
            terminate = False
            if isinstance(callback, Sequence):
                terminate = any([c(epoch, [solution_cost, candidate_cost], [solution, candidate]) for c in callback])
            else:
                terminate = callback(epoch, [solution_cost, candidate_cost], [solution, candidate])

            if terminate:
                break
    
    if cached:
        memory.clear(warn=False)

    if debug:
        return (solution, solution_cost, obj_cost_iter)
    else:
        return (solution, solution_cost)
