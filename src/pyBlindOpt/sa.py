# coding: utf-8


'''
Simulated annealing is a probabilistic technique for approximating the global 
optimum of a given function. Specifically, it is a metaheuristic to approximate
global optimization in a large search space for an optimization problem.
'''


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import tqdm
import joblib
import logging
import tempfile
import numpy as np
import pyBlindOpt.utils as utils


from collections.abc import Sequence


logger = logging.getLogger(__name__)


def simulated_annealing(objective:callable, bounds:list,
callback:"Sequence[callable] | callable"=None, n_iter:int=200, step_size:float=0.01, 
temp:float=20.0, cached:bool=False, debug:bool=False, verbose:bool=False, seed:int=42) -> tuple:
    '''
    Simulated annealing algorithm.

    Args:
        objective (callable): objective function used to evaluate the candidate solutions (lower is better)
        bounds (list): bounds that limit the search space
        callback (callable): callback function that is called at each epoch (deafult None)
        n_iter (int): the number of iterations (default 200)
        step_size (float): the step size (default 0.01)
        temp (float): initial temperature (default 20.0)
        cached (bool): controls if the objective function is cached by joblib (default False)
        debug (bool): controls if debug information is returned (default False)
        verbose (bool): controls the usage of tqdm as a progress bar (default False)
        seed (int): seed to init the random generator (default 42)

    Returns:
        tuple: the best solution
    '''
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
    best = utils.get_random_solution(bounds)
    
    # evaluate the initial point
    best_cost = objective_cache(best)
	
    # current working solution
    curr, curr_cost = best, best_cost
    
     # arrays to store the debug information
    if debug:
        cost_iter = []
	
    # run the algorithm
    for i in tqdm.tqdm(range(n_iter), disable=not verbose):
		# take a step
        candidate = curr + np.random.randn(len(bounds)) * step_size
        # fix out of bounds value
        candidate = utils.check_bounds(candidate, bounds)
		# evaluate candidate point
        candidate_cost = objective_cache(candidate)
		# check for new best solution
        if candidate_cost < best_cost:
			# store new best point
            best, best_cost = candidate, candidate_cost
            
		# difference between candidate and current point evaluation
        diff = candidate_cost - curr_cost
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = np.exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or np.random.rand() < metropolis:
            # store the new current point
            curr, curr_cost = candidate, candidate_cost
        
         ## Optional store the debug information
        if debug:
            # store the best cost
            cost_iter.append(best_cost)

        ## Optional execute the callback code
        if callback is not None:
            terminate = False
            if isinstance(callback, Sequence):
                terminate = any([c(i, [best_cost, candidate_cost], [best, candidate]) for c in callback])
            else:
                terminate = callback(i, [best_cost, candidate_cost], [best, candidate])

            if terminate:
                break

    if cached:
        memory.clear(warn=False)

    if debug:
        return (best, best_cost, cost_iter)
    else:
        return (best, best_cost)
