# coding: utf-8

'''
Random Search (RS) is a family of numerical optimization methods that do not require 
the gradient of the problem to be optimized, and RS can hence be used on functions 
that are not continuous or differentiable. Such methods are commonly known as 
metaheuristics as they make few or no assumptions about the problem being optimized.
'''


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import tqdm
import joblib
import logging
import tempfile
import statistics
import numpy as np
import pyBlindOpt.utils as utils


from collections.abc import Sequence


logger = logging.getLogger(__name__)


def random_search(objective:callable, bounds:np.ndarray, population:list=None, 
callback:"Sequence[callable] | callable"=None, n_iter:int=100, n_pop:int=10, 
n_jobs:int=-1, cached:bool=False, debug:bool=False, verbose:bool=False, 
seed:int=42) -> tuple:
    '''
    Computes the Random Search optimization algorithm.

    Args:
        objective (callable): objective function used to evaluate the candidate solutions (lower is better)
        bounds (np.ndarray): bounds that limit the search space
        population (list): optional list of candidate solutions (default None)
        callback (callable): callback function that is called at each epoch (deafult None)
        n_iter (int): the number of iterations (default 100)
        n_pop (int): the number of elements in the population (default 10)
        n_jobs (int): number of concurrent jobs (default -1)
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
    
    # check if the initial population is given
    if population is None:
        # initialise population of candidate solutions randomly within the specified bounds
        pop = [utils.get_random_solution(bounds) for _ in range(n_pop)]
    else:
        # initialise population of candidate and validate the bounds
        pop = [utils.check_bounds(p, bounds) for p in population]
        # overwrite the n_pop with the length of the given population
        n_pop = len(population)
    
    # evaluate initial population of candidate solutions
    obj_all = utils.compute_objective(pop, objective_cache, n_jobs)

    # find the best performing vector of initial population
    best_vector = pop[np.argmin(obj_all)]
    best_obj = min(obj_all)
    
    # arrays to store the debug information
    if debug:
        obj_avg_iter = []
        obj_best_iter = []
        obj_worst_iter = []
    
    # run iterations of the algorithm
    for epoch in tqdm.tqdm(range(n_iter), disable=not verbose):
        # In Random Search, every iteration generates a completely new random population
        # (excluding the global best if we wanted to be elitist, but pure RS is memoryless.
        # However, to fit the framework, we generate n_pop new solutions every epoch).
        pop = [utils.get_random_solution(bounds) for _ in range(n_pop)]
        
        # evaluate the new population
        obj_all = utils.compute_objective(pop, objective_cache, n_jobs)

        # find the best performing vector in this batch
        current_best_obj = min(obj_all)
        
        # update global best if found
        if current_best_obj < best_obj:
            best_obj = current_best_obj
            best_vector = pop[np.argmin(obj_all)]

        ## Optional store the debug information
        if debug:
            # store best, worst and average cost for the current batch
            obj_avg_iter.append(statistics.mean(obj_all))
            # Note: For debug plots, we usually want the global best history, 
            # but strictly this list tracks the "current iteration" stats. 
            # We append the global best to keep consistency with other algos usually showing convergence.
            obj_best_iter.append(best_obj) 
            obj_worst_iter.append(max(obj_all))
    
        ## Optional execute the callback code
        if callback is not None:
            terminate = False
            if isinstance(callback, Sequence):
                terminate = any([c(epoch, obj_all, pop) for c in callback])
            else:
                terminate = callback(epoch, obj_all, pop)

            if terminate:
                break

    # clean the cache
    if cached:
        memory.clear(warn=False)
    
    if debug:
        return (best_vector, best_obj, (obj_best_iter, obj_avg_iter, obj_worst_iter))
    else:
        return (best_vector, best_obj)