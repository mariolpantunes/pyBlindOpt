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


from tqdm import tqdm


import optimization.utils as utils


logger = logging.getLogger(__name__)


def get_random_solution(bounds:np.ndarray) -> np.ndarray:
    """
    Generates a random solutions that is within the bounds.

    Args:
        bounds (np.ndarray): the bounds of valid solutions

    Returns:
        np.ndarray: a random solutions that is within the bounds
    """
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    solution = np.minimum(solution, bounds.max(axis = 1))
    solution = np.maximum(solution, bounds.min(axis = 1))
    return solution


# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = np.random.randint(len(pop))
	for ix in np.random.randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]


# genetic algorithm
def genetic_algorithm(objective:typing.Callable, bounds:np.ndarray,
crossover:typing.Callable, mutation:typing.Callable, population:np.ndarray=None,
selection:typing.Callable=selection, callback:typing.Callable=None,
n_iter:int=200, n_pop:int=20, r_cross:float=0.9, r_mut:float=0.3,
cached=False, debug=False, verbose=False) -> list:
    """
    Genetic optimization algorithm.

    Args:
        objective (typing.Callable): objective fucntion
        bounds (np.ndarray): the bounds of valid solutions
        selection (typing.Callable): selection fucntion
        crossover (typing.Callable): crossover fucntion
        mutation (typing.Callable): mutation fucntion
        n_iter (int): the number of iterations (default 100)
        n_pop (int): the number of elements in the population (default 10)
        r_cross (float): ratio of crossover (default 0.9)
        r_mut (float): ratio of mutation (default 0.2)

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
    
    # check if the initial population is given
    if population is None:
        # initial population of random bitstring
        pop = [get_random_solution(bounds) for _ in range(n_pop)]
    else:
        # initialise population of candidate and validate the bounds
        pop = [utils.check_bounds(p, bounds) for p in population]

	# keep track of best solution
    best, best_eval = 0, objective_cache(pop[0])
	
    # arrays to store the debug information
    if debug:
        obj_avg_iter = []
        obj_best_iter = []
        obj_worst_iter = []
    
    # enumerate generations
    for epoch in tqdm(range(n_iter), disable=not verbose):
        # evaluate all candidates in the population
        scores = joblib.Parallel(n_jobs=-1)(joblib.delayed(objective_cache)(c) for c in pop)

        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                #logger.info('>%d, new best f(%s) = %.3f' % (gen,  pop[i], scores[i]))
        
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = []
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut, bounds)
                # store for next generation
                children.append(c)
        
        # replace population
        pop = [utils.check_bounds(c, bounds) for c in children]
    
        ## Optional execute the callback code
        if callback is not None:
            callback(epoch, obj_all)

        ## Optional store the debug information
        if debug:
            # store best, wort and average cost for all candidates
            obj_avg_iter.append(statistics.mean(obj_all))
            obj_best_iter.append(best_obj)
            obj_worst_iter.append(max(obj_all))
    # clean the cache
    if cached:
        memory.clear(warn=False)
    
    if debug:
        return (best, best_eval, (obj_best_iter, obj_avg_iter, obj_worst_iter))
    else:
        return (best, best_eval)