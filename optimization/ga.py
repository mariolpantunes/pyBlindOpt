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


# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = np.random.randint(len(pop))
	for ix in np.random.randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]


# random mutation operator
def random_mutation(candidate, r_mut, bounds):
    if np.random.rand() < r_mut:
        solution = utils.get_random_solution(bounds)
        for i in range(len(candidate)):
            candidate[i] = solution[i]


# linear crossover operator: two parents to create three children
def linear_crossover(p1, p2, r_cross):
    if np.random.rand() < r_cross:
        c1 = 0.5*p1 + 0.5*p2
        c2 = 1.5*p1 - 0.5*p2
        c3 = -0.5*p1 + 1.5*p2
        return [c1, c2, c3]
    else:
        return [p1, p2]


# blend crossover operator: two parents to create two children
def blend_crossover(p1, p2, r_cross, alpha=.5):
    if np.random.rand() < r_cross:
        c1 = p1 + alpha*(p2-p1)
        c2 = p2 - alpha*(p2-p1)
        return [c1, c2]
    else:
        return [p1, p2]


def genetic_algorithm(objective:typing.Callable, bounds:np.ndarray,
crossover:typing.Callable=blend_crossover, mutation:typing.Callable=random_mutation, 
population:np.ndarray=None, selection:typing.Callable=selection, 
callback:typing.Callable=None, n_iter:int=200, n_pop:int=20, r_cross:float=0.9, 
r_mut:float=0.3, n_jobs:int=-1, cached=False, debug=False, verbose=False, seed:int=42) -> tuple:
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
        # initial population of random bitstring
        pop = [utils.get_random_solution(bounds) for _ in range(n_pop)]
    else:
        # initialise population of candidate and validate the bounds
        pop = [utils.check_bounds(p, bounds) for p in population]
        # overwrite the n_pop with the length of the given population
        n_pop = len(population)

	# keep track of best solution
    best, best_eval = 0, objective_cache(pop[0])
	
    # arrays to store the debug information
    if debug:
        obj_avg_iter = []
        obj_best_iter = []
        obj_worst_iter = []
    
    # define the limit for the selection method (work with even size population)
    selection_limit = n_pop - (n_pop%2)

    # enumerate generations
    for epoch in tqdm.tqdm(range(n_iter), disable=not verbose):
        # evaluate all candidates in the population
        scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in pop)

        ## Optional execute the callback code
        if callback is not None:
            callback(epoch, scores)

        # TODO: optimize this code
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                #logger.info('>%d, new best f(%s) = %.3f' % (gen,  pop[i], scores[i]))
        
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = []
        for i in range(0, selection_limit, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut, bounds)
                # store for next generation
                children.append(c)
        # if one element is missing copy the last selection value
        if len(children) < n_pop:
            children.append(selected[-1])
        
        # replace population
        pop = [utils.check_bounds(c, bounds) for c in children]

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