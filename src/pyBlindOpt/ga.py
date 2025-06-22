# coding: utf-8

'''
Genetic algorithm is a metaheuristic inspired by the process of natural selection 
that belongs to the larger class of evolutionary algorithms (EA). 
Genetic algorithms are commonly used to generate high-quality solutions to 
optimization and search problems by relying on biologically inspired operators 
such as mutation, crossover, and selection.
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
import statistics
import numpy as np
import pyBlindOpt.utils as utils


from collections.abc import Sequence


logger = logging.getLogger(__name__)


# tournament selection
def tournament_selection(pop:list, scores:list, k:int=3) -> np.ndarray:
    '''
    Tournament selection.

    Args:
        pop (list): the population list
        scores (list): the scores of each solution candidate
        k (int): the number of participants in the tournament (default 3)
        
    Returns:
        np.ndarray: the tournament champion
    '''
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# random mutation operator
def random_mutation(candidate:np.ndarray, r_mut:float, bounds:list) -> np.ndarray:
    '''
    Random mutation operator.
    Generates a new solution from the bounds.

    Args:
        candidate (np.ndarray): the candidate that will be mutated
        r_mut (float): the mutation probability
        bounds (list): bounds that limit the search space
        
    Returns:
        np.ndarray: the mutated candidate solution
    '''
    if np.random.rand() < r_mut:
        solution = utils.get_random_solution(bounds)
        return solution
    else:
        return candidate


# linear crossover operator: two parents to create three children
def linear_crossover(p1:np.ndarray, p2:np.ndarray, r_cross:float) -> list:
    '''
    Linear crossover operator.

    Args:
        p1 (np.ndarray): the first parent
        p2 (np.ndarray): the second parent
        r_cross (float): the crossover probability
        
    Returns:
        list: list with the offspring
    '''
    if np.random.rand() < r_cross:
        c1 = 0.5*p1 + 0.5*p2
        c2 = 1.5*p1 - 0.5*p2
        c3 = -0.5*p1 + 1.5*p2
        return [c1, c2, c3]
    else:
        return [p1, p2]


# blend crossover operator: two parents to create two children
def blend_crossover(p1, p2, r_cross, alpha=.5):
    '''
    Blend crossover operator.

    Args:
        p1 (np.ndarray): the first parent
        p2 (np.ndarray): the second parent
        r_cross (float): the crossover probability
        alpha (float): weight that controls the blend crossover (default 0.5)
        
    Returns:
        list: list with the offspring
    '''
    if np.random.rand() < r_cross:
        c1 = p1 + alpha*(p2-p1)
        c2 = p2 - alpha*(p2-p1)
        return [c1, c2]
    else:
        return [p1, p2]


def genetic_algorithm(objective:callable, bounds:np.ndarray,
population:np.ndarray=None, selection:callable=tournament_selection,
crossover:callable=blend_crossover, mutation:callable=random_mutation,  
callback:"Sequence[callable] | callable"=None, n_iter:int=200, 
n_pop:int=20, r_cross:float=0.9, r_mut:float=0.3, n_jobs:int=-1, 
cached:bool=False, debug:bool=False, verbose:bool=False, seed:int=42) -> tuple:
    '''
    Computes the genetic algorithm optimization.

    Args:
        objective (callable): objective function used to evaluate the candidate solutions (lower is better)
        bounds (list): bounds that limit the search space
        population (list): optional list of candidate solutions (default None)
        selection (callable): selection operator (default tournament_selection)
        crossover (callable): crossover operator (default blend_crossover)
        mutation (callable): mutation operator (default random_mutation)
        callback (callable): callback function that is called at each epoch (deafult None)
        n_iter (int): the number of iterations (default 100)
        n_pop (int): the number of elements in the population (default 10)
        r_cross (float): ratio of crossover (default 0.9)
        r_mut (float): ratio of mutation (default 0.3)
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
        # initial population of random bitstring
        pop = [utils.get_random_solution(bounds) for _ in range(n_pop)]
    else:
        # initialise population of candidate and validate the bounds
        pop = [utils.check_bounds(p, bounds) for p in population]
        # overwrite the n_pop with the length of the given population
        n_pop = len(population)

    # keep track of best solution
    #scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in pop)
    scores = utils.compute_objective(pop, objective_cache, n_jobs)
    best_eval = min(scores)
    best = pop[scores.index(best_eval)]
    
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
        #scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in pop)
        scores = utils.compute_objective(pop, objective_cache, n_jobs)

        # check for new best solution
        best_eval = min(scores)
        best = pop[scores.index(best_eval)]
        
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = []
        for i in range(0, selection_limit, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # apply mutation and store for next generation
                children.append(mutation(c, r_mut, bounds))
        # if one element is missing copy the last selection value
        if len(children) < n_pop:
            children.append(selected[-1])
        
        # replace population
        pop = [utils.check_bounds(c, bounds) for c in children]

        ## Optional store the debug information
        if debug:
            # store best, wort and average cost for all candidates
            obj_avg_iter.append(statistics.mean(scores))
            obj_best_iter.append(best_eval)
            obj_worst_iter.append(max(scores))
        
        ## Optional execute the callback code
        if callback is not None:
            terminate = False
            if isinstance(callback, Sequence):
                terminate = any([c(epoch, scores, pop) for c in callback])
            else:
                terminate = callback(epoch, scores, pop)

            if terminate:
                break

    # clean the cache
    if cached:
        memory.clear(warn=False)
    
    if debug:
        return (best, best_eval, (obj_best_iter, obj_avg_iter, obj_worst_iter))
    else:
        return (best, best_eval)