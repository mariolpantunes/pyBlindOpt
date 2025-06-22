# coding: utf-8


'''
Differential evolution is a method that optimizes a problem by iteratively trying to 
improve a candidate solution with regard to a given measure of quality.
Such methods are commonly known as metaheuristics as they make few or no assumptions 
about the problem being optimized and can search very large spaces of candidate solutions.
'''


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import math
import tqdm
import random
import joblib
import logging
import tempfile
import statistics
import numpy as np
import pyBlindOpt.utils as utils


from collections.abc import Sequence


logger = logging.getLogger(__name__)


@enum.unique
class TargetVector(enum.Enum):
    '''
    Enum data type that represents how the target vector is selected
    '''
    best = 'best'
    rand = 'rand'

    def __str__(self):
        return self.value


@enum.unique
class CrossoverMethod(enum.Enum):
    '''
    Enum data type that represents the crossover method
    '''
    bin = 'bin'
    exp = 'exp'

    def __str__(self):
        return self.value


def mutation(x:np.ndarray, F:float) -> np.ndarray:
    '''
    Computes the mutation operation.

    Args:
        x (np.ndarray): a set (at least three) candidate solutions
        F (float): weight that controls the mutation operation

    Returns:
        np.ndarray: the mutated candidate solution
    '''
    diff = np.empty(x[0].shape)
    for i in range(1, len(x), 2):
        diff += x[i] - x[i+1]
    return x[0] + F * diff


def idx_bin(dims:int, cr:float) -> list:
    '''
    Computes crossover based on Binomial crossover.

    Args:
        dims (int): the size of the solution vector
        cr (float): weight that controls the crossover operation

    Returns:
        list: with the binary valued based on the binomial distribution
    '''
    j = random.randrange(dims)
    idx = [True if random.random() < cr or i == j else False for i in range(dims)]
    return idx


def idx_exp(dims:int, cr:float) -> list:
    '''
    Computes crossover based on Exponential crossover.

    Args:
        dims (int): the size of the solution vector
        cr (float): weight that controls the crossover operation

    Returns:
        list: with the binary valued based on the exponential distribution
    '''
    idx = []
    j = random.randrange(dims)
    idx.append(j)
    j = (j + 1) % dims
    while random.random() < cr and len(idx) < dims:
        idx.append(j)
        j = (j + 1)
    rv = [True if i in idx else False for i in range(dims)]
    return rv


def crossover(mutated:np.ndarray, target:np.ndarray,
dims:int, cr:float, cross_method:callable) -> np.ndarray:
    '''
    Applies the crossover operation based on the cross_method.

    Args:
        mutated (np.ndarray): mutated version of the target candidate
        target (np.ndarray): original target candidate
        dims (int): the size of the solution vector
        cr (float): weight that controls the crossover operation
        cross_method (callable): method that computes the crossover index list

    Returns:
        np.ndarray: the offstring of the target and mutated parents
    '''
    idx = cross_method(dims, cr)
    trial = [mutated[i] if idx[i] else target[i] for i in range(dims)]
    return np.asarray(trial)


def differential_evolution(objective:callable, bounds:np.ndarray, population:list=None, 
variant:str='best/1/bin', callback:"Sequence[callable] | callable"=None, n_iter:int=100, 
n_pop:int=10, F:float=0.5, cr:float=0.7, rt:int=10, n_jobs:int=-1, cached:bool=False, 
debug:bool=False, verbose:bool=False, seed:int=42) -> tuple:
    '''
    Computes the differential evolution optimization algorithm.

    Args:
        objective (callable): objective function used to evaluate the candidate solutions (lower is better)
        bounds (list): bounds that limit the search space
        population (list): optional list of candidate solutions (default None)
        variant (str): string that specifies the DE variant (default best/1/bin)
        callback (callable): callback function that is called at each epoch (deafult None)
        n_iter (int): the number of iterations (default 100)
        n_pop (int): the number of elements in the population (default 10)
        F (float): weight that controls the mutation operation (default 0.5)
        cr (float): weight that controls the crossover operation (default 0.7)
        rt (int): number of retries when refines initial population (default 10)
        n_jobs (int): number of concurrent jobs (default -1)
        cached (bool): controls if the objective function is cached by joblib (default False)
        debug (bool): controls if debug information is returned (default False)
        verbose (bool): controls the usage of tqdm as a progress bar (default False)
        seed (int): seed to init the random generator (default 42)

    Returns:
        tuple: the best solution
    '''

    try:
        v = variant.split('/')
        tv = TargetVector[v[0]]
        dv = int(v[1])
        cm = CrossoverMethod[v[2]]

        nc = 2*dv if tv is TargetVector.best else 2*dv+1
    except:
        raise ValueError('variant must be = [rand|best]/n/[bin|exp]')

    cross_method = {CrossoverMethod.bin: idx_bin, CrossoverMethod.exp: idx_exp}

    # define the seed of the random generation
    np.random.seed(seed)
    random.seed(seed)

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
        pop = bounds[:, 0] + (np.random.rand(n_pop, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
        pop = [utils.check_bounds(p, bounds) for p in pop]
        # evaluate initial population of candidate solutions
        #obj_all = joblib.Parallel(n_jobs=n_jobs, backend='loky')(joblib.delayed(objective_cache)(c) for c in pop)
        obj_all = utils.compute_objective(pop, objective_cache, n_jobs)
        # improve the quality of the initial solutions (avoid initial solutions with inf cost)
        r = 0
        while any(math.isinf(i) for i in obj_all) and r < rt:
            for i in range(n_pop):
                if math.isinf(obj_all[i]):
                    pop = bounds[:, 0] + (np.random.rand(n_pop, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
                    pop = [utils.check_bounds(p, bounds) for p in pop]
            #obj_all = joblib.Parallel(n_jobs=n_jobs, backend='loky')(joblib.delayed(objective_cache)(c) for c in pop)
            obj_all = utils.compute_objective(pop, objective_cache, n_jobs)
            r += 1
        # if after R repetitions it still has inf. cost
        if any(math.isinf(i) for i in obj_all):
            valid_idx = [i for i in range(n_pop) if not math.isinf(obj_all[i])]
            pop = pop[valid_idx]
            obj_all = [obj_all[i] for i in valid_idx]
            n_pop = len(valid_idx)
    else:
        # initialise population of candidate and validate the bounds
        pop = [utils.check_bounds(p, bounds) for p in population]
        # overwrite the n_pop with the length of the given population
        n_pop = len(population)
        # evaluate initial population of candidate solutions
        #obj_all = joblib.Parallel(n_jobs=n_jobs, backend='loky')(joblib.delayed(objective_cache)(c) for c in pop)
        obj_all = utils.compute_objective(pop, objective_cache, n_jobs)

    # find the best performing vector of initial population
    best_vector = pop[np.argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    
    # arrays to store the debug information
    if debug:
        obj_avg_iter = []
        obj_best_iter = []
        obj_worst_iter = []
    
    # run iterations of the algorithm
    for epoch in tqdm.tqdm(range(n_iter), disable=not verbose):
        # generate offspring
        offspring = []
        for j in tqdm.tqdm(range(n_pop), leave=False, disable=not verbose):
            # choose three candidates, a, b and c, that are not the current one
            candidates_idx = random.choices([candidate for candidate in range(n_pop) if candidate != j], k = nc)
            diff_candidates = [pop[i] for i in candidates_idx]
            
            if tv is TargetVector.best:
                candidates = [best_vector]
                candidates.extend(diff_candidates)
            else:
                candidates = diff_candidates

            # perform mutation
            mutated = mutation(candidates, F)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr, cross_method[cm])
            offspring.append(trial)
        # check that lower and upper bounds are retained after mutation and crossover
        offspring = [utils.check_bounds(trial, bounds) for trial in offspring]
        #obj_trial = joblib.Parallel(n_jobs=n_jobs, backend='loky')(joblib.delayed(objective_cache)(c) for c in offspring)
        obj_trial = utils.compute_objective(offspring, objective_cache, n_jobs)

        # iterate over all candidate solutions
        for j in range(n_pop):
            # perform selection
            if obj_trial[j] < obj_all[j]:
                # replace the target vector with the trial vector
                pop[j] = offspring[j]
                # store the new objective function value
                obj_all[j] = obj_trial[j]
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        
        # store the lowest objective function value
        if best_obj < prev_obj:
            best_vector = pop[np.argmin(obj_all)]
            prev_obj = best_obj

        ## Optional store the debug information
        if debug:
            # store best, wort and average cost for all candidates
            obj_avg_iter.append(statistics.mean(obj_all))
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