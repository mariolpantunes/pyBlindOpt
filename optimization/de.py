# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import math
import random
import typing
import joblib
import logging
import tempfile
import statistics
import numpy as np

from tqdm import tqdm


logger = logging.getLogger(__name__)

@enum.unique
class TargetVector(enum.Enum):
    best = 'best'
    rand = 'rand'

    def __str__(self):
        return self.value


@enum.unique
class CrossoverMethod(enum.Enum):
    bin = 'bin'
    exp = 'exp'

    def __str__(self):
        return self.value


# define mutation operation
def mutation(x, F):
    diff = np.empty(x[0].shape)
    for i in range(1, len(x), 2):
        diff += x[i] - x[i+1]
    return x[0] + F * diff


# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [np.clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound


def idx_bin(dims, cr):
    j = random.randrange(dims)
    idx = [True if random.random() < cr or i == j else False for i in range(dims)]
    return idx


def idx_exp(dims, cr):
    idx = []
    j = random.randrange(dims)
    idx.append(j)
    j = (j + 1) % dims
    while random.random() < cr and len(idx) < dims:
        idx.append(j)
        j = (j + 1)
    rv = [True if i in idx else False for i in range(dims)]
    return rv


def crossover(mutated, target, dims, cr, cross_method):
    idx = cross_method(dims, cr)
    trial = [mutated[i] if idx[i] else target[i] for i in range(dims)]
    return trial

 
def differential_evolution(objective:typing.Callable, bounds:np.ndarray, variant="best/1/bin", n_iter:int=200, n_pop:int=20, F=0.5, cr=0.7, rt=10, n_jobs=-1, cached=False, debug=False):
    try:
        v = variant.split('/')
        tv = TargetVector[v[0]]
        dv = int(v[1])
        cm = CrossoverMethod[v[2]]

        nc = 2*dv if tv is TargetVector.best else 2*dv+1
    except:
        raise ValueError('variant must be = [rand|best]/n/[bin|exp]')

    cross_method = {CrossoverMethod.bin: idx_bin, CrossoverMethod.exp: idx_exp}

    # cache the initial objective function
    if cached:
        # Cache from joblib
        location = tempfile.gettempdir()
        memory = joblib.Memory(location, verbose=0)
        objective_cache = memory.cache(objective)
    else:
        objective_cache = objective
    # initialise population of candidate solutions randomly within the specified bounds
    pop = bounds[:, 0] + (np.random.rand(n_pop, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    pop = np.array([check_bounds(p, bounds) for p in pop])
    # evaluate initial population of candidate solutions
    obj_all = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in pop)
    
    # improve the quality of the initial solutions (avoid initial solutions with inf cost)
    r = 0
    while any(math.isinf(i) for i in obj_all) and r < rt:
        for i in range(n_pop):
            if math.isinf(obj_all[i]):
                pop = bounds[:, 0] + (np.random.rand(n_pop, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
                pop = np.array([check_bounds(p, bounds) for p in pop])
        obj_all = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in pop)
        r += 1
    
    # if after R repetitions it still has inf. cost
    if any(math.isinf(i) for i in obj_all):
        valid_idx = [i for i in range(n_pop) if not math.isinf(obj_all[i])]
        pop = pop[valid_idx]
        obj_all = [obj_all[i] for i in valid_idx]
        n_pop = len(valid_idx)

    # find the best performing vector of initial population
    best_vector = pop[np.argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    obj_avg_iter = []
    obj_best_iter = []
    obj_worst_iter = []
    # run iterations of the algorithm
    for _ in tqdm(range(n_iter), disable=not debug):
        # generate offspring
        offspring = []
        for j in range(n_pop):
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
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr, cross_method[cm])
            offspring.append(trial)
        
        obj_trial = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in offspring)

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
        if debug:
            # store best, wort and average cost for all candidates
            obj_avg_iter.append(statistics.mean(obj_all))
            obj_best_iter.append(best_obj)
            obj_worst_iter.append(max(obj_all))
    if cached:
        memory.clear(warn=False)
    if debug:
        return (best_vector, best_obj, (obj_best_iter, obj_avg_iter, obj_worst_iter))
    else:
        return (best_vector, best_obj)