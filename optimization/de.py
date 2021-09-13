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


logger = logging.getLogger(__name__)


# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])
 
 
# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [np.clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound
 
 
# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = np.random.rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


def has_inf(l):
    return any(math.isinf(i) for i in l)

 
def differential_evolution(objective:typing.Callable, bounds:np.ndarray, n_iter:int=200, n_pop:int=20, F=0.5, cr=0.7, rt=10, n_jobs=-1, cached=True, debug=False):
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
    
    # improve que quality of the initial solutions (avoid initial solutions with inf cost)
    r = 0
    while(has_inf(obj_all) and r < rt):
        for i in range(n_pop):
            if math.isinf(obj_all[i]):
                pop = bounds[:, 0] + (np.random.rand(n_pop, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
                pop = np.array([check_bounds(p, bounds) for p in pop])
        obj_all = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in pop)
        r += 1
    
    # if after R repetitions it still has inf. cost
    if has_inf(obj_all):
        valid_idx = [i for i in range(n_pop) if not math.isinf(obj_all[i])]
        pop = pop[valid_idx]
        obj_all = [obj_all[i] for i in valid_idx]
        n_pop = len(valid_idx)

    # find the best performing vector of initial population
    best_vector = pop[np.argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    obj_iter = []
    # run iterations of the algorithm
    for _ in tqdm(range(n_iter), disable=not debug):
        # generate offspring
        offspring = []
        for j in range(n_pop):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(n_pop) if candidate != j]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
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
            obj_iter.append(best_obj)
    if cached:
        memory.clear(warn=False)
    if debug:
        return (best_vector, best_obj, obj_iter)
    else:
        return (best_vector, best_obj)