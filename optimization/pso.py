# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import tqdm
import typing
import joblib
import logging
import tempfile
import numpy as np
import optimization.utils as utils


logger = logging.getLogger(__name__)


def particle_swarm_optimization(objective:typing.Callable, bounds:np.ndarray,
population:np.ndarray=None, callback:typing.Callable=None,
n_iter:int=100, n_pop:int=10, c1:float=0.1, c2:float=0.1, w:float=0.8,
n_jobs:int=-1, cached=False, debug=False, verbose=False, seed:int=42) -> tuple:
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
        x = [utils.get_random_solution(bounds) for _ in range(n_pop)]
    else:
        # initialise population of candidate and validate the bounds
        x = [utils.check_bounds(p, bounds) for p in population]
        # overwrite the n_pop with the length of the given population
        n_pop = len(population)
    
    # compute the initial velocity values
    v = [np.random.randn(len(bounds))* 0.1 for _ in range(n_pop)]

    # Initialize data
    pbest = x
    pbest_obj = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in pbest)
    #print(f'Pbest Obj {pbest_obj}')
    gbest_obj = min(pbest_obj)
    #print(f'gbest_obj: {gbest_obj}')
    gbest = pbest[pbest_obj.index(gbest_obj)]
    #print(f'gbest: {gbest}')
    
    for epoch in tqdm.tqdm(range(n_iter), disable=not verbose):
        # Update params
        r1, r2 = np.random.rand(2)
        # Update V
        v = [w*v[i]+c1*r1*(pbest[i]-x[i])+c2*r2*(gbest-x[i]) for i in range(n_pop)]
        #print(f'V {v}')
        # Update X
        x = [x[i]+v[i] for i in range(n_pop)]
        x = [utils.check_bounds(p, bounds) for p in x]
        obj = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in x)
        # replace personal best
        # iterate over all candidate solutions
        for j in range(n_pop):
            # perform selection
            if obj[j] < pbest_obj[j]:
                # replace the target vector with the trial vector
                pbest[j] = x[j]
                # store the new objective function value
                pbest_obj[j] = obj[j]
        gbest_obj = min(pbest_obj)
        gbest = pbest[pbest_obj.index(gbest_obj)]

        ## Optional execute the callback code
        if callback is not None:
            callback(epoch, pbest_obj)

    # clean the cache
    if cached:
        memory.clear(warn=False)

    if debug:
        return (gbest, gbest_obj, (obj_best_iter, obj_avg_iter, obj_worst_iter))
    else:
        return (gbest, gbest_obj)
