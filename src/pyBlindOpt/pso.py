# coding: utf-8


'''
Particle swarm optimization is a computational method that optimizes 
a problem by iteratively trying to improve a candidate solution with 
regard to a given measure of quality. It solves a problem by having 
a population of candidate solutions, here dubbed particles, and 
moving these particles around in the search space according to simple 
mathematical formula over the particle's position and velocity. Each 
particle's movement is influenced by its local best-known position but 
is also guided toward the best-known positions in the search space, 
which are updated as better positions are found by other particles.
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


def particle_swarm_optimization(objective:callable, bounds:np.ndarray,
population:np.ndarray=None, callback:"Sequence[callable] | callable"=None,
n_iter:int=100, n_pop:int=10, c1:float=0.1, c2:float=0.1, w:float=0.8,
n_jobs:int=-1, cached=False, debug=False, verbose=False, seed:int=42) -> tuple:
    '''
    Computes the particle_swarm_optimization.

    Args:
        objective (callable): objective function used to evaluate the candidate solutions (lower is better)
        bounds (list): bounds that limit the search space
        population (list): optional list of candidate solutions (default None)
        callback (callable): callback function that is called at each epoch (deafult None)
        n_iter (int): the number of iterations (default 100)
        n_pop (int): the number of elements in the population (default 10)
        c1 (float): weight of personal best (default 0.1)
        c2 (float): weight of global best (default 0.1)
        w (float): weight of momentum (default 0.8)
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
    #pbest_obj = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in pbest)
    pbest_obj = utils.compute_objective(pbest, objective_cache, n_jobs)
    #print(f'Pbest Obj {pbest_obj}')
    gbest_obj = min(pbest_obj)
    #print(f'gbest_obj: {gbest_obj}')
    gbest = pbest[pbest_obj.index(gbest_obj)]
    #print(f'gbest: {gbest}')

    # arrays to store the debug information
    if debug:
        obj_avg_iter = []
        obj_best_iter = []
        obj_worst_iter = []
    
    for epoch in tqdm.tqdm(range(n_iter), disable=not verbose):
        # Update params
        r1, r2 = np.random.rand(2)
        # Update V
        v = [w*v[i]+c1*r1*(pbest[i]-x[i])+c2*r2*(gbest-x[i]) for i in range(n_pop)]
        #print(f'V {v}')
        # Update X
        x = [x[i]+v[i] for i in range(n_pop)]
        x = [utils.check_bounds(p, bounds) for p in x]
        #obj = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in x)
        obj = utils.compute_objective(x, objective_cache, n_jobs)
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
        
        ## Optional store the debug information
        if debug:
            # store best, wort and average cost for all candidates
            obj_avg_iter.append(statistics.mean(obj))
            obj_best_iter.append(gbest_obj)
            obj_worst_iter.append(max(obj))
        
        ## Optional execute the callback code
        if callback is not None:
            terminate = False
            if isinstance(callback, Sequence):
                terminate = any([c(epoch, pbest_obj, pbest) for c in callback])
            else:
                terminate = callback(epoch, pbest_obj, pbest)

            if terminate:
                break

    # clean the cache
    if cached:
        memory.clear(warn=False)

    if debug:
        return (gbest, gbest_obj, (obj_best_iter, obj_avg_iter, obj_worst_iter))
    else:
        return (gbest, gbest_obj)
