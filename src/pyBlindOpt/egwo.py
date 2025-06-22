# coding: utf-8

'''
Enhanced Grey Wolf Optimization (EGWO) is a population-based meta-heuristics 
algorithm that simulates the leadership hierarchy and hunting 
mechanism of grey wolves in nature.
'''

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import tqdm
import joblib
import random
import logging
import tempfile
import statistics
import numpy as np
import pyBlindOpt.utils as utils


from collections.abc import Sequence


logger = logging.getLogger(__name__)


def grey_wolf_optimization(objective:callable, bounds:np.ndarray, population:np.ndarray=None, 
callback:"Sequence[callable] | callable"=None, n_iter:int=100, n_pop:int=10, n_jobs:int=-1, 
cached=False, debug=False, verbose=False, seed:int=42) -> tuple:
    '''
    Computes the Enhanced Grey Wolf optimization algorithm.

    Args:
        objective (callable): objective function used to evaluate the candidate solutions (lower is better)
        bounds (list): bounds that limit the search space
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
    rnd = random.Random(seed)
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
    
    # compute the fitness and find the alfa, beta, gamma wolves
    #scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in pop)
    scores = utils.compute_objective(pop, objective_cache, n_jobs)
    alfa_score, beta_score, gamma_score = sorted(scores)[0:3]
    alfa_wolf = pop[scores.index(alfa_score)]
    beta_wolf = pop[scores.index(beta_score)]
    gamma_wolf = pop[scores.index(gamma_score)]

    # arrays to store the debug information
    if debug:
        obj_avg_iter = []
        obj_best_iter = []
        obj_worst_iter = []

    # run iterations of the algorithm
    for epoch in tqdm.tqdm(range(n_iter), disable=not verbose):
        # update the prey position
        #temp_weight = [random.uniform(1, 3) for _ in range(3)]
        #temp_weight = sorted(temp_weight, reverse=True)
        #sum_weight = sum(temp_weight)
        #omega = [temp_weight[i] / sum_weight for i in range(3)]
        omega = np.random.uniform(1,3,(3,))
        omega = omega / np.sum(omega)
        omega = np.sort(omega, kind='stable')[::-1]
        # the standard deviation of the simulated stochastic error
        epoch_std = math.exp(-100 * (epoch + 1) / n_iter)
        #prey = [omega[0] * alfa_wolf[i] + omega[1] * beta_wolf[i] + omega[2] * gamma_wolf[i] 
        #+ random.normalvariate(0, epoch_std) for i in range(bounds.shape[0])]
        prey = (omega[0] * alfa_wolf + omega[1] * beta_wolf + omega[2] * gamma_wolf) 
        + np.random.normal(0, epoch_std, (bounds.shape[0]),)

        # updating each population member with the help of best three members
        offspring = []
        for i in range(n_pop):
            #A1, A2, A3 = a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            #C1, C2, C3 = 2 * rnd.random(), 2*rnd.random(), 2*rnd.random()

            #X1 = alfa_wolf - A1 * np.abs(C1*alfa_wolf-pop[i])
            #X2 = beta_wolf - A2 * np.abs(C2*beta_wolf-pop[i])
            #X3 = gamma_wolf - A3 * np.abs(C3*gamma_wolf-pop[i])
            #Xnew = np.mean([X1, X2, X3], axis=0)
            #Xnew = np.array([pop[i][j] - random.uniform(-2, 2) * abs(prey_pos[j] - pop[i][j]) for j in range(bounds.shape[0])])
            Xnew = pop[i] - np.random.uniform(-2,2,(bounds.shape[0],)) * np.absolute(prey - pop[i])
            offspring.append(Xnew)
        
        # check that lower and upper bounds are retained
        offspring = [utils.check_bounds(trial, bounds) for trial in offspring]

        # compute the fitness and update the population
        #scores_offspring = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective_cache)(c) for c in offspring)
        scores_offspring = utils.compute_objective(offspring, objective_cache, n_jobs)
        for i in range(n_pop):
            if scores_offspring[i] < scores[i]:
                pop[i] = offspring[i]
                scores[i] = scores_offspring[i]
        alfa_score, beta_score, gamma_score = sorted(scores)[0:3]
        alfa_wolf = pop[scores.index(alfa_score)]
        beta_wolf = pop[scores.index(beta_score)]
        gamma_wolf = pop[scores.index(gamma_score)]

        ## Optional store the debug information
        if debug:
            # store best, wort and average cost for all candidates
            obj_avg_iter.append(statistics.mean(scores))
            obj_best_iter.append(alfa_score)
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
        return (alfa_wolf, alfa_score, (obj_best_iter, obj_avg_iter, obj_worst_iter))
    else:
        return alfa_wolf, alfa_score

    
