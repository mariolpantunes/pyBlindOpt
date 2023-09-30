# coding: utf-8


'''
Population initialization methods.
'''


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import joblib
import numpy as np
import pyBlindOpt.utils as utils


def opposition_based(objective:callable, bounds:np.ndarray,
population:np.ndarray=None, n_pop:int=20, n_jobs:int=-1) -> np.ndarray:
    '''
    '''

    # check if the initial population is given
    if population is None:
        # initial population of random bitstring
        pop = [utils.get_random_solution(bounds) for _ in range(n_pop)]
    else:
        # initialise population of candidate and validate the bounds
        pop = [utils.check_bounds(p, bounds) for p in population]
        # overwrite the n_pop with the length of the given population
        n_pop = len(population)
    
    # compute the fitness of the initial population
    scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective)(c) for c in pop)

    # compute the opposition population
    a = bounds[:,0]
    b = bounds[:,1]
    pop_opposition = [a+b-p for p in pop]
    
    # compute the fitness of the opposition population
    scores_opposition = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective)(c) for c in pop_opposition)

    # merge the results and filter
    results = list(zip(scores, pop)) + list(zip(scores_opposition, pop_opposition))
    results.sort(key=lambda x: x[0])

    return [results[i][1] for i in range (n_pop)]
