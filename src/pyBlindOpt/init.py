# coding: utf-8


'''
Population initialization methods.
'''


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import heapq
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


def round_init(objective:callable, bounds:np.ndarray, 
n_pop:int=20, n_rounds:int=5, n_jobs:int=-1) -> np.ndarray:

    samples = []
    fitness = []
    for i in range(n_rounds):
        sample = [utils.get_random_solution(bounds) for _ in range(n_pop)]
        sample_fitness = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective)(c) for c in sample)
        
        samples.extend(sample)
        fitness.extend(sample_fitness)
    
    # Additional code - get best n_pop points that are far away from each other
    # Optimal solution with pareto front too slow, use a simple heuristic
    # 1. Compute the global distance from one sample to all the others
    distances = utils.global_distances(samples)
    # 2. Invert the distance (since the want to maximize distance)
    max_distances = max(distances)
    inv_distances = max_distances - distances
    # 3. Scale booth inv_distance and fitness (so the range have less impact on the selection)
    scale_inv_dist, _, _ = utils.scale(inv_distances)
    scale_fitness, _, _ = utils.scale(fitness)
    # 4. Build a score metric that is the addition
    scores = scale_inv_dist + scale_fitness
    # 5. Use a minHeap to select the elements from the score points
    solution_scores = heapq.nsmallest(int(n_pop), scores)
    # 6. Get the indexes of the solution
    solution = [np.where(scores==s)[0][0] for s in solution_scores]
    # 7. Use the indexes to select the final population
    population = [samples[i] for i in solution]
    return population
