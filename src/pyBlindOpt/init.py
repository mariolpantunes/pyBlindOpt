# coding: utf-8


'''
Population initialization methods.
'''


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import ess
import logging
import numpy as np
import pyBlindOpt.utils as utils

import collections.abc

logger = logging.getLogger(__name__)


def get_initial_population(
    n_pop: int, bounds: np.ndarray, sampler: utils.Sampler
) -> np.ndarray:
    """
    Helper to generate the full population matrix (N_pop x D) at once.
    """
    return sampler.sample(n_pop, bounds)


def opposition_based(objective:collections.abc.Callable, 
    bounds:np.ndarray, population:np.ndarray|utils.Sampler|None = None,
    n_pop:int=10, n_jobs:int=1, seed:int|np.random.Generator|None = 42) -> np.ndarray:

    rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed

    if isinstance(population, utils.Sampler):
        pop = get_initial_population(n_pop, bounds, population)
    elif isinstance(population, np.ndarray):
        pop = utils.check_bounds(population, bounds)
        n_pop = pop.shape[0]
    elif population is None:
        sampler = utils.RandomSampler(rng)
        pop = get_initial_population(n_pop, bounds, sampler)
    else:
        raise ValueError("Population must be None, ndarray, or PopulationSampler.")

    # compute the fitness of the initial population
    scores = utils.compute_objective(pop, objective, n_jobs)

    # compute the opposition population
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    pop_opp = utils.check_bounds(lower + upper - pop, bounds)
    
    # compute the fitness of the opposition population
    scores_opp = utils.compute_objective(pop_opp, objective, n_jobs)

    # merge the results and filter
    combined_pop = np.vstack((pop, pop_opp))
    combined_scores = np.concatenate((scores, scores_opp))
    top_k_indices = np.argpartition(combined_scores, n_pop)[:n_pop] 
            
    return combined_pop[top_k_indices]


def round_init(objective:collections.abc.Callable, bounds:np.ndarray, 
    sampler: utils.Sampler, n_pop:int=10, n_rounds:int=3, 
    diversity_weight: float = 0.5, n_jobs:int=1) -> np.ndarray:

    total_candidates = n_pop * n_rounds
    full_pool = sampler.sample(total_candidates, bounds)
    
    fitness = np.zeros(total_candidates)
    for i in range(0, total_candidates, n_pop):
        batch = full_pool[i : i + n_pop]
        fitness[i : i + n_pop] = utils.compute_objective(batch, objective, n_jobs)
    
    prob_fitness = utils.score_2_probs(fitness)
    
    if diversity_weight > 0:
        crowding = utils.compute_crowding_distance(full_pool)
        prob_dist = utils.score_2_probs(-crowding)
    else:
        prob_dist = np.zeros_like(prob_fitness)

    final_probs = (1.0 - diversity_weight) * prob_fitness + diversity_weight * prob_dist
    # Normalize (Floating point math might make sum slightly != 1.0)
    final_probs /= np.sum(final_probs)
    
    selected_indices = sampler.rng.choice( total_candidates, 
        size=n_pop, replace=False, p=final_probs)
    
    return full_pool[selected_indices]


def oblesa(objective:collections.abc.Callable, bounds:np.ndarray,
    population:np.ndarray|utils.Sampler|None = None, n_pop:int=10, n_jobs:int=1, 
    epochs:int=1024, lr:float=0.01, search_mode: str = "radius", k:int|None=None, 
    decay: float = 0.9, batch_size: int = 50, tol: float = 1e-3, radius: float | None = None,
    border_strategy: str = "repulsive", metric: str | collections.abc.Callable = "softened_inverse",
    seed:int|np.random.Generator|None = 42, **metric_kwargs) -> np.ndarray:

    rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed

    if isinstance(population, utils.Sampler):
        ran_pop = get_initial_population(n_pop, bounds, population)
    elif isinstance(population, np.ndarray):
        ran_pop = utils.check_bounds(population, bounds)
        n_pop = ran_pop.shape[0]
    elif population is None:
        sampler = utils.RandomSampler(rng)
        ran_pop = get_initial_population(n_pop, bounds, sampler)
    else:
        raise ValueError("Population must be None, ndarray, or PopulationSampler.")

    lower, upper = bounds[:, 0], bounds[:, 1]
    opp_pop = utils.check_bounds(lower + upper - ran_pop, bounds)

    combined_samples = np.vstack((ran_pop, opp_pop))
    emp_pop = ess.esa(combined_samples, bounds, n=2*n_pop, epochs=epochs, lr=lr, k=k, 
        decay=decay, batch_size=batch_size, radius=radius, search_mode=search_mode,
        border_strategy=border_strategy, tol=tol, seed=rng, metric=metric, **metric_kwargs)
    
    population = np.vstack((ran_pop, opp_pop, emp_pop))
    scores = np.zeros(population.shape[0])
    
    for i in range(0, population.shape[0], n_pop):
        end = min(i + n_pop, population.shape[0])
        batch = population[i : end]
        scores[i : end] = utils.compute_objective(batch, objective, n_jobs)

    probs = utils.score_2_probs(scores)
    idx = rng.choice(population.shape[0], size=n_pop, replace=False, p=probs)

    return population[idx]