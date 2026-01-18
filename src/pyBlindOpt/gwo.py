# coding: utf-8

"""
Grey Wolf Optimization (GWO) is a population-based meta-heuristics
algorithm that simulates the leadership hierarchy and hunting
mechanism of grey wolves in nature.
"""

__author__ = "Mário Antunes"
__version__ = "0.1"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import collections.abc
import logging
import tempfile

import joblib
import numpy as np
import tqdm

import pyBlindOpt.init as init
import pyBlindOpt.utils as utils

logger = logging.getLogger(__name__)


def _update_leaders(current_pop, current_scores):
    # Efficiently find top 3 indices
    top_k_indices = np.argpartition(current_scores, 3)[:3]
    top_k_sorted = top_k_indices[np.argsort(current_scores[top_k_indices])]
    
    # Extract leaders (.copy() is crucial)
    a_idx, b_idx, g_idx = top_k_sorted[0], top_k_sorted[1], top_k_sorted[2]
    
    return (
        current_scores[a_idx], current_pop[a_idx].copy(),  # Alpha
        current_pop[b_idx].copy(),                         # Beta
        current_pop[g_idx].copy()                          # Gamma
    )


def grey_wolf_optimization(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    population: np.ndarray | None = None,
    callback: "list[collections.abc.Callable] | collections.abc.Callable | None" = None,
    n_iter: int = 100,
    n_pop: int = 10,
    n_jobs: int = 1,
    cached=False,
    debug=False,
    verbose=False,
    seed: int | np.random.Generator | utils.Sampler | None = None,
) -> tuple:
    """
    Computes the Grey Wolf optimization algorithm.

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
    """

    if isinstance(seed, utils.Sampler):
        rng = seed.rng
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # cache the initial objective function
    memory:joblib.Memory|None = None
    if cached:
        # Cache from joblib
        location = tempfile.gettempdir()
        memory = joblib.Memory(location, verbose=0)
        objective_cache = memory.cache(objective)
    else:
        objective_cache = objective

    # check if the initial population is given
    if population is None:
        if isinstance(seed, utils.Sampler):
            sampler = seed
        else:
            sampler = utils.RandomSampler(rng)
        pop = init.get_initial_population(n_pop, bounds, sampler)
    else:
        pop = np.clip(population, bounds[:, 0], bounds[:, 1])
        n_pop = pop.shape[0]

    # compute the fitness and find the alfa, beta, gamma wolves
    scores = utils.compute_objective(pop, objective_cache, n_jobs)
    alfa_score, alfa_wolf, beta_wolf, gamma_wolf = _update_leaders(pop, scores)

    # arrays to store the debug information
    history:np.ndarray|None = None
    if debug:
        history = np.zeros((n_iter, 3))

    # run iterations of the algorithm
    epoch:int = 0
    for epoch in tqdm.tqdm(range(n_iter), disable=not verbose):
        # linearly decreased from 2 to 0
        a = 2 * (1 - epoch / n_iter)

        dim = pop.shape[1]

        # Generate random vectors for the entire population (N, D)
        r1 = rng.random((n_pop, dim))
        r2 = rng.random((n_pop, dim))

        # Calculate A and C vectors (N, D)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        # Calculate distance to Alpha (Broadcasting: (1, D) - (N, D))
        D_alpha = np.abs(C1 * alfa_wolf - pop)
        X1 = alfa_wolf - A1 * D_alpha

        # Repeat for Beta
        r1 = rng.random((n_pop, dim))
        r2 = rng.random((n_pop, dim))
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = np.abs(C2 * beta_wolf - pop)
        X2 = beta_wolf - A2 * D_beta

        # Repeat for Gamma
        r1 = rng.random((n_pop, dim))
        r2 = rng.random((n_pop, dim))
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_gamma = np.abs(C3 * gamma_wolf - pop)
        X3 = gamma_wolf - A3 * D_gamma

        # Average to get new positions (N, D)
        offspring = (X1 + X2 + X3) / 3.0

        # Vectorized Clip (Bounds Check)
        offspring = np.clip(offspring, bounds[:, 0], bounds[:, 1])

        # compute the fitness and update the population
        scores_offspring = utils.compute_objective(offspring, objective_cache, n_jobs)

        # Greedy Selection
        improved_mask = scores_offspring < scores
        pop[improved_mask] = offspring[improved_mask]
        scores[improved_mask] = scores_offspring[improved_mask]

        # Update Leaders (Standard)
        alfa_score, alfa_wolf, beta_wolf, gamma_wolf = _update_leaders(pop, scores)

        if debug and history is not None:
            history[epoch, 0] = alfa_score
            history[epoch, 1] = np.mean(scores)
            history[epoch, 2] = np.max(scores)

        ## Optional execute the callback code
        if callback is not None:
            # Ensure callbacks are a list
            cbs = callback if isinstance(callback, collections.abc.Sequence) else [callback]
            
            stop_signal = False
            
            for c in cbs:
                # Snapshot before THIS specific callback runs
                pre_callback_pop = pop.copy()

                # Execute callback with currently aligned pop and scores
                res = c(epoch, scores, pop)
                
                # Check for stop signal
                if isinstance(res, (bool, np.bool_)) and res:
                    stop_signal = True
                    break
                
                # Check for population modification
                elif isinstance(res, np.ndarray):
                    if res.shape != pop.shape:
                        raise ValueError(f"Callback changed population shape from {pop.shape} to {res.shape}")
                    # Update population
                    pop = res
                    # Chain updates on the population for the following callbacks
                    changed_mask = np.any(pop != pre_callback_pop, axis=1)
                    
                    if np.any(changed_mask):
                        # Enforce bounds on mutated elements
                        pop[changed_mask] = np.clip(
                            pop[changed_mask], bounds[:, 0], bounds[:, 1]
                        )
                        # Evaluate only the changed wolves
                        new_scores = utils.compute_objective(pop[changed_mask], objective_cache, n_jobs)
                        scores[changed_mask] = new_scores
                        # Update Leaders Immediately (so next callback/loop sees correct state)
                        alfa_score, alfa_wolf, beta_wolf, gamma_wolf = _update_leaders(pop, scores)

            if stop_signal:
                break

        if debug and history is not None:
            history[epoch, 0] = alfa_score
            history[epoch, 1] = np.mean(scores)
            history[epoch, 2] = np.max(scores)

    # clean the cache
    if cached and memory is not None:
        memory.clear(warn=False)

    if debug and history is not None:
        # Slice to actual iterations (handles early stopping)
        actual_hist = history[:epoch+1]
        return (alfa_wolf, alfa_score, (
            actual_hist[:, 0], # Best
            actual_hist[:, 1], # Avg
            actual_hist[:, 2]  # Worst
        ))
    else:
        return alfa_wolf, alfa_score
