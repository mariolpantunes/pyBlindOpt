# coding: utf-8

"""
Random Search (RS) optimization implementation.

This module implements a memoryless metaheuristic that explores the search space purely
stochastically. It serves as a baseline to benchmark the performance of "intelligent"
optimizers.

**Mathematical Formulation:**
At every iteration $k$, the previous population is discarded, and a new set of solutions
$X_k$ is drawn from the search space $\\Omega$ according to a sampling strategy (e.g., Uniform, LHS).
$$ X_{k} \\sim \\text{Sampler}(\\Omega) $$
$$ x_{best} = \\min(X_0 \\cup X_1 \\cup ... \\cup X_k) $$

**Analogy:**
Like paratroopers being dropped into a foggy landscape at completely random coordinates
every hour. They report their altitude, and the mission control simply remembers the
lowest point anyone has ever landed on. They do not learn from previous drops.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import collections.abc
import logging

import numpy as np

import pyBlindOpt.init as init
import pyBlindOpt.utils as utils
from pyBlindOpt.optimizer import Optimizer

logger = logging.getLogger(__name__)


class RandomSearch(Optimizer):
    """
    Random Search Optimizer.

    Generates a completely new population at every iteration using a configured Sampler,
    independent of previous results.
    """

    def _init_population(self, population, seed):
        """
        Initialization override.

        Persists the `Sampler` instance derived from the seed to ensure consistent
        sampling logic (e.g., Latin Hypercube) is used across all iterations,
        not just the first one.
        """
        # 1. Store the sampler for reuse in _generate_offspring
        if isinstance(seed, utils.Sampler):
            self.sampler = seed
        else:
            # Default to RandomSampler using the optimizer's RNG
            self.sampler = utils.RandomSampler(self.rng)

        # 2. Call the Base class to actually generate the initial population
        super()._init_population(population, seed)

    def _initialize(self):
        """
        Initialization hook.

        No internal state setup required for Random Search.
        """
        pass

    def _update_iter_params(self, epoch: int):
        """
        Parameter update hook.

        Random Search is memoryless and parameter-free; it does not change behavior over time.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Global best update.

        Checks if the current random batch contains a solution better than the best
        one seen so far across all previous batches.

        Args:
            epoch (int): Current iteration.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Generates a new population.

        Unlike other algorithms that perturb existing solutions, RS requests a
        completely fresh batch from the sampler.

        Returns:
            np.ndarray: New random population.
        """
        return init.get_initial_population(self.n_pop, self.bounds, self.sampler)

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Unconditional Replacement.

        The entire previous population is discarded and replaced by the new random batch.
        No "survival of the fittest" logic applies between generations.

        Args:
            offspring (np.ndarray): New population.
            offspring_scores (np.ndarray): Scores of new population.
        """
        self.pop = offspring
        self.scores = offspring_scores


def random_search(
    objective: collections.abc.Callable, bounds: np.ndarray, **kwargs
) -> tuple:
    """
    Functional interface for Random Search.

    Args:
        objective (Callable): Function to minimize.
        bounds (np.ndarray): Search space bounds.

    Returns:
        tuple: (best_position, best_score).
    """
    optimizer = RandomSearch(objective=objective, bounds=bounds, **kwargs)
    return optimizer.optimize()
