# coding: utf-8

"""
Artificial Bee Colony (ABC) optimization algorithm.

This module implements the ABC metaheuristic, which mimics the foraging behavior of honey bees.
The colony consists of three groups of bees: employed bees, onlookers, and scouts.

**Analogy:**
* **Employed Bees:** Go to a specific food source (solution) and dance to share information about its quality (fitness). They try to find a better spot nearby.
* **Onlooker Bees:** Watch the dances and choose a food source to exploit based on the quality (probability). Better sources attract more onlookers.
* **Scout Bees:** If a food source is exhausted (limit reached without improvement), the employed bee abandons it and becomes a scout, searching randomly for a new source.

**Mathematical Formulation:**
A new candidate solution $v_{i,j}$ is generated from $x_{i,j}$ using a partner $x_{k,j}$ and random $\\phi \\in [-1, 1]$:
$$ v_{i,j} = x_{i,j} + \\phi_{i,j} (x_{i,j} - x_{k,j}) $$
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"


import numpy as np

import pyBlindOpt.utils as utils
from pyBlindOpt.optimizer import Optimizer


class ArtificialBeeColony(Optimizer):
    """
    Artificial Bee Colony Optimizer.
    """

    @utils.inherit_docs(Optimizer)
    def __init__(self, objective, bounds, limit: int = 50, **kwargs):
        """
        Artificial Bee Colony Optimizer.

        Args:
            limit (int, optional): The number of trials without improvement before a food source is abandoned (Scout phase). Defaults to 50.
        """
        self.limit = limit
        super().__init__(objective, bounds, **kwargs)

    def _initialize(self):
        """
        Initializes the trial counters.

        Sets up an array to track how many times each solution has failed to improve, used to trigger the Scout phase.
        """
        self.trials = np.zeros(self.n_pop, dtype=int)

    def _update_iter_params(self, epoch: int):
        """
        Parameter update hook.

        ABC does not have time-dependent parameters in this implementation.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Updates the global best solution.

        Scans the current population (food sources) for the highest nectar amount (lowest cost).

        Args:
            epoch (int): Current iteration.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _search_phase(self, indices):
        """
        Performs the neighbor search (Employed/Onlooker logic).

        Generates a new candidate position by perturbing the current position towards a randomly selected partner.

        $$ v_{i} = x_{i} + \\phi (x_{i} - x_{k}) $$

        Args:
            indices (np.ndarray): Indices of the bees performing the search.

        Returns:
            np.ndarray: The new candidate positions.
        """
        if len(indices) == 0:
            return np.array([])

        partners = self.rng.integers(0, self.n_pop, size=len(indices))
        # Ensure partner != self
        mask_same = partners == indices
        partners[mask_same] = (partners[mask_same] + 1) % self.n_pop

        phi = self.rng.uniform(-1, 1, size=(len(indices), self.bounds.shape[0]))

        # Change all dimensions (Vectorized Modified ABC) or 1 dimension?
        # Let's do all dimensions masked by random factor for standard behavior approximation
        # v_{ij} = x_{ij} + phi * (x_{ij} - x_{kj})
        new_pos = self.pop[indices] + phi * (self.pop[indices] - self.pop[partners])
        return self._check_bounds(new_pos)

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Executes the Employed, Onlooker, and Scout phases.

        **1. Employed Phase:** Every bee explores neighbors.
        **2. Onlooker Phase:** Bees select sources probabilistically based on fitness ($P_i \\propto \\text{fitness}_i$) and explore neighbors.
        **3. Scout Phase:** Sources with `trials > limit` are replaced by random solutions.

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The updated population after all phases.
        """
        # 1. Employed Phase
        employed_candidates = self._search_phase(np.arange(self.n_pop))
        employed_scores = self.evaluate(employed_candidates)

        # Greedy Update
        improved = employed_scores < self.scores
        self.pop[improved] = employed_candidates[improved]
        self.scores[improved] = employed_scores[improved]
        self.trials[improved] = 0
        self.trials[~improved] += 1

        # 2. Onlooker Phase
        # Probability calc (Minimization)
        # fitness = 1 / (1 + cost) is standard for cost >= 0
        # Robust inversion for costs that can be negative:
        max_score = np.max(self.scores)
        if max_score == np.min(self.scores):
            probs = np.ones(self.n_pop) / self.n_pop
        else:
            raw_fitness = max_score - self.scores + 1e-10
            probs = raw_fitness / np.sum(raw_fitness)

        onlooker_indices = self.rng.choice(self.n_pop, size=self.n_pop, p=probs)
        onlooker_candidates = self._search_phase(onlooker_indices)
        onlooker_scores = self.evaluate(onlooker_candidates)

        # Greedy Update (Targeting the SOURCE of the onlooker, not the bee index)
        for i in range(self.n_pop):
            source_idx = onlooker_indices[i]
            if onlooker_scores[i] < self.scores[source_idx]:
                self.pop[source_idx] = onlooker_candidates[i]
                self.scores[source_idx] = onlooker_scores[i]
                self.trials[source_idx] = 0
            else:
                self.trials[source_idx] += 1

        # 3. Scout Phase
        scout_mask = self.trials > self.limit
        if np.any(scout_mask):
            self.pop[scout_mask] = utils.get_random_solution(self.bounds, self.rng)
            # Evaluate immediately to keep self.scores consistent
            self.scores[scout_mask] = self.evaluate(self.pop[scout_mask])
            self.trials[scout_mask] = 0

        # Return the final state of the population as "offspring"
        # The base class will re-evaluate this, which is redundant but required
        # to satisfy the interface strictness.
        return self.pop

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Finalizes the population state.

        ABC performs greedy selection *inside* the phases. This method acts as a synchronization step.

        Args:
            offspring (np.ndarray): The final population from the epoch.
            offspring_scores (np.ndarray): The corresponding scores.
        """
        self.pop = offspring
        self.scores = offspring_scores


def artificial_bee_colony(objective, bounds, limit: int = 50, **kwargs):
    """
    Functional interface for Artificial Bee Colony optimization.

    Returns:
        tuple: (best_pos, best_score).
    """
    return ArtificialBeeColony(objective, bounds, limit, **kwargs).optimize()
