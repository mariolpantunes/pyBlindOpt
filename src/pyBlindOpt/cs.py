# coding: utf-8

"""
Cuckoo Search (CS) optimization.

This module implements CS, based on the brood parasitism of cuckoos.
Cuckoos lay eggs in other birds' nests. If the host discovers the egg, it throws it out or abandons the nest.

**Analogy:**
* **Cuckoo Breeding:** New solutions are generated via Lévy flights (long-range random walks).
* **Nest Abandonment:** A fraction $p_a$ of the worst nests are discovered and replaced by new random solutions (exploration).

**Mathematical Formulation:**
$$ x_{new} = x_{old} + \\alpha \\cdot \\text{Lévy}(\\beta) \\cdot (x_{old} - x_{best}) $$
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


class CuckooSearch(Optimizer):
    """
    Cuckoo Search (CS) optimization.
    """

    @utils.inherit_docs(Optimizer)
    def __init__(
        self,
        objective,
        bounds,
        pa: float = 0.25,
        alpha: float = 0.01,
        beta: float = 1.5,
        **kwargs,
    ):
        self.pa = pa
        self.alpha = alpha
        self.beta = beta
        super().__init__(objective, bounds, **kwargs)

    def _initialize(self):
        """
        Initialization hook.

        No specific state initialization required.
        """
        pass

    def _update_iter_params(self, epoch: int):
        """
        Parameter update hook.

        CS parameters are constant in this implementation.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Updates the global best nest.

        Args:
            epoch (int): Current iteration.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Generates new Cuckoo solutions via Lévy Flights.

        $$ \\text{step} = \\alpha \\cdot \\text{Lévy} \\cdot (x_{curr} - x_{best}) $$
        $$ x_{new} = x_{curr} + \\text{step} \\cdot \\mathcal{N}(0, 1) $$

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The new cuckoo eggs (solutions).
        """
        # 1. Generate Cuckoos via Lévy Flights
        # step = alpha * Levy * (Current - Best)
        levy = utils.levy_flight(self.n_pop, self.bounds.shape[0], self.beta, self.rng)
        step_size = self.alpha * levy * (self.pop - self.best_pos)

        # New candidate solutions
        offspring = self.pop + step_size * self.rng.standard_normal(self.pop.shape)
        return offspring

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Performs Greedy Selection and Nest Abandonment.

        1.  **Survival:** Keep $x_{new}$ if it is better than $x_{old}$.
        2.  **Abandonment:** Randomly discard a fraction $p_a$ of the population and replace them with new solutions via biased random walks.

        Args:
            offspring (np.ndarray): New cuckoo solutions.
            offspring_scores (np.ndarray): Scores.
        """
        # 1. Primary Greedy Selection (Survival of the fittest cuckoo)
        improved_mask = offspring_scores < self.scores
        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]

        # 2. Nest Abandonment (Discovery of alien eggs)
        # We must generate AND evaluate new solutions here manually,
        # as the Optimizer loop only does one pass of eval per epoch.
        abandon_mask = self.rng.random(self.n_pop) < self.pa

        if np.any(abandon_mask):
            # Biased Random Walk for abandoned nests
            p1 = self.rng.permutation(self.pop)
            p2 = self.rng.permutation(self.pop)

            step = self.rng.random() * (p1[abandon_mask] - p2[abandon_mask])
            new_nests = self.pop[abandon_mask] + step

            # Bound check & Evaluate manually
            new_nests = self._check_bounds(new_nests)
            new_scores = self.evaluate(new_nests)

            # Greedy update for abandoned nests
            better_abandon = new_scores < self.scores[abandon_mask]

            # Map back to full population
            # We get indices of rows where abandon_mask is True
            idxs = np.where(abandon_mask)[0]
            # Filter those by which actually improved
            update_idxs = idxs[better_abandon]

            if len(update_idxs) > 0:
                self.pop[update_idxs] = new_nests[better_abandon]
                self.scores[update_idxs] = new_scores[better_abandon]


def cuckoo_search(
    objective,
    bounds,
    pa: float = 0.25,
    alpha: float = 0.01,
    beta: float = 1.5,
    **kwargs,
):
    """
    Functional interface for Cuckoo Search.

    Returns:
        tuple: (best_pos, best_score).
    """
    return CuckooSearch(objective, bounds, pa, alpha, beta, **kwargs).optimize()
