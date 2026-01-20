# coding: utf-8

"""
Grey Wolf Optimization (GWO).

This module mimics the leadership hierarchy and hunting mechanism of grey wolves.


**Analogy:**
* **Alpha ($\\alpha$):** The leader (best solution).
* **Beta ($\\beta$):** The second best.
* **Gamma ($\\gamma$):** The third best.
* **Omega ($\\omega$):** The rest of the pack, which follows the leaders.

**Mathematical Formulation:**
The pack encircles the prey defined by the positions of $\\alpha, \\beta, \\gamma$.
$$ \\vec{D} = | \\vec{C} \\cdot \\vec{X}_{p} - \\vec{X} | $$
$$ \\vec{X}_{new} = \\vec{X}_{p} - \\vec{A} \\cdot \\vec{D} $$
The final position is the average of the moves towards $\\alpha, \\beta$, and $\\gamma$.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import collections.abc

import numpy as np

import pyBlindOpt.optimizer as optimizer


class GWO(optimizer.Optimizer):
    """
    Grey Wolf Optimizer.

    Maintains the top 3 solutions (Alpha, Beta, Gamma) to guide the search.
    """

    def _initialize(self):
        """
        Initializes the hierarchy.

        Allocates memory for Alpha, Beta, and Gamma positions and scores.
        """
        self.alpha_pos = np.zeros(self.pop.shape[1])
        self.alpha_score = np.inf
        self.beta_pos = np.zeros(self.pop.shape[1])
        self.gamma_pos = np.zeros(self.pop.shape[1])
        self.a = 2.0  # Will be updated

    def _update_iter_params(self, epoch: int):
        """
        Updates the convergence parameter 'a'.

        Linearly decreases $a$ from 2 to 0 over the course of iterations to transition from exploration to exploitation.
        $$ a = 2(1 - t/T) $$

        Args:
            epoch (int): Current iteration.
        """
        self.a = 2 * (1 - epoch / self.n_iter)

    def _update_best(self, epoch: int):
        """
        Identifies the Alpha, Beta, and Gamma wolves.

        Sorts the population by fitness and stores the top 3 as the leaders.

        Args:
            epoch (int): Current iteration.
        """
        # Top 3 indices
        top_k_indices = np.argpartition(self.scores, 3)[:3]
        top_k_sorted = top_k_indices[np.argsort(self.scores[top_k_indices])]

        a_idx, b_idx, g_idx = top_k_sorted[0], top_k_sorted[1], top_k_sorted[2]

        self.alpha_score = self.scores[a_idx]
        self.alpha_pos = self.pop[a_idx].copy()
        self.beta_pos = self.pop[b_idx].copy()
        self.gamma_pos = self.pop[g_idx].copy()

        # Update Base Class global best for return value
        self.best_score = self.alpha_score
        self.best_pos = self.alpha_pos.copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Updates wolf positions based on the leaders.

        Calculates the vector to Alpha, Beta, and Gamma separately and moves the omega wolves towards the centroid of the leaders.
        $$ \\vec{X}_{new} = \\frac{\vec{X}_1 + \\vec{X}_2 + \\vec{X}_3}{3} $$

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The new positions of the pack.
        """
        dim = self.pop.shape[1]

        def compute_X(leader_pos):
            r1 = self.rng.random((self.n_pop, dim))
            r2 = self.rng.random((self.n_pop, dim))
            A = 2 * self.a * r1 - self.a
            C = 2 * r2
            D_leader = np.abs(C * leader_pos - self.pop)
            return leader_pos - A * D_leader

        X1 = compute_X(self.alpha_pos)
        X2 = compute_X(self.beta_pos)
        X3 = compute_X(self.gamma_pos)

        return (X1 + X2 + X3) / 3.0

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Greedy Selection.

        Wolves only update their position if the move results in a better hunting spot.

        Args:
            offspring (np.ndarray): New positions.
            offspring_scores (np.ndarray): Scores.
        """
        improved_mask = offspring_scores < self.scores
        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]


def grey_wolf_optimization(
    objective: collections.abc.Callable, bounds: np.ndarray, **kwargs
) -> tuple:
    """
    Functional interface for Grey Wolf Optimization.

    Returns:
        tuple: (best_pos, best_score).
    """
    optimizer = GWO(objective=objective, bounds=bounds, **kwargs)
    return optimizer.optimize()
