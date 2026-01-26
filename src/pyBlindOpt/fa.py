# coding: utf-8


"""
Firefly Algorithm (FA) optimization.

This module implements the Firefly Algorithm, inspired by the flashing behavior of fireflies.
Fireflies are attracted to each other based on brightness (fitness), but light intensity decreases with distance.

**Analogy:**
Fireflies move towards brighter peers. If no one is brighter, they move randomly.
The attractiveness depends on distance, simulating light absorption in air.

**Mathematical Formulation:**
The movement of firefly $i$ towards $j$ is:
$$ x_i^{t+1} = x_i^t + \\beta_0 e^{-\\gamma r_{ij}^2} (x_j^t - x_i^t) + \\alpha (rand - 0.5) $$
where $r_{ij}$ is the distance, $\\beta_0$ is attractiveness at $r=0$, and $\\alpha$ is the randomization parameter.
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


class FireflyAlgorithm(Optimizer):
    """
    Firefly Algorithm Optimizer.
    """

    @utils.inherit_docs(Optimizer)
    def __init__(
        self,
        objective,
        bounds,
        alpha: float = 0.5,
        beta0: float = 1.0,
        gamma: float = 1.0,
        alpha_decay: float = 0.97,
        **kwargs,
    ):
        """
        Firefly Algorithm Optimizer.

        Args:
            alpha (float): Randomization parameter (step size). Defaults to 0.5.
            beta0 (float): Initial attractiveness at distance r=0. Defaults to 1.0.
            gamma (float): Light absorption coefficient. Controls convergence speed. Defaults to 1.0.
            alpha_decay (float): Geometric decay rate for alpha. Defaults to 0.97.
        """
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha_decay = alpha_decay
        super().__init__(objective, bounds, **kwargs)

    def _initialize(self):
        """
        Initialization hook.

        No specific state initialization required beyond parameters.
        """
        pass

    def _update_iter_params(self, epoch: int):
        """
        Updates the randomization parameter.

        Decays $\\alpha$ to reduce randomness as convergence approaches:
        $$ \\alpha_{t+1} = \\alpha_t \\cdot \\text{decay} $$

        Args:
            epoch (int): Current iteration.
        """
        self.alpha *= self.alpha_decay

    def _update_best(self, epoch: int):
        """
        Updates the global best firefly.

        Args:
            epoch (int): Current iteration.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Calculates firefly movements.

        Computes the pairwise attraction vector sum for every firefly $i$ towards all brighter fireflies $j$.

        **Attractiveness:**
        $$ \\beta = \\beta_0 e^{-\\gamma r^2} $$
        **Movement:**
        $$ x_{new} = x_{old} + \\sum_{j \\in \text{better}} \\beta_{ij}(x_j - x_i) + \\text{randomness} $$

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The new positions of the fireflies.
        """
        # 1. Compute pairwise distances (N, N)
        # using utils.global_distances or manual broadcasting
        # diff[i, j, d] = pop[j, d] - pop[i, d]
        pop = self.pop
        diff = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        r_sq = np.sum(diff**2, axis=-1)  # (N, N) distance squared

        # 2. Compute Attractiveness Beta (N, N)
        # beta[i, j] = beta0 * exp(-gamma * r^2)
        beta = self.beta0 * np.exp(-self.gamma * r_sq)

        # 3. Create Mask: Move i towards j ONLY IF score[j] < score[i]
        # mask[i, j] is True if j is better than i
        scores_flat = self.scores.flatten()
        mask = scores_flat[np.newaxis, :] < scores_flat[:, np.newaxis]

        # Zero out beta where j is not better than i
        beta_masked = (beta * mask)[:, :, np.newaxis]

        # 4. Compute Movement Steps
        # Sum over j: sum(beta * (x_j - x_i))
        # tensordot or sum(..., axis=1)
        # movement[i] = sum_j ( beta_masked[i, j] * diff[i, j] )
        movement = np.sum(beta_masked * diff, axis=1)

        # 5. Randomness
        bound_width = self.bounds[:, 1] - self.bounds[:, 0]
        random_step = self.alpha * (self.rng.random(pop.shape) - 0.5) * bound_width

        # 6. Apply
        new_pop = pop + movement + random_step

        return new_pop

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Greedy Selection.

        Updates the population only if the new position provides a better objective value.

        Args:
            offspring (np.ndarray): Candidate positions.
            offspring_scores (np.ndarray): Candidate scores.
        """
        improved_mask = offspring_scores < self.scores
        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]


def firefly_algorithm(
    objective,
    bounds,
    alpha: float = 0.5,
    beta0: float = 1.0,
    gamma: float = 1.0,
    alpha_decay: float = 0.97,
    **kwargs,
):
    """
    Functional interface for the Firefly Algorithm.

    Returns:
        tuple: (best_pos, best_score).
    """
    return FireflyAlgorithm(
        objective, bounds, alpha, beta0, gamma, alpha_decay, **kwargs
    ).optimize()
