# coding: utf-8

"""
Enhanced Grey Wolf Optimization (EGWO).

An improvement over GWO that addresses the balance between exploration and exploitation.
Instead of moving towards the centroid of Alpha, Beta, and Gamma, wolves move towards a weighted "Prey" position that includes stochastic error.

**Mathematical Formulation:**
$$ X_{prey} = w_1 X_\\alpha + w_2 X_\\beta + w_3 X_\\gamma + \\mathcal{N}(0, \\sigma^2) $$
where $\\sigma$ decays exponentially over time.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import collections.abc
import math

import numpy as np

from pyBlindOpt.gwo import GWO


class EGWO(GWO):
    """
    Enhanced Grey Wolf Optimization.

    Extends standard GWO by introducing a weighted prey position with stochastic error.
    """

    def _initialize(self):
        """
        Initializes GWO hierarchy and EGWO error parameter.

        Sets up `epoch_std` for the stochastic term.
        """
        super()._initialize()
        self.epoch_std = 0.0

    def _update_iter_params(self, epoch: int):
        """
        Updates the stochastic error standard deviation.

        Uses exponential decay:
        $$ \\sigma_t = \\exp(-100 \\frac{t+1}{T}) $$
        This ensures high randomness (exploration) at the start and low randomness (exploitation) at the end.

        Args:
            epoch (int): Current iteration.
        """
        self.epoch_std = math.exp(-100 * (epoch + 1) / self.n_iter)

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Generates new positions using Weighted Prey + Noise.

        1.  **Weights:** Randomly generates weights for $\\alpha, \\beta, \\gamma$.
        2.  **Target:** Calculates weighted center + Gaussian noise.
        3.  **Update:** Moves wolves towards this target.

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The new positions.
        """
        dim = self.pop.shape[1]

        # 1. Calculate Weights (Omega)
        # Random uniform [1, 3], normalized, and sorted descending
        omega = self.rng.uniform(1, 3, size=3)
        omega /= np.sum(omega)
        omega = np.sort(omega)[::-1]  # Sort descending (Alpha gets highest weight)

        # 2. Calculate "Prey" Position (Target)
        # Weighted sum of leaders + Stochastic Noise
        noise = self.rng.normal(0, self.epoch_std, size=dim)

        prey = (
            omega[0] * self.alpha_pos
            + omega[1] * self.beta_pos
            + omega[2] * self.gamma_pos
            + noise
        )

        # 3. Update Population Positions
        # X_new = X_old - Uniform(-2, 2) * |Prey - X_old|

        # Random factor shape: (N, D)
        rand_factor = self.rng.uniform(-2, 2, size=(self.n_pop, dim))

        # Broadcasting: prey (D,) - pop (N, D)
        distance_to_prey = np.abs(prey - self.pop)

        offspring = self.pop - rand_factor * distance_to_prey

        return offspring

    # _selection and _update_best are inherited from GWO
    # as they are identical (Greedy selection & Top-3 hierarchy)


def enhanced_grey_wolf_optimization(
    objective: collections.abc.Callable, bounds: np.ndarray, **kwargs
) -> tuple:
    """
    Functional interface for Enhanced GWO.

    Returns:
        tuple: (best_pos, best_score).
    """
    optimizer = EGWO(objective=objective, bounds=bounds, **kwargs)
    return optimizer.optimize()
