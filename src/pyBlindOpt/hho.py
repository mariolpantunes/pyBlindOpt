# coding: utf-8


"""
Harris Hawks Optimization (HHO).

This module mimics the cooperative hunting behavior of Harris' hawks (surprise pounce).
It features distinct exploration and exploitation phases controlled by the prey's escaping energy.

**Analogy:**
* **Exploration:** Hawks perch randomly or based on other hawks to find prey.
* **Exploitation:** Hawks besiege the rabbit (prey).
    * **Soft Besiege:** Rabbit has energy, hawks encircle slowly.
    * **Hard Besiege:** Rabbit is tired, hawks attack directly.
    * **Rapid Dives:** Hawks perform Lévy flight dives if the rabbit attempts to escape.

**Mathematical Formulation:**
Transitions are controlled by Escaping Energy $E$:
$$ E = 2 E_0 (1 - t/T) $$
where $E_0 \\in [-1, 1]$. $|E| \\ge 1$ triggers exploration, $|E| < 1$ triggers exploitation.
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


class HarrisHawksOptimization(Optimizer):
    """
    Harris Hawks Optimizer.

    Implements the 4 phases of HHO driven by the escaping energy $E$.
    """

    @utils.inherit_docs(Optimizer)
    def __init__(self, objective, bounds, **kwargs):
        super().__init__(objective, bounds, **kwargs)

    def _initialize(self):
        """
        Initialization hook.

        No specific internal state required.
        """
        pass

    def _update_iter_params(self, epoch: int):
        """
        Parameter update hook.

        The energy parameter $E$ is calculated dynamically per individual inside `_generate_offspring`.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Updates the Rabbit position (Global Best).

        In HHO, the global best is referred to as the "Rabbit".

        Args:
            epoch (int): Current iteration.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Generates new positions based on the HHO phases.

        **1. Exploration ($|E| \\ge 1$):**
        Move based on random hawk or average position.

        **2. Exploitation ($|E| < 1$):**
        * **Soft Besiege:** $x_{new} = \\text{Rabbit} - E | J \\cdot \\text{Rabbit} - x |$
        * **Hard Besiege:** $x_{new} = \\text{Rabbit} - E | \\text{Rabbit} - x |$
        * **Soft/Hard with Rapid Dives:** Uses Lévy flights ($LF$) to perform zig-zag movements if the besiege fails ($Z = Y + S \times LF$).

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The new positions of the hawks.
        """
        rabbit = self.best_pos
        E0 = 2 * self.rng.random(self.n_pop) - 1
        E = 2 * E0 * (1 - (epoch / self.n_iter))

        X_new = np.zeros_like(self.pop)
        mean_hawk = np.mean(self.pop, axis=0)

        # Iterate per hawk (vectorizing HHO's 4 branches is complex and prone to bugs)
        for i in range(self.n_pop):
            energy = E[i]
            x = self.pop[i]

            if abs(energy) >= 1:  # Exploration
                q = self.rng.random()
                if q >= 0.5:
                    rand_idx = self.rng.integers(0, self.n_pop)
                    rand_hawk = self.pop[rand_idx]
                    r1, r2 = self.rng.random(), self.rng.random()
                    X_new[i] = rand_hawk - r1 * np.abs(rand_hawk - 2 * r2 * x)
                else:
                    r3, r4 = self.rng.random(), self.rng.random()
                    term = self.bounds[:, 0] + r4 * (
                        self.bounds[:, 1] - self.bounds[:, 0]
                    )
                    X_new[i] = (rabbit - mean_hawk) - r3 * term
            else:  # Exploitation
                r = self.rng.random()

                # Soft Besiege
                if r >= 0.5 and abs(energy) >= 0.5:
                    J = 2 * (1 - self.rng.random())
                    X_new[i] = (rabbit - x) - energy * np.abs(J * rabbit - x)

                # Hard Besiege
                elif r >= 0.5 and abs(energy) < 0.5:
                    X_new[i] = rabbit - energy * np.abs(rabbit - x)

                # Rapid Dives (Soft & Hard)
                else:
                    # Base target Y
                    if abs(energy) >= 0.5:  # Phase 3 (Soft)
                        J = 2 * (1 - self.rng.random())
                        Y = rabbit - energy * np.abs(J * rabbit - x)
                    else:  # Phase 4 (Hard)
                        J = 2 * (1 - self.rng.random())
                        Y = rabbit - energy * np.abs(J * rabbit - mean_hawk)

                    # Dive target Z (Levi Flight)
                    dim = self.bounds.shape[0]
                    S = self.rng.random(dim)
                    levy = utils.levy_flight(1, dim, 1.5, self.rng)[0]
                    Z = Y + S * levy

                    # Selection internal to offspring generation
                    # We must evaluate to decide between Y and Z
                    Y = self._check_bounds(Y[np.newaxis, :])
                    Z = self._check_bounds(Z[np.newaxis, :])

                    score_Y = self.evaluate(Y)[0]
                    score_Z = self.evaluate(Z)[0]

                    # Greedy choice between Y, Z (and implicit X_old)
                    # We return the best of Y or Z.
                    # The base class _selection will compare this result vs X_old.
                    if score_Y < score_Z:
                        X_new[i] = Y[0]
                    else:
                        X_new[i] = Z[0]

                    # Note: If both Y and Z are worse than X_old, we still return one here.
                    # _selection will reject it later.

        return X_new

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Greedy Selection.

        Accepts the new position only if it improves upon the old one.

        Args:
            offspring (np.ndarray): New hawk positions.
            offspring_scores (np.ndarray): Scores.
        """
        improved_mask = offspring_scores < self.scores
        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]


def harris_hawks_optimization(objective, bounds, **kwargs):
    """
    Functional interface for Harris Hawks Optimization.

    Returns:
        tuple: (best_pos, best_score).
    """
    return HarrisHawksOptimization(objective, bounds, **kwargs).optimize()
