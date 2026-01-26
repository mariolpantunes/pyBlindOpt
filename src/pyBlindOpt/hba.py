# coding: utf-8

"""
Honey Badger Algorithm (HBA).

Mimics the intelligent foraging behavior of the honey badger.
It switches between digging (using smell) and eating honey (following a honeyguide bird).

**Analogy:**
* **Digging Phase:** Use smell intensity ($I$) to locate prey and dig in a cardioid shape motion.
* **Honey Phase:** Follow the guide bird directly to the hive.

**Mathematical Formulation:**
**Digging:** $x_{new} = x_{prey} + F \\cdot \\beta \\cdot I \\cdot x_{prey} + \\text{Cardioid}$
**Honey:** $x_{new} = x_{prey} + F \\cdot r \\cdot \\alpha \\cdot d$
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import math

import numpy as np

import pyBlindOpt.utils as utils
from pyBlindOpt.optimizer import Optimizer


class HoneyBadgerAlgorithm(Optimizer):
    """
    Honey Badger Algorithm.
    """

    @utils.inherit_docs(Optimizer)
    def __init__(
        self,
        objective,
        bounds,
        beta: float = 6.0,
        C: float = 2.0,
        **kwargs,
    ):
        """
        Honey Badger Algorithm.

        Args:
            beta (float): Ability of the badger to get food (digging intensity). Defaults to 6.0.
            C (float): Constant for density factor update. Defaults to 2.0.
        """
        self.beta = beta
        self.C = C
        super().__init__(objective, bounds, **kwargs)

    def _initialize(self):
        """
        Initialization hook.

        No specific state required.
        """
        pass

    def _update_iter_params(self, epoch: int):
        """
        Parameter update hook.

        Density factor $\\alpha$ is computed per epoch inside `_generate_offspring`.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Updates the Prey position (Global Best).

        Args:
            epoch (int): Current iteration.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Generates new positions based on Digging or Honey phases.

        Calculates smell intensity $I$ based on inverse square distance.
        Switches between phases probabilistically.

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: New badger positions.
        """
        alpha = self.C * math.exp(-epoch / self.n_iter)
        prey = self.best_pos
        new_pop = np.zeros_like(self.pop)

        for i in range(self.n_pop):
            # Intensity I
            dist = np.linalg.norm(self.pop[i] - prey)
            dist_sq = dist * dist if dist > 1e-10 else 1e-10
            In = self.rng.random() * self.C / (4 * math.pi * dist_sq) * alpha

            # Direction F
            F = 1 if self.rng.random() <= 0.5 else -1

            if self.rng.random() <= 0.5:
                r3, r4, r5 = self.rng.random(), self.rng.random(), self.rng.random()

                term_attract = F * self.beta * In * prey
                term_dist = (
                    F
                    * r3
                    * alpha
                    * dist
                    * np.abs(
                        math.cos(2 * math.pi * r4) * (1 - math.cos(2 * math.pi * r5))
                    )
                )

                new_pop[i] = prey + term_attract + term_dist
            else:  # Honey
                r7 = self.rng.random()
                new_pop[i] = prey + F * r7 * alpha * dist

        return new_pop

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Greedy Selection.

        Args:
            offspring (np.ndarray): New positions.
            offspring_scores (np.ndarray): New scores.
        """
        improved_mask = offspring_scores < self.scores
        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]


def honey_badger_algorithm(
    objective, bounds, beta: float = 6.0, C: float = 2.0, **kwargs
):
    """
    Functional interface for Honey Badger Algorithm.

    Returns:
        tuple: (best_pos, best_score).
    """
    return HoneyBadgerAlgorithm(objective, bounds, beta, C, **kwargs).optimize()
