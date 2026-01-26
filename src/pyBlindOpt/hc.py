# coding: utf-8

"""
Hill Climbing (HC) optimization algorithm.

This module implements the Hill Climbing metaheuristic, a local search algorithm
that continuously moves towards increasing value (or decreasing cost) to find a local optimum.
It is a "greedy" approach that never accepts a move that worsens the objective function.

**Mathematical Formulation:**
Given a current solution $x_{curr}$, a candidate $x_{new}$ is generated.
The selection criterion is:
$$
x_{next} =
\\begin{cases}
x_{new} & \\text{if } f(x_{new}) < f(x_{curr}) \\\\
x_{curr} & \\text{otherwise}
\\end{cases}
$$

**Analogy:**
Imagine a climber in a thick fog who can only see one step ahead. They take a step;
if it leads higher (for maximization) or lower (for minimization), they take it.
If it doesn't, they stay put and try a different direction. They stop when no step
leads to an improvement, possibly getting stuck on a small hill (local optimum)
rather than the highest peak.

**Parallel Execution:**
If `n_pop > 1`, this runs as "Parallel Hill Climbing", maintaining multiple
independent climbers starting from different locations to increase the chance of
finding the global optimum.
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

import pyBlindOpt.utils as utils
from pyBlindOpt.optimizer import Optimizer

logger = logging.getLogger(__name__)


class HillClimbing(Optimizer):
    """
    Hill Climbing Optimizer.

    A local search algorithm that iteratively perturbs the solution and accepts
    it only if it strictly improves the objective (Greedy).
    """

    @utils.inherit_docs(Optimizer)
    def __init__(
        self,
        objective: collections.abc.Callable,
        bounds: np.ndarray,
        step_size: float = 0.01,
        **kwargs,
    ):
        self.step_size = step_size
        super().__init__(objective=objective, bounds=bounds, **kwargs)

    def _initialize(self):
        """
        Initialization hook.

        No specific initialization is required for standard Hill Climbing beyond
        population generation handled by the base class.
        """
        pass

    def _update_iter_params(self, epoch: int):
        """
        Parameter update hook.

        Hill Climbing is a stationary algorithm with no dynamic parameters
        (like temperature or inertia) to update per epoch.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Updates the global best solution found by any of the climbers.

        Scans the current population to see if any climber has found a spot
        better than the historically stored `best_score`.

        Args:
            epoch (int): The current iteration index.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Generates candidate solutions via Gaussian perturbation.

        Creates a neighbor solution $x_{new}$ by adding normally distributed noise
        to the current solution $x_{curr}$:
        $$ x_{new} = x_{curr} + \\mathcal{N}(0, \\sigma^2) $$
        where $\\sigma$ is the `step_size`.

        Args:
            epoch (int): The current iteration index.

        Returns:
            np.ndarray: The candidate solutions (offspring).
        """
        noise = self.rng.normal(loc=0.0, scale=self.step_size, size=self.pop.shape)
        return self.pop + noise

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Greedy Selection (Local Search).

        Accepts a new solution only if it is strictly better than the current one.

        $$
        x_{t+1} = \\text{argmin}(f(x_t), f(x_{new}))
        $$

        Args:
            offspring (np.ndarray): The candidate solutions.
            offspring_scores (np.ndarray): The objective values of the candidates.
        """
        improved_mask = offspring_scores < self.scores

        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]


def hill_climbing(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    step_size: float = 0.01,
    **kwargs,
) -> tuple:
    """
    Functional interface for running Hill Climbing optimization.

    Args:
        objective (Callable): Function to minimize.
        bounds (np.ndarray): Search space bounds.
        step_size (float): Standard deviation of perturbation noise.

    Returns:
        tuple: (best_position, best_score) or (best_pos, best_score, history) if debug=True.
    """
    optimizer = HillClimbing(
        objective=objective, bounds=bounds, step_size=step_size, **kwargs
    )
    return optimizer.optimize()
