# coding: utf-8

"""
Simulated Annealing (SA) optimization algorithm.

A probabilistic local search mimicking the physical annealing process of solids.
Allows accepting worse solutions to escape local optima, controlled by a
decreasing temperature parameter.
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


class SimulatedAnnealing(Optimizer):
    """
    Simulated Annealing Optimizer.

    A probabilistic local search algorithm that approximates global optimization
    by accepting worse solutions with a probability that decreases over time (Temperature).

    If n_pop > 1, this acts as 'Parallel Simulated Annealing', maintaining 'n_pop'
    independent annealing chains.

    **Cooling Schedule:**
    Fast Annealing (Cauchy):
    $$ T_k = \\frac{T_0}{k + 1} $$
    """

    @utils.inherit_docs(Optimizer)
    def __init__(
        self,
        objective: collections.abc.Callable,
        bounds: np.ndarray,
        step_size: float = 0.01,
        temp: float = 20.0,
        **kwargs,
    ):
        self.step_size = step_size
        self.initial_temp = temp
        self.current_temp = temp

        super().__init__(objective=objective, bounds=bounds, **kwargs)

    def _initialize(self):
        """
        Resets temperature to initial value.
        """
        self.current_temp = self.initial_temp

    def _update_iter_params(self, epoch: int):
        """
        Updates temperature $T$.
        $$ T_{new} = T_0 / (epoch + 1) $$
        """
        self.current_temp = self.initial_temp / float(epoch + 1)

    def _update_best(self, epoch: int):
        """
        Updates global best from current population chains.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Gaussian perturbation.
        $$ x_{new} = x_{curr} + \\mathcal{N}(0, \\sigma^2) $$
        """
        noise = self.rng.normal(loc=0.0, scale=self.step_size, size=self.pop.shape)
        return self.pop + noise

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Metropolis Criterion.

        $$ P(\\text{accept}) = \\exp\\left(-\\frac{\\Delta E}{T}\\right) $$
        Accepts if Improved ($\\Delta E < 0$) OR if Random < $P(\\text{accept})$.
        """
        # Calculate Delta E (Candidate - Current)
        # Note: Optimization implies Minimization.
        diff = offspring_scores - self.scores

        # 1. Check for strict improvement
        # Mask where candidate is better (diff < 0)
        improve_mask = diff < 0

        # 2. Check for Metropolis acceptance (for worse solutions)
        # P = exp(-diff / T)
        # We only compute this where diff >= 0 to avoid warnings/unnecessary work
        metropolis_mask = np.zeros(self.n_pop, dtype=bool)

        worse_indices = ~improve_mask
        if np.any(worse_indices):
            # Probability of accepting worse solution
            # Ensure T is not zero to avoid division by zero error
            safe_temp = max(self.current_temp, 1e-10)
            acceptance_prob = np.exp(-diff[worse_indices] / safe_temp)

            # Generate random numbers [0, 1] for these indices
            count = int(np.sum(worse_indices))
            rand_vals = self.rng.random(size=count)

            # Accept if Random < Prob
            accepted_worse = rand_vals < acceptance_prob

            # Map back to full mask
            metropolis_mask[worse_indices] = accepted_worse

        # Combine: Accept if (Improved OR Metropolis)
        accept_mask = improve_mask | metropolis_mask

        # Update population
        self.pop[accept_mask] = offspring[accept_mask]
        self.scores[accept_mask] = offspring_scores[accept_mask]


def simulated_annealing(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    step_size: float = 0.01,
    temp: float = 20.0,
    **kwargs,
) -> tuple:
    """
    Functional interface for Simulated Annealing.

    Helper function to run Simulated Annealing.
    Set n_pop > 1 to enable Parallel Simulated Annealing.
    """
    optimizer = SimulatedAnnealing(
        objective=objective, bounds=bounds, step_size=step_size, temp=temp, **kwargs
    )
    return optimizer.optimize()
