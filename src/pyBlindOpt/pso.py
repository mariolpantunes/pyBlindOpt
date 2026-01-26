# coding: utf-8

"""
Particle Swarm Optimization (PSO).

A classic metaheuristic simulating a flock of birds or school of fish.
Particles fly through the search space with a velocity adjusted by their own history and the swarm's best known position.


**Mathematical Formulation:**
$$ v_{t+1} = w v_t + c_1 r_1 (p_{best} - x_t) + c_2 r_2 (g_{best} - x_t) $$
$$ x_{t+1} = x_t + v_{t+1} $$
where $w$ is inertia, $c_1$ is cognitive (personal) weight, and $c_2$ is social (swarm) weight.
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


class ParticleSwarmOptimization(Optimizer):
    """
    Particle Swarm Optimization (PSO).

    A population-based metaheuristic where particles move through the search space
    guided by their own best known position (pbest) and the swarm's best known position (gbest).
    """

    @utils.inherit_docs(Optimizer)
    def __init__(
        self,
        objective: collections.abc.Callable,
        bounds: np.ndarray,
        c1: float = 0.1,  # Cognitive parameter
        c2: float = 0.1,  # Social parameter
        w: float = 0.8,  # Inertia weight
        **kwargs,
    ):
        """
        Particle Swarm Optimization.

        Args:
            c1 (float): Cognitive parameter. Pulls particle towards its own personal best. Defaults to 0.1.
            c2 (float): Social parameter. Pulls particle towards the swarm's global best. Defaults to 0.1.
            w (float): Inertia weight. Keeps the particle moving in its previous direction. Defaults to 0.8.
        """
        self.c1 = c1
        self.c2 = c2
        self.w = w

        super().__init__(objective=objective, bounds=bounds, **kwargs)

    def _init_population(self, population, seed):
        """
        Initializes population and velocities.

        Sets initial velocities to 10% of the bound width to prevent immediate explosion.

        Args:
            population: Initial positions.
            seed: Random seed.
        """
        # 1. Initialize Positions (using Base implementation)
        super()._init_population(population, seed)

        # 2. Initialize Velocities
        # V is initialized as 10% of the bound width to provide initial momentum
        # but prevent immediate explosion.
        bound_width = self.bounds[:, 1] - self.bounds[:, 0]
        self.v = self.rng.uniform(
            -0.1 * bound_width, 0.1 * bound_width, size=self.pop.shape
        )

    def _initialize(self):
        """
        Initializes Personal Bests ($P_{best}$).

        Sets $P_{best}$ to the initial positions at the start of the run.
        """
        self.pbest = self.pop.copy()
        self.pbest_scores = self.scores.copy()

    def _update_iter_params(self, epoch: int):
        """
        Parameter update hook.

        This implementation uses constant $w, c_1, c_2$, so no update is performed.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Updates the Global Best ($G_{best}$).

        The global best is determined by the best value ever found in the $P_{best}$ history.

        Args:
            epoch (int): Current iteration.
        """
        best_idx = np.argmin(self.pbest_scores)
        if self.pbest_scores[best_idx] < self.best_score:
            self.best_score = self.pbest_scores[best_idx]
            self.best_pos = self.pbest[best_idx].copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Updates Velocities and Positions.

        $$ v_{new} = w v + c_1 r_1 (P_{best} - x) + c_2 r_2 (G_{best} - x) $$
        $$ x_{new} = x + v_{new} $$

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The new positions of the particles.
        """
        # Random coefficients r1, r2 (one pair per particle or per dimension?)
        # Standard PSO usually does it per dimension for diversity.
        r1 = self.rng.random(size=self.pop.shape)
        r2 = self.rng.random(size=self.pop.shape)

        # Cognitive Term (pbest - current)
        cognitive = self.c1 * r1 * (self.pbest - self.pop)

        # Social Term (gbest - current)
        # We need to broadcast gbest (1D) to match pop shape (N, D)
        social = self.c2 * r2 * (self.best_pos - self.pop)

        # Update Velocity
        self.v = (self.w * self.v) + cognitive + social

        # Update Position
        new_pos = self.pop + self.v
        return new_pos

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Updates Personal Bests ($P_{best}$) and moves the swarm.

        1.  Always accepts the new position $x_{new}$ as the current state (particles keep moving).
        2.  Updates $P_{best}$ if $x_{new}$ is better than the previous $P_{best}$.

        Args:
            offspring (np.ndarray): New positions.
            offspring_scores (np.ndarray): New scores.
        """
        # 1. Update Current Population (Particles move regardless of improvement)
        # In PSO, the 'population' tracks the current *exploring* point.
        self.pop = offspring
        self.scores = offspring_scores

        # 2. Update Personal Bests
        # Check where new score < old pbest score
        improved_mask = offspring_scores < self.pbest_scores

        self.pbest[improved_mask] = offspring[improved_mask]
        self.pbest_scores[improved_mask] = offspring_scores[improved_mask]


def particle_swarm_optimization(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    c1: float = 0.1,
    c2: float = 0.1,
    w: float = 0.8,
    **kwargs,
) -> tuple:
    """
    Functional interface for Particle Swarm Optimization.

    Returns:
        tuple: (best_pos, best_score).
    """
    optimizer = ParticleSwarmOptimization(
        objective=objective, bounds=bounds, c1=c1, c2=c2, w=w, **kwargs
    )
    return optimizer.optimize()
