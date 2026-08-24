

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

    def _update_iter_params(self, epoch: int):
        """
        Parameter update hook.

        The energy parameter $E$ is calculated dynamically per individual inside `_generate_offspring`.
        """

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
        # (index, Y, Z) for each hawk taking the rapid-dive branch. Collected
        # here and scored together below rather than evaluated inside the
        # loop, where each dive asked the objective for a *single row*. The
        # group size is the contract; the call count is not. Making fewer
        # calls is a side effect here, never the goal -- see below for the
        # packing that would reduce it further and must not be done.
        dives = []

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

                    dives.append((i, Y[0], Z[0]))

        if dives:
            # Two calls, each of exactly `n_pop` rows: the diving hawks carry
            # their candidate, everyone else carries their current position
            # and their score is discarded. The choice below sees the same Y
            # and Z values it saw before, so the search is unchanged -- what
            # changes is that the objective is asked once for a full
            # population instead of twice per diving hawk.
            #
            # That is the contract this package is built to: an objective may
            # be a live match of `n_pop` agents, and a server that starts
            # only when `n_pop` players are connected cannot be asked to run
            # a one-player game.
            #
            # Do NOT collapse these into one call by packing Y and Z into a
            # single population when fewer than half the hawks dive. It
            # would fit -- and it would put hawk `i`'s Y and its Z into the
            # same match. A collective objective scores an agent against the
            # others present, so the two candidates would face different
            # opponents from each other and interference from every other
            # diver's pair, and the greedy choice below would no longer be
            # comparing what HHO says it compares. Two calls of `n_pop` is
            # what keeps both candidates against the same opponents.
            probe_y = self.pop.copy()
            probe_z = self.pop.copy()
            for i, y, z in dives:
                probe_y[i] = y
                probe_z[i] = z
            scores_y = self.evaluate(probe_y)
            scores_z = self.evaluate(probe_z)
            for i, y, z in dives:
                # Greedy choice between Y and Z; the base class `_selection`
                # still compares the winner against X_old afterwards.
                X_new[i] = y if scores_y[i] < scores_z[i] else z

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
