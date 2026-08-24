
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

Variant:
    `GWO._selection` applies **greedy acceptance**, which the 2014 algorithm
    does not: Mirjalili et al. replace the pack unconditionally each iteration.
    What is implemented here is the greedy variant of Akbari et al. (2021).
    It is measurably better and is kept for that reason, but any comparison
    against published GWO numbers should say which one it is. See
    `GWO._selection` for the measurements and for what the choice does *not*
    explain.

Reference:
    Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer.
    *Advances in Engineering Software*, 69, 46-61.
"""


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
        Greedy selection: a wolf moves only if the move improves its score.

        **This is not canonical GWO.** Mirjalili et al. (2014), Fig. 6, replaces
        the pack unconditionally -- compute the new position, accept it, re-rank.
        Greedy acceptance is a published *variant* (Akbari et al. 2021), and it
        is the one implemented here, so results from this class belong to the
        variant rather than to the 2014 algorithm.

        Kept because it is worth a great deal. Median final value, greedy
        against canonical, 5 shifted landscapes x 20 seeds:

        =========  ====  ========  ===========
        optimizer  d     greedy    canonical
        =========  ====  ========  ===========
        EGWO       8       0.0957         4.72
        EGWO       32        78.0          144
        GWO        8         5.00         4.70
        GWO        32        83.4          122
        =========  ====  ========  ===========

        It does **not** explain why this family is unresponsive to
        initialization: restoring unconditional replacement leaves the response
        to OBLESA indistinguishable from zero at d=8 and still slightly
        *negative* at d=32 under both rules. See `pyBlindOpt.egwo` for that.

        Both rules evaluate exactly `n_pop` offspring per generation, so the
        choice is free for callers that require batched evaluation.

        Args:
            offspring (np.ndarray): New positions.
            offspring_scores (np.ndarray): Scores.

        References:
            Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf
            Optimizer. *Advances in Engineering Software*, 69, 46-61.

            Akbari, E., Rahmani, M., & Zarrabi, H. (2021). A greedy
            non-hierarchical grey wolf optimizer for real-world optimization.
            *Electronics Letters*, 57(13), 499-501.
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
