# coding: utf-8

"""
Differential Evolution (DE).

A powerful evolutionary algorithm that uses the differences between randomly selected vectors to perturb the population.

**Analogy:**
Imagine a group of agents. Each agent looks at other agents, takes the difference between them, scales it, and adds it to a target vector. This creates a "mutant". If the mutant is better than the current agent, it replaces it.

**Mathematical Formulation:**
**Mutation (DE/best/1):**
$$ v_i = x_{best} + F \\cdot (x_{r1} - x_{r2}) $$
**Crossover:**
Mixes the target vector $x_i$ and mutant $v_i$ with probability $CR$.

**Key Concepts:**
* **Mutation:** Generating a new vector from differences of others.
* **Crossover:** Mixing the mutant with the current individual.
* **Selection:** Greedy survival (child replaces parent if better).
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.3.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import collections.abc

import numpy as np

import pyBlindOpt.utils as utils
from pyBlindOpt.optimizer import Optimizer


# ==============================================================================
# 1. Mutation Operators
# ==============================================================================
def mutation_rand_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/rand/1: $v = r1 + F(r2 - r3)$
    """
    return candidates[0] + F * (candidates[1] - candidates[2])


def mutation_best_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/best/1: $v = best + F(r1 - r2)$
    """
    return best + F * (candidates[0] - candidates[1])


def mutation_rand_2(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/rand/2: $v = r1 + F(r2 - r3) + F(r4 - r5)$
    """
    return (
        candidates[0]
        + F * (candidates[1] - candidates[2])
        + F * (candidates[3] - candidates[4])
    )


def mutation_best_2(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/best/2: $v = best + F(r1 - r2) + F(r3 - r4)$
    """
    return (
        best + F * (candidates[0] - candidates[1]) + F * (candidates[2] - candidates[3])
    )


def mutation_current_to_best_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/current-to-best/1: $v = current + F(best - current) + F(r1 - r2)$
    """
    return current + F * (best - current) + F * (candidates[0] - candidates[1])


def mutation_current_to_rand_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/current-to-rand/1: $v = current + F(r1 - current) + F(r2 - r3)$
    """
    return current + F * (candidates[0] - current) + F * (candidates[1] - candidates[2])


# ==============================================================================
# 2. Crossover Operators
# ==============================================================================
def crossover_bin(
    target: np.ndarray, mutant: np.ndarray, cr: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Binomial Crossover.
    Each gene is swapped with probability $CR$. Ensures at least one gene is changed.
    """
    dim = target.shape[0]
    mask = rng.random(dim) < cr
    # Force at least one index to change (standard DE guarantee)
    j_rand = rng.integers(0, dim)
    mask[j_rand] = True
    return np.where(mask, mutant, target)


def crossover_exp(
    target: np.ndarray, mutant: np.ndarray, cr: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Exponential Crossover.
    Swaps a contiguous block of genes starting from a random index.
    """
    dim = target.shape[0]
    trial = target.copy()
    j = rng.integers(0, dim)
    L = 0
    while rng.random() < cr and L < dim:
        trial[j] = mutant[j]
        j = (j + 1) % dim
        L += 1
    return trial


# ==============================================================================
# 3. Optimizer Class
# ==============================================================================
class DifferentialEvolution(Optimizer):
    """
    Differential Evolution Optimizer.

    A versatile DE implementation supporting multiple mutation strategies and crossover methods
    via a configuration string.

    **Supported Variants:**
    * `rand/1`: Standard DE. Good diversity.
    * `best/1`: Converges fast, greedy.
    * `rand/2`: Robust for difficult landscapes.
    * `best/2`: Trade-off between greedy and robust.
    * `current-to-best/1`: Rotationally invariant, modern standard.
    * `current-to-rand/1`: Rotationally invariant, exploratory.

    **Supported Crossovers:**
    * `bin`: Binomial (independent swaps).
    * `exp`: Exponential (block swaps).
    """

    # Strategy Mapping: "Name" -> (Function, Required_Sample_Count)
    _STRATEGIES = {
        "rand/1": (mutation_rand_1, 3),
        "best/1": (mutation_best_1, 2),
        "rand/2": (mutation_rand_2, 5),
        "best/2": (mutation_best_2, 4),
        "current-to-best/1": (mutation_current_to_best_1, 2),
        "current-to-rand/1": (mutation_current_to_rand_1, 3),
    }

    _CROSSOVERS = {"bin": crossover_bin, "exp": crossover_exp}

    @utils.inherit_docs(Optimizer)
    def __init__(
        self,
        objective: collections.abc.Callable,
        bounds: np.ndarray,
        variant: str = "best/1/bin",
        parent_selection: str = "rand",
        F: float = 0.5,
        cr: float = 0.7,
        **kwargs,
    ):
        """
        Differential Evolution Optimizer.

        Args:
            variant (str): Strategy format 'target/num_diffs/crossover'.
                           Examples: 'rand/1/bin', 'best/2/exp', 'current-to-best/1/bin'.
                           Defaults to 'best/1/bin'.
            parent_selection (str): Method to select the base vector ($r1$).
                                    Options: 'rand' (Standard), 'tournament'.
                                    Defaults to 'rand'.
            F (float): Differential weight (scaling factor). Defaults to 0.5.
            cr (float): Crossover probability. Defaults to 0.7.
        """
        self.F = F
        self.cr = cr
        self.parent_selection = parent_selection
        self.variant_name = variant

        # Parse Variant String
        try:
            # Expect format: "strategy_base/strategy_num/crossover"
            # We join base and num to look up strategy (e.g. "rand/1")
            parts = variant.split("/")
            if len(parts) != 3:
                raise ValueError("Variant format must be 'base/n/cross'")

            strategy_key = f"{parts[0]}/{parts[1]}"
            crossover_key = parts[2]

            if strategy_key not in self._STRATEGIES:
                raise KeyError(f"Unknown strategy: {strategy_key}")
            if crossover_key not in self._CROSSOVERS:
                raise KeyError(f"Unknown crossover: {crossover_key}")

            self.mutation_op, self.samples_needed = self._STRATEGIES[strategy_key]
            self.crossover_op = self._CROSSOVERS[crossover_key]

        except (KeyError, IndexError, ValueError) as e:
            valid_strats = list(self._STRATEGIES.keys())
            valid_cross = list(self._CROSSOVERS.keys())
            raise ValueError(
                f"Invalid variant '{variant}'.\n"
                f"Supported Strategies: {valid_strats}\n"
                f"Supported Crossovers: {valid_cross}\n"
                f"Error: {e}"
            )

        super().__init__(objective, bounds, **kwargs)

    def _initialize(self):
        """
        Initialization hook.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Updates the global best solution.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Generates the Trial Population.

        1.  **Selection:** Picks random distinct vectors ($r1, r2, ...$).
            Supports Tournament selection for the base vector ($r1$).
        2.  **Mutation:** Creates mutant vectors.
        3.  **Crossover:** Combines mutant and target vectors.
        """
        offspring = np.zeros_like(self.pop)
        n_pop = self.n_pop

        for j in range(n_pop):
            # 1. Identify valid pool (cannot include self)
            available_indices = np.delete(np.arange(n_pop), j)

            # 2. Select Candidates
            if self.parent_selection == "tournament":
                # Tournament for the first candidate (r1 / base vector)
                # This increases selection pressure for the mutation base.

                # a. Perform Tournament
                k_tournament = 3
                if len(available_indices) < k_tournament:
                    # Fallback for small pops
                    tourn_inds = self.rng.choice(
                        available_indices, size=len(available_indices), replace=False
                    )
                else:
                    tourn_inds = self.rng.choice(
                        available_indices, size=k_tournament, replace=False
                    )

                # Winner has lowest score (minimization)
                winner_idx = tourn_inds[np.argmin(self.scores[tourn_inds])]

                # b. Select remaining candidates randomly
                needed_others = self.samples_needed - 1
                remaining_pool = np.setdiff1d(available_indices, [winner_idx])

                if len(remaining_pool) < needed_others:
                    # Fallback (allow replacement if strictly needed)
                    others = self.rng.choice(
                        remaining_pool, size=needed_others, replace=True
                    )
                else:
                    others = self.rng.choice(
                        remaining_pool, size=needed_others, replace=False
                    )

                # Combine: [Winner, r2, r3...]
                choices = np.concatenate(([winner_idx], others))

            else:
                # Standard Random Selection
                if self.samples_needed > len(available_indices):
                    choices = self.rng.choice(
                        available_indices, size=self.samples_needed, replace=True
                    )
                else:
                    choices = self.rng.choice(
                        available_indices, size=self.samples_needed, replace=False
                    )

            candidates = self.pop[choices]

            # 3. Mutation
            # Note: For 'best/...' strategies, 'best' is used as base, and 'candidates' are just diffs.
            # Tournament selection above mainly benefits 'rand/...' strategies where candidates[0] is base.
            mutant = self.mutation_op(self.pop[j], self.best_pos, candidates, self.F)

            # 4. Crossover
            trial = self.crossover_op(self.pop[j], mutant, self.cr, self.rng)
            offspring[j] = trial

        return offspring

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Greedy Survivor Selection.

        The child replaces the parent if and only if it is better or equal.
        """
        improved_mask = offspring_scores <= self.scores
        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]


def differential_evolution(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    variant: str = "best/1/bin",
    parent_selection: str = "rand",
    F: float = 0.5,
    cr: float = 0.7,
    **kwargs,
) -> tuple:
    """
    Functional interface for Differential Evolution.

    Args:
        objective (Callable): The function to minimize.
        bounds (np.ndarray): Search bounds (min, max).
        variant (str): Strategy string (e.g., 'rand/1/bin').
        parent_selection (str): 'rand' or 'tournament'.
        F (float): Mutation factor.
        cr (float): Crossover probability.

    Returns:
        tuple: (best_pos, best_score).
    """
    optimizer = DifferentialEvolution(
        objective=objective,
        bounds=bounds,
        variant=variant,
        parent_selection=parent_selection,
        F=F,
        cr=cr,
        **kwargs,
    )
    return optimizer.optimize()
