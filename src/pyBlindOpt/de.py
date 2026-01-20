# coding: utf-8

"""
Differential Evolution (DE).

A powerful evolutionary algorithm that uses the differences between randomly selected vectors to perturb the population.

**Analogy:**
Imagine a group of agents. Each agent looks at three others, takes the difference between two of them, scales it, and adds it to the third. This creates a "mutant" vector. If the mutant is better, the agent adopts it.

**Mathematical Formulation:**
**Mutation (DE/best/1):**
$$ v_i = x_{best} + F \\cdot (x_{r1} - x_{r2}) $$
**Crossover:**
Mixes the target vector $x_i$ and mutant $v_i$ with probability $CR$.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import collections.abc
import enum
import logging

import numpy as np

from pyBlindOpt.optimizer import Optimizer

logger = logging.getLogger(__name__)


@enum.unique
class TargetVector(enum.Enum):
    best = "best"
    rand = "rand"


@enum.unique
class CrossoverMethod(enum.Enum):
    bin = "bin"
    exp = "exp"


class DifferentialEvolution(Optimizer):
    """
    Differential Evolution Optimizer.

    Supports configurable strategies via the `variant` string (e.g., 'best/1/bin', 'rand/1/exp').
    """

    def __init__(
        self,
        objective: collections.abc.Callable,
        bounds: np.ndarray,
        population: np.ndarray | None = None,
        variant: str = "best/1/bin",
        callback: list[collections.abc.Callable]
        | collections.abc.Callable
        | None = None,
        n_iter: int = 100,
        n_pop: int = 10,
        F: float = 0.5,
        cr: float = 0.7,
        n_jobs: int = 1,
        cached: bool = False,
        debug: bool = False,
        verbose: bool = False,
        seed: int = 42,
    ):
        """
        Differential Evolution Optimizer.

        Supports configurable strategies via the `variant` string (e.g., 'best/1/bin', 'rand/1/exp').

        Args:
            variant (str): Strategy format 'target/num_diffs/crossover'. Defaults to 'best/1/bin'.
            F (float): Differential weight (scaling factor). Defaults to 0.5.
            cr (float): Crossover probability. Defaults to 0.7.
        """

        self.F = F
        self.cr = cr

        # Parse Variant String (e.g., "best/1/bin")
        try:
            v = variant.split("/")
            self.tv = TargetVector[v[0]]
            self.dv = int(v[1])
            self.cm = CrossoverMethod[v[2]]
            # Number of candidates needed:
            # if 'best': we need 'dv' pairs (2*dv)
            # if 'rand': we need 1 base + 'dv' pairs (1 + 2*dv)
            self.nc = 2 * self.dv if self.tv is TargetVector.best else 1 + 2 * self.dv
        except (KeyError, IndexError, ValueError):
            raise ValueError(
                "Variant must be format: '[rand|best]/n/[bin|exp]' (e.g., 'best/1/bin')"
            )

        super().__init__(
            objective=objective,
            bounds=bounds,
            population=population,
            callback=callback,
            n_iter=n_iter,
            n_pop=n_pop,
            n_jobs=n_jobs,
            cached=cached,
            debug=debug,
            verbose=verbose,
            seed=seed,
        )

    def _initialize(self):
        """
        Initialization hook.

        No specific state required per iteration.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Updates the global best solution.

        Args:
            epoch (int): Current iteration.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _mutation(self, candidates: np.ndarray) -> np.ndarray:
        """
        Applies Differential Mutation.

        $$ v = x_{base} + F \\sum (x_{rA} - x_{rB}) $$

        Args:
            candidates (np.ndarray): Array where index 0 is the base vector and subsequent indices are pairs for difference calculation.

        Returns:
            np.ndarray: The mutant vector.
        """
        diff_sum = np.zeros_like(candidates[0])

        # Iterate pairs: (1,2), (3,4), etc.
        for i in range(1, len(candidates), 2):
            diff_sum += candidates[i] - candidates[i + 1]

        return candidates[0] + self.F * diff_sum

    def _crossover_binom(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Binomial Crossover.

        Each gene is swapped with probability $CR$. Ensures at least one gene is changed.

        Args:
            target (np.ndarray): The parent vector.
            mutant (np.ndarray): The donor vector.

        Returns:
            np.ndarray: The trial vector.
        """
        dim = target.shape[0]
        # 1. Generate random mask
        mask = self.rng.random(dim) < self.cr
        # 2. Force at least one index to change (standard DE guarantee)
        j_rand = self.rng.integers(0, dim)
        mask[j_rand] = True

        trial = np.where(mask, mutant, target)
        return trial

    def _crossover_exp(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Exponential Crossover.

        Swaps a contiguous block of genes starting from a random index.

        Args:
            target (np.ndarray): The parent vector.
            mutant (np.ndarray): The donor vector.

        Returns:
            np.ndarray: The trial vector.
        """
        dim = target.shape[0]
        trial = target.copy()

        j = self.rng.integers(0, dim)
        L = 0
        while self.rng.random() < self.cr and L < dim:
            trial[j] = mutant[j]
            j = (j + 1) % dim
            L += 1

        return trial

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Generates the Trial Population.

        1.  **Selection:** Picks random distinct vectors ($r1, r2, ...$) for each individual.
        2.  **Mutation:** Creates mutant vectors.
        3.  **Crossover:** Combines mutant and target vectors.

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The population of trial vectors.
        """
        offspring = np.zeros_like(self.pop)

        for j in range(self.n_pop):
            available_indices = np.delete(np.arange(self.n_pop), j)
            count_needed = (
                2 * self.dv if self.tv is TargetVector.best else 1 + 2 * self.dv
            )
            if count_needed > len(available_indices):
                # Fallback for very small populations
                choices = self.rng.choice(
                    available_indices, size=count_needed, replace=True
                )
            else:
                choices = self.rng.choice(
                    available_indices, size=count_needed, replace=False
                )

            picked_vecs = self.pop[choices]

            # Construct the mutation candidate list
            # Format: [Base, Pair1_A, Pair1_B, Pair2_A, Pair2_B...]
            if self.tv is TargetVector.best:
                candidates = np.vstack([self.best_pos, picked_vecs])
            else:
                candidates = picked_vecs

            # 2. Mutation
            mutated_vec = self._mutation(candidates)

            # 3. Crossover
            if self.cm is CrossoverMethod.bin:
                trial_vec = self._crossover_binom(self.pop[j], mutated_vec)
            else:
                trial_vec = self._crossover_exp(self.pop[j], mutated_vec)

            offspring[j] = trial_vec

        return offspring

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Greedy Selection.

        Survivor selection: The child replaces the parent if and only if it is better.

        Args:
            offspring (np.ndarray): Trial vectors.
            offspring_scores (np.ndarray): Trial scores.
        """
        improved_mask = offspring_scores < self.scores
        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]


def differential_evolution(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    population: np.ndarray | None = None,
    variant: str = "best/1/bin",
    callback: list[collections.abc.Callable] | collections.abc.Callable | None = None,
    n_iter: int = 100,
    n_pop: int = 10,
    F: float = 0.5,
    cr: float = 0.7,
    rt: int = 10,
    n_jobs: int = 1,
    cached: bool = False,
    debug: bool = False,
    verbose: bool = False,
    seed: int = 42,
) -> tuple:
    """
    Functional interface for Differential Evolution.

    Returns:
        tuple: (best_pos, best_score).
    """
    # Convert list population to array if provided
    pop_arr = np.array(population) if population is not None else None

    optimizer = DifferentialEvolution(
        objective=objective,
        bounds=bounds,
        population=pop_arr,
        variant=variant,
        callback=callback,
        n_iter=n_iter,
        n_pop=n_pop,
        F=F,
        cr=cr,
        n_jobs=n_jobs,
        cached=cached,
        debug=debug,
        verbose=verbose,
        seed=seed,
    )
    return optimizer.optimize()
