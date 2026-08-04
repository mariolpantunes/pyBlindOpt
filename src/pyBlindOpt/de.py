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
import typing

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


def mutation_current_to_pbest_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    r"""
    DE/current-to-pbest/1: $v = current + F(x_{pbest} - current) + F(r1 - r2)$

    The arithmetic is `mutation_current_to_best_1`'s, and deliberately so --
    what differs is not the formula but the vector the caller supplies as
    `best`. Here it is $x_{pbest}$, drawn afresh for each individual from the
    fittest $p \cdot N$ of the population, rather than the single global best
    shared by everyone.

    That one substitution is the whole idea behind JADE's mutation. Pulling
    every individual toward the *same* point is what makes `current-to-best/1`
    greedy and prone to premature convergence; pulling each toward a
    *different* good point keeps the population spread while still moving it
    somewhere useful. The two coincide at $p = 1/N$, and at $p = 1$ the pull
    is toward a uniformly random member, which is `current-to-rand/1`.

    See `DifferentialEvolution._base_vector`, which does the drawing.
    """
    return mutation_current_to_best_1(current, best, candidates, F)


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
# 3. Parameter and Strategy Policies
# ==============================================================================
class Proposal(typing.NamedTuple):
    """What a policy decides for one trial round, one entry per individual.

    Arrays rather than scalars because the adaptive variants choose `F` and
    `cr` *per individual* and learn from which choices succeeded. Holding them
    as arrays means the generic loop indexes rather than branches, and the
    same arrays go straight back to `Policy.observe`.
    """

    F: np.ndarray
    cr: np.ndarray
    ops: list
    samples: list


class Policy:
    """How `F`, `cr` and the mutation strategy are chosen each generation.

    Separating this from `variant` is deliberate. `variant` is the textbook
    DE taxonomy -- *which* mutation and *which* crossover -- and it is a fixed
    property of a run. JADE, SHADE, SaDE and CoDE do not add mutations; they
    decide, generation by generation, which mutation and which parameters to
    use and what to learn from the outcome. Two ideas, two arguments.

    The generic loop calls three hooks and knows nothing else:

    * `begin` -- per-individual parameters for one round of trials;
    * `pool`  -- the indices difference vectors may be drawn from, which the
      archive-based policies widen;
    * `observe` -- which trials survived selection, with what parameters, so
      the policy can adapt.

    `n_trials` is how many complete populations of offspring are produced and
    evaluated per generation. It is 1 for everything except CoDE, which
    builds three and keeps the best of each triple. Crucially the loop
    evaluates them as `n_trials` separate batches of `n_pop`, never one wide
    batch, so a caller bound to exactly `n_pop` live evaluation slots keeps
    working.
    """

    n_trials = 1

    def begin(self, pop, scores, rng, trial):
        """Parameters for every individual in this round of trials."""
        raise NotImplementedError

    def pool(self, j, n_pop, rng):
        """Indices individual `j` may draw difference vectors from."""
        return np.delete(np.arange(n_pop), j)

    def observe(self, improved, proposal, delta):
        """Report the outcome. `improved` is the survivor mask, `delta` the
        score improvement (positive = better). Stateless policies ignore it."""


class FixedPolicy(Policy):
    """Constant `F` and `cr`, one strategy: classical DE.

    The default, and the compatibility anchor -- it must reproduce the
    pre-policy implementation byte for byte, including how much randomness is
    consumed, because a shift there silently moves every seeded run in every
    exercise built on this library.

    It therefore draws no random numbers of its own.
    """

    def __init__(self, F, cr, mutation_op, samples_needed):
        self.F = F
        self.cr = cr
        self.mutation_op = mutation_op
        self.samples_needed = samples_needed

    def begin(self, pop, scores, rng, trial):
        n = len(pop)
        return Proposal(
            F=np.full(n, self.F),
            cr=np.full(n, self.cr),
            ops=[self.mutation_op] * n,
            samples=[self.samples_needed] * n,
        )


# ==============================================================================
# 4. Optimizer Class
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
    * `current-to-pbest/1`: JADE's mutation. Like `current-to-best/1` but each
      individual is pulled toward its own draw from the fittest `p` fraction,
      so the population is not funnelled through a single point. Set `p`.
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
        "current-to-pbest/1": (mutation_current_to_pbest_1, 2),
        "current-to-rand/1": (mutation_current_to_rand_1, 3),
    }

    #: Strategies whose base vector is drawn per individual from the fittest
    #: `p` fraction rather than being the single global best.
    _PBEST_STRATEGIES = frozenset({"current-to-pbest/1"})

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
        p: float = 0.1,
        **kwargs,
    ):
        r"""
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
            p (float): Fraction of the population that counts as "best" for
                `current-to-pbest/1`, in $(0, 1]$. Ignored by every other
                variant. Defaults to 0.1, i.e. the fittest tenth.

                It interpolates between two variants already here: the pool is
                floored at one individual, so $p \le 1/N$ is exactly
                `current-to-best/1`, and $p = 1$ draws from the whole
                population, which is `current-to-rand/1`'s pull. Values around
                0.05-0.2 are the usual range.
        """
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")

        self.F = F
        self.cr = cr
        self.p = p
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
            self.uses_pbest = strategy_key in self._PBEST_STRATEGIES

        except (KeyError, IndexError, ValueError) as e:
            valid_strats = list(self._STRATEGIES.keys())
            valid_cross = list(self._CROSSOVERS.keys())
            raise ValueError(
                f"Invalid variant '{variant}'.\n"
                f"Supported Strategies: {valid_strats}\n"
                f"Supported Crossovers: {valid_cross}\n"
                f"Error: {e}"
            )

        self.policy = FixedPolicy(F, cr, self.mutation_op, self.samples_needed)
        #: Filled by `_generate_offspring`, consumed by `_selection`.
        self._proposal = None

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

        # The policy decides F, cr and the strategy for every individual
        # before any of them is built, so an adaptive policy can look at the
        # whole population once rather than per individual. `FixedPolicy`
        # returns constants and draws no randomness, which is what keeps this
        # path byte-identical to the pre-policy implementation.
        proposal = self.policy.begin(self.pop, self.scores, self.rng, 0)
        self._proposal = proposal

        for j in range(n_pop):
            # 1. Identify valid pool (cannot include self)
            available_indices = self.policy.pool(j, n_pop, self.rng)
            samples_needed = proposal.samples[j]

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
                needed_others = samples_needed - 1
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
                if samples_needed > len(available_indices):
                    choices = self.rng.choice(
                        available_indices, size=samples_needed, replace=True
                    )
                else:
                    choices = self.rng.choice(
                        available_indices, size=samples_needed, replace=False
                    )

            candidates = self.pop[choices]

            # 3. Mutation
            # Note: For 'best/...' strategies, 'best' is used as base, and 'candidates' are just diffs.
            # Tournament selection above mainly benefits 'rand/...' strategies where candidates[0] is base.
            mutant = proposal.ops[j](
                self.pop[j], self._base_vector(j), candidates, proposal.F[j]
            )

            # 4. Crossover
            trial = self.crossover_op(
                self.pop[j], mutant, proposal.cr[j], self.rng
            )
            offspring[j] = trial

        return offspring

    def _base_vector(self, j: int) -> np.ndarray:
        r"""The vector the mutation pulls toward, for individual `j`.

        Ordinary strategies pull everyone toward the same global best. The
        p-best ones draw a fresh $x_{pbest}$ per individual from the fittest
        $\lfloor p N \rfloor$, which is what stops the population being
        funnelled through a single point.

        Two details that are easy to get wrong and change the algorithm:

        * the pool is floored at **one** individual, so small populations
          degrade to `current-to-best/1` instead of raising on an empty
          selection;
        * the draw is per individual **per generation**. Sharing one draw
          across the population reintroduces exactly the single attractor
          the variant exists to avoid.

        Args:
            j (int): Index of the individual being mutated.

        Returns:
            np.ndarray: The base vector, shape (D,).
        """
        if not self.uses_pbest:
            return self.best_pos

        n_best = max(1, int(self.p * self.n_pop))
        top = np.argpartition(self.scores, n_best - 1)[:n_best]
        return self.pop[self.rng.choice(top)]

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Greedy Survivor Selection.

        The child replaces the parent if and only if it is better or equal.

        The survivor mask and the size of each improvement are reported to
        the policy before they are discarded. Every adaptive variant learns
        from exactly this: JADE and SHADE from the `F` and `cr` that produced
        surviving trials, SaDE from which strategy did. Computing it and
        throwing it away, as this did, is why none of them could be built.
        """
        improved_mask = offspring_scores <= self.scores
        delta = self.scores - offspring_scores

        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]

        if self._proposal is not None:
            self.policy.observe(improved_mask, self._proposal, delta)


def differential_evolution(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    variant: str = "best/1/bin",
    parent_selection: str = "rand",
    F: float = 0.5,
    cr: float = 0.7,
    p: float = 0.1,
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
        p=p,
        F=F,
        cr=cr,
        **kwargs,
    )
    return optimizer.optimize()
