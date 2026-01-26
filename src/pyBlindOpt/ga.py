# coding: utf-8

"""
Genetic Algorithm (GA).

A population-based metaheuristic inspired by natural selection.
Evolves a population using operators: Selection, Crossover (Recombination), and Mutation.


**Analogy:**
Survival of the fittest. Individuals compete to reproduce. The best traits are combined to create offspring, and random mutations introduce diversity to prevent stagnation.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import collections.abc

import numpy as np

import pyBlindOpt.utils as utils
from pyBlindOpt.optimizer import Optimizer


# ==============================================================================
# Default Operators
# ==============================================================================
def tournament_selection(
    pop: np.ndarray,
    scores: np.ndarray,
    k: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Tournament Selection.

    Selects the best individual from a random pool of $k$ competitors.

    Args:
        pop (np.ndarray): Population.
        scores (np.ndarray): Fitness scores.
        k (int): Tournament size.

    Returns:
        np.ndarray: The selected winner.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(pop)
    # Select k random indices
    selection_ix = rng.integers(0, n, size=k)

    # Get the scores of these k candidates
    candidate_scores = scores[selection_ix]

    # Find the index (0 to k-1) of the best score
    best_local_idx = np.argmin(candidate_scores)

    # Return the actual individual
    return pop[selection_ix[best_local_idx]]


def random_mutation(
    candidate: np.ndarray,
    r_mut: float,
    bounds: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Random Mutation.

    Completely replaces the individual with a random solution with probability $r_{mut}$.

    Args:
        candidate (np.ndarray): Individual.
        r_mut (float): Mutation probability.
        rng: Random generator.

    Returns:
        np.ndarray: Mutated individual.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < r_mut:
        return utils.get_random_solution(bounds, rng)
    else:
        return candidate


def gaussian_mutation(
    candidate: np.ndarray,
    r_mut: float,
    bounds: np.ndarray,
    scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Gaussian Mutation.

    Adds Gaussian noise to the individual.
    $$ x' = x + \\mathcal{N}(0, \\sigma^2) $$

    Args:
        candidate: The vector to mutate.
        r_mut: Mutation probability (applied per individual or per gene depending on logic).
               Here, we treat it as: "If mutation occurs, apply noise".
        bounds: Search space bounds.
        scale: Standard deviation of the noise (relative to bound width or absolute).
               Here we treat it as a fraction of the bound width.
        rng: Random generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Apply mutation with probability r_mut
    if rng.random() < r_mut:
        # Calculate dynamic scale based on bounds
        # shape: (D,)
        bound_width = bounds[:, 1] - bounds[:, 0]
        sigma = bound_width * scale

        # Generate noise
        noise = rng.normal(0, sigma, size=candidate.shape)

        # Apply and Clamp
        mutated = candidate + noise
        return utils.check_bounds(mutated, bounds)
    else:
        return candidate


def polynomial_mutation(
    candidate: np.ndarray,
    r_mut: float,
    bounds: np.ndarray,
    eta: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Polynomial Mutation (Deb et al., NSGA-II).

    Favor small perturbations but allows occasional large jumps based on 'eta'.
    Uses a polynomial distribution to perturb genes, favoring small changes for fine-tuning.

    Args:
        candidate: Individual to mutate.
        r_mut: Probability of mutation per gene/dimension (usually 1/D).
        bounds: Search space bounds.
        eta: Distribution index. High value (~20) = Local search. Low value (~5) = Random search.
        rng: Random generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Create a copy to avoid mutating parent in place
    mutant = candidate.copy()
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # Iterate over each gene (dimension)
    for i in range(len(candidate)):
        if rng.random() < r_mut:
            y = candidate[i]
            yl, yu = lower[i], upper[i]
            delta_max = yu - yl

            # Generate random number u
            u = rng.random()

            if u <= 0.5:
                delta_q = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))

            # Apply mutation
            mutant[i] = y + delta_q * delta_max

            # Clamp
            mutant[i] = max(yl, min(yu, mutant[i]))

    return mutant


def blend_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    r_cross: float,
    alpha: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """
    Blend Crossover (BLX-alpha).

    Creates offspring in the range $[min - I\\alpha, max + I\\alpha]$ where $I = |p1 - p2|$.

    Args:
        alpha (float): Expansion factor.

    Returns:
        list[np.ndarray]: Two children.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < r_cross:
        diff = p2 - p1
        c1 = p1 + alpha * diff
        c2 = p2 - alpha * diff
        return [c1, c2]
    else:
        return [p1, p2]


def linear_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    r_cross: float,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """
    Linear crossover operator.
    Returns 3 children (as per original definition), but GA loop usually expects 2.
    Generates linear combinations of parents: $0.5(p1+p2)$, $1.5p1 - 0.5p2$, etc.
    The GA class handles variable length returns by appending to the new pool.

    Returns:
            list[np.ndarray]: Three children.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < r_cross:
        c1 = 0.5 * p1 + 0.5 * p2
        c2 = 1.5 * p1 - 0.5 * p2
        c3 = -0.5 * p1 + 1.5 * p2
        return [c1, c2, c3]
    else:
        return [p1, p2]


# ==============================================================================
# Genetic Algorithm Class
# ==============================================================================
class GeneticAlgorithm(Optimizer):
    """
    Genetic Algorithm (GA).

    A population-based metaheuristic that evolves solutions using biologically
    inspired operators.

    This implementation delegates the evolutionary logic to external callables
    (selection, crossover, mutation), allowing full customization.
    """

    @utils.inherit_docs(Optimizer)
    def __init__(
        self,
        objective: collections.abc.Callable,
        bounds: np.ndarray,
        selection: collections.abc.Callable = tournament_selection,
        crossover: collections.abc.Callable = blend_crossover,
        mutation: collections.abc.Callable = random_mutation,
        r_cross: float = 0.9,
        r_mut: float = 0.3,
        **kwargs,
    ):
        """
        Genetic Algorithm Optimizer.

        Delegates evolutionary logic to callable operators for flexibility.

        Args:
            selection (Callable): Selection operator. Defaults to `tournament_selection`.
            crossover (Callable): Crossover operator. Defaults to `blend_crossover`.
            mutation (Callable): Mutation operator. Defaults to `random_mutation`.
            r_cross (float): Crossover probability. Defaults to 0.9.
            r_mut (float): Mutation probability. Defaults to 0.3.
        """
        # Store Operators
        self.selection_op = selection
        self.crossover_op = crossover
        self.mutation_op = mutation

        # Store Parameters
        self.r_cross = r_cross
        self.r_mut = r_mut

        super().__init__(objective=objective, bounds=bounds, **kwargs)

    def _initialize(self):
        """
        Initialization hook.
        """
        pass

    def _update_iter_params(self, epoch: int):
        """
        Parameter update hook.
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

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Executes the GA Loop.

        1.  **Selection:** Creates a mating pool of size $N$.
        2.  **Crossover:** Pairs parents and produces children.
        3.  **Mutation:** Mutates children.

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The next generation.
        """
        # 1. Selection
        # Select n_pop parents
        # Note: We pass self.rng to ensure reproducibility if operators support it
        selected = []
        for _ in range(self.n_pop):
            # Check if operator accepts rng
            try:
                s = self.selection_op(self.pop, self.scores, rng=self.rng)
            except TypeError:
                s = self.selection_op(self.pop, self.scores)
            selected.append(s)

        # 2. Crossover & Mutation
        children = []

        # Work in pairs (0,1), (2,3), etc.
        # Ensure we don't go out of bounds if n_pop is odd
        limit = self.n_pop - (self.n_pop % 2)

        for i in range(0, limit, 2):
            p1, p2 = selected[i], selected[i + 1]

            # Apply Crossover
            try:
                offspring_list = self.crossover_op(p1, p2, self.r_cross, rng=self.rng)
            except TypeError:
                offspring_list = self.crossover_op(p1, p2, self.r_cross)

            # Apply Mutation to each child
            for child in offspring_list:
                # We stop adding if we reached n_pop (e.g., linear crossover produces 3 children)
                if len(children) >= self.n_pop:
                    break

                try:
                    mutant = self.mutation_op(
                        child, self.r_mut, self.bounds, rng=self.rng
                    )
                except TypeError:
                    mutant = self.mutation_op(child, self.r_mut, self.bounds)

                children.append(mutant)

            if len(children) >= self.n_pop:
                break

        # 3. Fill remaining spots (if any)
        # If crossover produced fewer children or n_pop was odd
        while len(children) < self.n_pop:
            # Fallback: copy the last selected parent or random
            children.append(selected[-1].copy())

        return np.array(children)

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Generational Replacement.

        Completely replaces the old population with the new offspring.

        Args:
            offspring (np.ndarray): New population.
            offspring_scores (np.ndarray): New scores.
        """
        self.pop = offspring
        self.scores = offspring_scores


def genetic_algorithm(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    selection: collections.abc.Callable = tournament_selection,
    crossover: collections.abc.Callable = blend_crossover,
    mutation: collections.abc.Callable = random_mutation,
    r_cross: float = 0.9,
    r_mut: float = 0.3,
    **kwargs,
) -> tuple:
    """
    Functional interface for Genetic Algorithm.

    Returns:
        tuple: (best_pos, best_score).
    """
    optimizer = GeneticAlgorithm(
        objective=objective,
        bounds=bounds,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        r_cross=r_cross,
        r_mut=r_mut,
        **kwargs,
    )
    return optimizer.optimize()
