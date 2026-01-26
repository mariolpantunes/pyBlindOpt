# coding: utf-8

"""
Population initialization strategies.

Includes advanced sampling and initialization techniques beyond simple randomness,
such as Opposition-Based Learning (OBL) and ESA-based strategies to improve
initial convergence.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import collections.abc
import logging

import ess
import numpy as np

import pyBlindOpt.utils as utils

logger = logging.getLogger(__name__)


def get_initial_population(
    n_pop: int, bounds: np.ndarray, sampler: utils.Sampler
) -> np.ndarray:
    """
    Generates a population matrix using a specified Sampler.

    Args:
        n_pop (int): Number of individuals.
        bounds (np.ndarray): Search space bounds.
        sampler (utils.Sampler): The sampling strategy (Random, LHS, Sobol, etc.).

    Returns:
        np.ndarray: Population matrix $(N, D)$.
    """
    return sampler.sample(n_pop, bounds)


def _parse_population_arg(
    population: np.ndarray | utils.Sampler | None,
    n_pop: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """
    Standardizes the population argument handling.

    Converts user input (None, Array, or Sampler) into a concrete population array.
    """
    if isinstance(population, utils.Sampler):
        pop = get_initial_population(n_pop, bounds, population)
    elif isinstance(population, np.ndarray):
        pop = utils.check_bounds(population, bounds)
        n_pop = pop.shape[0]  # Update n_pop to match provided array
    elif population is None:
        sampler = utils.RandomSampler(rng)
        pop = get_initial_population(n_pop, bounds, sampler)
    else:
        raise ValueError("Population must be None, ndarray, or PopulationSampler.")

    return pop, n_pop


def opposition_based(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    population: np.ndarray | utils.Sampler | None = None,
    n_pop: int = 10,
    n_jobs: int = 1,
    seed: int | np.random.Generator | None = 42,
) -> np.ndarray:
    """
    Opposition-Based Learning (OBL) Initialization.

    Generates a population and its "opposite" in the search space, then selects
    the best $N$ individuals from the combined pool ($2N$).

    **Opposite Point Formula:**
    For a point $x \\in [a, b]$, the opposite $\\breve{x}$ is:
    $$ \\breve{x} = a + b - x $$

    **Analogy:**
    If looking for gold, check your current spot, but also check the exact
    opposite side of the map. Often, if one side is bad, the opposite is promising.

    Returns:
        np.ndarray: The fittest $N$ individuals from the union of random and opposite populations.
    """
    rng = (
        np.random.default_rng(seed)
        if not isinstance(seed, np.random.Generator)
        else seed
    )

    pop, n_pop = _parse_population_arg(population, n_pop, bounds, rng)

    # compute the fitness of the initial population
    scores = utils.compute_objective(pop, objective, n_jobs)

    # compute the opposition population
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    pop_opp = utils.check_bounds(lower + upper - pop, bounds)

    # compute the fitness of the opposition population
    scores_opp = utils.compute_objective(pop_opp, objective, n_jobs)

    # merge the results and filter
    combined_pop = np.vstack((pop, pop_opp))
    combined_scores = np.concatenate((scores, scores_opp))
    top_k_indices = np.argpartition(combined_scores, n_pop)[:n_pop]

    return combined_pop[top_k_indices]


def round_init(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    sampler: utils.Sampler,
    n_pop: int = 10,
    n_rounds: int = 3,
    diversity_weight: float = 0.5,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Tournament-like Initialization.

    Samples a larger pool ($N \\times \\text{rounds}$), evaluates them, and selects
    the final $N$ based on a weighted combination of Fitness and Diversity (Crowding Distance).

    **Selection Probability:**
    $$ P(x) \\propto (1 - w) \\cdot P_{fitness}(x) + w \\cdot P_{diversity}(x) $$

    Args:
        n_rounds (int): Multiplier for pool size.
        diversity_weight (float): Trade-off between quality (0.0) and spread (1.0).

    Returns:
        np.ndarray: Selected population.
    """
    total_candidates = n_pop * n_rounds
    full_pool = sampler.sample(total_candidates, bounds)

    fitness = np.zeros(total_candidates)
    for i in range(0, total_candidates, n_pop):
        batch = full_pool[i : i + n_pop]
        fitness[i : i + n_pop] = utils.compute_objective(batch, objective, n_jobs)

    prob_fitness = utils.score_2_probs(fitness)

    if diversity_weight > 0:
        crowding = utils.compute_crowding_distance(full_pool)
        prob_dist = utils.score_2_probs(-crowding)
    else:
        prob_dist = np.zeros_like(prob_fitness)

    final_probs = (1.0 - diversity_weight) * prob_fitness + diversity_weight * prob_dist
    # Normalize (Floating point math might make sum slightly != 1.0)
    final_probs /= np.sum(final_probs)

    selected_indices = sampler.rng.choice(
        total_candidates, size=n_pop, replace=False, p=final_probs
    )

    return full_pool[selected_indices]


@utils.inherit_docs(ess.esa)
def oblesa(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    *,
    population: np.ndarray | utils.Sampler | None = None,
    n_pop: int = 10,
    n_jobs: int = 1,
    seed: int | np.random.Generator | None = None,
    selection: str = "best",
    diversity_weight: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    OBLESA (Opposition-Based Learning with Empty Space Search) Initialization.

    Combines OBL with Empty Space Search (`ess.esa`) to ensure
    the population is not only high-quality but also maximally distributed
    (low potential energy configuration).

    Args:
        objective (Callable): The objective function to minimize.
        bounds (np.ndarray): Search space boundaries of shape (D, 2).
        population (ndarray | Sampler | None): Initial population or Sampler.
            If None, RandomSampler is used.
        n_pop (int): Number of individuals to select for the final population.
        n_jobs (int): Number of parallel jobs for objective evaluation.
        seed (int | Generator | None): Random seed or Generator instance.
        selection (str): Selection strategy, either 'best' (greedy) or 'random'
            (stochastic selection based on fitness/diversity).
        diversity_weight (float): Trade-off between fitness (0.0) and spatial
            diversity (1.0) using crowding distance.
        **kwargs: Arguments passed directly to `ess.esa` for the repulsion simulation.

    Returns:
        np.ndarray: Optimized population of shape (n_pop, D).
    """
    rng = (
        np.random.default_rng(seed)
        if not isinstance(seed, np.random.Generator)
        else seed
    )

    ran_pop, n_pop = _parse_population_arg(population, n_pop, bounds, rng)

    lower, upper = bounds[:, 0], bounds[:, 1]
    opp_pop = utils.check_bounds(lower + upper - ran_pop, bounds)

    combined_samples = np.vstack((ran_pop, opp_pop))
    emp_pop = ess.esa(combined_samples, bounds, n=2 * n_pop, seed=rng, **kwargs)

    population = np.vstack((ran_pop, opp_pop, emp_pop))
    scores = np.zeros(population.shape[0])

    for i in range(0, population.shape[0], n_pop):
        end = min(i + n_pop, population.shape[0])
        batch = population[i:end]
        scores[i:end] = utils.compute_objective(batch, objective, n_jobs)

    prob_fitness = utils.score_2_probs(scores)

    if diversity_weight > 0:
        crowding = utils.compute_crowding_distance(population)
        prob_dist = utils.score_2_probs(-crowding)
    else:
        prob_dist = np.zeros_like(prob_fitness)

    final_probs = (1.0 - diversity_weight) * prob_fitness + diversity_weight * prob_dist
    final_probs /= np.sum(final_probs)

    if selection == "best":
        idx = np.argpartition(final_probs, n_pop)[-n_pop:]
    else:
        try:
            idx = rng.choice(
                population.shape[0], size=n_pop, replace=False, p=final_probs
            )
        except ValueError:
            idx = np.argpartition(final_probs, n_pop)[-n_pop:]

    return population[idx]


def quasi_opposition_based(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    population: np.ndarray | utils.Sampler | None = None,
    n_pop: int = 10,
    n_jobs: int = 1,
    seed: int | np.random.Generator | None = 42,
) -> np.ndarray:
    """
    Quasi-Opposition Based Learning (QOBL) Initialization.

    An extension of OBL. Instead of checking the exact opposite point, it samples
    a random point between the search space center $C$ and the opposite point $\\breve{x}$.

    **Formula:**
    $$ C = \\frac{a + b}{2}, \\quad \\breve{x} = a + b - x $$
    $$ x_{q} \\sim U(\\min(C, \\breve{x}), \\max(C, \\breve{x})) $$

    Ref: "A comprehensive study of opposition-based learning" (2014).

    Returns:
        np.ndarray: The fittest $N$ individuals from the combined pool.
    """
    rng = (
        np.random.default_rng(seed)
        if not isinstance(seed, np.random.Generator)
        else seed
    )

    # 1. Base Population
    pop, n_pop = _parse_population_arg(population, n_pop, bounds, rng)
    scores = utils.compute_objective(pop, objective, n_jobs)

    # 2. Compute Center and Opposite
    lower, upper = bounds[:, 0], bounds[:, 1]
    center = (lower + upper) / 2.0

    # Standard Opposition: x_opp = a + b - x
    pop_opp = lower + upper - pop

    # 3. Quasi-Opposition Logic
    # We sample uniformly between [Center, Opposite]
    # Note: center and pop_opp are arrays (N, D).
    # We need element-wise min/max to ensure correct random range
    low_bound = np.minimum(center, pop_opp)
    high_bound = np.maximum(center, pop_opp)

    pop_quasi = rng.uniform(low_bound, high_bound)

    # Check bounds (QOBL can sometimes drift slightly, though theoretically safe here)
    pop_quasi = utils.check_bounds(pop_quasi, bounds)

    # 4. Evaluate
    scores_quasi = utils.compute_objective(pop_quasi, objective, n_jobs)

    # 5. Selection (Greedy vs Combined)
    # Standard QOBL usually merges and picks top N
    combined_pop = np.vstack((pop, pop_quasi))
    combined_scores = np.concatenate((scores, scores_quasi))

    top_k_indices = np.argpartition(combined_scores, n_pop)[:n_pop]

    return combined_pop[top_k_indices]
