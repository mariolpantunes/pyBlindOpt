
"""
Population initialization strategies.

Includes advanced sampling and initialization techniques beyond simple randomness,
such as Opposition-Based Learning (OBL) and ESA-based strategies to improve
initial convergence.
"""


import collections.abc
import logging

import ess  # type: ignore[reportMissingImports]
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
    pop_opp = utils.check_bounds(lower + (upper - pop), bounds)

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
    center = lower + (upper - lower) / 2.0

    # Standard Opposition: x_opp = a + b - x
    pop_opp = lower + (upper - pop)

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


def _oppose(
    pop: np.ndarray,
    bounds: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Reflect `pop` through the centre of `bounds`.

    `mode` is 'standard' for the exact reflection or 'quasi' for a point drawn
    uniformly between the centre and that reflection. Shared by the base
    population and the empty-space block so the two cannot drift apart.

    Note the frame is whatever `bounds` is handed. Opposing the probes about
    the *domain* centre when they were placed inside an elite sub-box throws
    them straight back out of it; reflecting them within their own search box
    keeps them where the restriction intended.
    """
    lower, upper = bounds[:, 0], bounds[:, 1]
    opp = lower + (upper - pop)
    if mode == "quasi":
        center = lower + (upper - lower) / 2.0
        opp = rng.uniform(np.minimum(center, opp), np.maximum(center, opp))
    return utils.check_bounds(opp, bounds)


#: The empty-space backend behind the `force` knob of :func:`oblesa`.
def _ess_engine(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    n: int,
    seed=None,
    scores: np.ndarray | None = None,
    attraction_weight: float = 0.5,
    placement_weight: float | None = None,
    att_model: str = "auto",
    k_cand: int = 64,
    **ignored,
) -> np.ndarray:
    """The EmptySpaceSearch relaxation, behind this module's engine contract.

    `ess.esa` places the points, then relaxes them under a force field guided
    by the fitness already measured on the pool. This adapts it rather than
    changing it, because the `accepts`
    capability protocol is pyBlindOpt's convention and ESS should not have to
    know about it.

    Two mappings the adapter owns:

    * **Polarity.** ESS's contract is *higher is more attractive*; it is not
      told whether the caller minimises, and cannot guess. OBLESA minimises,
      so the scores are negated here. Getting this backwards would pull probes
      toward the *worst* regions and still return a plausible-looking
      population, which makes it the one line in the substitution worth
      staring at.
    * **`force_weight` arrives as `attraction_weight`, in ESS's units.** It
      scales a pairwise force bounded by a collapse condition, so it is refused
      at or above 2.5. `k_cand` maps to `init_pool`; both are candidates per
      placement.

    The attraction law is `cauchy` rather than ESS's default of "same as the
    repulsion", because two identical laws are proportional and can never
    cross -- attraction would only weaken the push instead of pulling. See
    `ess.esa`.

    Args:
        samples (np.ndarray): Points already occupying the space, shape (M, D).
        bounds (np.ndarray): Search space bounds, shape (D, 2).
        n (int): How many points to place.
        seed: Random seed or Generator.
        scores (np.ndarray | None): Objective values for `samples`, lower
            better. Without them this is pure repulsion.
        attraction_weight (float): Pull strength in ESS's units. The default
            is ESS's own measured optimum; it is **not** `force_weight`, see
            above.
        att_model (str): How ESS estimates the attractiveness of a position it
            has no measurement for -- 'fourier' fits one function of position
            and evaluates it, 'idw' weights the nearest measured points,
            'detrended' does both. The default 'auto' cross-validates them on
            the pool that was already evaluated, so it costs no objective
            calls; which one wins depends on whether the objective is
            separable, and that is not readable from the dimension.
        placement_weight (float | None): Attraction weight for ESS's placement
            step alone. None pairs it with `attraction_weight`, which is the
            sensible default; they are separable so the guided placement and
            the guided relaxation can be measured apart.
        k_cand (int): Candidates per placement, forwarded as `init_pool`.
        **ignored: Accepted and dropped, for signature compatibility.

    Returns:
        np.ndarray: The `n` placed points, shape (n, D).

    """
    del ignored

    kw = {}
    if scores is not None:
        kw = {
            "attractiveness": -np.asarray(scores, dtype=float),
            "attraction_weight": attraction_weight,
            "placement_weight": placement_weight,
            "att_model": att_model,
            "attraction_metric": "cauchy",
            "attraction_kwargs": {"power": 1.0},
        }
    return ess.esa(samples, bounds, n=n, seed=seed, init_pool=k_cand, **kw)


_ess_engine.accepts = frozenset(  # type: ignore[reportFunctionMemberAccess]
    {"scores", "k_cand", "attraction_weight"})


# One engine: OBLESA's empty-space stage is ESS. Controls an experiment needs
# belong to that experiment and arrive through `engine=`.
_FORCES = {
    "guided": _ess_engine,
    "ess": _ess_engine,
}


def oblesa(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    *,
    population: np.ndarray | utils.Sampler | None = None,
    n_pop: int = 10,
    selection: str = "best",
    opp: str = "quasi",
    opp_ess: bool = False,
    force: str = "guided",
    force_weight: float = 0.5,
    seed: int | np.random.Generator | None = None,
    n_jobs: int = 1,
    n_ess: int | None = None,
    k_cand: int = 64,
    engine: collections.abc.Callable | None = None,
    diversity_weight: float = 0.25,
    info: dict | None = None,
) -> np.ndarray:
    """
    OBLESA (Opposition-Based Learning with Empty Space Search) Initialization.

    Combines OBL with Empty Space Search (`ess.esa`) to ensure
    the population is not only high-quality but also maximally distributed
    (low potential energy configuration).

    The pipeline is four stages, each with its own knob::

        P_0    <- sample                        n_pop points
        P_obl  <- oppose(P_0)         `opp`     n_pop points
        P_ess  <- probe empty space   `force`   n_ess points
        P_eop  <- oppose(P_ess)       `opp_ess` n_ess points
        return select(P_0 u P_obl u P_ess u P_eop)   `selection`

    so the candidate pool is `2 * n_pop + 2 * n_ess` at most: 2N for plain
    OBL (`n_ess=0`), the paper's 3N by default, 4N with `opp_ess=True`.

    **How much of the empty-space block survives is a selection question,
    not a reserved-slot one.** The empty-space candidates are probes: one that
    is discarded found an unpromising region cheaply, which is the mechanism
    working, and one that is kept found something. How aggressively the pool
    is filtered is controlled by `selection` -- `'best'` for greedy, `'prob'`
    for roulette, `'maximin'` for spread over the fittest -- together with
    `diversity_weight`. Those cover the range without forcing any block into
    the population regardless of what it found.

    Args:
        objective (Callable): The objective function to minimize.
        bounds (np.ndarray): Search space boundaries of shape (D, 2).
        population (ndarray | Sampler | None): Initial population or Sampler.
            If None, RandomSampler is used.
        n_pop (int): Number of individuals to select for the final population.
        selection (str): Selection strategy: 'best' (greedy), 'prob' (roulette
            over the blended score -- *not* uniform sampling) or 'maximin'
            (sequential maximin over the fittest candidates). See
            :func:`pyBlindOpt.utils.select_indices`.
        opp (str): Opposition applied to the base sample: 'none', 'standard'
            (exact reflection) or 'quasi' (stochastic, between centre and
            reflection). 'none' makes this random + empty-space with no
            opposition stage at all.
        opp_ess (bool): Apply the same transform to the empty-space block,
            appending the result to the pool. The premise is qOBL's, one level
            down: an empty-region centroid is a guess, and the segment from the
            centre toward its opposite is where quasi-opposition says the
            payoff tends to sit. Ignored when `opp='none'` or `n_ess=0`.
            Default False: worth 1.6 points of acceleration rate on the
            factorial, against 10.6 for `opp` and 8.4 for `selection`, and it
            *compresses* what the guided search itself earns -- the margin over
            the uniform null falls from +6.1 to +3.6 when it is on. It competes
            with the attraction term rather than adding to it.
        force (str): Which force field the probes feel. 'guided' (and its
            explicit spelling 'ess') is the EmptySpaceSearch relaxation: it
            places each probe on a blend of novelty and the attractiveness the
            position is *expected* to have, then relaxes the block under
            repulsion and attraction together.

            That is the only backend, and the knob is kept for the name rather
            than for a choice. Anything else -- a null that spends the same
            candidates without searching, some other placement rule -- is a
            control for an experiment, so it belongs to the harness running it
            and arrives through `engine`.

            Ignored when `engine` is given.
        force_weight (float): Attraction strength, as ESS's
            `attraction_weight`. Bounded by a collapse condition: ESS refuses
            anything at or above 2.5. Zero reduces the placement to pure
            novelty, which makes a sweep over this an ablation rather than a
            comparison of two methods.
        seed (int | Generator | None): Random seed or Generator instance.
        n_jobs (int): Number of parallel jobs for objective evaluation.
        n_ess (int | None): Size of the empty-space block. Defaults to `n_pop`.
            Zero disables the stage, reducing this to OBL under `selection`.
        k_cand (int): Candidates the probe search draws per placed point,
            reaching `ess.esa` as `init_pool`. Accuracy knob of the placement;
            higher is closer to the exact largest empty sphere and linearly
            more expensive. ESS relaxes the block afterwards, so the
            placement only has to be a reasonable starting point: raising this
            to 2048 costs 4.5-7.1x and returns the same population.
        engine (Callable | None): Empty-space backend override, called as
            `engine(samples, bounds, n=..., seed=..., ...)`. When None it is
            chosen by `force`. Pass `ess.esa` for the published implementation;
            engines declare which extra keywords they accept via an `accepts`
            attribute, and anything without one receives only the four
            positional-equivalent arguments.
        diversity_weight (float): Trade-off between fitness (0.0) and spatial
            diversity (1.0) using crowding distance. The default is an interior
            optimum, not a corner: 26.7 at 0.0, 29.4 at 0.25, 24.0 at 0.5. It
            exists only under `selection='best'` -- the other rules already
            spend their own randomness on spread.
        info (dict | None): If given, filled in place with `pool_size`.

    Returns:
        np.ndarray: Optimized population of shape (n_pop, D), ordered by
        ascending score.
    """
    rng = (
        np.random.default_rng(seed)
        if not isinstance(seed, np.random.Generator)
        else seed
    )

    if opp not in ("none", "standard", "quasi"):
        raise ValueError(f"opp must be 'none', 'standard' or 'quasi', got {opp!r}")
    if engine is None and force not in _FORCES:
        raise ValueError(f"force must be one of {sorted(_FORCES)}, got {force!r}")
    probe = _FORCES[force] if engine is None else engine

    ran_pop, n_pop = _parse_population_arg(population, n_pop, bounds, rng)

    if opp == "none":
        combined_samples = ran_pop
    else:
        combined_samples = np.vstack((ran_pop, _oppose(ran_pop, bounds, opp, rng)))
    n_ess = n_pop if n_ess is None else int(n_ess)

    # One call per `n_pop` rows, never a bigger block. Every optimizer here
    # evaluates a generation of exactly `n_pop`, and an objective may be
    # sized for it -- a simulator with a fixed job width, a model with a
    # pinned device batch, a licence metered per call. This stage is the
    # sampler population stacked with its opposition, so it is `2 * n_pop`
    # rows and used to go over in a single call.
    obl_scores = np.concatenate([
        utils.compute_objective(combined_samples[i:i + n_pop], objective, n_jobs)
        for i in range(0, combined_samples.shape[0], n_pop)])

    if n_ess > 0:
        # Only keywords the engine declares are forwarded. `ess.esa` declares
        # nothing and pushes anything it does not recognise into its metric
        # kernel, where it dies; this is what keeps it substitutable.
        accepts = getattr(probe, "accepts", frozenset())
        eng_kw = {}
        if "k_cand" in accepts:
            eng_kw["k_cand"] = k_cand
        # `force_weight` under whichever name the engine calls it. ESS scales a
        # pairwise force; an external engine scoring a candidate cloud calls the
        # same knob `lam`. One `force` level, one attraction strength, whatever
        # the backend spells it.
        if "attraction_weight" in accepts:
            eng_kw["attraction_weight"] = force_weight
        elif "lam" in accepts:
            eng_kw["lam"] = force_weight
        if "scores" in accepts:
            eng_kw["scores"] = obl_scores
        emp_pop = probe(combined_samples, bounds, n=n_ess, seed=rng, **eng_kw)
    else:
        emp_pop = np.empty((0, bounds.shape[0]))

    if opp_ess and opp != "none" and emp_pop.shape[0]:
        emp_pop = np.vstack((emp_pop, _oppose(emp_pop, bounds, opp, rng)))

    population = np.vstack((combined_samples, emp_pop))
    scores = np.concatenate([obl_scores] + [
        utils.compute_objective(emp_pop[i:i + n_pop], objective, n_jobs)
        for i in range(0, emp_pop.shape[0], n_pop)])

    idx = utils.select_indices(
        population=population,
        scores=scores,
        n_pop=n_pop,
        selection=selection,
        diversity_weight=diversity_weight,
        rng=rng,
    )

    if info is not None:
        info["pool_size"] = int(population.shape[0])

    return population[idx]
