
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
    if not isinstance(population, np.ndarray) and n_pop < 1:
        # Only a drawn population is sized by `n_pop`; a supplied array
        # overrides it below, so it is not this function's business then.
        raise ValueError(f"n_pop must be >= 1, got {n_pop}")

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
    k_att: int = 8,
    att_power: float = 2.0,
    search_mode: str = "k_nn",
    radius: float = 0.0,
    radius_target: int = 2,
    att_search_mode: str = "k_nn",
    att_radius: float = 0.0,
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
        k_att (int): Neighbours the attractiveness estimate averages over,
            and `att_power` its inverse-distance exponent. Together they are
            the whole of the estimator: `att_model` is pinned to `'idw'`
            below, so these are the only knobs it has.
        search_mode (str): ``'k_nn'`` or ``'radius'`` for the repulsion, and
            `att_search_mode` the same for the attractiveness estimate. They
            are separate because they are separate uses of the index.
        radius (float): Normalized interaction radius in ``(0, 1]`` for
            ``search_mode='radius'``, and `att_radius` the same for the
            attraction. ``0`` means derive it.

            The scale is a fraction of the torus diameter, and that is the
            only scale OBLESA can speak in: it hands ESS a set of points and
            deliberately knows nothing about the geometry they are placed
            under, so an absolute cutoff is not a number it could form. Pass
            `ess.radius_for_target(dim, n_points)` to specify one by
            neighbour count instead -- the useful band is narrow and moves
            with dimension.
        radius_target (int): Neighbours the auto-derived radius should
            contain, when `radius` is 0. The tuning knob of radius mode.
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
            # Pinned, not defaulted. ESS's own default is `'auto'`, which
            # cross-validates and can land on a fit carrying `2d`
            # coefficients -- and OBLESA supplies `2 * n_pop` anchors, a count
            # set by the population rather than by the dimension. Dropping
            # this line does not remove a choice, it silently makes the other
            # one. See `oblesa`'s docstring for the measured difference.
            "att_model": "idw",
            "k_att": k_att,
            "att_power": att_power,
            "attraction_metric": "cauchy",
            "attraction_kwargs": {"power": 1.0},
            # Only meaningful alongside an attractiveness field, so they sit
            # inside the same guard that builds one.
            "att_search_mode": att_search_mode,
            "att_radius": att_radius,
        }
    return ess.esa(samples, bounds, n=n, seed=seed, init_pool=k_cand,
                   search_mode=search_mode, radius=radius,
                   radius_target=radius_target, **kw)


_ess_engine.accepts = frozenset(  # type: ignore[reportFunctionMemberAccess]
    {"scores", "k_cand", "attraction_weight", "k_att", "att_power",
     "search_mode", "radius", "radius_target",
     "att_search_mode", "att_radius"})


def uniform_engine(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    n: int,
    seed=None,
    **ignored,
) -> np.ndarray:
    """OBLESA's pipeline with the empty-space stage replaced by uniform draws.

    A diagnostic, not an alternative. Swapping it in holds the pool's *shape*
    fixed -- same rounds, same opposition, same anchor set, same selection --
    and varies only where the probe block lands, which is the one substitution
    that separates what the pipeline contributes from what the placement does.
    `random4x` and its relatives match the pool's *size* and cannot do this:
    they replace the whole pipeline, so a difference against them could come
    from anywhere in it.

    It declares no capabilities, so the dispatch in `oblesa` hands it nothing
    and the draws stay unguided by fitness. That is deliberate: a null that
    quietly read `scores` would be a cheap surrogate search rather than a null.

    Args:
        samples (np.ndarray): The anchors, ignored -- that is the point.
        bounds (np.ndarray): Search space bounds, shape (D, 2).
        n (int): How many points to draw.
        seed: Random seed or Generator.
        **ignored: Accepted and dropped, for signature compatibility.

    Returns:
        np.ndarray: `n` uniform points, shape (n, D).
    """
    del samples, ignored
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, bounds.shape[0]))


#: Empty, so `oblesa` forwards it nothing at all. See `uniform_engine`.
uniform_engine.accepts = frozenset()  # type: ignore[reportFunctionMemberAccess]


def oblesa_pool_size(
    n_pop: int,
    *,
    n_ess: int | None = None,
    rounds: int = 1,
    opp: str = "quasi",
    opp_ess: bool = False,
) -> int:
    """
    How many points `oblesa` will evaluate, from the knobs alone.

    The pool size is fully determined before anything runs, so it is a
    function of the arguments rather than something a caller has to discover
    by executing a search and reading a value back out. Budget accounting
    needs it up front -- an experiment sizing a run against a fixed evaluation
    budget cannot afford to spend the budget to learn what it cost.

    Args:
        n_pop (int): Population size, and the size of every objective call.
        n_ess (int | None): Empty-space block per round. Defaults to `n_pop`;
            zero disables the stage.
        rounds (int): How many times the empty-space stage runs. The defaults
            here mirror `oblesa`'s, so calling both bare describes one run.
        opp (str): 'none', 'standard' or 'quasi'.
        opp_ess (bool): Whether each probe block is opposed as well.

    Returns:
        int: Total points evaluated, which is also the pool `selection` picks
        `n_pop` rows from.
    """
    block = n_pop if opp == "none" else 2 * n_pop
    n_ess = n_pop if n_ess is None else int(n_ess)
    if n_ess <= 0:
        return block
    per_round = n_ess * (2 if (opp_ess and opp != "none") else 1)
    return block + rounds * per_round


def oblesa(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    *,
    population: np.ndarray | utils.Sampler | None = None,
    n_pop: int = 30,
    selection: str = "best",
    opp: str = "quasi",
    opp_ess: bool = False,
    force_weight: float = 0.5,
    seed: int | np.random.Generator | None = None,
    n_jobs: int = 1,
    n_ess: int | None = None,
    rounds: int = 1,
    k_cand: int = 64,
    k_att: int = 8,
    att_power: float = 2.0,
    search_mode: str = "k_nn",
    radius: float = 0.0,
    radius_target: int = 2,
    att_search_mode: str = "k_nn",
    att_radius: float = 0.0,
    engine: collections.abc.Callable | None = None,
    diversity_weight: float = 0.25,
) -> np.ndarray:
    """
    OBLESA (Opposition-Based Learning with Empty Space Search) Initialization.

    Combines OBL with Empty Space Search (`ess.esa`) to ensure
    the population is not only high-quality but also maximally distributed
    (low potential energy configuration).

    The pipeline is four stages, each with its own knob::

        P_0    <- sample                        n_pop points
        P_obl  <- oppose(P_0)         `opp`     n_pop points
        A      <- P_0 u P_obl                            (the anchors)
        repeat `rounds` times:
            P_ess  <- probe empty space in A   `engine`  n_ess points
            P_eop  <- oppose(P_ess)            `opp_ess` n_ess points
            A      <- A u P_ess u P_eop                  (now measured)
        return select(A)                       `selection`

    so the candidate pool is `2 * n_pop + rounds * 2 * n_ess` at most: 2N for
    plain OBL (`n_ess=0`), the paper's 3N by default, 4N with `opp_ess=True`.

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
        n_pop (int): Number of individuals to select for the final population,
            and the size of every objective call this function makes.
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
            Default False, and that default is now the conservative choice
            rather than the measured one. The figures it was set from (worth
            1.6 points of acceleration rate; *compressing* the guided margin
            from +6.1 to +3.6) were taken on a different empty-space backend
            that no longer exists, and this knob was then held fixed at False
            through every sweep since, so nothing re-checked it against
            `ess.esa` until now.

            Against `ess.esa` it helps, and it helps more as dimension rises.
            On `cs`, acceleration rate at `force_weight=2`, 5 landscapes
            x 8 seeds:

            | d              |   8 |  16 |  32 |  64 | 100 |
            |----------------|-----|-----|-----|-----|-----|
            | `opp_ess=False`| 32.5| 29.5| 31.0| 44.0| 52.0|
            | `opp_ess=True` | 32.5| 29.5| 33.2| 52.0| 62.0|

            which is the opposite sign to the note it replaced: at d>=64 it
            buys more than doubling `n_ess` does, for the same 4N. It still
            does nothing on `de` or `egwo`. Left False because flipping a
            default changes every downstream comparison, not because the
            evidence favours False.
        force_weight (float): Attraction strength, as ESS's
            `attraction_weight`. Bounded by a collapse condition: ESS refuses
            anything at or above 2.5. Zero reduces the placement to pure
            novelty, which makes a sweep over this an ablation rather than a
            comparison of two methods.
        seed (int | Generator | None): Random seed or Generator instance.
        n_jobs (int): Number of parallel jobs for objective evaluation.
        n_ess (int | None): Size of the empty-space block. Defaults to `n_pop`.
            Zero disables the stage, reducing this to OBL under `selection`.

            Must be a whole number of populations -- `n_ess`, or `2 * n_ess`
            when `opp_ess` is set, has to divide by `n_pop`. Every objective
            call this function makes is exactly `n_pop` rows, and a block that
            does not divide evenly leaves a short final call, which breaks
            callers whose objective is sized for a fixed batch. Rejected up
            front rather than left to surface as an odd-shaped call.
        rounds (int): How many times the empty-space stage runs, each round
            probing against everything the previous rounds placed **and
            measured**. `rounds=1` is the single-pass pipeline.

            **The default is 1, which is OBLESA as published.** Additional
            rounds exist so a caller can take better advantage of the
            attraction model, and that is a decision for them: each one costs
            a further `n_ess` objective evaluations, and a default that spends
            three times the budget is not a default, it is a policy imposed on
            everyone who did not read this paragraph.

            Two sweeps do measure 3 as the strongest setting at every
            dimension on every optimizer that responds to initialization at
            all -- it was the largest single effect found -- so a caller with
            budget to spend should raise it. That is a recommendation, and it
            reads better as one than as a number nobody chose.

            This is not the same purchase as a larger `n_ess`, though it costs
            the same evaluations: `n_ess=2*n_pop, rounds=1` places 2N probes
            against 2N anchors in one shot, while `n_ess=n_pop, rounds=2`
            places N, *evaluates them*, and places the second N against 3N
            anchors that now include N points in the regions the first round
            chose.

            That difference is the whole point, and it is a high-dimensional
            one. The attraction field is fitted once per round from the
            anchors (see `force_weight`); with only 2N anchors it is
            extrapolating everywhere the probes actually go, and above roughly
            d=32 there are not enough of them to pin a field down at all.
            A round converts the previous round's guesses into measurements
            sited exactly where the field was least sure, which is the cheapest
            available way to buy back identifiability.

            Rounds keep the `n_pop` evaluation-group contract intact: each one
            adds whole `n_pop`-sized calls, never a wider batch.

            The cost is serial. `rounds` rounds mean `rounds` separate
            relaxations and `rounds` field fits, which cannot overlap, so wall
            clock grows with it even where the evaluation count does not.

        k_cand (int): Candidates the probe search draws per placed point,
            reaching `ess.esa` as `init_pool`. Accuracy knob of the placement;
            higher is closer to the exact largest empty sphere and linearly
            more expensive. ESS relaxes the block afterwards, so the
            placement only has to be a reasonable starting point: raising this
            to 2048 costs 4.5-7.1x and returns the same population.
        k_att (int): Neighbours the attractiveness of an unmeasured position
            is averaged over, and `att_power` the inverse-distance exponent
            weighting them. Forwarded to the engine when it declares them in
            its `accepts`; ignored by engines that do not.
        search_mode (str): How the ESS relaxation finds neighbours --
            ``'k_nn'`` fixes the neighbour *count*, ``'radius'`` fixes the
            *volume*. `att_search_mode` is the same choice for the
            attractiveness estimate, and is separate on purpose: they are two
            different uses of the index and nothing requires a run to make
            the same choice twice.
        radius (float): Normalized interaction radius in ``(0, 1]`` when
            `search_mode` is ``'radius'``, and `att_radius` the same for
            `att_search_mode`. ``0`` derives one.

            Normalized rather than given in the units `bounds` is in,
            because for this metric there is no such number. ESS min-maxes
            each axis onto [0, 1] *independently* and its distance is an L1
            sum over all of them, so the radius is a sum of `dim`
            dimensionless per-axis fractions -- a length in the caller's
            units only if every axis shares a unit and a width, which
            `bounds` need not. It is also the only scale that crosses the
            boundary cleanly: OBLESA holds points ESS produced and not the
            geometry it produced them under.

            Per axis it reads directly: ``radius / 2`` is the mean fraction
            of each axis's own range the ball reaches. On bounds of
            [-5, 5], ``radius=0.2`` reaches 1.0 in the caller's units on a
            typical axis.

            Do not set it by hand in high dimension. The value holding a
            fixed neighbour count converges on 1/2, so at dim=1000 the whole
            range from 1 to 64 neighbours spans 0.474 to 0.491.
            `ess.radius_for_target(dim, n)` turns a neighbour count into a
            value for this argument, which is the usable way in.
        radius_target (int): Neighbours the auto-derived radius should
            contain. This is how radius mode is meant to be tuned, and it is
            the whole of its appeal: `k_nn` needs a `k` chosen per problem,
            where radius mode derives its own from the point density and
            only asks how *many* neighbours are wanted -- a question with a
            sensible answer that does not move with the dimension. It is
            near-parametric rather than parameter-free, but the parameter is
            one a user can actually reason about.

            Forwarded only to engines that declare these in their `accepts`.

            **These are the estimator, now that there is only one.** The model
            was a parameter until a 468-arm factorial settled it, and the
            answer was not close enough to keep the choice open. Fitted models
            carry `2d` coefficients while OBLESA supplies `2 * n_pop` anchors
            -- a count set by the population, not by the dimension -- so
            identifiability needs `n_pop > d`, a population growing linearly
            in dimension where every anchor is a paid objective call.

            Measured as the share of the selected population won by the
            relaxed block at `force_weight=2`, where 33% is parity:

            ============  =====  =====  =====  =====
            estimator     d=8    d=16   d=32   d=64
            ============  =====  =====  =====  =====
            detrended     70.4   41.9   18.0   13.8
            projection    71.4   56.8   41.1   27.9
            inverse dist  74.9   73.1   72.1   71.0
            auto          70.9   72.2   70.0   64.1
            ============  =====  =====  =====  =====

            Inverse-distance weighting is also the only one bounded by
            construction -- a convex combination of measured values cannot
            leave their range -- and the sweep found that is what protects the
            spread repulsion produced: at `d=100` the closest pair in the
            selected population sits at 3.116 under it against 2.797 under a
            detrended fit, a 10.2% collapse that grows with dimension.

        engine (Callable | None): The empty-space backend, called as
            `engine(samples, bounds, n=..., seed=..., ...)`. None selects
            EmptySpaceSearch, which places each probe on a blend of novelty
            and the attractiveness the position is *expected* to have, then
            relaxes the block under repulsion and attraction together.

            This is the only extension point, and the single one deliberately.
            Anything else -- a null that spends the same candidates without
            searching, some other placement rule -- is a control for an
            experiment, so it belongs to the harness running it rather than to
            a mode string here.

            Engines declare which extra keywords they accept via an `accepts`
            attribute; one without it receives only the four
            positional-equivalent arguments.
        diversity_weight (float): Trade-off between fitness (0.0) and spatial
            diversity (1.0) using crowding distance. The default is an interior
            optimum, not a corner: 26.7 at 0.0, 29.4 at 0.25, 24.0 at 0.5. It
            exists only under `selection='best'` -- the other rules already
            spend their own randomness on spread.
    Returns:
        np.ndarray: Optimized population of shape (n_pop, D), ordered by
        ascending score.

    See Also:
        oblesa_pool_size: how many points this will evaluate, from the knobs
        alone, without running anything.
    """
    rng = (
        np.random.default_rng(seed)
        if not isinstance(seed, np.random.Generator)
        else seed
    )

    if opp not in ("none", "standard", "quasi"):
        raise ValueError(f"opp must be 'none', 'standard' or 'quasi', got {opp!r}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError(
            "diversity_weight is a mixing fraction and must lie in [0, 1], "
            f"got {diversity_weight}"
        )
    if force_weight < 0.0:
        raise ValueError(
            "force_weight is an attraction strength and must be >= 0; a "
            f"negative value would pull probes toward the worst regions, got "
            f"{force_weight}"
        )
    if k_cand < 1:
        raise ValueError(f"k_cand must be >= 1, got {k_cand}")
    probe = _ess_engine if engine is None else engine

    ran_pop, n_pop = _parse_population_arg(population, n_pop, bounds, rng)
    if n_pop < 1:
        raise ValueError(f"n_pop must be >= 1, got {n_pop}")

    if opp == "none":
        combined_samples = ran_pop
    else:
        combined_samples = np.vstack((ran_pop, _oppose(ran_pop, bounds, opp, rng)))
    n_ess = n_pop if n_ess is None else int(n_ess)
    if n_ess < 0:
        raise ValueError(f"n_ess must be >= 0, got {n_ess}")

    # The batch invariant, enforced rather than hoped for. Every objective
    # call below is a slice of `n_pop` rows, so a probe block that is not a
    # whole number of populations leaves a short final call -- which breaks
    # callers whose objective is sized for a fixed batch. `opp_ess` doubles
    # the block, so it can mask the problem for some `n_ess` and expose it
    # for others; that is worse than failing, so both are checked together.
    probe_block = n_ess * (2 if (opp_ess and opp != "none") else 1)
    if probe_block % n_pop:
        raise ValueError(
            f"the empty-space block must be a whole number of populations so "
            f"every objective call is exactly n_pop rows: n_ess={n_ess}"
            + (" doubled by opp_ess" if probe_block != n_ess else "")
            + f" gives {probe_block}, which is not a multiple of n_pop={n_pop}"
        )

    # One call per `n_pop` rows, never a bigger block. Every optimizer here
    # evaluates a generation of exactly `n_pop`, and an objective may be
    # sized for it -- a simulator with a fixed job width, a model with a
    # pinned device batch, a licence metered per call. This stage is the
    # sampler population stacked with its opposition, so it is `2 * n_pop`
    # rows and used to go over in a single call.
    obl_scores = np.concatenate([
        utils.compute_objective(combined_samples[i:i + n_pop], objective, n_jobs)
        for i in range(0, combined_samples.shape[0], n_pop)])

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
    # The rest pass through under their own names, so they are a table
    # rather than a line each -- the two above are here as `if`s because
    # they are the ones that get *renamed* on the way.
    for name, value in (("k_att", k_att), ("att_power", att_power),
                        ("search_mode", search_mode), ("radius", radius),
                        ("radius_target", radius_target),
                        ("att_search_mode", att_search_mode),
                        ("att_radius", att_radius)):
        if name in accepts:
            eng_kw[name] = value

    # The pool *is* the anchor set: every round probes against everything
    # placed so far and hands back points that join it. At `rounds=1` this
    # loop runs once and the result is the single-pass pipeline unchanged.
    population = combined_samples
    scores = obl_scores
    for _ in range(rounds if n_ess > 0 else 0):
        if "scores" in accepts:
            # Measured, never inferred. The whole reason a round is worth more
            # than the same budget spent in one larger block is that the
            # previous round's probes enter the next field fit at the same
            # standing as the sampler's own points.
            eng_kw["scores"] = scores
        emp_pop = probe(population, bounds, n=n_ess, seed=rng, **eng_kw)

        if opp_ess and opp != "none" and emp_pop.shape[0]:
            emp_pop = np.vstack((emp_pop, _oppose(emp_pop, bounds, opp, rng)))

        emp_scores = np.concatenate([
            utils.compute_objective(emp_pop[i:i + n_pop], objective, n_jobs)
            for i in range(0, emp_pop.shape[0], n_pop)]) if emp_pop.shape[0] \
            else np.empty(0)

        population = np.vstack((population, emp_pop))
        scores = np.concatenate((scores, emp_scores))

    idx = utils.select_indices(
        population=population,
        scores=scores,
        n_pop=n_pop,
        selection=selection,
        diversity_weight=diversity_weight,
        rng=rng,
    )

    return population[idx]
