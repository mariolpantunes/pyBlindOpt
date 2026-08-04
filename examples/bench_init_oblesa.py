"""Does OBLESA initialization actually make the optimizer find better optima?

Everything measured in `ess` and `torann` so far is *intrinsic* to the
sampler or the index — Clark-Evans, toroidal separation, recall, ms/epoch.
None of it shows that a population initialized this way leads an optimizer
anywhere better. For an evolutionary-computation audience that is the only
claim that counts, so this is the experiment the rest of the programme is
subordinate to.

It is also the experiment the field disagrees about. Kazimipour, Li & Qin's
survey (CEC 2014) records that studies dispute whether the benefit of a
uniform initial population *grows* with dimension or **evaporates above
roughly 12 dimensions**, that nearly all studies stay under 60 dimensions,
and that most compare fewer than four techniques. So the dimension sweep is
the point, not a robustness check.

**Initialization is preprocessing, and is not charged.** The arms do not cost
the same: random/LHS/Sobol spend nothing to initialize, OBL spends `2*n_pop`
and OBLESA `3*n_pop`. An earlier version of this script subtracted that from
the iteration budget, which was wrong in a way that produced the exact
artefact it was meant to prevent. `AR` is defined over *iterations*
(GECCO Companion '26 Eq. 3), and §4.3 of that paper states outright that
weighing initialization overhead against the iterations saved is a separate,
future analysis. Charging it here meant `oblesa-quasi`, whose population is
bit-identical to `qobl`'s, ran 196 generations against `qobl`'s 198 — so it
could only ever come out equal-or-worse, and duly did, by the 0.002 that two
generations of DE are worth. Every arm now gets the **same iteration count**;
`init_evals` is still measured and reported, for whoever does that end-to-end
analysis. (Caching the objective would shrink it further, which is the other
reason it is not a fair charge.)

**What the cost controls are for.** OBLESA's pool is a strict superset of
OBL's, so its *initial population* cannot be worse — that is arithmetic, and
`test_init.py` pins it. What that does not settle is whether the empty-space
stage contributes anything, because a bigger pool alone gives greedy selection
more to choose from. `random4x`, `obl2x` and above all `oblesa-rand` — OBLESA's
exact pipeline with the empty-space engine swapped for uniform noise — spend
the same candidates without searching for empty space, so the margin over them
is the margin attributable to the search.

**Two metrics decide it: acceleration rate and final fitness.** Nothing else.
Counting how often an empty-space candidate survives into the population is
not one of them and is not reported: those points are *probes*. Evaluating one
in an unpromising region is a cheap discard that was never meant to be kept,
and the case where the region turns out to be good already shows up in the two
metrics above. A survival count would penalise the mechanism for working as
intended.

This is the deliberately dirty first pass: one optimizer, few seeds, stock
defaults, the current L1 torann. It answers "does the effect exist at all"
before anyone designs the full factorial.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import pyBlindOpt.de as de
import pyBlindOpt.functions as functions
import pyBlindOpt.init as init
import pyBlindOpt.utils as utils

# Eight functions in two halves, because the halves test different things.
#
# The first four are EVEN -- f(x) == f(-x) to machine precision -- and three of
# them are periodic on top of that. On a symmetric box the opposite of x is
# exactly -x and carries *identical* fitness, so an OBL pool is half mirror
# pairs: measured on rastrigin at d=10, 30 of 30 selected points had their
# mirror also present and only 15 distinct |x| values survived. Shifting the
# optimum off centre fixes the *location* of the basin but not the symmetry of
# the landscape, and periodicity leaves a lattice of near-equivalent basins
# behind. So on their own these four cannot support a conclusion about
# opposition, however far the optimum is moved.
#
# The second four break that. Rosenbrock is a twisted valley; Styblinski-Tang
# is multimodal with an odd term; Levy is multimodal with no symmetry of any
# kind; Dixon-Price is a non-separable valley whose leading term is asymmetric.
# None of them satisfies f(x) == f(-x), so a reflected point is a genuinely
# different candidate rather than a duplicate with the same score.
#
# Each entry carries the location and value of its optimum, which `shifted`
# needs: an offset applied blind would push an optimum that does not sit at
# the origin (Rosenbrock's is at 1, Styblinski-Tang's at -2.9) straight out of
# the box, and the alpha success criterion needs f* normalised to 0.
_ONES = lambda d: np.ones(d)
_ZEROS = lambda d: np.zeros(d)
# Styblinski-Tang's per-coordinate minimum, from 4x^3 - 32x + 5 = 0, solved
# rather than quoted: the literature's -39.16599 is a rounded value and using
# it as f* would leave the "optimum" reachable by a hair, which the alpha
# criterion would score as a miss forever.
_ST_ROOTS = np.roots([4.0, 0.0, -32.0, 5.0])
_ST_ROOTS = _ST_ROOTS[np.isreal(_ST_ROOTS)].real
_ST_X = float(_ST_ROOTS[np.argmin(0.5 * (_ST_ROOTS**4 - 16 * _ST_ROOTS**2
                                         + 5 * _ST_ROOTS))])
_ST_F = float(0.5 * (_ST_X**4 - 16 * _ST_X**2 + 5 * _ST_X))

# name -> (callable, x_opt(d), f_opt(d))
FUNCTIONS = {
    # --- even, and three of them periodic ---
    "sphere":     (functions.sphere,     _ZEROS, lambda d: 0.0),
    "rastrigin":  (functions.rastrigin,  _ZEROS, lambda d: 0.0),
    "ackley":     (functions.ackley,     _ZEROS, lambda d: 0.0),
    "griewank":   (functions.griewank,   _ZEROS, lambda d: 0.0),
    # --- asymmetric ---
    "rosenbrock": (functions.rosenbrock, _ONES,  lambda d: 0.0),
    "styblinski": (functions.styblinski_tang,
                   lambda d: np.full(d, _ST_X), lambda d: _ST_F * d),
    "levy":       (functions.levy,       _ONES,  lambda d: 0.0),
    "dixon":      (functions.dixon_price,
                   lambda d: 2.0 ** (-(2.0 ** np.arange(1, d + 1) - 2.0)
                                     / 2.0 ** np.arange(1, d + 1)),
                   lambda d: 0.0),
    # Even, but neither separable nor permutation-invariant, and increasingly
    # ill-conditioned. Not in the default eight; name it on --functions.
    "zakharov":   (functions.zakharov,   _ZEROS, lambda d: 0.0),
}
DEFAULT_FUNCTIONS = ("sphere", "rastrigin", "ackley", "griewank",
                     "rosenbrock", "styblinski", "levy", "dixon")


def shifted(name, d, frac=0.8, bounds_half=5.0):
    """`FUNCTIONS[name]` with its optimum moved, and its value normalised to 0.

    The optimum is *placed*, not displaced. Drawing an offset and adding it
    moves an optimum that already sits away from the origin by that much
    again -- Styblinski-Tang's is at -2.9, so a -4 offset would put it at -6.9,
    outside the box, and the run would be measuring how fast an optimizer
    reaches a wall. Here a target is drawn inside the box and the offset is
    whatever carries the true optimum to it, so the optimum is at the target
    by construction whatever the function.

    `frac` is the target range as a fraction of the half-width, so 0.8 puts it
    anywhere in +/-4 of a +/-5 box -- the COCO/bbob convention. The old default
    of 0.25 kept it within 12.5% of the centre, which is not far enough to be a
    fix: quasi-opposition maps every point into the band between the centre and
    its opposite, so a near-central optimum rewards QOBL for its centre bias
    rather than for opposition working.

    Subtracting `f_opt` makes `f* = 0` for every function, which is what the
    `alpha` success criterion assumes.
    """
    fn, x_opt, f_opt = FUNCTIONS[name]
    rng = np.random.default_rng(abs(hash(name)) % (2**32) + d)
    target = rng.uniform(-frac, frac, d) * bounds_half
    off = target - x_opt(d)
    bias = f_opt(d)

    def g(x):
        return fn(np.asarray(x) - off) - bias

    g.__name__ = f"{name}_shifted"
    return g


def unshifted(name, d):
    """`FUNCTIONS[name]` in its textbook position, value still normalised."""
    fn, _, f_opt = FUNCTIONS[name]
    bias = f_opt(d)

    def g(x):
        return fn(np.asarray(x)) - bias

    g.__name__ = name
    return g


# Straddling the region the EC literature disagrees about (~12 d) and the
# region torann measured its own wall in (~8 d).
# 2-40 matches the dimensions of "Active Initialization in Population-Based
# Optimizers" (GECCO Companion '26, Table 3) so the acceleration rates are
# directly comparable to its published figures. 64 and 100 go past them,
# because that is the regime OBLESA is designed for and the published sweep
# shows AR shrinking towards it (38.37% at d=2 down to 4.74% at d=40) -- the
# question of whether it keeps shrinking or flattens is only answerable above
# 40. Nothing caps the range any more: `utils.SobolSampler` used to stop at 40
# and now generates its own direction numbers.
DIMS = (2, 5, 10, 20, 40, 64, 100)

# Four groups, and the middle two are the point.
#
#   baselines      what the field already does
#   cost controls  the same candidate count as OBLESA, but with no
#                  empty-space search -- these say whether ESS finds genuinely
#                  empty regions or merely hands the selector more candidates
#   quota arms     slots reserved for the empty-space block. A knob, not a
#                  fix: forcing probe points into the population is as likely
#                  to carry an unpromising region forward as a good one, so
#                  these have to earn their place on AR and fitness like
#                  everything else
#   oblesa knobs   OBLESA exposes controls the other arms do not have, so
#                  comparing it only at defaults understates it
# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
# The OBLESA arms are *generated* from the knobs of `init.oblesa` rather than
# written out, so the sweep and the signature cannot drift apart: every arm is
# the same function with different keyword arguments, and the factorial is a
# full crossing of the five stage knobs. That also means every arm is a
# control for the arms that differ from it in one knob, which is what makes
# main effects and interactions readable off the results without a separate
# ablation run.
#
#   opp        base opposition: standard or quasi
#   opp_ess    extend it to the probe block (3N pool -> 4N)
#   force      what pulls the probes: uniform null, pure repulsion, or
#              repulsion plus attraction toward low predicted objective at
#              weight `force_weight`
#   selection  how the pool is filtered, with `diversity_weight`
#
# `force='uniform'` is the null: OBLESA's pool shape and candidate count with
# no empty-space search at all. Its presence at every point of the grid is
# what lets a margin be attributed to the search rather than to pool size --
# the comparison OBLESA-versus-OBL on its own cannot make.

BASELINE_ARMS = (
    "random", "sobol", "lhs",       # samplers, N calls
    "obl", "qobl",                  # opposition, 2N
    "random4x", "obl2x",            # equal-space cost controls, 4N
)

BASELINE = "random"   # the arm AR is measured against, GECCO Companion '26 Eq. 3
SHIFT = [True]        # cleared by --no-shift; module-level so one_run sees it
SHIFT_FRAC = [0.8]

#: force level -> (force, force_weight). `repulsive` is `guided` at weight 0
#: and is kept as a separate label only because it names the ablation's origin.
_FORCE_LEVELS = {
    "u": ("uniform", 0.0),
    "r": ("repulsive", 0.0),
    "g01": ("guided", 1.0),
    "g04": ("guided", 4.0),
    "g08": ("guided", 8.0),
    "g16": ("guided", 16.0),
    "g32": ("guided", 32.0),
}

#: selection label -> (selection, diversity_weight). `maximin` at weight 0 is
#: omitted deliberately: `keep = 1 + 3w` then considers exactly `n_pop`
#: candidates and must take all of them, making it identical to `best` at
#: weight 0 -- the same arm under two names, not a second measurement.
_SELECTION_LEVELS = {
    "best00": ("best", 0.00),
    "best25": ("best", 0.25),
    "best50": ("best", 0.50),
    "prob00": ("prob", 0.00),
    "prob50": ("prob", 0.50),
    "mmin25": ("maximin", 0.25),
    "mmin50": ("maximin", 0.50),
}

OBLESA_KNOBS = {}
for _opp, _o in (("standard", "s"), ("quasi", "q")):
    for _oe in (False, True):
        for _flab, (_force, _fw) in _FORCE_LEVELS.items():
            for _slab, (_sel, _dw) in _SELECTION_LEVELS.items():
                _name = f"ob_{_o}{'e' if _oe else '_'}_{_flab}_{_slab}"
                OBLESA_KNOBS[_name] = {
                    "opp": _opp, "opp_ess": _oe,
                    "force": _force, "force_weight": _fw,
                    "selection": _sel, "diversity_weight": _dw,
                }

ARMS = BASELINE_ARMS + tuple(OBLESA_KNOBS)

#: What each arm reports as its engine, for grouping in the report.
ENGINE_LABEL = {
    name: {"uniform": "uniform-random-null",
           "repulsive": "dart-largest-empty-sphere",
           "guided": "dart-fitness-guided"}[kw["force"]]
    for name, kw in OBLESA_KNOBS.items()
}



class _Counted:
    """Objective wrapper that counts evaluations however it is called."""

    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += x.shape[0] if x.ndim == 2 else 1
        return self.fn(x)


class _Trace:
    """Records best-so-far per generation. Never alters the search.

    The endpoint alone is the wrong statistic here. An initializer acts on
    generation zero, and 200 generations of DE wash most of that out, so a
    final score is mostly optimizer variance with the effect under test
    buried inside it. The curve keeps the early budget — where the effect
    should live if it exists at all — separable from the late budget.
    """

    def __init__(self):
        self.best = []

    def __call__(self, epoch, fitness, population):
        b = float(np.nanmin(fitness))
        self.best.append(min(b, self.best[-1]) if self.best else b)
        return False


def centered_discrepancy(unit):
    r"""Centered $L_2$ discrepancy (Hickernell) — uniformity in a **box**.

    The ESS-layer score for this benchmark, and it must not be the toroidal
    one. `ess.utils.wrap_around_discrepancy` identifies opposite faces of
    the cube, which is correct inside `esa` — the relaxation runs on a
    torus — and *false* here: OBLESA hands the optimizer a bounded box, in
    which a point at 0.01 and one at 0.99 are far apart, not adjacent. A
    wrap-around measure would score a design that hugs two opposite walls
    as perfectly uniform.

    $$ CD^2 = \left(\tfrac{13}{12}\right)^{d}
       - \frac{2}{n}\sum_i \prod_k \left(1 + \tfrac{1}{2}|u_{ik} - \tfrac12|
         - \tfrac{1}{2}|u_{ik} - \tfrac12|^2\right)
       + \frac{1}{n^2}\sum_{i,j} \prod_k \left(1 + \tfrac12|u_{ik}-\tfrac12|
         + \tfrac12|u_{jk}-\tfrac12| - \tfrac12|u_{ik}-u_{jk}|\right) $$

    Unlike the wrap-around form it is *not* invariant under shifting the
    design, which is the property that makes it right for a box: distance
    to the boundary matters.

    Args:
        unit: ``(n, d)`` design already scaled into $[0, 1]^d$.

    Returns:
        float: $CD^2$. Lower is more uniform; report against the random
        baseline measured in the same cell, since it grows with `d`.
    """
    u = np.asarray(unit, dtype=float)
    n, d = u.shape
    c = np.abs(u - 0.5)
    term2 = np.prod(1.0 + 0.5 * c - 0.5 * c * c, axis=1).sum()
    total = 0.0
    for i in range(0, n, 128):                       # blocked: (n, n, d)
        blk = u[i:i + 128]
        cb = c[i:i + 128]
        total += float(np.prod(
            1.0 + 0.5 * cb[:, None, :] + 0.5 * c[None, :, :]
            - 0.5 * np.abs(blk[:, None, :] - u[None, :, :]), axis=2).sum())
    return float((13.0 / 12.0) ** d - 2.0 * term2 / n + total / (n * n))


def population_dispersion(pop, bounds):
    """ESS-layer scores for one initial population, in bounded space."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    unit = np.clip((np.asarray(pop, float) - lo) / (hi - lo), 0.0, 1.0)
    n = unit.shape[0]
    # Mean nearest-neighbour Euclidean gap: the "are points spread out"
    # question the empty-space stage is supposed to answer, measured in the
    # optimizer's own metric rather than the relaxation's.
    diff = unit[:, None, :] - unit[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    return {
        "cd2": centered_discrepancy(unit),
        "nn_mean": float(nn.mean()),
        "nn_min": float(nn.min()),
    }


def _best_of(pool, objective, n_pop):
    """Greedy best `n_pop` of an evaluated pool — OBL's own selection rule."""
    scores = utils.compute_objective(pool, objective, 1)
    return pool[np.argpartition(scores, n_pop)[:n_pop]]


def initial_population(arm, objective, bounds, n_pop, rng, stats=None, info=None):
    """One arm's initial population. Evaluations are counted, not charged.

    The cost controls exist because OBLESA's advantage has two candidate
    explanations that the default arms cannot separate: the empty-space stage
    may be locating genuinely under-explored regions, or a larger pool may
    simply give greedy selection more to work with. `random4x` and `obl2x`
    spend comparable candidates without it, and `oblesa-rand` runs OBLESA's
    exact pipeline with the engine swapped for uniform noise -- the only one of
    the three that holds pool *shape* fixed as well as pool size.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    if arm == "random":
        return utils.RandomSampler(rng).sample(n_pop, bounds)
    if arm == "lhs":
        return utils.HLCSampler(rng).sample(n_pop, bounds)
    if arm == "sobol":
        return utils.SobolSampler(rng).sample(n_pop, bounds)
    if arm in ("random3x", "random4x"):
        mult = 3 if arm == "random3x" else 4
        return _best_of(utils.RandomSampler(rng).sample(mult * n_pop, bounds),
                        objective, n_pop)
    if arm in ("obl15x", "obl2x"):
        # 1.5N or 2N random points plus their opposites: 3N or 4N candidates,
        # matching OBLESA's pool size without any empty-space stage.
        half = (3 * n_pop) // 2 if arm == "obl15x" else 2 * n_pop
        base = utils.RandomSampler(rng).sample(half, bounds)
        opp = utils.check_bounds(lo + hi - base, bounds)
        return _best_of(np.vstack([base, opp]), objective, n_pop)
    if arm == "obl":
        return init.opposition_based(objective, bounds, n_pop=n_pop, seed=rng)
    if arm == "qobl":
        return init.quasi_opposition_based(
            objective, bounds, n_pop=n_pop, seed=rng)
    if arm in OBLESA_KNOBS:
        # Every OBLESA arm goes through the same `init.oblesa` call; the arm
        # name only selects keyword arguments. The empty-space engine included
        # -- it is chosen by the `force` knob rather than by a monkeypatched
        # module attribute -- so the opposition step, the pool shape and the
        # selection rule stay bit-for-bit identical across arms that share
        # them, and each knob is genuinely the only thing that varies along
        # its own axis.
        kw = dict(OBLESA_KNOBS[arm])
        if stats is not None and arm in ENGINE_LABEL:
            stats["engine"] = ENGINE_LABEL[arm]
        if info is not None:
            kw["info"] = info
        return init.oblesa(objective, bounds, n_pop=n_pop, seed=rng, **kw)
    raise ValueError(f"unknown arm {arm!r}")


def objective_for(fname, d):
    """The landscape for one cell, shifted or not, always with f* = 0."""
    return (shifted(fname, d, SHIFT_FRAC[0]) if SHIFT[0]
            else unshifted(fname, d))


def one_run(arm, fname, d, seed, n_pop, n_iter):
    """One (arm, function, dimension, seed) cell, at a fixed iteration count.

    **Two independent generators, and the reason matters.** The arms consume
    wildly different amounts of randomness during initialization — ESS draws
    for every particle of every epoch, plain sampling draws once. Feeding the
    optimizer whatever generator state the initializer happened to leave
    behind means each arm runs DE on a different random trajectory, so a
    paired comparison would be measuring two changes at once and attributing
    both to the initializer. `rng_opt` is therefore seeded identically for
    every arm: the initial population becomes the only thing that differs.

    **Every arm gets the same `n_iter`.** Initialization is preprocessing and
    is measured but not charged — see the module docstring for why subtracting
    it manufactured the result it was supposed to guard against.
    """
    bounds = np.array([[-5.0, 5.0]] * d)
    counted = _Counted(objective_for(fname, d))
    rng_init = np.random.default_rng(seed)
    rng_opt = np.random.default_rng(2**31 - seed)

    t0 = time.perf_counter()
    ess_stats, ess_info = {}, {}
    pop = initial_population(
        arm, counted, bounds, n_pop, rng_init, ess_stats, ess_info)
    t_init = time.perf_counter() - t0
    used = counted.n

    init_scores = utils.compute_objective(pop, objective_for(fname, d), 1)

    trace = _Trace()
    _, score = de.differential_evolution(
        counted, bounds, population=pop, n_pop=n_pop, n_iter=n_iter,
        seed=rng_opt, callback=trace)
    return {
        "arm": arm, "function": fname, "d": d, "seed": seed,
        "score": float(score), "init_evals": int(used),
        "total_evals": int(counted.n), "n_iter": int(n_iter),
        "init_seconds": t_init, "curve": trace.best,
        # Generation-zero quality, so what the initializer handed over stays
        # separable from what DE then did with it.
        "pop_best": float(np.min(init_scores)),
        "pop_median": float(np.median(init_scores)),
        # Recorded but not reported. Empty-space candidates are probes: one
        # that is discarded did its job, so a survival count is not evidence
        # either way and must not be read as one. Kept in the JSON only so a
        # later question can be asked of an existing run.
        "ess_share": ess_info.get("ess_share"),
        "pool_size": ess_info.get("pool_size"),
        # ESS layer, scored in bounded space (see `centered_discrepancy`).
        **{f"pop_{k}": v for k, v in
           population_dispersion(pop, bounds).items()},
        # ESS/torann layer: epochs and the cost split, empty for non-ESS arms.
        "ess": {k: v for k, v in ess_stats.items()
                if k in ("epochs_total", "radius", "engine", "query_s",
                         "force_s", "step_s", "update_s", "setup_s")},
    }


def at_gen(row, g):
    """Best-so-far at generation `g` — the equal-iterations comparison."""
    c = row["curve"]
    return c[min(g, len(c) - 1)] if c else row["score"]


def vtr_for_cell(base_rows):
    """The value-to-reach for one (function, dimension) cell.

    A fixed absolute tolerance cannot work here: the same `alpha = 0.1`
    that is unreachable for rastrigin at d=32 is passed by sphere at d=2
    within a couple of generations, so it censors the hard cells and
    saturates the easy ones. Instead the target is *what the baseline
    actually achieves* — the median final best-so-far of the random arm.
    Acceleration then answers a question with a stable meaning in every
    cell: how much sooner does this arm reach the quality random ends at?
    By construction roughly half the baseline runs reach it, so no cell is
    all-censored and none is trivially saturated.
    """
    return float(np.median([r["curve"][-1] if r["curve"] else r["score"]
                            for r in base_rows]))


def to_target(row, vtr):
    """`(iterations, reached)` to first hit `vtr`.

    Acceleration is defined over iterations (GECCO Companion '26 Eq. 3), and a
    run that never reaches the target contributes its full iteration count --
    the censoring convention the paper uses, and the reason the success rate
    has to be reported next to AR rather than instead of it.
    """
    for g, v in enumerate(row["curve"]):
        if v <= vtr:
            return g + 1, True
    return len(row["curve"]), False


def wilcoxon_signed_rank(a, b):
    """Paired Wilcoxon signed-rank, normal approximation with tie correction.

    Hand-rolled: scipy is not a dependency of this repo. Returns the
    two-sided p-value, or 1.0 when every pair is tied.
    """
    diff = np.asarray(a, float) - np.asarray(b, float)
    diff = diff[diff != 0.0]
    n = diff.size
    if n < 1:
        return 1.0
    order = np.argsort(np.abs(diff))
    ranks = np.empty(n, float)
    absd = np.abs(diff)[order]
    i = 0
    while i < n:                       # average ranks within tied groups
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    signs = np.sign(diff)[order]
    w = float(np.sum(ranks[signs > 0]) - np.sum(ranks[signs < 0]))
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 6.0)
    if sd == 0.0:
        return 1.0
    z = abs(w) / sd
    # two-sided normal tail without scipy
    return float(np.exp(-0.717 * z - 0.416 * z * z))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    # Population size is an axis, not a constant. Kazimipour et al. found
    # initialization matters more at small populations; and ESS only reaches
    # torann's LSH path above the brute-force crossover (512 points), so at
    # n_pop=30 the index under test is never exercised at all.
    ap.add_argument("--n-pop", type=int, nargs="+", default=[30])
    ap.add_argument("--iters", type=int, default=200,
                    help="iterations per run, identical for every arm; "
                         "initialization is measured but not charged")
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--functions", nargs="+", default=list(DEFAULT_FUNCTIONS),
                    help=f"any of: {' '.join(FUNCTIONS)}")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="examples/out/bench_init_oblesa.json")
    ap.add_argument("--no-shift", action="store_true",
                    help="leave each optimum at the domain centre. Off by "
                         "default: four of the five functions are even, so "
                         "with a central optimum the opposite of x carries "
                         "identical fitness and no conclusion about "
                         "opposition survives")
    ap.add_argument("--shift-frac", type=float, default=SHIFT_FRAC[0],
                    help="optimum offset as a fraction of the half-width")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="success threshold f_min - f* <= alpha; f* is 0 for "
                         "every function here, shifted or not")
    ap.add_argument("--arm-index", type=int, default=None,
                    help="run exactly one arm, selected by position in --arms, "
                         "and write it to its own file under --out-dir. This is "
                         "the Slurm array mode: one task per arm, no shared "
                         "output file, and resume for free")
    ap.add_argument("--out-dir", default="examples/out/sweep",
                    help="directory for per-arm JSONL, used by --arm-index")
    ap.add_argument("--list-arms", action="store_true",
                    help="print the arm count and exit, so the array size is "
                         "read off the same table the run uses")
    ap.add_argument("--force", action="store_true",
                    help="recompute an arm whose output file is already complete")
    args = ap.parse_args()
    SHIFT[0] = not args.no_shift
    SHIFT_FRAC[0] = args.shift_frac

    if args.list_arms:
        print(len(args.arms))
        return

    if args.arm_index is not None:
        run_one_arm(args)
        return

    rows = []
    for n_pop in args.n_pop:
        for fname in args.functions:
            for d in args.dims:
                for arm in args.arms:
                    t0 = time.perf_counter()
                    for seed in range(args.seeds):
                        row = one_run(arm, fname, d, seed, n_pop, args.iters)
                        row["n_pop"] = n_pop
                        rows.append(row)
                    cell = rows[-args.seeds:]
                    print(f"  n_pop={n_pop:<5} {fname:<11} d={d:<3} {arm:<22} "
                          f"median={np.median([r['score'] for r in cell]):<13.6g} "
                          f"pop_best={np.median([r['pop_best'] for r in cell]):<12.5g} "
                          f"init={rows[-1]['init_evals']:<6} "
                          f"{time.perf_counter() - t0:5.1f}s", flush=True)
                    # Written per cell, not at the end: a run this long
                    # should never lose its per-seed data to a crash, and
                    # the results should be inspectable while it runs.
                    # Written via a temporary file and renamed, because a
                    # plain truncate-and-rewrite of a file this size leaves
                    # a long window in which a reader sees half a document.
                    tmp = args.out + ".tmp"
                    with open(tmp, "w") as fh:
                        json.dump({"config": vars(args), "rows": rows}, fh)
                    os.replace(tmp, args.out)

    report(rows, args)


def run_one_arm(args):
    """One arm over the whole grid, into its own file. The Slurm array unit.

    Splitting the sweep by arm rather than by cell is what makes this scale.
    The shared-file design rewrote the entire results document after every
    cell, so at a few hundred arms the serialization cost would have overtaken
    the optimization it was recording. Here each task owns one file, writes it
    once, and never contends with another task.

    It also gives resume for nothing: a task whose file already holds the
    expected number of rows is finished, so resubmitting the identical array
    recomputes exactly the tasks that did not complete. A dropped connection
    or a pre-empted node stops costing anything but the arms that were
    actually in flight.
    """
    arm = args.arms[args.arm_index]
    expected = (len(args.n_pop) * len(args.functions)
                * len(args.dims) * args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{arm}.jsonl")

    if not args.force and os.path.exists(path):
        with open(path) as fh:
            done = sum(1 for line in fh if line.strip())
        if done >= expected:
            print(f"[skip] {arm}: {done}/{expected} rows already present")
            return
        print(f"[redo] {arm}: {done}/{expected} rows, recomputing")

    t_arm = time.perf_counter()
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for n_pop in args.n_pop:
            for fname in args.functions:
                for d in args.dims:
                    t0 = time.perf_counter()
                    for seed in range(args.seeds):
                        row = one_run(arm, fname, d, seed, n_pop, args.iters)
                        row["n_pop"] = n_pop
                        row["arm_knobs"] = OBLESA_KNOBS.get(arm, {})
                        fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"  {arm:<24} {fname:<12} d={d:<4} n_pop={n_pop:<4} "
                          f"{time.perf_counter() - t0:6.1f}s", flush=True)
    os.replace(tmp, path)
    print(f"[done] {arm}: {expected} rows in "
          f"{(time.perf_counter() - t_arm) / 60.0:.1f} min -> {path}")


def report(rows, args):
    """Two questions, kept separate because they are different questions.

    *Quality* — where is each arm at the same generation, and where did its
    initial population start? Every arm gets the same iteration count, so this
    is a straight comparison with no accounting in it.
    *Speed* — how many iterations to reach the target? That is the acceleration
    rate, `AR = (1 - sum(iters_m) / sum(iters_random)) * 100`, GECCO
    Companion '26 Eq. 3, together with the success rate that has to be read
    next to it.
    Lower objective values are better throughout.
    """
    by = {}
    for r in rows:
        by.setdefault((r["n_pop"], r["function"], r["d"], r["arm"]), []).append(r)
    for k in by:
        by[k].sort(key=lambda r: r["seed"])
    cells = sorted({k[:3] for k in by
                    if all((k[0], k[1], k[2], a) in by for a in args.arms)})
    gen = min(len(r["curve"]) for r in rows) - 1
    width = max(len(a) for a in args.arms) + 2

    def col(text):
        return f"{text:>{width}}"

    for n_pop in args.n_pop:
        sub = [c for c in cells if c[0] == n_pop]
        if not sub:
            continue

        print(f"\n{'=' * 78}\nQUALITY at generation {gen + 1} "
              f"(n_pop={n_pop}) — median, lower better; * = p<0.05 vs random")
        print(f"{'function':<11}{'d':>4}  " + "".join(col(a) for a in args.arms))
        for (_, f, d) in sub:
            base = np.array([at_gen(r, gen) for r in by[(n_pop, f, d, BASELINE)]])
            line = f"{f:<11}{d:>4}  "
            for a in args.arms:
                v = np.array([at_gen(r, gen) for r in by[(n_pop, f, d, a)]])
                p = wilcoxon_signed_rank(v, base)
                mark = "*" if (p < 0.05 and np.median(v) < np.median(base)) else " "
                line += f"{np.median(v):>{width - 1}.4g}{mark}"
            print(line)

        print(f"\nINITIAL POPULATION (n_pop={n_pop}) — median best at "
              f"generation 0, lower better")
        print(f"{'function':<11}{'d':>4}  " + "".join(col(a) for a in args.arms))
        for (_, f, d) in sub:
            line = f"{f:<11}{d:>4}  "
            for a in args.arms:
                line += col(f"{np.median([r['pop_best'] for r in by[(n_pop, f, d, a)]]):.4g}")
            print(line)

        # Two targets, because neither works alone. `alpha` is the paper's
        # criterion (Eq. 2) and is directly comparable to its published
        # tables, but a fixed threshold saturates sphere at d=2 and is
        # unreachable on rastrigin at d=40, so it censors the hard cells and
        # flattens the easy ones. The baseline median is what the random arm
        # actually achieves in that cell, so by construction about half its
        # runs reach it and every cell yields signal -- at the cost of not
        # being comparable across papers.
        for label, target_of in (
            (f"alpha={args.alpha} (paper Eq. 2, f*=0)",
             lambda base_rows: args.alpha),
            ("baseline median (per cell)", vtr_for_cell),
        ):
            agg = {a: {"it": 0, "ok": 0, "n": 0} for a in args.arms}
            print(f"\nSPEED (n_pop={n_pop}) — AR% by iterations vs {BASELINE}, "
                  f"target = {label}")
            print(f"{'function':<11}{'d':>4}  " + "".join(col(a) for a in args.arms))
            for (_, f, d) in sub:
                vtr = target_of(by[(n_pop, f, d, BASELINE)])
                st = {}
                for a in args.arms:
                    pairs = [to_target(r, vtr) for r in by[(n_pop, f, d, a)]]
                    st[a] = (sum(x[0] for x in pairs), sum(x[1] for x in pairs),
                             len(pairs))
                    agg[a]["it"] += st[a][0]
                    agg[a]["ok"] += st[a][1]
                    agg[a]["n"] += st[a][2]
                base_it = st[BASELINE][0]
                line = f"{f:<11}{d:>4}  "
                for a in args.arms:
                    line += col(f"{(1 - st[a][0] / base_it) * 100:.1f}")
                print(line)

            print(f"\nAGGREGATE (n_pop={n_pop}, target = {label})")
            print(f"{'arm':<24}{'AR% iters':>11}{'success':>10}{'init_evals':>12}")
            for a in args.arms:
                init_ev = np.mean([r["init_evals"] for c in sub
                                   for r in by[(n_pop, c[1], c[2], a)]])
                print(f"{a:<24}"
                      f"{(1 - agg[a]['it'] / agg[BASELINE]['it']) * 100:>11.2f}"
                      f"{100 * agg[a]['ok'] / agg[a]['n']:>9.1f}%"
                      f"{init_ev:>12.0f}")

if __name__ == "__main__":
    main()
