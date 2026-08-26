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

**The optimizer is an axis, not a constant.** The 86-arm sweep found the
engine effect 2.6x larger under `best/1/bin` DE than under JADE — 0.066 of
normalised rank against 0.025 — which is the whole difficulty with a
single-optimizer initialization result: a large effect can mean the
initializer is good or only that the optimizer cannot recover from a bad
start, and one optimizer cannot separate those. Five are run here, spanning
that recovery ability from `de` and `ga`, which adapt nothing, through `cs`
and `egwo`, which adapt their step size, to `jade`, which adapts `F` and `CR`
per individual from its own success history. All five see the *same* initial
population, and each is scored only against its own random baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib

import numpy as np

import pyBlindOpt.cs as cs
import pyBlindOpt.de as de
import pyBlindOpt.egwo as egwo
import pyBlindOpt.functions as functions
import pyBlindOpt.ga as ga
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

# Schwefel's per-coordinate optimum, solved for the same reason. The quoted
# 420.9687 is rounded at the seventh digit, which leaves f* about 9e-9 above 0
# at d=32 -- small, but the alpha criterion measures a *relative* deviation
# from f* = 0, so a floor that is not actually 0 is a floor no run can reach.
# Bisection rather than a root-finder because scipy is not a dependency here.
def _schwefel_x():
    """Stationary point of x sin(sqrt(x)), i.e. sin(u) + u cos(u)/2 = 0."""
    df = lambda t: np.sin(np.sqrt(t)) + np.sqrt(t) * np.cos(np.sqrt(t)) / 2.0
    lo, hi = 400.0, 440.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if df(lo) * df(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# /100 because `functions.schwefel` is posed on the module's [-5, 5] box.
_SCHWEFEL_X = _schwefel_x() / 100.0

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
    # ill-conditioned. Not in the default set; name it on --functions.
    "zakharov":   (functions.zakharov,   _ZEROS, lambda d: 0.0),
    # --- weak global structure; see the block below ---
    "schwefel":   (functions.schwefel,
                   lambda d: np.full(d, _SCHWEFEL_X), lambda d: 0.0),
    "lunacek":    (functions.lunacek_bi_rastrigin,
                   lambda d: np.full(d, 1.25), lambda d: 0.0),
}

# ---------------------------------------------------------------------------
# What the set has to span, and what it did not
# ---------------------------------------------------------------------------
# A benchmark set generalises an *initializer* only if it varies the two
# properties that decide whether initialisation can matter at all.
#
#   coupling  fraction of coordinate pairs with a non-zero mixed second
#             difference. Zero for every pair iff f is additively separable.
#   fdc_loc   correlation between a *local* optimum's fitness and its distance
#             to the global one, over 200 descents. Near 1 means one funnel:
#             descend from anywhere and you arrive, so where the population
#             started cannot change where it ends.
#
# fdc_loc is measured on the local optima rather than on uniform samples of
# the box, because almost every landscape here falls toward its optimum *on
# average* -- box-wide FDC reads 0.71-1.00 across the original eight and says
# nothing. It is also budget-sensitive where it matters: Lunacek at D=32 reads
# 0.947 under a 250-evaluation descent and 0.683 under 4000, because a weak
# local search never finds the second funnel and then reports that there is
# only one. The figures below use 1000.
#
#   landscape        cpl(32)  fdc_loc(8)  fdc_loc(32)
#   sphere              0.00       0.990        0.997
#   rastrigin           0.00       0.983        0.978
#   griewank            0.00       0.250        0.238
#   styblinski          0.00       0.818        0.821
#   levy                0.00       0.870        0.834
#   dixon               0.01       0.447        0.895
#   rosenbrock          0.03       0.580        0.687
#   ackley              0.58       0.998        0.988
#   rot_rastrigin       0.99       0.979        0.982
#   rot_styblinski      0.56       0.849        0.763
#   rot_levy            0.96       0.946        0.988
#   rot_dixon           0.69       0.407        0.938
#   schwefel            0.00       0.287        0.293
#   lunacek             0.00       0.327        0.718
#
# **The gap that mattered was coupling.** Seven of the original eight are
# separable at D=32, so each coordinate can be optimised independently and a
# coordinate-marginal design -- LHS, qOBL, plain uniform -- is already close to
# optimal. OBLESA's premise is *joint* coverage, and on a separable landscape
# joint coverage buys nothing marginal coverage does not already have. The set
# was therefore structurally unable to show the thing it was built to measure,
# and no number of seeds would have helped: the missing quantity was variance
# across landscapes, not precision within one. Checked directly on sweep 8565
# (1.58M runs) -- with ackley the only coupled landscape, the
# separable-minus-coupled margin came out +0.022 for `cs` and -0.031 for `de`,
# opposite signs and both inside noise. Unanswerable, rather than answered.
#
# It also got *worse* with dimension -- griewank 0.65 -> 0.00, dixon 0.19 ->
# 0.01, rosenbrock 0.20 -> 0.03 from D=8 to D=32 -- which is exactly the regime
# OBLESA targets. Some part of "the advantage thins out at high d" is the
# benchmark thinning out.
#
# Global structure was the *smaller* gap, and an earlier version of this note
# overstated it by reading box-wide FDC. The original eight already spanned
# 0.25 (griewank) to 1.00, so weak-funnel landscapes were represented. What
# they had none of was a *deceptive* one -- an optimum pinned against the
# boundary, which is where constrained problems put theirs.
#
# `rot_*` supplies the coupling: a per-instance random rotation turns any
# separable function into a densely coupled one (0.00 -> 0.96-1.00 at D=8)
# without touching its modality, conditioning or optimum value. Sphere is
# deliberately not rotated -- it is rotation-invariant, so the unrotated pair
# is a null control. `schwefel` and `lunacek` fill in the weak-funnel corner
# at 0.29 and 0.33.
#
# The default set is a 2x2 over {separable, coupled} x {one funnel, weak
# funnel}, which is what makes a per-class breakdown readable. Trim it with
# --functions if the budget will not carry 14; `test_bench_landscape` fails if
# a trim drops the coupled class below four.

#: Landscapes whose instances come from per-coordinate sign flips rather than
#: a translation, because their optimum's *position* is the property under
#: test. See `shifted`.
SIGN_FLIP_ONLY = ("schwefel",)

#: Landscapes that get a per-instance random rotation. Sphere is excluded on
#: purpose: rotating it changes nothing, which is the point of having it.
ROTATED = ("rastrigin", "styblinski", "levy", "dixon", "lunacek")
for _r in ROTATED:
    FUNCTIONS[f"rot_{_r}"] = FUNCTIONS[_r]

DEFAULT_FUNCTIONS = (
    # separable / one funnel
    "sphere", "rastrigin", "ackley", "griewank",
    "rosenbrock", "styblinski", "levy", "dixon",
    # coupled / one funnel
    "rot_rastrigin", "rot_styblinski", "rot_levy", "rot_dixon",
    # weak global structure
    "schwefel", "lunacek",
)


def shifted(name, d, seed, frac=0.8, bounds_half=5.0):
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

    **The offset is keyed on `seed`, and drawn from `zlib.crc32` rather than
    `hash`.** Both halves of that were bugs, and both were load-bearing.

    `hash(str)` is salted per interpreter process unless `PYTHONHASHSEED` is
    set, and it is set nowhere here. The sweep runs one process per
    `(arm, dimension)` task, so every arm was scoring a *different* landscape:
    measured, three processes gave `f(0) = 30.54 / 71.23 / 56.57` for sphere
    at `d=8`. Arms were therefore never paired, and a comparison between two
    of them carried the whole variance of the offset draw. `crc32` is
    specified, so the same key gives the same landscape in every process, on
    every machine, forever.

    Keying on `seed` is the other half. Without it the offset is one draw per
    `(function, dimension)` and the seeds are repetitions of a single problem
    instance, not independent instances -- so 100 seeds buy precision about
    one landscape rather than evidence about landscapes, and the `frac` guard
    below never averages over anything. With it, `--seeds 100` means 100
    instances, which is what every claim in the report needs it to mean.
    """
    fn, x_opt, f_opt = FUNCTIONS[name]
    rng = np.random.default_rng(np.random.SeedSequence(
        [zlib.crc32(name.encode()), int(d), int(seed)]))
    target = rng.uniform(-frac, frac, d) * bounds_half
    bias = f_opt(d)
    x_star = x_opt(d)

    if name in SIGN_FLIP_ONLY:
        # Translating this one destroys the property it was added for. Its
        # optimum sits at 0.84 of the half-width *by design*, so carrying it
        # to an interior target drags the whole box outward: measured, 96.8%
        # of box points then have at least one coordinate outside the native
        # domain, where the boundary penalty dominates. The penalty is a
        # convex bowl, so the landscape the optimizer actually sees becomes
        # single-funnel -- fitness-distance correlation over local optima rose
        # from 0.05 to 0.43, i.e. the deception was translated away.
        #
        # Per-coordinate sign flips generate instances instead. The term
        # x sin(sqrt|x|) is odd, so a flip is a genuinely different landscape
        # rather than a relabelling, the optimum stays hard against the
        # boundary at +/-4.21, and nothing ever leaves the native domain.
        sign = rng.choice((-1.0, 1.0), size=d)

        def g(x):
            return fn(sign * np.asarray(x)) - bias
    elif not name.startswith("rot_"):
        off = target - x_star

        def g(x):
            return fn(np.asarray(x) - off) - bias
    else:
        # Rotation is taken about the optimum, so the optimum stays exactly at
        # `target` and `f* = 0` still holds -- a rotation about the origin
        # would move it, and the alpha criterion would then be chasing a
        # value the landscape no longer attains.
        #
        # Haar-uniform via QR, with the sign correction: `np.linalg.qr` fixes
        # no sign convention, so taking Q raw gives a distribution biased
        # toward particular reflections. Multiplying by sign(diag(R)) is what
        # makes it uniform over O(d), which matters because a biased rotation
        # is a *fixed* structure that an arm could exploit across instances.
        #
        # Keyed on (name, d, seed) like the shift, so every instance gets its
        # own rotation and no arm can be measured against a single one.
        q_rng = np.random.default_rng(np.random.SeedSequence(
            [zlib.crc32(f"rot::{name}".encode()), int(d), int(seed)]))
        q, r = np.linalg.qr(q_rng.standard_normal((d, d)))
        q = q * np.sign(np.diag(r))

        def g(x):
            # `(x - target) @ q.T`, not `q @ (x - target)`: the objective is
            # handed a whole population of shape (n, d) as often as a single
            # point of shape (d,), and the matmul form only accepts the
            # latter. It does not merely fail on a batch -- it fails *loudly*
            # for n != d and silently returns the wrong thing for n == d,
            # rotating across individuals instead of within one.
            return fn((np.asarray(x) - target) @ q.T + x_star) - bias

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
#   engine     what pulls the probes: the uniform null, or ESS's repulsion
#              plus attraction toward low predicted objective at weight
#              `force_weight`
#   selection  how the pool is filtered, with `diversity_weight`
#
# `_uniform_null` is the null: OBLESA's pool shape and candidate count with no
# empty-space search at all. Its presence at every point of the grid is what
# lets a margin be attributed to the search rather than to pool size -- the
# comparison OBLESA-versus-OBL on its own cannot make.

BASELINE_ARMS = (
    "random", "sobol", "lhs",       # samplers, N calls
    "obl", "qobl",                  # opposition, 2N
    "random4x", "obl2x",            # equal-space cost controls, 4N
)

BASELINE = "random"   # the arm AR is measured against, GECCO Companion '26 Eq. 3


def baseline_for(arm):
    """The random arm this one is compared against — **its own optimizer's**.

    AR (Eq. 3) is a ratio of iteration counts, and iterations are not
    commensurate across optimizers: one GA generation and one JADE generation
    do different amounts of work and reach the target at different rates for
    reasons that have nothing to do with initialization. Dividing every arm by
    a single `random@de` would fold that difference into every non-DE number
    and report it as an initializer effect. Each optimizer is therefore its own
    control, which is also what makes "does a dumber search gain more from a
    better start" a readable comparison rather than an artefact.
    """
    _, opt = split_arm(arm)
    return f"{BASELINE}@{opt}" if "@" in arm else BASELINE
SHIFT = [True]        # cleared by --no-shift; module-level so one_run sees it
SHIFT_FRAC = [0.8]

#: The no-search null: `n` uniform points, same pool shape, nothing sought.
#:
#: This is the control the whole design rests on. OBLESA against OBL confounds
#: two things -- a larger candidate pool, and a pool with points chosen by
#: searching empty space -- and only an arm that spends the same candidates
#: without searching separates them.
#:
#: It used to be four lines here rather than a call out, on the grounds that
#: the null must not move when ESS changes or it stops being a fixed
#: reference. That argument held while the only place to call was ESS. It now
#: lives in `pyBlindOpt.init`, versioned with `oblesa` itself and covered by
#: its tests, so a copy here would drift from the thing it is meant to
#: control rather than be protected from it. Verified identical to the copy
#: it replaces, including that a passed Generator advances rather than being
#: reseeded.
_uniform_null = init.uniform_engine




#: Engine level -> the knobs that select it.
#:
#: **The attraction ladder.** Every rung is ESS; what varies is how hard the
#: probes are pulled toward positions the pool's own fitness says should be
#: good. `a000` is novelty alone, so the ladder is an ablation of attraction
#: rather than a comparison of two methods, and `null` at the bottom is the
#: no-search control that separates "the search found something" from "a
#: bigger pool gave the selector more to choose from".
#:
#: `force_weight` is ESS's `attraction_weight`: it scales a pairwise force
#: bounded by a collapse condition, and ESS refuses it at or above 2.5, so the
#: ladder stops at 2.0.
#:
#: No number from the earlier sweep sets these levels. That sweep ran on a
#: dart stand-in built to answer "is OBLESA worth pursuing" before ESS existed
#: -- a different engine on a different scale, whose lambda has no conversion
#: into this one.
_ENGINE_LEVELS = {
    "null": {"engine": _uniform_null},
    "a000": {"force_weight": 0.00},     # ESS, novelty only
    "a025": {"force_weight": 0.25},
    "a050": {"force_weight": 0.50},     # ESS's own default
    "a100": {"force_weight": 1.00},
    "a200": {"force_weight": 2.00},     # just under refusal
}

#: How ESS estimates attractiveness where it has no measurement. Crossed with
#: the ladder because this is where the dimension question lives.
#:
#: The estimate exists so the pairwise force balance is well posed for points
#: that were never measured -- **not** to name a position better than anything
#: measured. See `ess.attraction`'s module docstring; reasoning about it as an
#: extrapolating surrogate gets every conclusion here backwards, `idw`'s in
#: particular. This comment used to say `idw` "flattens to the pool mean as
#: distances concentrate", which is true of the values and false as a verdict:
#: measured at `force_weight=2`, `idw` holds 74.9 / 73.1 / 72.1 / 71.0 percent
#: of the selected population at d = 8 / 16 / 32 / 64, while `detrended` falls
#: 70.4 -> 13.8 over the same range.
#:
#: **`projection` and `auto` were missing, and their absence is why the last
#: sweep could say nothing about `d >= 32`.** OBLESA hands ESS the sampler and
#: (q)OBL points as measured sources -- `M = 2 * n_pop = 60`, *at every
#: dimension* -- while `fourier` and `detrended` carry `2d + 1` coefficients.
#: From `d = 32` (65 coefficients against 60 points) the fit is
#: underdetermined: it reproduces its own training points and generalises to
#: nothing. Held-out error at `M = 60`, normalised so 1.0 is what predicting
#: the mean scores:
#:
#: | d | 2d+1 | idw | fourier | detrended | projection |
#: |---|---|---|---|---|---|
#: | 8 | 17 | 0.592 | **0.381** | 0.403 | 0.514 |
#: | 16 | 33 | 0.768 | 0.521 | **0.514** | 0.707 |
#: | 32 | 65 | 0.706 | 0.805 | 0.805 | **0.665** |
#: | 64 | 129 | 0.764 | 0.716 | 0.716 | 0.777 |
#: | 100 | 201 | 0.773 | 0.745 | 0.745 | **0.740** |
#:
#: `projection` fits the same basis by correlation instead of by solving, so
#: it stays defined at any source count; `auto` cross-validates the candidates
#: on sources already paid for and keeps the winner -- it is **ESS's own
#: default**, and hard-setting this knob to one of the other three is what
#: disabled it. Both belong on the ladder before any claim about high `d`.
_ATT_MODELS = ("fourier", "idw", "detrended", "projection", "auto")

#: Probe-block size as a multiple of `n_pop`: 1N, the paper's 3N pool. The 2N
#: the earlier sweep preferred was measured on a different engine; `opp_ess`
#: reaches the same 4N as `[N, N_opp, N_ess, N_ess_opp]`.
_N_ESS_MULT = 1.0

#: Selection rule with its diversity weight, crossed against attraction
#: rather than swept for its own sake. The 86-arm sweep found this axis
#: **spent**: 0.5015 / 0.4913 / 0.5072 under `de` and 0.4962 / 0.4995 /
#: 0.5043 under `jade`, a 0.016 spread against attraction's 0.066, and the
#: per-dimension breakdown reverses sign -- `s50` is best at d=16 and d=32
#: under `de` and worst at d=8 and d=100. That is noise, not an optimum.
#:
#: It stays crossed because "spent" is a claim about the *main* effect. If
#: the best attraction weight moves with the diversity weight, the two knobs
#: have to be tuned together and neither can be read alone; the 86-arm sweep
#: could not see that, having only two attraction levels to move between.
_SELECTION_LEVELS = {
    "s00": ("best", 0.00),
    "s25": ("best", 0.25),
    "s50": ("best", 0.50),
}

#: The optimizer itself, as `(entry point, fixed keywords)`.
#:
#: **Every OBLESA measurement before the 86-arm sweep used `best/1/bin`** --
#: the only DE arm that fails outright on multimodal landscapes, and the one
#: most sensitive to where the population starts, which is exactly what an
#: initializer changes. That sweep then showed the choice is not incidental:
#: the engine axis spans 0.066 of normalised rank under `de` and 0.025 under
#: `jade`, so a single optimizer cannot separate "this initializer is good"
#: from "this optimizer cannot recover from a bad start".
#:
#: The five here span that recovery ability rather than sampling one point of
#: it. `de` and `ga` carry no adaptation at all; `cs` and `egwo` adapt their
#: step but not their parameters; `jade` adapts `F` and `CR` per individual
#: from its own success history and is the strongest recoverer in the repo.
#: If initialization matters more the dumber the search, the effect must
#: order along that axis -- and if it does not, that is the finding.
#:
#: Every arm is compared only against the *same optimizer's* random baseline
#: (AR, Eq. 3, is a within-optimizer ratio), so the five needing different
#: numbers of evaluations per iteration costs nothing here.
OPTIMIZERS = {
    "de": (de.differential_evolution,
           {"variant": "best/1/bin", "policy": "fixed"}),
    "jade": (de.differential_evolution,
             {"variant": "current-to-pbest/1/bin", "policy": "jade"}),
    "ga": (ga.genetic_algorithm, {}),
    "cs": (cs.cuckoo_search, {}),
    "egwo": (egwo.enhanced_grey_wolf_optimization, {}),
}

#: Knobs held fixed so the budget goes to what is genuinely unknown.
#:
#: **Both were fixed on evidence from the retired dart engine, and neither
#: has been re-measured on ESS.** `opp` is the one that matters: it was frozen
#: at 31.6 against 21.0 for standard, but a later 12-seed probe under `ga` put
#: quasi at +2.39 against standard's +39.09 at d=32. That reverses the
#: decision on the optimizer least able to recover from a bad start. See
#: TODO.md -- returning `opp` to the swept axis is the first thing the next
#: sweep should do.
_FIXED = {
    "opp": "quasi",
    "opp_ess": False,      # worth 1.6, and compresses the guided margin
}

#: Only crossed where there is attraction to estimate: crossing `null` and
#: `a000` would ship two bit-identical arms under two names.
_MODEL_SUFFIX = {"fourier": "f", "idw": "i", "detrended": "d",
                 "projection": "p", "auto": "x"}

OBLESA_KNOBS = {}
for _elab, _eng in _ENGINE_LEVELS.items():
    _guided = _eng.get("force_weight", 0) > 0
    _variants = ([(_elab + _MODEL_SUFFIX[m], m) for m in _ATT_MODELS]
                 if _guided else [(_elab, None)])
    for _label, _model in _variants:
        for _slab, (_sel, _dw) in _SELECTION_LEVELS.items():
            OBLESA_KNOBS[f"ob_{_label}_{_slab}"] = dict(
                _FIXED, selection=_sel, diversity_weight=_dw,
                _n_ess_mult=_N_ESS_MULT, _att_model=_model, **_eng,
            )

#: **The budget axis, added after sweep 8200.** Two knobs were pinned on
#: dart-era evidence and both are implicated by that sweep's own results, so
#: they are crossed here on the two settings worth carrying rather than across
#: the whole ladder -- 8 arms instead of 292.
#:
#: `_n_ess_mult=2` takes the pool from 3N to 4N. 8200 showed `ga` responding
#: almost entirely to *budget*: `obl2x` and `random4x` scored within a point of
#: each other at 4N (AR 30.5/48.8/52.8/62.5/60.5 against 29.3/49.8/52.2/60.0/
#: 60.5), so opposition contributed nothing there and the 33% extra evaluations
#: contributed everything -- while oblesa was being asked to beat them from 3N.
#: At 4N the comparison is like-for-like. The earlier 86-arm sweep also
#: preferred 2N under both optimizers it ran.
#:
#: `opp_ess=True` opposes the empty-space block as well, which is the other
#: route to 4N -- and it has **never been measured on `ess.esa`**. The figures
#: in `oblesa`'s docstring (worth 1.6 points of AR, and compressing the guided
#: margin from +6.1 to +3.6) are all from `emptyspace.dart_esa`, deleted in
#: 2bc9590. Crossing it against `_n_ess_mult` separates "more candidates" from
#: "opposed candidates", which a single 4N arm cannot.
_FOCUS_BASE = {
    "a050x": {"force_weight": 0.50, "_att_model": "auto"},
    "a200i": {"force_weight": 2.00, "_att_model": "idw"},
}
for _flab, _fkw in _FOCUS_BASE.items():
    for _mult, _mlab in ((1.0, ""), (2.0, "n2")):
        for _oe, _olab in ((False, ""), (True, "oe")):
            if not _mlab and not _olab:
                continue            # already in the ladder above
            OBLESA_KNOBS[f"ob_{_flab}{_mlab}{_olab}_s00"] = dict(
                _FIXED, selection="best", diversity_weight=0.0,
                _n_ess_mult=_mult, **_fkw, opp_ess=_oe,
            )

#: `rounds` spends the same evaluations as `_n_ess_mult` and buys something
#: different with them. `r2` is `n2`'s budget-matched twin -- both 4N -- but
#: places its second N of probes against anchors that now *include* the first
#: N, measured. The two therefore separate exactly the question 8200 could not
#: answer: whether the empty-space stage fails at high d for want of
#: candidates or for want of anchors to fit a field against.
#:
#: Locally, on `cs`, at the same 4N: `r2` beats `n2` at every dimension tried
#: (dlog10 vs qOBL -0.215/-0.054/-0.046 against -0.052/-0.051/-0.024 at
#: d=8/32/64), and the share of the selected population that came from an ESS
#: round holds at 86.7% under `r2` against 73.3% under `n2` at d=64. That is a
#: 24-run local read on 3 landscapes; these arms are what settles it.
for _flab, _fkw in _FOCUS_BASE.items():
    for _r, _rlab in ((2, "r2"), (3, "r3")):
        for _oe, _olab in ((False, ""), (True, "oe")):
            if _r == 3 and _oe:
                continue            # 7N; past the budget the controls cover
            OBLESA_KNOBS[f"ob_{_flab}{_rlab}{_olab}_s00"] = dict(
                _FIXED, selection="best", diversity_weight=0.0,
                _n_ess_mult=1.0, rounds=_r, **_fkw, opp_ess=_oe,
            )

#: The initializers. Every one is run under every optimizer in `OPTIMIZERS`,
#: on the *same* initial population, so the two axes can be read jointly; a
#: result row is named `<initializer>@<optimizer>`. The crossing is implicit
#: rather than an arm list because the population is what costs, and it is
#: built once per initializer -- see `one_cell`.
ARMS_INIT = BASELINE_ARMS + tuple(OBLESA_KNOBS)

#: What each arm reports as its engine, for grouping in the report.
ENGINE_LABEL = {
    name: "uniform-random-null" if kw.get("engine") is _uniform_null
    else "ess-relaxation-w{:g}-{}".format(
        kw.get("force_weight", 0), kw.get("_att_model") or "none")
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
    unit.shape[0]
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
    """Greedy best `n_pop` of an evaluated pool -- OBL's own selection rule.

    Evaluated one `n_pop` group at a time, matching `init.oblesa`. These are
    the cost controls the whole comparison rests on, so they have to be
    comparable in *shape* as well as in total evaluations: they used to hand
    the objective their entire 4N pool in a single call, which is a different
    contract from the one every OBLESA arm honours and would break a caller
    whose objective is sized for a generation.
    """
    scores = np.concatenate([
        utils.compute_objective(pool[i:i + n_pop], objective, 1)
        for i in range(0, pool.shape[0], n_pop)])
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
    arm, _ = split_arm(arm)
    lo, hi = bounds[:, 0], bounds[:, 1]
    if arm == "random":
        return utils.RandomSampler(rng).sample(n_pop, bounds)
    if arm == "lhs":
        return utils.HLCSampler(rng).sample(n_pop, bounds)
    if arm == "sobol":
        return utils.SobolSampler(rng).sample(n_pop, bounds)
    if arm in RANDOM_KX:
        return _best_of(
            utils.RandomSampler(rng).sample(RANDOM_KX[arm] * n_pop, bounds),
            objective, n_pop)
    if arm in OBL_KX:
        # Half the pool drawn at random, half its opposites, so the total
        # matches an OBLESA pool of the same size with no empty-space stage.
        num, den = OBL_KX[arm]
        base = utils.RandomSampler(rng).sample((num * n_pop) // den, bounds)
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
        # -- it arrives as `force` or `engine` rather than as a monkeypatched
        # module attribute -- so the opposition step, the pool shape and the
        # selection rule stay bit-for-bit identical across arms that share
        # them, and each knob is genuinely the only thing that varies along
        # its own axis.
        kw = dict(OBLESA_KNOBS[arm])
        # `_n_ess_mult` is an arm-table convention, not an `oblesa` argument.
        mult = kw.pop("_n_ess_mult", 1.0)
        kw["n_ess"] = max(1, round(mult * n_pop))
        # The attraction model was an `oblesa` argument while the factorial
        # was deciding it; the answer was `idw` and the parameter is gone, so
        # the table's entry is now only a name in the arm string. Refuse any
        # other value rather than dropping it: an arm called `_mdet_` that
        # quietly ran `idw` would land in the same directory as the real
        # `_mdet_` rows and the two would be indistinguishable.
        model = kw.pop("_att_model", "idw")
        if model != "idw":
            raise ValueError(
                f"arm {arm!r} names att_model={model!r}, which oblesa no "
                "longer accepts; those rows are historical, re-run them "
                "against the commit that measured them")
        if stats is not None and arm in ENGINE_LABEL:
            stats["engine"] = ENGINE_LABEL[arm]
        if info is not None:
            info["pool_size"] = init.oblesa_pool_size(
                n_pop,
                n_ess=kw.get("n_ess"),
                rounds=kw.get("rounds", 1),
                opp=kw.get("opp", "quasi"),
                opp_ess=kw.get("opp_ess", False))
        return init.oblesa(objective, bounds, n_pop=n_pop, seed=rng, **kw)
    raise ValueError(f"unknown arm {arm!r}")


def split_arm(arm):
    """`"<initializer>@<optimizer>"` -> the two parts.

    The two axes are crossed rather than merged so an initializer can be read
    across optimizers and vice versa. Bare names default to the classical
    optimizer, which keeps older arm lists working.
    """
    init_arm, _, opt = arm.partition("@")
    return init_arm, (opt or "de")


def objective_for(fname, d, seed):
    """The landscape for one cell, shifted or not, always with f* = 0.

    `seed` selects the instance. Every arm at the same `(fname, d, seed)` gets
    the identical landscape, which is what makes the arm axis paired; a
    different `seed` is a different instance, which is what makes the seed
    axis independent.
    """
    return (shifted(fname, d, seed, SHIFT_FRAC[0]) if SHIFT[0]
            else unshifted(fname, d))


#: **Sweep v8's configuration space.** A flat product over the four knobs
#: that survived v7 as genuinely open questions, named so the arm string can
#: be parsed back into its settings: `f8_w050_midw_r3_s25_oq_e0` is `force_weight=0.5, att_model='idw',
#: rounds=3, diversity_weight=0.25, opp='quasi', opp_ess=False`.
#:
#: The v7 arms above are kept unchanged so the two sweeps stay comparable;
#: this is a separate table rather than an extension of that naming, which had
#: run out of room.
#:
#: `att_model` is deliberately *not* an axis here. It is a decision to be
#: retired, not a factor to cross with everything else: Stage D varies it
#: alone at each cell's tuned operating point, and if one model wins or ties
#: everywhere the parameter is deleted.
#: **The full factorial.** Six knobs crossed, because the per-optimizer
#: pattern is the thing being looked for and a fractional design cannot show
#: an interaction it did not vary. Concretely: whether the best attraction
#: weight moves with the model, whether opposition interacts with the probe
#: block, whether any of it depends on the optimizer.
F8_W = {"w000": 0.0, "w050": 0.5, "w100": 1.0, "w200": 2.0}
#: One entry, and the token stays in the arm name on purpose. The axis is
#: decided -- `idw` won at every dimension and is the only estimator bounded
#: by construction -- but 1650 task files on disk are named with this token,
#: and dropping it would orphan every one of them from `--list-arms` and from
#: the report's arm parser. Cheaper to keep a one-entry table than to rename
#: a finished sweep.
F8_M = {"midw": "idw"}
F8_R = {"r1": 1, "r2": 2, "r3": 3}
F8_S = {"s00": 0.0, "s25": 0.25, "s50": 0.5}
F8_O = {"oq": "quasi", "os": "standard"}
F8_E = {"e0": False, "e1": True}

for _wl, _w in F8_W.items():
    for _ml in F8_M:
        for _rl, _r in F8_R.items():
            for _sl, _s in F8_S.items():
                for _ol, _o in F8_O.items():
                    for _el, _e in F8_E.items():
                        OBLESA_KNOBS[
                            f"f8_{_wl}_{_ml}_{_rl}_{_sl}_{_ol}_{_el}"] = {
                            "selection": "best", "force_weight": _w,
                            "rounds": _r, "diversity_weight": _s,
                            "opp": _o, "opp_ess": _e, "_n_ess_mult": 1.0}

F8_ARMS = [a for a in OBLESA_KNOBS if a.startswith("f8_")]

#: **The estimator knobs, at the two operating points the factorial settled.**
#: With the attraction model pinned to inverse-distance weighting, `k_att` and
#: `att_power` are the whole of it -- and neither has ever been measured,
#: because until now they were unreachable from `oblesa`.
#:
#: Two bases rather than one: every optimizer but `cs` wants `force_weight=1`
#: with standard opposition, and `cs` wants `2.0` with quasi. Sweeping the
#: estimator at a point no optimizer occupies would measure a configuration
#: nobody runs. `rounds=1` throughout, because that is the default and the
#: corner where an under-resourced estimate has the least to work with.
F9_BASE = {
    "b1": {"force_weight": 1.0, "opp": "standard"},
    "b2": {"force_weight": 2.0, "opp": "quasi"},
}
F9_K = {"k02": 2, "k04": 4, "k08": 8, "k16": 16, "k32": 32}
F9_P = {"p1": 1.0, "p2": 2.0, "p3": 3.0}

for _bl, _base in F9_BASE.items():
    for _kl, _k in F9_K.items():
        for _pl, _p in F9_P.items():
            OBLESA_KNOBS[f"f9_{_bl}_{_kl}_{_pl}"] = {
                "selection": "best", "rounds": 1, "diversity_weight": 0.0,
                "opp_ess": False, "k_att": _k, "att_power": _p,
                "_n_ess_mult": 1.0, **_base}

F9_ARMS = [a for a in OBLESA_KNOBS if a.startswith("f9_")]

#: **The null engine at matched pool size.** `random_kx` matches the number of
#: candidates but replaces the entire pipeline, so a difference against it
#: could come from anywhere in it -- the opposition block, the rounds
#: structure, the selection rule. These arms keep all of that and swap only
#: where the probe block lands, which is the one substitution that isolates
#: the placement. Both opposition modes, because the sweep found `opp` is the
#: only knob whose winner moves with the optimizer.
#:
#: `force_weight` is absent on purpose: `uniform_engine` declares no
#: capabilities, so the dispatch hands it nothing and the draws stay unguided
#: by fitness. A null that read `scores` would be a cheap surrogate search.
F10_OPP = {"os": "standard", "oq": "quasi"}

for _rl, _r in F8_R.items():
    for _ol, _o in F10_OPP.items():
        OBLESA_KNOBS[f"f10_null_{_rl}_{_ol}"] = {
            "selection": "best", "rounds": _r, "diversity_weight": 0.0,
            "opp": _o, "opp_ess": False, "_n_ess_mult": 1.0,
            "engine": init.uniform_engine}

F10_ARMS = [a for a in OBLESA_KNOBS if a.startswith("f10_")]

#: `random4x`/`obl2x` are the budget-matched controls the whole claim rests
#: on: is any of this beating "sample more and keep the best"? `qobl` and
#: `obl` separate quasi- from exact opposition without any empty-space stage
#: at all, which is the comparison the GWO-family question turns on.
#: Cost controls, by the pool size they match. `oblesa_pool_size` is
#: `2N + rounds * n_ess`, and `opp_ess` doubles the per-round block, so the
#: 468-arm factorial spends 3N, 4N, 5N, 6N or 8N depending on `rounds` and
#: `opp_ess` -- five distinct budgets, against which only 4N had a control.
#: Comparing a `rounds=1` arm (3N) to `random4x` charged it for a block it
#: never drew; comparing a `rounds=3` arm (5N) to the same control handed it
#: one for free. Neither is a fair test of whether the empty-space stage is
#: doing anything, which is the only thing these arms exist to decide.
RANDOM_KX = {"random3x": 3, "random4x": 4, "random5x": 5,
             "random6x": 6, "random8x": 8}
#: `(numerator, denominator)` of the random half; the opposites double it.
#: `obl15x` keeps its original `(3 * n_pop) // 2` so the name still means what
#: it meant, and `obl2x` is unchanged at exactly 2N + 2N.
OBL_KX = {"obl15x": (3, 2), "obl2x": (2, 1), "obl25x": (5, 2),
          "obl3x": (3, 1), "obl4x": (4, 1)}

V8_BASELINES = ["random", "lhs", "sobol", "obl", "qobl", "obl2x", "random4x"]

#: The controls sweep v8 was missing. Submitted separately once the pool-size
#: mismatch was found, rather than folded into `V8_BASELINES`, so the arms
#: already on disk keep their identity and resume for free.
V8_COST_CONTROLS = ["random3x", "obl15x", "random5x", "obl25x",
                    "random6x", "obl3x", "random8x", "obl4x"]


#: **Population rules, the axis every earlier sweep pinned.** `n_pop=30` was
#: held from d=8 to d=100, so anchors-per-dimension fell by more than an order
#: of magnitude across the range and the decaying returns with dimension were
#: partly a measurement of that rather than of the initializer. Measured
#: offline, anchor count is the single largest lever available: raising it
#: from 60 to 480 buys +0.167 top-decile surrogate precision at d=32 against
#: +0.013 for the best attraction-model change.
#:
#: The four rules span the literature. `log` is CMA-ES's default shape, and it
#: is here as the null -- if it wins, OBLESA's gains were never about
#: population size.
N_POP_RULES = {
    "fixed30": lambda d: 30,
    "log": lambda d: math.ceil(8 + 6 * math.log(d)),
    "root": lambda d: math.ceil(10 * math.sqrt(d)),
    "linear": lambda d: 2 * d,
}


def population_for(args, d):
    """The population sizes to run at dimension `d`.

    A rule replaces the explicit `--n-pop` list rather than multiplying it:
    the point of the rule is that one number per dimension is *derived*, so
    crossing it with a hand-written list would defeat the comparison.
    """
    if getattr(args, "n_pop_rule", None):
        return [N_POP_RULES[args.n_pop_rule](d)]
    return list(args.n_pop)


def iters_for(args, arm, d, n_pop):
    """Iterations from an evaluation budget, not a fixed count.

    Once `n_pop` varies with dimension, "200 iterations" stops being a budget
    -- it is 6k evaluations at n_pop=30 and 40k at n_pop=200, so a rule that
    raises the population would be handed proportionally more objective calls
    and win on that alone. Fix total evaluations instead and derive the
    iteration count, charging initialization against the same budget.

    `BUDGET_PER_DIM * d` follows the BBOB convention of scaling budget with
    dimension. Returns `--iters` unchanged when no budget is set, so every
    existing invocation reproduces bit for bit.
    """
    if not getattr(args, "budget_per_dim", 0):
        return args.iters
    budget = args.budget_per_dim * d
    n_iter = (budget - init_cost(arm, n_pop)) // n_pop
    if n_iter < 1:
        raise ValueError(
            f"budget {budget} at d={d} does not cover initialization for "
            f"{arm!r} at n_pop={n_pop} ({init_cost(arm, n_pop)} evaluations); "
            f"raise --budget-per-dim or lower n_pop")
    return int(n_iter)


def init_cost(arm, n_pop):
    """Evaluations an arm spends before the optimizer starts.

    Known from the knobs, so the budget can be split before anything runs --
    see `init.oblesa_pool_size`. The samplers are one population; the cost
    controls are their stated multiple; OBLESA reads off its own stage knobs.
    """
    base, _ = split_arm(arm)
    if base in ("random", "sobol", "lhs"):
        # Pure samplers: they draw a population and evaluate none of it. The
        # optimizer pays for the first generation either way, so charging
        # them `n_pop` here would double-count it and hand every other arm a
        # free population's worth of budget.
        return 0
    if base in ("obl", "qobl"):
        return 2 * n_pop
    if base in RANDOM_KX:
        return RANDOM_KX[base] * n_pop
    if base in OBL_KX:
        num, den = OBL_KX[base]
        return 2 * ((num * n_pop) // den)
    if base in OBLESA_KNOBS:
        kw = dict(OBLESA_KNOBS[base])
        mult = kw.pop("_n_ess_mult", 1.0)
        return init.oblesa_pool_size(
            n_pop,
            n_ess=max(1, round(mult * n_pop)),
            rounds=kw.get("rounds", 1),
            opp=kw.get("opp", "quasi"),
            opp_ess=kw.get("opp_ess", False))
    raise ValueError(f"unknown arm {arm!r}")


def one_cell(init_arm, fname, d, seed, n_pop, n_iter, optimizers):
    """One initial population, run through each named optimizer.

    **The population is built once and shared, and that is not merely a
    saving.** It is 97-98% of a run's wall clock at d=100 — 8.2 s of an 8.4 s
    run — and it is a deterministic function of the initializer, the landscape
    and the seed, so recomputing it per optimizer would spend five times the
    sweep's budget reproducing the same array. It also makes the optimizer
    axis exactly paired: every optimizer sees the identical `n_pop x d` matrix,
    so a difference between them cannot be initialization noise.

    **Two independent generators, and the reason matters.** The arms consume
    wildly different amounts of randomness during initialization — the probe
    search draws `k_cand` candidates per placement, plain sampling draws once.
    Feeding the optimizer whatever generator state the initializer happened to
    leave behind means each arm runs on a different random trajectory, so a
    paired comparison would be measuring two changes at once and attributing
    both to the initializer. `rng_opt` is therefore reseeded identically for
    every arm *and* every optimizer: the initial population becomes the only
    thing that differs.

    **Every arm gets the same `n_iter`.** Initialization is preprocessing and
    is measured but not charged — see the module docstring for why subtracting
    it manufactured the result it was supposed to guard against.

    Yields:
        dict: one result row per optimizer, in `optimizers` order.
    """
    bounds = np.array([[-5.0, 5.0]] * d)
    objective = objective_for(fname, d, seed)
    counted = _Counted(objective)
    rng_init = np.random.default_rng(seed)

    t0 = time.perf_counter()
    ess_stats, ess_info = {}, {}
    pop = initial_population(
        init_arm, counted, bounds, n_pop, rng_init, ess_stats, ess_info)
    t_init = time.perf_counter() - t0
    init_evals = counted.n

    init_scores = utils.compute_objective(pop, objective, 1)
    shared = {
        "function": fname, "d": d, "seed": seed, "n_pop": n_pop,
        "init_evals": int(init_evals), "n_iter": int(n_iter),
        "init_seconds": t_init,
        # Generation-zero quality, so what the initializer handed over stays
        # separable from what the optimizer then did with it.
        "pop_best": float(np.min(init_scores)),
        "pop_median": float(np.median(init_scores)),
        # Recorded but not reported. Empty-space candidates are probes: one
        # that is discarded did its job, so a survival count is not evidence
        # either way and must not be read as one. Kept in the JSON only so a
        # later question can be asked of an existing run.
        "pool_size": ess_info.get("pool_size"),
        # ESS layer, scored in bounded space (see `centered_discrepancy`).
        **{f"pop_{k}": v for k, v in
           population_dispersion(pop, bounds).items()},
        # ESS/torann layer: epochs and the cost split, empty for non-ESS arms.
        "ess": {k: v for k, v in ess_stats.items()
                if k in ("epochs_total", "radius", "engine", "query_s",
                         "force_s", "step_s", "update_s", "setup_s")},
    }

    for opt_key in optimizers:
        optimize, opt_kw = OPTIMIZERS[opt_key]
        trace = _Trace()
        # A fresh counter per optimizer: `total_evals` is that optimizer's
        # own budget, and a shared one would accumulate across the five.
        counted = _Counted(objective)
        counted.n = init_evals
        # A copy, though `test_optimizer.py` pins that none of them writes
        # through: the sharing below is the point of this function, and it
        # should not depend on a property of thirteen other modules holding.
        _, score = optimize(
            counted, bounds, population=pop.copy(), n_pop=n_pop, n_iter=n_iter,
            seed=np.random.default_rng(2**31 - seed), callback=trace, **opt_kw)
        yield {
            "arm": f"{init_arm}@{opt_key}", "optimizer": opt_key,
            "score": float(score), "total_evals": int(counted.n),
            "curve": trace.best, **shared,
        }


def one_run(arm, fname, d, seed, n_pop, n_iter):
    """One `<initializer>@<optimizer>` cell — the serial path's single row."""
    init_arm, opt_key = split_arm(arm)
    row, = one_cell(init_arm, fname, d, seed, n_pop, n_iter, [opt_key])
    row["arm"] = arm
    return row


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
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed, so tuning and evaluation can run on "
                         "disjoint instances: --seed-start 0 --seeds 50 "
                         "selects a config, --seed-start 50 --seeds 50 "
                         "reports it on seeds it was never chosen on")
    # Population size is an axis, not a constant. Kazimipour et al. found
    # initialization matters more at small populations; and ESS only reaches
    # torann's LSH path above the brute-force crossover (512 points), so at
    # n_pop=30 the index under test is never exercised at all.
    ap.add_argument("--n-pop", type=int, nargs="+", default=[30])
    ap.add_argument("--n-pop-rule", choices=sorted(N_POP_RULES),
                    default=None,
                    help="derive n_pop from the dimension instead of taking "
                         "--n-pop; replaces that list rather than crossing "
                         "with it")
    ap.add_argument("--budget-per-dim", type=int, default=0,
                    help="total objective evaluations per run, as this many "
                         "times the dimension (BBOB convention). Iterations "
                         "are derived from it and initialization is charged "
                         "against it, which is what makes population rules "
                         "comparable. 0 keeps the fixed --iters.")
    ap.add_argument("--iters", type=int, default=200,
                    help="iterations per run, identical for every arm; "
                         "initialization is measured but not charged")
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--functions", nargs="+", default=list(DEFAULT_FUNCTIONS),
                    help=f"any of: {' '.join(FUNCTIONS)}")
    ap.add_argument("--arms", nargs="+", default=list(ARMS_INIT),
                    help="initializers. Each is crossed with every "
                         "--optimizers entry on one shared initial "
                         "population, so results are named init@optimizer")
    ap.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS),
                    help=f"any of: {' '.join(OPTIMIZERS)}")
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
                    help="run exactly one task, selected by position in the "
                         "(initializer, dimension) list --list-arm-names "
                         "prints, and write it to its own file under "
                         "--out-dir. This is the Slurm array mode: no shared "
                         "output file, and resume for free")
    ap.add_argument("--out-dir", default="examples/out/sweep",
                    help="directory for per-task JSONL, used by --arm-index")
    ap.add_argument("--list-arms", action="store_true",
                    help="print the task count and exit, so the array size is "
                         "read off the same table the run uses")
    ap.add_argument("--list-arm-names", action="store_true",
                    help="print one task name per line and exit. Use this, not "
                         "--list-arms, to decide which files in --out-dir "
                         "belong to the current grid: --list-arms prints a "
                         "count, so filtering against it silently matches "
                         "nothing and sweeps every file into the stale pile")
    ap.add_argument("--force", action="store_true",
                    help="recompute a task whose output file is already complete")
    args = ap.parse_args()
    SHIFT[0] = not args.no_shift
    SHIFT_FRAC[0] = args.shift_frac

    if args.list_arms:
        print(len(sweep_tasks(args)))
        return

    if args.list_arm_names:
        print("\n".join(f"{a}_d{d}" for a, d in sweep_tasks(args)))
        return

    if args.arm_index is not None:
        run_one_arm(args)
        return

    rows = []
    for fname in args.functions:
        for d in args.dims:
            for n_pop in population_for(args, d):
                for arm in args.arms:
                    t0 = time.perf_counter()
                    n_iter = iters_for(args, arm, d, n_pop)
                    cell = []
                    for seed in range(args.seed_start,
                                      args.seed_start + args.seeds):
                        cell.extend(one_cell(arm, fname, d, seed, n_pop,
                                             n_iter, args.optimizers))
                    rows.extend(cell)
                    # One median per optimizer, not one across them: the five
                    # reach wildly different values on the same population,
                    # so a pooled median would be a statistic of the optimizer
                    # mix rather than of the initializer under test.
                    med = " ".join(
                        f"{o}={np.median([r['score'] for r in cell if r['optimizer'] == o]):.4g}"
                        for o in args.optimizers)
                    print(f"  n_pop={n_pop:<5} {fname:<11} d={d:<3} {arm:<22} "
                          f"pop_best={np.median([r['pop_best'] for r in cell]):<12.5g} "
                          f"init={cell[-1]['init_evals']:<6} "
                          f"{time.perf_counter() - t0:5.1f}s  {med}", flush=True)
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


def sweep_tasks(args):
    """The array's task list: one entry per `(initializer, dimension)`.

    **Not one per arm.** An arm is `<initializer>@<optimizer>`, and splitting
    on that would recompute the same initial population once per optimizer —
    97-98% of a run's cost at d=100, for an array that runs five optimizers.
    The initializer is the unit that owns the expensive work, so it is the
    unit the array is cut along, and `one_cell` runs every optimizer on the
    population it built.

    Dimension is the second half of the key because the initializer alone
    would leave the array too narrow to fill a node — 31 tasks against 62
    usable threads — and because cost varies ~7x across the dimension range,
    so mixing dimensions into one task makes every task as slow as its worst.
    Crossed, they give 155 balanced-enough tasks.
    """
    return [(a, d) for a in args.arms for d in args.dims]


def run_one_arm(args):
    """One `(initializer, dimension)` over the grid, into its own file.

    Splitting the sweep by task rather than by cell is what makes this scale.
    The shared-file design rewrote the entire results document after every
    cell, so at a few hundred arms the serialization cost would have overtaken
    the optimization it was recording. Here each task owns one file, writes it
    once, and never contends with another task.

    It also gives resume for nothing: a task whose file already holds the
    expected number of rows is finished, so resubmitting the identical array
    recomputes exactly the tasks that did not complete. A dropped connection
    or a pre-empted node stops costing anything but the tasks that were
    actually in flight.
    """
    tasks = sweep_tasks(args)
    init_arm, d = tasks[args.arm_index]
    optimizers = list(args.optimizers)
    expected = (len(population_for(args, d)) * len(args.functions)
                * args.seeds * len(optimizers))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{init_arm}_d{d}.jsonl")
    label = f"{init_arm} d={d}"

    if not args.force and os.path.exists(path):
        with open(path) as fh:
            done = sum(1 for line in fh if line.strip())
        if done >= expected:
            print(f"[skip] {label}: {done}/{expected} rows already present")
            return
        print(f"[redo] {label}: {done}/{expected} rows, recomputing")

    t_arm = time.perf_counter()
    # Recorded per row, so a file says which arm produced it without needing
    # this script's tables to reconstruct it. Callables have to be named
    # rather than embedded: the engine-backed arms hold a function under
    # `engine`, and `json.dumps` refuses it -- which killed **every `null`
    # task** of the first 245-arm run, 15 of 15, seconds in. The null is the
    # no-search control the whole design rests on ("its presence at every
    # point of the grid is what lets a margin be attributed to the search"),
    # so the sweep completed 230/245 having dropped precisely the arm without
    # which nothing else can be attributed.
    knobs = {
        k: (getattr(v, "__name__", repr(v)) if callable(v) else v)
        for k, v in OBLESA_KNOBS.get(init_arm, {}).items()
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for n_pop in population_for(args, d):
            n_iter = iters_for(args, init_arm, d, n_pop)
            for fname in args.functions:
                t0 = time.perf_counter()
                for seed in range(args.seed_start,
                                  args.seed_start + args.seeds):
                    for row in one_cell(init_arm, fname, d, seed, n_pop,
                                        n_iter, optimizers):
                        row["arm_knobs"] = knobs
                        fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"  {label:<24} {fname:<12} n_pop={n_pop:<4} "
                      f"n_iter={n_iter:<5} "
                      f"{time.perf_counter() - t0:6.1f}s", flush=True)
    os.replace(tmp, path)
    print(f"[done] {label}: {expected} rows in "
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
    # `args.arms` names initializers; a *column* is one crossed with one
    # optimizer, which is the unit every table below compares.
    columns = [f"{a}@{o}" for a in args.arms for o in args.optimizers]

    by = {}
    for r in rows:
        by.setdefault((r["n_pop"], r["function"], r["d"], r["arm"]), []).append(r)
    for group in by.values():
        group.sort(key=lambda r: r["seed"])

    # Everything below is measured against a baseline arm, so a column whose
    # baseline was not run has no comparison to report. Saying so beats
    # crashing on a KeyError, and beats silently reporting it against some
    # other optimizer's random.
    have = set(columns)
    orphans = [c for c in columns if baseline_for(c) not in have]
    if orphans:
        print(f"note: no baseline arm for {', '.join(orphans)} — omitted")
        columns = [c for c in columns if baseline_for(c) in have]
    if not columns:
        print("note: nothing to report — no arm has its baseline in this run")
        return

    cells = sorted({k[:3] for k in by
                    if all((k[0], k[1], k[2], c) in by for c in columns)})
    gen = min(len(r["curve"]) for r in rows) - 1
    width = max(len(c) for c in columns) + 2

    def col(text):
        return f"{text:>{width}}"

    for n_pop in args.n_pop:
        sub = [c for c in cells if c[0] == n_pop]
        if not sub:
            continue

        print(f"\n{'=' * 78}\nQUALITY at generation {gen + 1} "
              f"(n_pop={n_pop}) — median, lower better; * = p<0.05 vs random")
        print(f"{'function':<11}{'d':>4}  " + "".join(col(a) for a in columns))
        for (_, f, d) in sub:
            line = f"{f:<11}{d:>4}  "
            for a in columns:
                base = np.array([at_gen(r, gen)
                                 for r in by[(n_pop, f, d, baseline_for(a))]])
                v = np.array([at_gen(r, gen) for r in by[(n_pop, f, d, a)]])
                p = wilcoxon_signed_rank(v, base)
                mark = "*" if (p < 0.05 and np.median(v) < np.median(base)) else " "
                line += f"{np.median(v):>{width - 1}.4g}{mark}"
            print(line)

        print(f"\nINITIAL POPULATION (n_pop={n_pop}) — median best at "
              f"generation 0, lower better")
        print(f"{'function':<11}{'d':>4}  " + "".join(col(a) for a in columns))
        for (_, f, d) in sub:
            line = f"{f:<11}{d:>4}  "
            for a in columns:
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
            agg = {a: {"it": 0, "ok": 0, "n": 0} for a in columns}
            print(f"\nSPEED (n_pop={n_pop}) — AR% by iterations vs each "
                  f"optimizer's own random arm, target = {label}")
            print(f"{'function':<11}{'d':>4}  " + "".join(col(a) for a in columns))
            for (_, f, d) in sub:
                st = {}
                for a in columns:
                    # The target is set by the arm's own baseline too: a cell's
                    # "what random reaches here" differs per optimizer, so one
                    # shared VTR would be unreachable for the weaker searches
                    # and trivial for the stronger ones.
                    vtr = target_of(by[(n_pop, f, d, baseline_for(a))])
                    pairs = [to_target(r, vtr) for r in by[(n_pop, f, d, a)]]
                    st[a] = (sum(x[0] for x in pairs), sum(x[1] for x in pairs),
                             len(pairs))
                    agg[a]["it"] += st[a][0]
                    agg[a]["ok"] += st[a][1]
                    agg[a]["n"] += st[a][2]
                line = f"{f:<11}{d:>4}  "
                for a in columns:
                    line += col(f"{(1 - st[a][0] / st[baseline_for(a)][0]) * 100:.1f}")
                print(line)

            print(f"\nAGGREGATE (n_pop={n_pop}, target = {label})")
            print(f"{'arm':<24}{'AR% iters':>11}{'success':>10}{'init_evals':>12}")
            for a in columns:
                init_ev = np.mean([r["init_evals"] for c in sub
                                   for r in by[(n_pop, c[1], c[2], a)]])
                print(f"{a:<24}"
                      f"{(1 - agg[a]['it'] / agg[baseline_for(a)]['it']) * 100:>11.2f}"
                      f"{100 * agg[a]['ok'] / agg[a]['n']:>9.1f}%"
                      f"{init_ev:>12.0f}")

if __name__ == "__main__":
    main()
