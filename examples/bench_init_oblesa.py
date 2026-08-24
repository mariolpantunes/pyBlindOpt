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
import functools
import json
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

def _uniform_null(samples, bounds, *, n, seed=None, **ignored):
    """The no-search null: `n` uniform points, same pool shape, nothing sought.

    This is the control the whole design rests on. OBLESA against OBL confounds
    two things -- a larger candidate pool, and a pool with points chosen by
    searching empty space -- and only an arm that spends the same candidates
    without searching separates them.

    Four lines here rather than a call into ESS: the null must not move when
    ESS changes, or it stops being a fixed reference.
    """
    del samples, ignored
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(int(n), bounds.shape[0]))




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
    "a000": {"force": "guided", "force_weight": 0.00},     # ESS, novelty only
    "a025": {"force": "guided", "force_weight": 0.25},
    "a050": {"force": "guided", "force_weight": 0.50},     # ESS's own default
    "a100": {"force": "guided", "force_weight": 1.00},
    "a200": {"force": "guided", "force_weight": 2.00},     # just under refusal
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
    _guided = _eng.get("force") == "guided" and _eng.get("force_weight", 0) > 0
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
    "a050x": {"force": "guided", "force_weight": 0.50, "_att_model": "auto"},
    "a200i": {"force": "guided", "force_weight": 2.00, "_att_model": "idw"},
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
    arm, _ = split_arm(arm)
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
        # -- it arrives as `force` or `engine` rather than as a monkeypatched
        # module attribute -- so the opposition step, the pool shape and the
        # selection rule stay bit-for-bit identical across arms that share
        # them, and each knob is genuinely the only thing that varies along
        # its own axis.
        kw = dict(OBLESA_KNOBS[arm])
        # `_n_ess_mult` is an arm-table convention, not an `oblesa` argument.
        mult = kw.pop("_n_ess_mult", 1.0)
        kw["n_ess"] = max(1, round(mult * n_pop))
        # `oblesa` forwards `force_weight`; `att_model` is ESS's own knob,
        # below the engine contract, so bind it through `engine=`.
        model = kw.pop("_att_model", None)
        if model is not None:
            kw["engine"] = functools.partial(
                init._ess_engine, att_model=model)
            kw["engine"].accepts = (  # type: ignore[reportAttributeAccessIssue]
                init._ess_engine.accepts)  # type: ignore[reportFunctionMemberAccess]
            kw.pop("force", None)
        if stats is not None and arm in ENGINE_LABEL:
            stats["engine"] = ENGINE_LABEL[arm]
        if info is not None:
            kw["info"] = info
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
    for n_pop in args.n_pop:
        for fname in args.functions:
            for d in args.dims:
                for arm in args.arms:
                    t0 = time.perf_counter()
                    cell = []
                    for seed in range(args.seeds):
                        cell.extend(one_cell(arm, fname, d, seed, n_pop,
                                             args.iters, args.optimizers))
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
    expected = (len(args.n_pop) * len(args.functions)
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
        for n_pop in args.n_pop:
            for fname in args.functions:
                t0 = time.perf_counter()
                for seed in range(args.seeds):
                    for row in one_cell(init_arm, fname, d, seed, n_pop,
                                        args.iters, optimizers):
                        row["arm_knobs"] = knobs
                        fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"  {label:<24} {fname:<12} n_pop={n_pop:<4} "
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
