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

**Budget parity is the whole methodology.** The arms do not cost the same:
random/LHS/Sobol spend nothing to initialize, OBL spends `2*n_pop`, and
OBLESA spends `4*n_pop` — random plus opposite plus the `2*n_pop` points
`ess.esa` returns. Reporting "best after `n_iter` generations" would hand
OBLESA a free head start and prove nothing. Here every arm gets the **same
total evaluation budget** and pays for its own initialization out of it:
generations are whatever is left. Evaluations are *counted*, not assumed —
`_Counted` wraps the objective, because `compute_objective` may call it
either row-wise or on the whole matrix, and because the `4*n_pop` above was
itself a measurement that corrected a wrong guess of `3*n_pop`.

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

FUNCTIONS = {
    "sphere": functions.sphere,
    "rastrigin": functions.rastrigin,
    "ackley": functions.ackley,
    "griewank": functions.griewank,
    "rosenbrock": functions.rosenbrock,
}

# Straddling the region the EC literature disagrees about (~12 d) and the
# region torann measured its own wall in (~8 d).
# Matching the dimensions of "Active Initialization in Population-Based
# Optimizers" (GECCO Companion '26, Table 3) so the acceleration rates are
# directly comparable to its published figures. 40 is also the ceiling of
# pyBlindOpt's Sobol sampler (Joe & Kuo direction numbers), so no arm has to
# be dropped or substituted at the top of the range.
DIMS = (2, 5, 10, 20, 40)

# Three groups, and the middle one is the point.
#
#   baselines      what the field already does
#   cost controls  4N candidates like OBLESA, but without ESS -- these say
#                  whether ESS finds genuinely empty regions or merely hands
#                  the selector more candidates to choose from
#   oblesa knobs   OBLESA exposes controls the other arms do not have, so
#                  comparing it only at defaults understates it
ARMS = (
    "random", "sobol",                              # baselines, N calls
    "random4x", "obl2x",                            # cost controls, 4N calls
    "obl", "qobl",                                  # opposition, 2N calls
    "oblesa", "oblesa-quasi",                       # OBLESA on fitness alone
    "oblesa-div25",                                 # incumbent: crowding blend
    "oblesa-mm25", "oblesa-mm50",                   # sequential maximin
    "oblesa-quasi-mm25", "oblesa-quasi-mm50",       # quasi + maximin
)
BASELINE = "random"

# Knob settings per OBLESA arm; everything else is the pyBlindOpt default.
OBLESA_KNOBS = {
    "oblesa": {},
    "oblesa-quasi": {"opp": "quasi"},
    # The incumbent diversity rule: NSGA-II crowding distance blended with
    # fitness as probabilities. Kept as the control the replacement has to
    # beat, since bench_selection.py showed it is Pareto dominated.
    "oblesa-div25": {"diversity_weight": 0.25},
    # Sequential maximin over a fitness-truncated pool. diversity_weight is
    # reinterpreted as the truncation, keep = 1 + 3w, so 0.25 keeps the
    # fittest 1.75x n_pop and 0.50 keeps 2.5x.
    "oblesa-mm25": {"selection": "maximin", "diversity_weight": 0.25},
    "oblesa-mm50": {"selection": "maximin", "diversity_weight": 0.50},
    "oblesa-quasi-mm25": {"opp": "quasi", "selection": "maximin",
                          "diversity_weight": 0.25},
    "oblesa-quasi-mm50": {"opp": "quasi", "selection": "maximin",
                          "diversity_weight": 0.50},
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


def _best_of(pool, objective, n_pop):
    """Greedy best `n_pop` of an evaluated pool — OBL's own selection rule."""
    scores = utils.compute_objective(pool, objective, 1)
    return pool[np.argpartition(scores, n_pop)[:n_pop]]


def initial_population(arm, objective, bounds, n_pop, rng):
    """One arm's initial population. Cost is charged to `objective`.

    The cost controls exist because OBLESA's advantage has two candidate
    explanations that the default arms cannot separate: ESS may be locating
    genuinely under-explored regions, or a 4N pool may simply give greedy
    selection more to work with. `random4x` and `obl2x` spend the same 4N
    calls without ESS, so any margin OBLESA keeps over them is attributable
    to the empty-space search itself.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    if arm == "random":
        return utils.RandomSampler(rng).sample(n_pop, bounds)
    if arm == "lhs":
        return utils.HLCSampler(rng).sample(n_pop, bounds)
    if arm == "sobol":
        return utils.SobolSampler(rng).sample(n_pop, bounds)
    if arm == "random4x":
        return _best_of(utils.RandomSampler(rng).sample(4 * n_pop, bounds),
                        objective, n_pop)
    if arm == "obl2x":
        base = utils.RandomSampler(rng).sample(2 * n_pop, bounds)
        opp = utils.check_bounds(lo + hi - base, bounds)
        return _best_of(np.vstack([base, opp]), objective, n_pop)
    if arm == "obl":
        return init.opposition_based(objective, bounds, n_pop=n_pop, seed=rng)
    if arm == "qobl":
        return init.quasi_opposition_based(
            objective, bounds, n_pop=n_pop, seed=rng)
    if arm in OBLESA_KNOBS:
        return init.oblesa(objective, bounds, n_pop=n_pop, seed=rng,
                           **OBLESA_KNOBS[arm])
    raise ValueError(f"unknown arm {arm!r}")


def one_run(arm, fname, d, seed, n_pop, budget):
    """One (arm, function, dimension, seed) cell, at a fixed total budget.

    **Two independent generators, and the reason matters.** The arms consume
    wildly different amounts of randomness during initialization — ESS draws
    for every particle of every epoch, plain sampling draws once. Feeding the
    optimizer whatever generator state the initializer happened to leave
    behind means each arm runs DE on a different random trajectory, so a
    paired comparison would be measuring two changes at once and attributing
    both to the initializer. `rng_opt` is therefore seeded identically for
    every arm: the initial population becomes the only thing that differs.
    """
    bounds = np.array([[-5.0, 5.0]] * d)
    counted = _Counted(FUNCTIONS[fname])
    rng_init = np.random.default_rng(seed)
    rng_opt = np.random.default_rng(2**31 - seed)

    t0 = time.perf_counter()
    pop = initial_population(arm, counted, bounds, n_pop, rng_init)
    t_init = time.perf_counter() - t0
    used = counted.n

    # Whatever the arm did not spend on initialization buys generations.
    # The population itself is re-scored on the first generation, so the
    # budget left over is (budget - used) and each generation costs n_pop.
    n_iter = max(1, (budget - used) // n_pop)

    trace = _Trace()
    _, score = de.differential_evolution(
        counted, bounds, population=pop, n_pop=n_pop, n_iter=n_iter,
        seed=rng_opt, callback=trace)
    return {
        "arm": arm, "function": fname, "d": d, "seed": seed,
        "score": float(score), "init_evals": int(used),
        "total_evals": int(counted.n), "n_iter": int(n_iter),
        "init_seconds": t_init, "curve": trace.best,
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


def to_target(row, n_pop, budget, vtr):
    """(iterations, function calls, reached) to first hit `vtr`."""
    for g, v in enumerate(row["curve"]):
        if v <= vtr:
            return g + 1, row["init_evals"] + (g + 1) * n_pop, True
    return len(row["curve"]), budget, False


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
                    help="budget in generations; total evals = iters * n_pop, "
                         "identical for every arm")
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--functions", nargs="+", default=list(FUNCTIONS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="examples/out/bench_init_oblesa.json")
    args = ap.parse_args()

    rows = []
    for n_pop in args.n_pop:
        budget = args.iters * n_pop
        for fname in args.functions:
            for d in args.dims:
                for arm in args.arms:
                    t0 = time.perf_counter()
                    for seed in range(args.seeds):
                        row = one_run(arm, fname, d, seed, n_pop, budget)
                        row["n_pop"] = n_pop
                        rows.append(row)
                    print(f"  n_pop={n_pop:<5} {fname:<11} d={d:<3} {arm:<7} "
                          f"median={np.median([r['score'] for r in rows[-args.seeds:]]):<13.6g} "
                          f"init={rows[-1]['init_evals']:<6} "
                          f"iters={rows[-1]['n_iter']:<5} "
                          f"evals={rows[-1]['total_evals']:<8} "
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


def report(rows, args):
    """Two questions, kept separate because they are different questions.

    *Quality* — where is each arm at the same generation? That isolates the
    initial population from every accounting choice.
    *Speed* — how many generations, and how many function calls, to reach the
    quality the baseline ends at? That is the acceleration rate.

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

    for n_pop in args.n_pop:
        budget = args.iters * n_pop
        sub = [c for c in cells if c[0] == n_pop]
        if not sub:
            continue

        print(f"\n{'=' * 78}\nQUALITY at generation {gen + 1} "
              f"(n_pop={n_pop}) — median, lower better; * = p<0.05 vs random")
        print(f"{'function':<11}{'d':>4}  " + "".join(f"{a:>15}" for a in args.arms))
        for (_, f, d) in sub:
            base = np.array([at_gen(r, gen) for r in by[(n_pop, f, d, BASELINE)]])
            line = f"{f:<11}{d:>4}  "
            for a in args.arms:
                v = np.array([at_gen(r, gen) for r in by[(n_pop, f, d, a)]])
                p = wilcoxon_signed_rank(v, base)
                mark = "*" if (p < 0.05 and np.median(v) < np.median(base)) else " "
                line += f"{np.median(v):>14.4g}{mark}"
            print(line)

        agg = {a: {"it": 0, "nfc": 0, "ok": 0, "n": 0} for a in args.arms}
        print(f"\nSPEED (n_pop={n_pop}) — acceleration rate % vs {BASELINE}, "
              f"per cell, by function calls")
        print(f"{'function':<11}{'d':>4}  " + "".join(f"{a:>15}" for a in args.arms))
        for (_, f, d) in sub:
            vtr = vtr_for_cell(by[(n_pop, f, d, BASELINE)])
            st = {}
            for a in args.arms:
                trip = [to_target(r, n_pop, budget, vtr)
                        for r in by[(n_pop, f, d, a)]]
                st[a] = (sum(x[0] for x in trip), sum(x[1] for x in trip),
                         sum(x[2] for x in trip), len(trip))
                for i, key in enumerate(("it", "nfc", "ok")):
                    agg[a][key] += st[a][i]
                agg[a]["n"] += st[a][3]
            base_nfc = st[BASELINE][1]
            line = f"{f:<11}{d:>4}  "
            for a in args.arms:
                line += f"{(1 - st[a][1] / base_nfc) * 100:>15.1f}"
            print(line)

        print(f"\nAGGREGATE (n_pop={n_pop})")
        print(f"{'arm':<15}{'AR% iters':>11}{'AR% calls':>11}{'success':>10}")
        for a in args.arms:
            print(f"{a:<15}"
                  f"{(1 - agg[a]['it'] / agg[BASELINE]['it']) * 100:>11.2f}"
                  f"{(1 - agg[a]['nfc'] / agg[BASELINE]['nfc']) * 100:>11.2f}"
                  f"{100 * agg[a]['ok'] / agg[a]['n']:>9.1f}%")


if __name__ == "__main__":
    main()
