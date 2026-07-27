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
DIMS = (2, 4, 8, 16, 32)
ARMS = ("random", "lhs", "sobol", "obl", "oblesa")


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


def initial_population(arm, objective, bounds, n_pop, rng):
    """One arm's initial population. Cost is charged to `objective`."""
    if arm == "random":
        return utils.RandomSampler(rng).sample(n_pop, bounds)
    if arm == "lhs":
        return utils.HLCSampler(rng).sample(n_pop, bounds)
    if arm == "sobol":
        return utils.SobolSampler(rng).sample(n_pop, bounds)
    if arm == "obl":
        return init.opposition_based(
            objective, bounds, n_pop=n_pop, seed=rng)
    if arm == "oblesa":
        return init.oblesa(objective, bounds, n_pop=n_pop, seed=rng)
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


def at_budget(row, n_pop, budget):
    """Best-so-far once `budget` evaluations have been spent, or `inf`.

    Generation `g` of a run has consumed `init_evals + (g+1)*n_pop`, so arms
    that paid more to initialize reach a given budget later. This is what
    makes the curves comparable across arms that do not start level.
    """
    curve = row["curve"]
    if not curve:
        return row["score"]
    g = (budget - row["init_evals"]) // n_pop - 1
    if g < 0:
        return float("inf")            # arm had not started by this budget
    return curve[min(int(g), len(curve) - 1)]


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
                    # the p-values should be inspectable while it runs.
                    with open(args.out, "w") as fh:
                        json.dump({"config": vars(args), "rows": rows}, fh)

    report(rows, args)


def report(rows, args):
    """Arms compared at matched evaluation budgets, not just at the end.

    Lower is better throughout. Each block fixes a fraction of the budget and
    asks where every arm stood once it had spent that many evaluations, so an
    initializer that buys an early lead and then loses it is visible as such
    instead of averaging into nothing.
    """
    fracs = (0.1, 0.25, 0.5, 1.0)

    def sel(n_pop, fname, d, arm, budget):
        got = [r for r in rows if r["function"] == fname and r["d"] == d
               and r["arm"] == arm and r["n_pop"] == n_pop]
        got.sort(key=lambda r: r["seed"])
        return np.array([at_budget(r, n_pop, budget) for r in got])

    for n_pop in args.n_pop:
        full = args.iters * n_pop
        for frac in fracs:
            budget = int(full * frac)
            print(f"\n=== n_pop={n_pop}, {int(frac * 100)}% budget "
                  f"({budget} evals), {args.seeds} seeds, lower is better ===")
            print(f"{'function':<11} {'d':>3} "
                  + "".join(f"{a:>13}" for a in args.arms)
                  + f"{'p vs rand':>11}{'p vs obl':>10}{'winner':>9}")
            for fname in args.functions:
                for d in args.dims:
                    meds, cols = {}, ""
                    for arm in args.arms:
                        s = sel(n_pop, fname, d, arm, budget)
                        meds[arm] = float(np.median(s)) if s.size else float("nan")
                        cols += f"{meds[arm]:>13.5g}"
                    best = min(meds, key=lambda a: meds[a])
                    p_rand = p_obl = float("nan")
                    if "oblesa" in args.arms:
                        o = sel(n_pop, fname, d, "oblesa", budget)
                        if "random" in args.arms:
                            p_rand = wilcoxon_signed_rank(
                                o, sel(n_pop, fname, d, "random", budget))
                        if "obl" in args.arms:
                            p_obl = wilcoxon_signed_rank(
                                o, sel(n_pop, fname, d, "obl", budget))
                    print(f"{fname:<11} {d:>3} {cols}{p_rand:>11.3f}"
                          f"{p_obl:>10.3f}{best:>9}")


if __name__ == "__main__":
    main()
