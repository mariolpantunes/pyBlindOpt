"""Which diversity metric, and is a linear fitness/diversity blend the right
way to combine it?

`bench_init_oblesa.py` measured that OBLESA's diversity knobs *hurt*, and
increasingly so with dimension: at d=40 plain `oblesa` reaches 34.1%
acceleration where `oblesa-div25` reaches 15.9%. This script asks why, and
what to use instead. Two separable questions:

**The metric.** NSGA-II crowding distance sums a normalised per-axis gap
over every dimension. It fails twice in high dimension. It assigns `inf` to
the extreme point of *each* axis, so a point is flagged maximally isolated
if it is extremal in any one of `d` axes — at d=40 with a 120-point pool
that is ~48% of the pool, all then collapsed to one value. And the sum of
`d` gaps concentrates, so the survivors are nearly indistinguishable. It was
designed for *objective* space in multi-objective problems; OBLESA applies
it to *decision* space.

**The combination.** The current rule blends two probability vectors,
`(1-w)*P_fitness + w*P_diversity`, and truncates. Beyond the arbitrary
comparability of the two scales, it has a structural flaw independent of
dimension: **diversity is scored once against the whole pool and never
against the set being selected.** Points that are each isolated relative to
120 candidates can be tightly clustered relative to each other. Sequential
schemes do not have this problem, because every choice is scored against
what has already been taken.

Reported per selector: the fitness it achieves and the spread it achieves,
so the trade-off is visible rather than collapsed into one number. Spread is
reported two ways on purpose — `min_nn`, the smallest nearest-neighbour
distance in the selected set, and `cover`, the mean per-axis coverage of the
marginals. In high dimension the joint space cannot be filled at all, but
the *projections* can be, which is the property that makes Latin hypercube
sampling survive where distance-based measures stop discriminating.

Everything is in the bounded space OBLESA actually selects in — no toroidal
wrap; that belongs inside `esa` and nowhere else.

Usage::

    python examples/bench_selection.py
    python examples/bench_selection.py --dims 2 10 40 --trials 40
"""

from __future__ import annotations

import argparse

import numpy as np

import pyBlindOpt.functions as functions
import pyBlindOpt.utils as utils

FUNCTIONS = {"rastrigin": functions.rastrigin, "ackley": functions.ackley}


# --------------------------------------------------------------------- #
# Diversity scores: higher means "more isolated / more useful for spread"
# --------------------------------------------------------------------- #

def div_crowding(X):
    """NSGA-II crowding distance — the incumbent."""
    return utils.compute_crowding_distance(X)


def div_knn(X, k=3):
    """Distance to the k-th nearest neighbour: a local sparsity estimate.

    Distance-based, so it inherits concentration, but it has no boundary
    atom and it measures the quantity the name promises.
    """
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    kk = min(k, X.shape[0] - 1)
    return np.partition(D, kk - 1, axis=1)[:, kk - 1]


def div_marginal(X, bins=None):
    """Per-axis marginal rarity: how under-populated each point's cell is.

    The one score here that does not concentrate, because it never sums a
    growing number of noisy terms — it averages them. In high dimension the
    joint space cannot be covered, but the marginals can, which is exactly
    the property Latin hypercube sampling exploits.
    """
    n, d = X.shape
    bins = bins or max(4, int(np.sqrt(n)))
    lo, hi = X.min(0), X.max(0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    idx = np.clip(((X - lo) / rng * bins).astype(int), 0, bins - 1)
    score = np.zeros(n)
    for j in range(d):
        counts = np.bincount(idx[:, j], minlength=bins)
        score += 1.0 / counts[idx[:, j]]
    return score / d


DIVERSITY = {"crowding": div_crowding, "knn": div_knn, "marginal": div_marginal}


# --------------------------------------------------------------------- #
# Selection schemes
# --------------------------------------------------------------------- #

def sel_fitness(X, y, n, div=None):
    """Greedy best fitness. OBLESA's default (diversity_weight = 0)."""
    return np.argpartition(y, n)[:n]


def sel_blend(X, y, n, div, w=0.25):
    """The incumbent: linear blend of two probability vectors, truncated."""
    pf = utils.score_2_probs(y)
    pd = utils.score_2_probs(-div(X))
    p = (1.0 - w) * pf + w * pd
    return np.argpartition(p, -n)[-n:]


def sel_twostage(X, y, n, div, keep=2.0):
    """Truncate to the fittest `keep*n`, then take the most isolated `n`.

    Fitness is a hard filter rather than a term to be traded away, so a
    point cannot buy its way in on spread alone.
    """
    m = min(len(y), int(keep * n))
    cand = np.argpartition(y, m - 1)[:m]
    s = div(X[cand])
    return cand[np.argpartition(s, -n)[-n:]]


def sel_greedy_maximin(X, y, n, div=None, keep=2.0):
    """Sequential maximin over the fittest `keep*n`, seeded by the best point.

    The only scheme here that scores diversity against *the set being built*
    rather than against the pool, which is the flaw every one-shot scheme
    shares.
    """
    m = min(len(y), int(keep * n))
    cand = np.argpartition(y, m - 1)[:m]
    P = X[cand]
    chosen = [int(np.argmin(y[cand]))]
    d2 = np.sum((P - P[chosen[0]]) ** 2, axis=1)
    for _ in range(n - 1):
        d2[chosen] = -1.0
        nxt = int(np.argmax(d2))
        chosen.append(nxt)
        d2 = np.minimum(d2, np.sum((P - P[nxt]) ** 2, axis=1))
    return cand[np.array(chosen)]


def sel_pareto(X, y, n, div):
    """Non-dominated sorting on (fitness, -diversity); no weight to choose."""
    s = -div(X)
    order = np.lexsort((s, y))
    chosen, best = [], np.inf
    for i in order:                     # front 1: nothing better on both
        if s[i] < best:
            chosen.append(i)
            best = s[i]
        if len(chosen) == n:
            break
    if len(chosen) < n:                 # top up on fitness
        rest = [i for i in order if i not in set(chosen)]
        chosen += rest[: n - len(chosen)]
    return np.array(chosen[:n])


SCHEMES = {
    "fitness-only": (sel_fitness, False),
    "blend(w=.25)": (sel_blend, True),
    "two-stage": (sel_twostage, True),
    "greedy-maximin": (sel_greedy_maximin, False),
    "pareto": (sel_pareto, True),
}


# --------------------------------------------------------------------- #

def spread(X):
    """(smallest nearest-neighbour gap, mean per-axis marginal coverage)."""
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    n, d = X.shape
    bins = max(4, int(np.sqrt(n)))
    occupied = 0.0
    for j in range(d):
        lo, hi = X[:, j].min(), X[:, j].max()
        rg = hi - lo if hi > lo else 1.0
        idx = np.clip(((X[:, j] - lo) / rg * bins).astype(int), 0, bins - 1)
        occupied += len(np.unique(idx)) / bins
    return float(D.min(1).min()), float(occupied / d)


def pool(fn, d, n_pop, rng):
    """An OBLESA-shaped pool: random N, opposites N, plus 2N filler."""
    b = np.array([[-5.0, 5.0]] * d)
    base = utils.RandomSampler(rng).sample(n_pop, b)
    opp = utils.check_bounds(b[:, 0] + (b[:, 1] - base), b)
    extra = utils.RandomSampler(rng).sample(2 * n_pop, b)
    X = np.vstack([base, opp, extra])
    return X, fn(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20, 40])
    ap.add_argument("--n-pop", type=int, default=30)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--function", default="rastrigin", choices=list(FUNCTIONS))
    args = ap.parse_args()
    fn = FUNCTIONS[args.function]

    print(f"{args.function}, pool = 4x{args.n_pop}, selecting {args.n_pop}, "
          f"{args.trials} trials. Bounded space, no toroidal wrap.")
    print("fitness: median of selected (lower better).  "
          "min_nn: smallest nearest-neighbour gap (higher better).  "
          "cover: mean per-axis marginal coverage (higher better).")

    for d in args.dims:
        print(f"\n--- d = {d} ---")
        print(f"{'scheme':<16}{'diversity':<11}{'fitness':>11}{'min_nn':>10}"
              f"{'cover':>9}")
        rows = {}
        for sname, (fnsel, needs_div) in SCHEMES.items():
            divs = DIVERSITY if needs_div else {"-": None}
            for dname, dfn in divs.items():
                f_, m_, c_ = [], [], []
                for t in range(args.trials):
                    rng = np.random.default_rng(1000 * d + t)
                    X, y = pool(fn, d, args.n_pop, rng)
                    idx = (fnsel(X, y, args.n_pop, dfn) if needs_div
                           else fnsel(X, y, args.n_pop))
                    S = X[idx]
                    f_.append(np.median(y[idx]))
                    mn, cv = spread(S)
                    m_.append(mn); c_.append(cv)
                rows[(sname, dname)] = (np.mean(f_), np.mean(m_), np.mean(c_))
                print(f"{sname:<16}{dname:<11}{np.mean(f_):>11.4g}"
                      f"{np.mean(m_):>10.3f}{np.mean(c_):>9.3f}")


if __name__ == "__main__":
    main()
