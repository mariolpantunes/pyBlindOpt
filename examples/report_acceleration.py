"""Acceleration rate and best fitness, with effect sizes, per dimension.

The cross-optimizer view of a finished sweep. `report_init_oblesa.py` renders
one optimizer at a time into HTML; this answers whether an effect *holds
across* optimizers and dimensions.

**Two metrics, and only two.** An initializer either gets the optimizer to a
given quality sooner, or it ends up somewhere better. Everything else here is
a way of putting an error bar on one of those.

  AR %          Acceleration rate. Per landscape, the target is the value the
                `random` baseline *finished* at on that same landscape, and
                AR is `(1 - iterations_to_reach / budget) * 100`. Positive
                means the arm reached random's final quality with that share
                of the budget left. A run that never reaches it is censored at
                the full budget and scores 0, which is the paper's convention.

  dlog10 fit    Best fitness achieved, as `log10(arm / random)` on the same
                landscape. **Negative is better**: -1.0 is a final value one
                order of magnitude below what random reached. A ratio is the
                only scale-free comparison available when the eight functions
                finish anywhere between 1e-83 and 1e3, and the log makes it
                symmetric so a 10x win and a 10x loss weigh alike.

**Why not a win rate or a rank.** Both discard the margin. "Better in 54% of
cells" is compatible with winning by nothing and losing by a lot, and it was
how an earlier reading of this sweep talked itself into an effect. Report the
size of the difference, or do not report it.

**The unit of independence is the landscape.** `shifted()` draws the optimum
from `(function, dimension, seed)`, so every seed is a different problem
instance and the confidence intervals below are bootstrapped over instances,
not over repetitions of one. That was not true before commit 451d3c2: the
offset came from `hash()`, which is salted per process, so each arm was scored
on a landscape of its own and no comparison between two arms meant anything.

8 functions x 100 seeds = 800 landscapes per (dimension, optimizer) cell.

Run against a completed sweep, from the repository root::

    python examples/report_acceleration.py
    python examples/report_acceleration.py --sweep-dir examples/out/sweep
    python examples/report_acceleration.py --arms qobl ob_a200i_s00
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

DIMS = [8, 16, 32, 64, 100]
OPTS = ["de", "jade", "ga", "cs", "egwo"]
BASE = "random"

#: Default arms: the cost-matched baselines, then the attraction ladder at the
#: models that stay identifiable. `x` is `auto`, ESS's own default.
SHOW = ["lhs", "qobl", "obl2x", "random4x", "ob_null_s00", "ob_a000_s00",
        "ob_a050d_s00", "ob_a050i_s00", "ob_a050x_s00",
        "ob_a200d_s00", "ob_a200p_s00", "ob_a200i_s00", "ob_a200x_s00"]
PRETTY = {
    "lhs": "lhs", "qobl": "qobl", "obl2x": "obl2x (4N)",
    "random4x": "random4x (4N)", "ob_null_s00": "oblesa null",
    "ob_a000_s00": "oblesa w=0",
    "ob_a050d_s00": "oblesa w=.5 detr", "ob_a050i_s00": "oblesa w=.5 idw",
    "ob_a050x_s00": "oblesa w=.5 auto",
    "ob_a200d_s00": "oblesa w=2 detr", "ob_a200p_s00": "oblesa w=2 proj",
    "ob_a200i_s00": "oblesa w=2 idw", "ob_a200x_s00": "oblesa w=2 auto",
}
FLOOR = 1e-30           # both sides of the ratio, so log10 stays finite

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--sweep-dir", default="examples/out/sweep",
                 help="directory of per-task JSONL written by the sweep")
_ap.add_argument("--arms", nargs="+", default=None,
                 help="override the default arm list")
_ap.add_argument("--boot", type=int, default=2000,
                 help="bootstrap resamples for the confidence intervals")
_args = _ap.parse_args()
if _args.arms:
    SHOW = list(_args.arms)
    PRETTY = {a: PRETTY.get(a, a) for a in SHOW}

WANT = set(SHOW) | {BASE}
cell = collections.defaultdict(dict)      # (fn, d, seed, opt) -> {arm: row}
for _p in sorted(glob.glob(os.path.join(_args.sweep_dir, "*.jsonl"))):
    with open(_p) as fh:
        for line in fh:
            r = json.loads(line)
            arm, _, opt = r["arm"].partition("@")
            if arm in WANT:
                cell[(r["function"], r["d"], r["seed"], opt)][arm] = r
if not cell:
    raise SystemExit(f"no sweep output under {_args.sweep_dir!r}")


def reached(row, vtr):
    """Iterations until the curve first reaches `vtr`, censored at the budget."""
    for g, v in enumerate(row["curve"]):
        if v <= vtr:
            return g + 1, True
    return len(row["curve"]), False


def ci(vals, boot, rng):
    """Median and a 95% percentile bootstrap interval over landscapes."""
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, a.size, size=(boot, a.size))
    meds = np.median(a[idx], axis=1)
    return (float(np.median(a)), float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)))


print(__doc__)
_rng = np.random.default_rng(0)
for opt in OPTS:
    print(f"\n{'=' * 104}\noptimizer: {opt}\n{'=' * 104}")
    print(f"{'arm':<20}" + "".join(f"{'d=' + str(d):>16}" for d in DIMS)
          + f"{'reached':>9}")
    print(f"{'':<20}" + "".join(f"{'AR%   dlog10':>16}" for _ in DIMS)
          + f"{'%':>9}")
    print("-" * 104)
    for arm in SHOW:
        line, hit, tot = [], 0, 0
        for d in DIMS:
            ars, dls = [], []
            for (fn, dd, seed, oo), v in cell.items():
                if dd != d or oo != opt or arm not in v or BASE not in v:
                    continue
                base, cur = v[BASE], v[arm]
                budget = len(base["curve"])
                if not budget:
                    continue
                # The target is what random ended at on THIS landscape, so the
                # comparison never crosses instances of different difficulty.
                vtr = base["curve"][-1]
                g, ok = reached(cur, vtr)
                ars.append((1 - g / budget) * 100)
                hit += ok
                tot += 1
                dls.append(np.log10(max(cur["score"], FLOOR)
                                    / max(base["score"], FLOOR)))
            ar, _, _ = ci(ars, _args.boot, _rng)
            dl, lo, hi = ci(dls, _args.boot, _rng)
            sig = (lo < 0) == (hi < 0) and not np.isnan(dl)
            star = "*" if sig else " "
            line.append(f"{ar:6.1f} {dl:+7.2f}{star}")
        rate = 100.0 * hit / tot if tot else float("nan")
        print(f"{PRETTY.get(arm, arm):<20}"
              + "".join(f"{c:>16}" for c in line) + f"{rate:9.1f}")

print("\n* the 95% bootstrap interval for dlog10 excludes 0 "
      "(resampled over landscapes).")
print("AR% > 0: reached random's final value with that share of the budget "
      "unspent.   dlog10 < 0: ended below random by that many orders of "
      "magnitude.")
