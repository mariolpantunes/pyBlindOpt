"""Acceleration rate and best fitness, per dimension, for every optimizer.

The cross-optimizer view of a finished sweep. `report_init_oblesa.py` renders
one optimizer at a time into HTML and is the analysis of record; this answers
the different question of whether an effect *holds across* optimizers and
dimensions, in one table you can read at a glance.

Uses this project's own definitions (`report_init_oblesa.py`):

  target (VTR)  the MEDIAN FINAL VALUE the `random` baseline reaches in that
                (function, dimension, optimizer) cell. A fixed tolerance
                cannot work across functions spanning 1e-83 to 1e3 -- the
                same alpha is unreachable for rastrigin at d=32 and passed in
                two generations by sphere. Targeting what the baseline
                actually achieves keeps acceleration meaningful everywhere,
                and guarantees ~half the baseline runs reach it.

  AR %          (1 - iterations_arm / iterations_random) * 100, summed over
                the cells of a dimension. POSITIVE = reaches the target in
                fewer iterations than random. A run that never reaches it is
                censored at the full budget, which is the paper's convention.

  success %     share of runs that reached the target at all.

  best fitness  mean normalised rank of the final value, 0.00 = best of the
                compared arms, 1.00 = worst. Ranked because raw values are
                not comparable across functions and dimensions.

8 functions x 5 dimensions x 100 seeds per arm per optimizer.

Run against a completed sweep, from the repository root::

    python examples/report_acceleration.py
    python examples/report_acceleration.py --sweep-dir examples/out/sweep
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
SHOW = ["lhs", "qobl", "ob_null_s00", "ob_a000_s00", "ob_a050f_s00",
        "ob_a100f_s00", "ob_a200f_s00"]
PRETTY = {"lhs": "lhs", "qobl": "qobl", "ob_null_s00": "OBLESA null",
          "ob_a000_s00": "OBLESA w=0", "ob_a050f_s00": "OBLESA w=0.5",
          "ob_a100f_s00": "OBLESA w=1", "ob_a200f_s00": "OBLESA w=2"}
N_POP, ITERS = 30, 200
BUDGET = N_POP * ITERS

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--sweep-dir", default="examples/out/sweep",
                 help="directory of per-task JSONL written by the sweep")
_args = _ap.parse_args()

rows = collections.defaultdict(list)          # (fn, d, opt, arm) -> rows
_files = sorted(glob.glob(os.path.join(_args.sweep_dir, "*.jsonl")))
if not _files:
    raise SystemExit(f"no sweep output under {_args.sweep_dir!r}")
for p in _files:
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            arm, _, opt = r["arm"].partition("@")
            if arm in SHOW or arm == BASE:
                rows[(r["function"], r["d"], opt, arm)].append(r)


def trip(r, vtr):
    for g, v in enumerate(r["curve"]):
        if v <= vtr:
            return g + 1, True
    return len(r["curve"]), False


print(__doc__)
for opt in OPTS:
    fns = sorted({k[0] for k in rows if k[2] == opt})
    print(f"\n{'=' * 92}\noptimizer: {opt}   functions: {len(fns)}\n{'=' * 92}")
    print(f"{'arm':<14}" + "".join(f"{'d=' + str(d):>13}" for d in DIMS)
          + f"{'success':>10}")
    print(f"{'':<14}" + "".join(f"{'AR%  fit':>13}" for _ in DIMS) + f"{'%':>10}")
    print("-" * 92)
    for arm in SHOW:
        cells_line, ok_tot, n_tot = [], 0, 0
        for d in DIMS:
            it_a = it_b = 0
            ranks = []
            for f in fns:
                base = rows.get((f, d, opt, BASE))
                cur = rows.get((f, d, opt, arm))
                if not base or not cur:
                    continue
                vtr = float(np.median([r["curve"][-1] if r["curve"]
                                       else r["score"] for r in base]))
                for r in base:
                    it_b += trip(r, vtr)[0]
                for r in cur:
                    g, ok = trip(r, vtr)
                    it_a += g
                    ok_tot += ok
                    n_tot += 1
                # rank the final value of this arm among all shown arms
                allv = {a: rows[(f, d, opt, a)] for a in [BASE] + SHOW
                        if rows.get((f, d, opt, a))}
                per = sorted(allv)
                med = {a: float(np.median([x["score"] for x in allv[a]]))
                       for a in per}
                order = sorted(per, key=lambda a: med[a])
                ranks.append(order.index(arm) / (len(per) - 1))
            ar = (1 - it_a / it_b) * 100 if it_b else float("nan")
            cells_line.append(f"{ar:6.1f} {np.mean(ranks):5.2f}")
        succ = 100.0 * ok_tot / n_tot if n_tot else float("nan")
        print(f"{PRETTY[arm]:<14}"
              + "".join(f"{c:>13}" for c in cells_line) + f"{succ:10.1f}")
