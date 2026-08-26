"""Stage A selection: which setting of each OBLESA knob wins, per optimizer.

`report_acceleration.py` compares a hand-written list of arms. Sweep v8 has
468 of them -- six knobs fully crossed -- and an arm table that long answers
nothing. The question the factorial was run to answer is *marginal*: holding
everything else fixed, does raising the attraction weight help this optimizer,
and does the answer move with the attraction model or with dimension.

**The estimator is a paired contrast, not a group mean.** For a knob `K`,
every arm has an exact sibling differing in `K` alone, because the design is
full. So the effect of `K` is measured within sibling pairs on the *same*
landscape and then aggregated, which cancels the landscape, the optimizer, the
population rule and the other five knobs. Comparing the mean of all `w200`
arms against the mean of all `w050` arms would instead average over an
unbalanced remainder: `force_weight=0` carries one attraction model rather
than four, so the model marginal is confounded with the weight unless the
comparison is paired.

**The unit of independence is the landscape**, as everywhere else in this
repository: `shifted()` draws the optimum from `(function, dimension, seed)`,
so 14 functions x 50 seeds is 700 distinct problems per (dimension, optimizer)
cell, not 50 repetitions of 14. Sibling deltas are averaged within a landscape
first, then bootstrapped across landscapes; bootstrapping across pairs would
count one landscape as many independent observations and shrink every interval
by roughly the number of sibling groups.

Metrics are the two `report_acceleration.py` defines, unchanged:

  dlog10  `log10(arm / random)` on the same landscape. Negative is better.
  AR %    `(1 - iterations_to_reach / budget) * 100` against the value the
          `random` baseline finished at on that landscape. Positive is better.

The contrast tables report dlog10 deltas: **negative means the level beats the
reference level**.

Two phases, because the sweep is 31 GB of JSONL and most of it is curves::

    python examples/report_v8_selection.py --build     # jsonl -> per-arm npz
    python examples/report_v8_selection.py             # npz -> tables

`--build` is incremental and safe to run against a live sweep: a task's file
is cached only once it holds all `EXPECTED` rows, so a file still being
appended to is left for the next pass. Re-running costs one `stat` per arm.
"""

import argparse
import collections
import glob
import json
import multiprocessing
import os
import re
import sys

import numpy as np

try:                                    # 3-5x faster on files this shape
    import orjson  # type: ignore[reportMissingImports]
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

DIMS = [8, 16, 32, 64, 100]
OPTS = ["de", "jade", "ga", "cs", "egwo"]
FUNCTIONS = ["ackley", "dixon", "griewank", "levy", "lunacek", "rastrigin",
             "rosenbrock", "rot_dixon", "rot_levy", "rot_rastrigin",
             "rot_styblinski", "schwefel", "sphere", "styblinski"]
SEEDS = 50
BASE = "random"
FLOOR = 1e-30                           # both sides of the ratio stay finite
EXPECTED = len(FUNCTIONS) * SEEDS * len(OPTS)
N_LAND = len(FUNCTIONS) * SEEDS
#: Bumped whenever the npz columns change, so a stale cache rebuilds itself
#: instead of being read with the wrong schema.
CACHE_V = 2

_FN_IX = {f: i for i, f in enumerate(FUNCTIONS)}
_OPT_IX = {o: i for i, o in enumerate(OPTS)}

#: The six crossed knobs, in the order the arm name spells them. The value is
#: `(pretty name, {token: label})`, and the *first* label is the contrast's
#: reference level -- chosen as the off/neutral setting so a negative delta
#: always reads as "turning this on helped".
KNOBS = collections.OrderedDict([
    ("w", ("force_weight", {"w000": "0.0", "w050": "0.5",
                            "w100": "1.0", "w200": "2.0"})),
    ("m", ("att_model", {"midw": "idw", "maut": "auto",
                         "mdet": "detrended", "mprj": "projection"})),
    ("r", ("rounds", {"r1": "1", "r2": "2", "r3": "3"})),
    ("s", ("diversity_weight", {"s00": "0.00", "s25": "0.25",
                                "s50": "0.50"})),
    ("o", ("opp", {"os": "standard", "oq": "quasi"})),
    ("e", ("opp_ess", {"e0": "False", "e1": "True"})),
])
KNOB_KEYS = list(KNOBS)
ARM_RE = re.compile(r"^f8_(w\d{3})_(m\w{3})_(r\d)_(s\d{2})_(o[qs])_(e[01])$")

#: The controls the whole claim rests on. Shown beside every selected arm,
#: because beating `obl` is arithmetic -- OBLESA's pool is a superset of it --
#: while beating a same-sized pool of unguided draws is not.
CONTROLS = ["random3x", "obl15x", "random4x", "obl2x", "random5x", "obl25x",
            "random6x", "obl3x", "random8x", "obl4x",
            "qobl", "obl", "lhs", "sobol"]

#: Pool multiple -> the two controls that spend it. `oblesa_pool_size` is
#: `2N + rounds * n_ess`, and `opp_ess` doubles the per-round block, so an arm
#: at `rounds=1` draws 3N and one at `rounds=3, opp_ess=True` draws 8N.
#: Comparing either against the 4N control -- the only one Stage A shipped --
#: charges the first for a block it never drew and hands the second one free.
POOL_CONTROLS = {3: ("random3x", "obl15x"), 4: ("random4x", "obl2x"),
                 5: ("random5x", "obl25x"), 6: ("random6x", "obl3x"),
                 8: ("random8x", "obl4x")}


def pool_multiple(tok):
    """How many `n_pop` blocks an arm's knobs make it evaluate."""
    per_round = 2 if tok["e"] == "e1" else 1
    return 2 + int(KNOBS["r"][1][tok["r"]]) * per_round


def parse_arm(arm):
    """`{knob: token}` for an f8 arm name, or None if it is not one."""
    m = ARM_RE.match(arm)
    return dict(zip(KNOB_KEYS, m.groups())) if m else None


# --------------------------------------------------------------------------
# phase 1: JSONL -> npz
# --------------------------------------------------------------------------

def reached(curve, vtr):
    """Iterations until `curve` first reaches `vtr`, censored at the budget."""
    for g, v in enumerate(curve):
        if v <= vtr:
            return g + 1, True
    return len(curve), False


def _slot(fn, seed):
    return _FN_IX[fn] * SEEDS + seed


def read_rows(path):
    """Every row of one task file, or None if the file is still being written.

    A task appends one cell at a time, so a short file is not corrupt -- it is
    unfinished, and caching it would freeze a partial arm into the report.
    """
    rows = []
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                rows.append(_loads(line))
    return rows if len(rows) == EXPECTED else None


def build_baselines(sweep_dir):
    """`(d, opt) -> (vtr[N_LAND], score[N_LAND])` from the `random` arm.

    Every AR% target and every dlog10 denominator comes from here, so a
    missing dimension is fatal rather than skipped: it would silently drop a
    fifth of the sweep from the report.
    """
    out = {}
    for d in DIMS:
        path = os.path.join(sweep_dir, f"{BASE}_d{d}.jsonl")
        rows = read_rows(path) if os.path.exists(path) else None
        if rows is None:
            raise SystemExit(
                f"baseline {BASE}_d{d}.jsonl is missing or incomplete; "
                "AR% and dlog10 are undefined without it")
        for opt in OPTS:
            out[(d, opt)] = (np.full(N_LAND, np.nan),
                             np.full(N_LAND, np.nan),
                             np.zeros(N_LAND, dtype=np.int32))
        for r in rows:
            vtr, score, budget = out[(r["d"], r["optimizer"])]
            i = _slot(r["function"], r["seed"])
            vtr[i] = r["curve"][-1]
            score[i] = r["score"]
            budget[i] = len(r["curve"])
    return out


def _build_one(job):
    """Cache one task file as `(land, opt, dl, ar, hit)` columns."""
    path, cache_dir, base = job
    name = os.path.basename(path)[:-len(".jsonl")]
    dst = os.path.join(cache_dir, name + ".npz")
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(path):
        try:
            if int(np.load(dst)["ver"]) == CACHE_V:
                return name, "cached"
        except (KeyError, ValueError, OSError):
            pass
    rows = read_rows(path)
    if rows is None:
        return name, "partial"
    n = len(rows)
    land = np.empty(n, dtype=np.int32)
    opt = np.empty(n, dtype=np.int8)
    dl = np.empty(n, dtype=np.float32)
    ar = np.empty(n, dtype=np.float32)
    hit = np.zeros(n, dtype=bool)
    # The cost side. `n_iter` is what the arm could actually afford after
    # `iters_for` charged initialization against the evaluation budget, and
    # `secs` is the wall-clock the budget does *not* charge for.
    nit = np.empty(n, dtype=np.int32)
    iev = np.empty(n, dtype=np.int32)
    secs = np.empty(n, dtype=np.float32)
    for k, r in enumerate(rows):
        i = _slot(r["function"], r["seed"])
        b_vtr, b_score, b_budget = base[(r["d"], r["optimizer"])]
        land[k] = i
        opt[k] = _OPT_IX[r["optimizer"]]
        dl[k] = np.log10(max(r["score"], FLOOR) / max(b_score[i], FLOOR))
        g, ok = reached(r["curve"], b_vtr[i])
        ar[k] = (1.0 - g / b_budget[i]) * 100.0
        hit[k] = ok
        nit[k] = r["n_iter"]
        iev[k] = r["init_evals"]
        secs[k] = r["init_seconds"]
    # `savez_compressed` appends `.npz` unless the name already ends in it,
    # so the temp name has to carry the suffix or the rename misses the file.
    tmp = dst + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, land=land, opt=opt, dl=dl, ar=ar, hit=hit,
                        nit=nit, iev=iev, secs=secs, ver=np.int32(CACHE_V))
    os.replace(tmp, dst)
    return name, "built"


def build(sweep_dir, cache_dir, procs):
    os.makedirs(cache_dir, exist_ok=True)
    base = build_baselines(sweep_dir)
    paths = sorted(glob.glob(os.path.join(sweep_dir, "*.jsonl")))
    jobs = [(p, cache_dir, base) for p in paths]
    tally = collections.Counter()
    with multiprocessing.Pool(procs) as pool:
        for k, (name, what) in enumerate(pool.imap_unordered(_build_one, jobs), 1):
            tally[what] += 1
            if k % 50 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  " + "  ".join(
                    f"{v} {kk}" for kk, v in sorted(tally.items())),
                    file=sys.stderr, flush=True)
    return tally


# --------------------------------------------------------------------------
# phase 2: npz -> tables
# --------------------------------------------------------------------------

#: Column order in the tuple `load` returns, so a section can say which one
#: it wants by name instead of by an index nobody can read.
COL = {"dl": 0, "ar": 1, "nit": 2, "iev": 3, "secs": 4}


def load(cache_dir):
    """`(arm, d, opt) -> (dl, ar, nit, iev, secs)`, NaN where a cell is absent."""
    data = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        name = os.path.basename(path)[:-len(".npz")]
        arm, _, dtok = name.rpartition("_d")
        if not dtok.isdigit():
            continue
        d = int(dtok)
        z = np.load(path)
        if int(z.get("ver", 0)) != CACHE_V:
            continue
        land, opt = z["land"], z["opt"]
        cols = [z["dl"], z["ar"], z["nit"], z["iev"], z["secs"]]
        for oi, o in enumerate(OPTS):
            m = opt == oi
            if not m.any():
                continue
            out = []
            for c in cols:
                a = np.full(N_LAND, np.nan, dtype=np.float32)
                a[land[m]] = c[m]
                out.append(a)
            data[(arm, d, o)] = tuple(out)
    return data


def ci(vals, boot, rng):
    """Median and a 95% percentile bootstrap interval over landscapes."""
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    idx = rng.integers(0, a.size, size=(boot, a.size))
    meds = np.median(a[idx], axis=1)
    return (float(np.median(a)), float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)), int(a.size))


def sibling_groups(arms, knob):
    """Arms bucketed by every knob *except* `knob`.

    A bucket with one arm contributes no pair, which is exactly how the
    `force_weight=0` corner drops out of the attraction-model contrast: with
    the field off all four models are bit-identical, so that corner carries a
    single model and has no sibling to be paired against.
    """
    rest = [k for k in KNOB_KEYS if k != knob]
    groups = collections.defaultdict(dict)
    for arm, tok in arms.items():
        groups[tuple(tok[k] for k in rest)][tok[knob]] = arm
    return [g for g in groups.values() if len(g) > 1]


def contrast_raw(data, arms, knob, opt, d, ref=None):
    """Per-landscape paired delta per level of `knob`, against `ref`.

    Returns `{level_token: array[N_LAND]}`. Each landscape contributes one
    number -- the mean delta over its sibling pairs -- which is what makes the
    landscape, not the pair, the unit everything downstream resamples over.
    """
    tokens = list(KNOBS[knob][1])
    ref = ref or tokens[0]
    others = [t for t in tokens if t != ref]
    acc = {t: (np.zeros(N_LAND), np.zeros(N_LAND)) for t in others}
    for g in sibling_groups(arms, knob):
        if ref not in g:
            continue
        base = data.get((g[ref], d, opt))
        if base is None:
            continue
        for t in others:
            if t not in g:
                continue
            cur = data.get((g[t], d, opt))
            if cur is None:
                continue
            delta = cur[0] - base[0]
            ok = np.isfinite(delta)
            tot, cnt = acc[t]
            tot[ok] += delta[ok]
            cnt[ok] += 1
    out = {}
    for t in others:
        tot, cnt = acc[t]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out


def contrast(data, arms, knob, opt, d, boot, rng):
    """`contrast_raw` reduced to `{level: (median, lo, hi, n_landscapes)}`."""
    tokens = list(KNOBS[knob][1])
    raw = contrast_raw(data, arms, knob, opt, d)
    return tokens[0], {t: ci(v, boot, rng) for t, v in raw.items()}


def pooled(data, arm, opt, dims=DIMS, col=0):
    """Every landscape of `arm` for `opt`, concatenated across dimensions."""
    parts = [data[(arm, d, opt)][col] for d in dims if (arm, d, opt) in data]
    return np.concatenate(parts) if parts else np.array([])


def fmt(v, star=""):
    return "     --" if not np.isfinite(v) else f"{v:+7.3f}{star}"


def report(data, boot):
    rng = np.random.default_rng(0)
    arms = {}
    for (arm, _, _) in data:
        if arm not in arms:
            tok = parse_arm(arm)
            if tok:
                arms[arm] = tok
    have = {(a, d) for (a, d, _) in data if a in arms}
    print(__doc__)
    print(f"arms cached: {len(arms)} of 468   "
          f"(arm, dimension) cells: {len(have)} of {468 * len(DIMS)}")
    print("negative dlog10 is better; a delta marked * has a 95% bootstrap "
          "interval clear of 0.")
    print("a pooled delta marked ! is missing a dimension the reference has, "
          "so it averages over an easier subset.\n")

    for opt in OPTS:
        print("=" * 100)
        print(f"KNOB CONTRASTS -- optimizer: {opt}   "
              "(paired dlog10 delta vs the reference level)")
        print("=" * 100)
        print(f"{'knob':<16}{'level':<12}"
              + "".join(f"{'d=' + str(d):>13}" for d in DIMS)
              + f"{'pooled':>13}")
        print("-" * 100)
        for knob in KNOB_KEYS:
            pretty, levels = KNOBS[knob]
            ref = next(iter(levels))
            print(f"{pretty:<16}{levels[ref] + ' (ref)':<12}"
                  + "".join(f"{'.':>13}" for _ in DIMS) + f"{'.':>13}")
            per_d = {}
            for d in DIMS:
                _, res = contrast(data, arms, knob, opt, d, boot, rng)
                per_d[d] = res
            # Pooled: the same contrast run on every dimension's landscapes
            # at once, which is the number Stage B selects on.
            pool_res = {}
            for t in list(levels)[1:]:
                vals = []
                for d in DIMS:
                    m, lo, hi, n = per_d[d][t]
                    if n:
                        vals.append((m, n))
                pool_res[t] = ((np.average([v for v, _ in vals],
                                           weights=[n for _, n in vals])
                                if vals else float("nan")),
                               sum(n for _, n in vals))
            for t in list(levels)[1:]:
                cells = []
                for d in DIMS:
                    m, lo, hi, n = per_d[d][t]
                    sig = n and np.isfinite(m) and (lo < 0) == (hi < 0)
                    cells.append(fmt(m, "*" if sig else " "))
                pm, _ = pool_res[t]
                # `!` marks a level that is missing a dimension the reference
                # has: its pooled number is an average over an easier subset.
                cover = "!" if any(per_d[d][t][3] == 0 for d in DIMS) else " "
                print(f"{'':<16}{levels[t]:<12}"
                      + "".join(f"{c:>13}" for c in cells)
                      + f"{fmt(pm, cover):>13}")
        print()

        # The interaction the sweep exists to settle: does the best attraction
        # weight move with the model? If it does not, `att_model` is a knob
        # with no decision behind it and the parameter goes.
        print(f"{'w x model (dlog10 vs random, pooled over dims)':<100}")
        print(f"{'':<16}" + "".join(f"{m:>13}" for m in KNOBS['m'][1].values()))
        for wt, wl in KNOBS["w"][1].items():
            if wt == "w000":
                continue
            cells = []
            for mt in KNOBS["m"][1]:
                vals = [pooled(data, a, opt) for a, tok in arms.items()
                        if tok["w"] == wt and tok["m"] == mt]
                vals = np.concatenate(vals) if vals else np.array([])
                vals = vals[np.isfinite(vals)]
                cells.append(fmt(np.median(vals)) if vals.size else "     --")
            print(f"{'w=' + wl:<16}" + "".join(f"{c:>13}" for c in cells))
        print()

        # Selection: the arm Stage B carries forward, and the controls that
        # decide whether carrying it forward means anything.
        # Only arms measured on *every* dimension may be ranked together.
        # The sweep finishes a task at a time, so a half-done arm carries its
        # easy dimensions and not its hard ones, and pooling that against a
        # complete arm reads as a win that is entirely missing d=100.
        full = [d for d in DIMS if (BASE, d, opt) in data]
        rows, thin = [], 0
        for arm in list(arms) + CONTROLS:
            if any((arm, d, opt) not in data for d in full):
                thin += 1
                continue
            v = pooled(data, arm, opt, dims=full)
            v = v[np.isfinite(v)]
            if v.size < len(full) * N_LAND:
                thin += 1
                continue
            a = pooled(data, arm, opt, dims=full, col=1)
            rows.append((float(np.median(v)), arm, v.size,
                         float(np.nanmedian(a)) if a.size else float("nan")))
        rows.sort()
        print(f"TOP ARMS -- {opt}   (dims {full}; "
              f"{thin} arms excluded for incomplete coverage)")
        print(f"{'#':<4}{'arm':<34}{'dlog10':>10}{'AR%':>9}{'n':>9}")
        print("-" * 66)
        for i, (m, arm, n, ar) in enumerate(rows[:12], 1):
            print(f"{i:<4}{arm:<34}{m:>10.3f}{ar:>9.1f}{n:>9}")
        ctrl = [r for r in rows if r[1] in CONTROLS]
        if ctrl:
            print("  -- controls --")
            for m, arm, n, ar in sorted(ctrl):
                rank = rows.index((m, arm, n, ar)) + 1
                print(f"{rank:<4}{arm:<34}{m:>10.3f}{ar:>9.1f}{n:>9}")
        print()
        matched(data, arms, opt)
        verdict(data, arms, opt)
        cost(data, arms, opt)
        robustness(data, arms, opt)



def robustness(data, arms, opt):
    """Tail behaviour of the attraction-model choice, not just its median.

    A median contrast of -0.014 is compatible with a model that is slightly
    better almost always and catastrophic occasionally, and for `detrended`
    that is the failure worth fearing rather than a hypothetical.
    `Detrended` is `HarmonicRidge` plus `InverseDistance` on the residuals,
    and `attraction.py` states the ridge's own condition outright: it is
    "dangerous below `M > 2 * d`". Two regimes follow from that, and only one
    of them is safe.

    Far below `M = 2d` the penalty shrinks the coefficients to zero, the trend
    degrades to the global mean, and -- because a constant washes out of
    `_rank_normalise` -- `detrended` collapses back onto plain `idw`. That is
    the graceful corner.

    The hazard is the *band just above* identifiability, where the fit is
    weakly determined but no longer shrunk: the coefficients are noise, and
    unlike `idw` -- a convex combination that provably cannot leave the
    measured range -- a harmonic trend is unbounded and can assert a pull
    toward a value never observed. So this reports the worst decile and
    percentile of the paired delta, which is where that would show.

    `rounds` decides which regime a run is in, because each round adds
    measured anchors. At `rounds=1` a user gets `M = 2 * n_pop` anchors
    against `2d` unknowns, so the ratio, not the dimension, is the axis.
    """
    print(f"MODEL ROBUSTNESS -- {opt}   (paired dlog10 delta vs idw; "
          "negative is better)")
    # Both tails, because a fat upper tail alone proves nothing: symmetric
    # noise would produce one. What would indict a model is asymmetry -- it
    # loses more when it loses than it gains when it gains.
    print(f"{'rounds':<8}{'model':<12}{'d':>5}{'M/2d':>7}"
          + f"{'best':>9}{'p01':>9}{'p10':>9}{'p50':>9}"
          + f"{'p90':>9}{'p99':>9}{'worst':>9}{'skew':>8}")
    print("-" * 101)
    for rtok in KNOBS["r"][1]:
        sub = {a: t for a, t in arms.items() if t["r"] == rtok}
        if not sub:
            continue
        for mtok, mname in KNOBS["m"][1].items():
            if mtok == "midw":
                continue
            for d in DIMS:
                raw = contrast_raw(data, sub, "m", opt, d, ref="midw")
                v = raw.get(mtok)
                if v is None:
                    continue
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                # Anchors the field is fitted from: the measured Random+OBL
                # set, one block per round, against the ridge's 2d unknowns.
                n_pop = int(np.ceil(10 * np.sqrt(d)))
                ratio = (2 * n_pop * int(KNOBS["r"][1][rtok])) / (2 * d)
                q = {k: float(np.percentile(v, k)) for k in (1, 10, 50, 90, 99)}
                # Positive skew = the losses outweigh the wins at matched
                # rarity, which is the shape that disqualifies a default.
                skew = (q[99] + q[1]) / 2.0
                print(f"{KNOBS['r'][1][rtok]:<8}{mname:<12}{d:>5}{ratio:>7.2f}"
                      + f"{v.min():>9.3f}{q[1]:>9.3f}{q[10]:>9.3f}"
                      + f"{q[50]:>9.3f}{q[90]:>9.3f}{q[99]:>9.3f}"
                      + f"{v.max():>9.3f}{skew:>+8.3f}")
    print()


def _rounds_cell(data, arms, opt, tok):
    """Median init seconds, generations forgone and dlog10 at `rounds=tok`.

    `force_weight=0` arms are excluded: with the field off the probe block is
    pure novelty, so they measure the repulsion's cost rather than the
    stage's, and they are the cheapest arms at every round count.
    """
    sel = [a for a, t in arms.items()
           if t["r"] == tok and t["w"] != "w000"]
    secs, gens, dls = [], [], []
    for a in sel:
        for d in DIMS:
            if (a, d, opt) not in data or (BASE, d, opt) not in data:
                continue
            cur, base = data[(a, d, opt)], data[(BASE, d, opt)]
            secs.append(cur[COL["secs"]])
            gens.append(base[COL["nit"]] - cur[COL["nit"]])
            dls.append(cur[COL["dl"]])
    if not secs:
        return None
    return tuple(float(np.nanmedian(np.concatenate(v)))
                 for v in (secs, gens, dls))


def cost(data, arms, opt):
    """What `rounds` charges, in the two currencies that are not the same.

    Evaluations are already charged: `iters_for` sets
    `n_iter = (budget - init_cost) // n_pop`, so a larger pool is paid for in
    generations the optimizer does not get to run, and every dlog10 in this
    report is therefore *net* of it. Wall-clock is charged nowhere, and
    `rounds` is where it goes -- each one is a separate relaxation and a
    separate field fit, which cannot overlap.

    `rounds` is the only knob costed here because it is the only one that
    spends anything. The attraction model was the other candidate and it no
    longer exists: the factorial settled it on inverse-distance weighting,
    which has no fit to pay for.
    """
    print(f"COST -- {opt}   (median over landscapes and dimensions)")
    print(f"{'rounds':<10}{'init s':>10}{'x r=1':>9}"
          + f"{'gens lost':>11}{'dlog10':>9}")
    print("-" * 49)
    ref = None
    for tok, name in KNOBS["r"][1].items():
        cell = _rounds_cell(data, arms, opt, tok)
        if cell is None:
            continue
        secs, gens, dl = cell
        ref = secs if ref is None else ref
        print(f"{name:<10}{secs:>10.3f}{secs / ref:>9.2f}"
              + f"{gens:>11.0f}{dl:>9.3f}")
    print()


def verdict(data, arms, opt):
    """Does OBLESA pay for itself at the settings a user will actually run?

    The tuned arm is not the honest number: it assumes someone swept six
    knobs on their own problem, which nobody does. So this reads the untuned
    corner -- `rounds=1`, `diversity_weight=0`, stock opposition -- against
    the two controls that share its evaluation budget. `random4x` spends the
    same 4N on candidates with no structure at all; `obl2x` spends it on
    opposition without the empty-space stage. If OBLESA cannot beat those
    from its default corner, the empty-space search is not what is working.
    """
    full = [d for d in DIMS if (BASE, d, opt) in data]

    def med(arm, col="dl"):
        v = [data[(arm, d, opt)][COL[col]] for d in full
             if (arm, d, opt) in data]
        if len(v) < len(full):
            return float("nan")
        v = np.concatenate(v)
        return float(np.nanmedian(v))

    print(f"VERDICT -- {opt}   (dims {full})")
    print(f"{'configuration':<40}{'dlog10':>10}{'AR%':>9}{'init s':>10}")
    print("-" * 69)
    rows = [
        ("untuned corner  w=0.5 r=1 s=0 idw", "f8_w050_midw_r1_s00_os_e0"),
        ("stock-ish       w=0.5 r=1 s=0 auto", "f8_w050_maut_r1_s00_os_e0"),
        ("weight only     w=1.0 r=1 s=0 idw", "f8_w100_midw_r1_s00_os_e0"),
        ("weight+rounds   w=1.0 r=3 s=0 idw", "f8_w100_midw_r3_s00_os_e0"),
        ("pure novelty    w=0   r=1 s=0", "f8_w000_midw_r1_s00_os_e0"),
    ]
    for label, arm in rows:
        if arm not in arms:
            continue
        print(f"{label:<40}{med(arm):>10.3f}{med(arm, 'ar'):>9.1f}"
              + f"{med(arm, 'secs'):>10.3f}")
    print("  -- budget-matched controls --")
    for arm in ("random4x", "obl2x", "qobl", "obl"):
        if (arm, full[0], opt) in data:
            print(f"{arm:<40}{med(arm):>10.3f}{med(arm, 'ar'):>9.1f}"
                  + f"{med(arm, 'secs'):>10.3f}")
    print()


def _pooled_median(data, arm, opt, dims, col):
    """Median over every landscape of `arm`, or None if a dimension is absent.

    Absent rather than sparse: an arm measured on four dimensions out of five
    is missing the hardest one, and pooling it against a complete arm reads as
    a win that is entirely the missing cell.
    """
    if any((arm, d, opt) not in data for d in dims):
        return None
    v = np.concatenate([data[(arm, d, opt)][COL[col]] for d in dims])
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def matched(data, arms, opt):
    """Best arm at each pool size, against the controls that spend the same.

    This is the comparison the whole sweep exists to make. `obl` and `qobl`
    are not it -- OBLESA's pool is a strict superset of theirs, so beating
    them is arithmetic. Beating `random_kx` at the *same* `k` is not: it draws
    the same number of candidates and keeps the best, with no empty-space
    stage and no opposition, so a margin over it is what the placement bought.
    """
    dims = [d for d in DIMS if (BASE, d, opt) in data]
    print(f"MATCHED-POOL COMPARISON -- {opt}   (dims {dims})")
    print(f"{'pool':<6}{'best arm':<30}{'dlog10':>9}{'AR%':>7}   "
          f"{'control':<11}{'dlog10':>9}{'AR%':>7}{'margin':>9}")
    print("-" * 90)
    for mult, controls in sorted(POOL_CONTROLS.items()):
        ranked = []
        for arm, tok in arms.items():
            if pool_multiple(tok) != mult:
                continue
            dl = _pooled_median(data, arm, opt, dims, "dl")
            if dl is not None:
                ranked.append((dl, arm))
        if not ranked:
            continue
        ranked.sort()
        best_dl, best = ranked[0]
        best_ar = _pooled_median(data, best, opt, dims, "ar")

        # The stronger of the two controls, so the margin is never flattered
        # by picking whichever happened to do worse.
        scored = [(c, _pooled_median(data, c, opt, dims, "dl")) for c in controls]
        scored = [(c, v) for c, v in scored if v is not None]
        if not scored:
            print(f"{mult}N{'':<4}{best:<30}{best_dl:>9.3f}{best_ar:>7.1f}   "
                  f"{'(not run)':<11}{'--':>9}{'--':>7}{'--':>9}")
            continue
        cname, cdl = min(scored, key=lambda p: p[1])
        car = _pooled_median(data, cname, opt, dims, "ar")
        print(f"{mult}N{'':<4}{best:<30}{best_dl:>9.3f}{best_ar:>7.1f}   "
              f"{cname:<11}{cdl:>9.3f}{car:>7.1f}{best_dl - cdl:>9.3f}")
    print()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir",
                    default=os.path.join(here, "out", "sweep_v8_a_root"))
    ap.add_argument("--cache-dir", default=None,
                    help="default: <sweep-dir>_cache")
    ap.add_argument("--build", action="store_true",
                    help="parse JSONL into the npz cache, then stop")
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    cache = args.cache_dir or args.sweep_dir.rstrip("/") + "_cache"
    if args.build:
        tally = build(args.sweep_dir, cache, args.procs)
        print("  ".join(f"{v} {k}" for k, v in sorted(tally.items())))
        return
    if not os.path.isdir(cache):
        raise SystemExit(f"no cache at {cache!r}; run --build first")
    report(load(cache), args.boot)


if __name__ == "__main__":
    main()
