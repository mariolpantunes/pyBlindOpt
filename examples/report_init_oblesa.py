"""Render `bench_init_oblesa.py`'s JSON as a standalone HTML report.

Reads ``out/bench_np30.json`` and writes ``out/bench_init_oblesa.html`` — one
self-contained file, no external assets, readable in light or dark.

The report is built around one structural fact: OBLESA selects greedily from
OBL's candidate pool *plus* the ESS batch, so it cannot be worse than OBL on
the population it starts from. Any measurement showing otherwise is a defect
in the harness, and this report leads with that check because three separate
versions of this benchmark failed it for three different reasons.

Usage::

    python examples/report_init_oblesa.py
    python examples/report_init_oblesa.py --json out/bench_np30.json
    python examples/report_init_oblesa.py --sweep-dir out/sweep

The third form is the Slurm array output. `--arm-index` gives every arm its own
JSONL file so a 344,000-run sweep resumes for free and no two tasks contend for
one handle, but nothing then reassembled them -- the report only ever read the
single JSON that a serial run produces. `--sweep-dir` is that missing step: it
concatenates the per-arm files into the same row list `--json` would have
supplied, so both paths reach an identical report.
"""

import argparse
import glob
import html
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "out")
# Grouped, because the grouping is the argument: the cost controls spend
# OBLESA's candidates without any empty-space search, so they are what decides
# whether the search earns its cost or merely enlarges the selection pool.
# `oblesa-rand` is the sharpest of them -- OBLESA's exact pipeline with the
# engine swapped for uniform noise, so pool *shape* is held fixed too.
GROUPS = (
    ("baselines (N calls)", ("random", "sobol")),
    ("cost controls, 3N candidates", ("random3x", "obl15x")),
    ("cost controls, 4N candidates", ("random4x", "obl2x")),
    ("opposition (2N calls)", ("obl", "qobl")),
    ("OBLESA (3N / 4N pool)",
     ("oblesa", "oblesa-quasi")),
    ("OBLESA, selection knobs",
     ("oblesa-div25", "oblesa-div50", "oblesa-mm25", "oblesa-mm50",
      "oblesa-quasi-mm25", "oblesa-quasi-mm50")),
    ("uniform-noise null, matched settings",
     ("oblesa-rand", "oblesa-quasi-rand")),
)
ARMS = tuple(a for _, g in GROUPS for a in g)
BASELINE = "random"
#: Reference arms for the invariant check and the verdict. Resolved against
#: the data in `main`, because the sweep's arm names are generated from a
#: factorial rather than hand-written, and set to None when absent.
OBLESA_REF = "oblesa"
OPP_REF = "obl"
CTRL_ARMS = ("random4x", "obl2x")
ARM_GLOSS = {
    "random": "Uniform random sampling. The reference every acceleration "
              "rate is measured against.",
    "lhs": "Latin hypercube: stratified per dimension, no fitness used.",
    "sobol": "Sobol low-discrepancy sequence. Quasi-random, no fitness used.",
    "random3x": "3N uniform random candidates, keep the best N. Spends "
                "exactly what a 3N-pool OBLESA spends, with no structure.",
    "obl15x": "1.5N random plus their 1.5N opposites, keep the best N: the "
              "3N-pool cost control with opposition but no empty-space stage.",
    "random4x": "4N uniform random candidates, keep the best N. The cost "
                "control for the 4N-pool OBLESA arms.",
    "obl2x": "2N random plus their 2N opposites, keep the best N. OBLESA's "
             "budget and OBLESA's opposition, without the empty-space stage.",
    "obl": "Opposition-based learning: N sampled, N reflected, best N of 2N.",
    "qobl": "Quasi-opposition: the opposite is drawn between the centre and "
            "the reflected point rather than at it.",
    "oblesa": "OBL plus Empty-Space Search, stock defaults. ESS adds 2N "
              "points in under-explored regions of the Random+OBL set; best "
              "N of 4N are kept on fitness alone.",
    "oblesa-quasi": "OBLESA with quasi-opposition.",
    "oblesa-div25": "OBLESA selecting on fitness blended with crowding "
                    "distance at weight 0.25.",
    "oblesa-div50": "The same at weight 0.50 — half fitness, half spread.",
    "oblesa-mm25": "OBLESA selecting by sequential maximin over the fittest "
                   "1.75x n_pop: each point chosen is the one furthest from "
                   "everything already selected.",
    "oblesa-mm50": "The same, spreading over the fittest 2.5x n_pop.",
    "oblesa-quasi-mm25": "Quasi-opposition with maximin selection at 1.75x.",
    "oblesa-quasi-mm50": "Quasi-opposition with maximin selection at 2.5x.",
    "oblesa-rand": "OBLESA's pipeline with the empty-space engine replaced by "
                   "uniform noise. Identical pool shape, candidate count and "
                   "selection rule, so any OBLESA margin over this arm belongs "
                   "to the empty-space search and to nothing else.",
    "oblesa-quasi-rand": "Quasi-opposition null at the 3N pool size.",
}


#: Engine label -> what placed the empty-space block, for arms named by the
#: sweep convention `ob_<engine>_<selection>@<optimizer>`.
#:
#: Only labels that are *not* a rung of the attraction ladder need a table
#: entry. A rung is `a<weight x100>` with an optional attractiveness-model
#: suffix, and is described from its own number, so adding or moving a rung
#: does not mean editing a dictionary here.
_ENGINE_GLOSS = {
    "null": "uniform noise in place of any empty-space search — the control "
            "that separates “the search found something” from "
            "“a bigger pool gave the selector more to choose from”",
}
_SEL_GLOSS = {"s00": "greedy on fitness", "s25": "fitness blended with "
              "crowding distance at 0.25", "s50": "blended at 0.50"}
#: Short form of the same, for section headings.
_ENGINE_SHORT = {"null": "uniform-noise control"}
#: Attractiveness-model suffix on a ladder rung -> how ESS estimates the
#: value of a position it never evaluated.
_MODEL_GLOSS = {
    "f": ("Fourier", ("a periodic function of position fitted to the whole "
                     "measured set")),
    "i": ("IDW", "a distance weighting of the nearest measured points"),
    "d": ("detrended", ("a fitted trend plus a distance weighting of what it "
                       "leaves behind")),
    "a": ("auto", "whichever of the above cross-validates best on the pool"),
}


def _rung_of(engine):
    """`"a050f"` -> `(0.5, "f")`, the attraction weight and model. None if not a rung."""
    if not engine.startswith("a"):
        return None
    body = engine[1:]
    suffix = ""
    if body and body[-1].isalpha():
        body, suffix = body[:-1], body[-1]
    return (float(body) / 100.0, suffix) if body.isdigit() else None


def engine_gloss(engine, short=False):
    """Describe an engine label, from the table or from its own rung."""
    rung = _rung_of(engine)
    if rung is None:
        table = _ENGINE_SHORT if short else _ENGINE_GLOSS
        return table.get(engine, engine)
    weight, suffix = rung
    if weight == 0:
        return ("ESS, novelty only" if short else
                "the EmptySpaceSearch relaxation with the attraction term "
                "switched off — the ablation end of the ladder, where probes "
                "are placed on novelty alone")
    name, how = _MODEL_GLOSS.get(suffix, ("", ""))
    if short:
        return f"ESS, attraction {weight:g}" + (f", {name}" if name else "")
    return (f"the EmptySpaceSearch relaxation at attraction weight "
            f"{weight:g}, estimating the attractiveness of unmeasured "
            f"positions with {how}" if name else
            f"the EmptySpaceSearch relaxation at attraction weight {weight:g}")


def parse_arm(a):
    """`ob_<engine>_<sel>@<opt>` -> its parts, or None if not one.

    Also accepts the earlier `ob_<engine>_<n_ess>_<sel>` shape, so a report
    can still be rendered against the sweep that used it.
    """
    init, _, opt = a.partition("@")
    parts = init.split("_")
    if not init.startswith("ob_") or len(parts) not in (3, 4):
        return None
    n_ess = parts[2] if len(parts) == 4 else "n2"
    return {"engine": parts[1], "n_ess": n_ess, "sel": parts[-1],
            "opt": opt, "init": init}


def gloss(a):
    """Description for an arm, from the table or built from its name.

    The hardcoded table covers the hand-named baselines. The OBLESA arms are
    generated by a factorial in the benchmark, so describing them is done from
    the name rather than by keeping a parallel dictionary in step with a grid
    that changes every time a knob is added.
    """
    init = a.partition("@")[0]
    if a in ARM_GLOSS:
        return ARM_GLOSS[a]
    if init in ARM_GLOSS:
        return ARM_GLOSS[init]
    p = parse_arm(a)
    if not p:
        return f"{init}, driven by {a.partition('@')[2] or 'DE'}."
    mult = {"n1": "1", "n2": "2"}.get(p["n_ess"], p["n_ess"])
    return (f"OBLESA with {engine_gloss(p['engine'])}; "
            f"empty-space block of {mult}N; selection "
            f"{_SEL_GLOSS.get(p['sel'], p['sel'])}; optimizer {p['opt']}.")


def derive_groups(arms):
    """Group whatever arms the data contains, rather than a fixed list.

    `GROUPS` names the arms of the sweep it was written for, and `organise`
    keeps only cells present for *every* arm in it, so a report run against a
    later grid silently finds no complete cell and renders nothing. Grouping
    what is actually in the file removes that coupling.
    """
    known = {a for _, g in GROUPS for a in g}
    if set(arms) <= known:
        return GROUPS
    base, byeng = [], {}
    for a in sorted(arms):
        p = parse_arm(a)
        if p is None:
            base.append(a)
        else:
            byeng.setdefault(p["engine"], []).append(a)
    out = []
    if base:
        out.append(("baselines and cost controls", tuple(base)))
    for eng in sorted(byeng):
        out.append((f"OBLESA, engine ‘{eng}’ ({engine_gloss(eng, short=True)})",
                    tuple(byeng[eng])))
    return tuple(out)


def vtr_for_cell(base_rows):
    """Value-to-reach for one cell: the median final value of the baseline.

    A fixed absolute tolerance cannot work across functions spanning 1e-83
    to 1e3 — the same alpha is unreachable for rastrigin at d=32 and passed
    within two generations by sphere at d=2, censoring the hard cells and
    saturating the easy ones. Targeting what the baseline actually achieves
    gives acceleration a stable meaning in every cell, and guarantees that
    roughly half the baseline runs reach it.
    """
    return float(np.median([r["curve"][-1] if r["curve"] else r["score"]
                            for r in base_rows]))


def wilcoxon_signed_rank(a, b):
    """Paired Wilcoxon signed-rank, normal approximation with tie correction."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    diff = diff[diff != 0.0]
    n = diff.size
    if n < 1:
        return 1.0
    order = np.argsort(np.abs(diff))
    ranks = np.empty(n, float)
    absd = np.abs(diff)[order]
    i = 0
    while i < n:
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
    return float(np.exp(-0.717 * z - 0.416 * z * z))


def to_target(row, n_pop, budget, vtr):
    """(iterations, function calls, reached) to first hit `vtr`."""
    for g, v in enumerate(row["curve"]):
        if v <= vtr:
            return g + 1, row["init_evals"] + (g + 1) * n_pop, True
    return len(row["curve"]), budget, False


def organise(rows):
    cells, by = {}, {}
    for r in rows:
        by.setdefault((r["function"], r["d"], r["arm"]), []).append(r)
        cells.setdefault((r["function"], r["d"]), set()).add(r["arm"])
    full = sorted(k for k, v in cells.items() if set(ARMS) <= v)
    for group in by.values():
        group.sort(key=lambda r: r["seed"])
    return full, by


def esc(s):
    return html.escape(str(s))


def fmt(v, digits=4):
    if v == 0:
        return "0"
    if abs(v) < 1e-3 or abs(v) >= 1e5:
        return f"{v:.2e}"
    return f"{v:.{digits}g}"


CSS = """
:root{ color-scheme: light;
  --ground:#f7f8fa; --card:#fff; --ink:#10141b; --ink-2:#545c6b;
  --muted:#7b8394; --rule:#dfe3ea; --accent:#2a78d6; --good:#1f8a54;
  --bad:#e34948; --track:#eceff4; }
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme: dark;
  --ground:#0e1116; --card:#161b22; --ink:#eef1f6; --ink-2:#aab3c0;
  --muted:#7d8797; --rule:#262d38; --accent:#3987e5; --good:#3fb27a;
  --bad:#e66767; --track:#1e242d; } }
:root[data-theme="dark"]{ color-scheme: dark;
  --ground:#0e1116; --card:#161b22; --ink:#eef1f6; --ink-2:#aab3c0;
  --muted:#7d8797; --rule:#262d38; --accent:#3987e5; --good:#3fb27a;
  --bad:#e66767; --track:#1e242d; }
*{box-sizing:border-box}
body{margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.62; -webkit-font-smoothing:antialiased;}
.wrap{max-width:1080px; margin:0 auto;
  padding:clamp(28px,5vw,72px) clamp(18px,4vw,40px) 96px;
  display:flex; flex-direction:column; gap:44px;}
.prose{max-width:68ch; display:flex; flex-direction:column; gap:16px;}
h1{font-family:ui-serif,Georgia,"Times New Roman",serif; font-weight:600;
  font-size:clamp(30px,4.4vw,46px); line-height:1.12;
  letter-spacing:-0.015em; margin:0; text-wrap:balance;}
h2{font-family:ui-serif,Georgia,serif; font-weight:600;
  font-size:clamp(21px,2.6vw,27px); letter-spacing:-0.01em; margin:0;
  text-wrap:balance;}
h3{font-size:15px; font-weight:650; margin:0;}
p{margin:0; color:var(--ink-2);} .prose p{max-width:68ch}
strong{color:var(--ink); font-weight:640}
.eyebrow{font-family:ui-monospace,Menlo,monospace; font-size:11.5px;
  letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0;}
.sub{font-size:17.5px; color:var(--ink-2); max-width:64ch; margin:0}
.rule{height:1px; background:var(--rule); border:0; margin:0}
section{display:flex; flex-direction:column; gap:20px}
.verdict{background:var(--card); border:1px solid var(--rule);
  border-left:3px solid var(--accent); border-radius:3px; padding:24px 26px;
  display:flex; flex-direction:column; gap:12px; max-width:72ch;}
.verdict p{color:var(--ink)}
.scroll{overflow-x:auto; -webkit-overflow-scrolling:touch;
  border:1px solid var(--rule); border-radius:3px; background:var(--card);}
table{border-collapse:collapse; width:100%; font-size:13px;}
th,td{padding:7px 12px; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--rule);
  font-family:ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums;}
th{font-family:system-ui,sans-serif; font-size:11.5px; font-weight:600;
  color:var(--muted); text-transform:uppercase; letter-spacing:.06em;
  position:sticky; top:0; background:var(--card);}
td.l,th.l{text-align:left}
tbody tr:last-child td{border-bottom:0}
tr.tot td{font-weight:700; color:var(--ink); border-top:2px solid var(--rule);}
.good{color:var(--good); font-weight:650}
.bad{color:var(--bad); font-weight:650}
.dim{color:var(--muted)}
.cards{display:grid; gap:14px;
  grid-template-columns:repeat(auto-fit,minmax(210px,1fr));}
.card{background:var(--card); border:1px solid var(--rule); border-radius:3px;
  padding:16px 18px; display:flex; flex-direction:column; gap:5px;}
.card .k{font-family:ui-monospace,Menlo,monospace; font-size:26px;
  font-weight:640; color:var(--ink); letter-spacing:-.02em;}
.card .n{font-size:12px; color:var(--muted); line-height:1.45}
ul{margin:0; padding-left:1.15em; color:var(--ink-2);
  display:flex; flex-direction:column; gap:9px; max-width:68ch}
li::marker{color:var(--muted)}
.foot{color:var(--muted); font-size:13px; max-width:68ch}
"""


def build(data, path):
    cfg, rows = data["config"], data["rows"]
    n_pop = cfg["n_pop"][0] if isinstance(cfg["n_pop"], list) else cfg["n_pop"]
    budget = cfg["iters"] * n_pop
    seeds = cfg["seeds"]
    cells, by = organise(rows)
    dims = sorted({d for _, d in cells})
    gen = min(len(r["curve"]) for r in rows) - 1

    # ---- what initialization actually costs in seconds -------------------
    # Acceleration is counted in iterations, so it charges nothing for the
    # time spent building the population -- and the engines differ by more
    # than 3x there. Reported beside it rather than instead of it: for the
    # live-agent case, where one evaluation costs seconds, initialization is
    # free and iterations are the right unit. On a cheap analytic benchmark
    # it is most of the wall clock, and AR alone would flatter the slowest
    # engine most.
    init_s = {a: [] for a in ARMS}
    for r in rows:
        if r["arm"] in init_s:
            init_s[r["arm"]].append(float(r.get("init_seconds", 0.0)))
    init_med = {a: (float(np.median(v)) if v else 0.0)
                for a, v in init_s.items()}
    init_by_dim = {d: {} for d in dims}
    for d in dims:
        for a in ARMS:
            v = [float(r.get("init_seconds", 0.0))
                 for (f, dd) in cells if dd == d for r in by[(f, d, a)]]
            init_by_dim[d][a] = float(np.median(v)) if v else 0.0

    # ---- speed: acceleration to the per-cell target ---------------------
    agg = {a: {"it": 0, "nfc": 0, "ok": 0, "n": 0} for a in ARMS}
    by_dim = {d: {a: {"it": 0, "nfc": 0} for a in ARMS} for d in dims}
    per_cell = {}
    for (f, d) in cells:
        vtr = vtr_for_cell(by[(f, d, BASELINE)])
        st = {}
        for a in ARMS:
            trip = [to_target(r, n_pop, budget, vtr) for r in by[(f, d, a)]]
            st[a] = (sum(x[0] for x in trip), sum(x[1] for x in trip),
                     sum(x[2] for x in trip), len(trip))
            agg[a]["it"] += st[a][0]; agg[a]["nfc"] += st[a][1]
            agg[a]["ok"] += st[a][2]; agg[a]["n"] += st[a][3]
            by_dim[d][a]["it"] += st[a][0]; by_dim[d][a]["nfc"] += st[a][1]
        per_cell[(f, d)] = st

    def ar(arm, field, table=None):
        tb = table or agg
        base = tb[BASELINE][field]
        return (1 - tb[arm][field] / base) * 100 if base else float("nan")

    # ---- quality at equal generations -----------------------------------
    def at(f, d, a):
        return np.array([r["curve"][min(gen, len(r["curve"]) - 1)]
                         for r in by[(f, d, a)]])

    wins = {a: 0 for a in ARMS}          # significantly better than random
    viol = 0
    qual_rows = []
    for (f, d) in cells:
        base = at(f, d, BASELINE)
        row = {}
        for a in ARMS:
            v = at(f, d, a)
            p_ = wilcoxon_signed_rank(v, base)
            better = np.median(v) < np.median(base) and p_ < 0.05
            wins[a] += bool(better)
            row[a] = (float(np.median(v)), p_, better)
        # Invariant: OBLESA must not be significantly worse than the
        # opposition arm whose pool it is a superset of. Skipped when either
        # reference is absent from this data.
        #
        # Note this is a *final-score* check, not the strict population-level
        # one. `min(score(P_oblesa)) <= min(score(P_obl))` holds only while
        # selection is greedy on fitness; the blended arms (s25, s50) trade
        # some fitness for spread by construction, so a strict reading would
        # flag them for doing exactly what they are supposed to do.
        if OBLESA_REF and OPP_REF:
            e, o = at(f, d, OBLESA_REF), at(f, d, OPP_REF)
            if (np.median(e) > np.median(o)
                    and wilcoxon_signed_rank(e, o) < 0.05):
                viol += 1
        qual_rows.append((f, d, row))

    sr = {a: 100 * agg[a]["ok"] / agg[a]["n"] for a in ARMS}
    best_arm = max(ARMS, key=lambda a: ar(a, "it"))

    P = []
    P.append(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OBLESA initialization &mdash; knobs and cost controls</title>
<style>{CSS}</style></head><body><div class="wrap">""")

    P.append(f"""<header class="prose">
<p class="eyebrow">pyBlindOpt &middot; initialization benchmark</p>
<h1>Does empty-space search earn its cost?</h1>
<p class="sub">{len(ARMS)} initialization strategies driving differential
evolution over {len(cells)} function&times;dimension cells,
{seeds} paired seeds each &mdash; {len(rows):,} runs. The question is not
whether OBLESA beats random: it is whether OBLESA beats spending the same
4N function calls <em>without</em> ESS.</p></header><hr class="rule">""")

    present_ctrl = [c for c in CTRL_ARMS if c in ARMS]
    ctrl_txt = ', '.join(f"<code>{esc(c)}</code> {ar(c, 'it'):.2f}%"
                         for c in present_ctrl) or 'none present'
    ctrl = max((ar(c, "it") for c in present_ctrl), default=float("-inf"))
    obe = ar(OBLESA_REF, "it") if OBLESA_REF else float("nan")
    best = ar(best_arm, "it")
    verdict = ("clears" if best > ctrl else "does not clear")
    P.append(f"""<section><h2>Verdict</h2><div class="verdict">
<p><strong>The best OBLESA configuration ({esc(best_arm)}) reaches
{best:.2f}% acceleration; the strongest ESS-free control at the same 4N
budget reaches {ctrl:.2f}%.</strong> On this evidence OBLESA
<strong>{verdict}</strong> its own cost controls. Default OBLESA sits at
{obe:.2f}%, so configuration matters more than the method label.</p>
<p><strong>Controls:</strong> {ctrl_txt} &mdash; each spends exactly what
OBLESA spends, none uses ESS. The gap between the best OBLESA arm and the best
of these is the part attributable to empty-space search rather than to a larger
selection pool.</p>
<p><strong>Structural invariant:</strong> {viol} of {len(cells)} cells where
{esc(OBLESA_REF or 'OBLESA')} is significantly worse than
{esc(OPP_REF or 'OBL')} at equal generations. OBLESA selects
greedily from a superset of OBL's candidates, so a correct harness must show
zero; this check caught three separate defects in earlier versions.</p>
</div></section>""")

    P.append('<section><h2>Headline</h2><div class="cards">')
    for k, n in ((f"{best:.2f}%", f"best OBLESA arm ({best_arm}), acceleration vs random, iterations"),
                 (f"{ctrl:.2f}%", "best ESS-free control at the same 4N budget"),
                 (f"{obe:.2f}%", "OBLESA at stock defaults"),
                 (f"{ar('obl','it'):.2f}%", "plain OBL, 2N calls"),
                 (f"{viol}", "cells where OBLESA is significantly worse than OBL"),
                 (f"{len(rows):,}", f"runs: {len(cells)} cells x {len(ARMS)} arms x {seeds} seeds")):
        P.append(f'<div class="card"><span class="k">{esc(k)}</span>'
                 f'<span class="n">{esc(n)}</span></div>')
    P.append("</div></section>")

    P.append("<section><h2>The arms</h2>")
    for label, group in GROUPS:
        P.append(f"<h3>{esc(label)}</h3><ul>")
        for a in group:
            P.append(f"<li><strong>{esc(a)}</strong> &mdash; "
                     f"{esc(gloss(a))}</li>")
        P.append("</ul>")
    P.append("</section>")

    P.append(f"""<section><h2>Acceleration and success rate</h2>
<p class="prose">Acceleration rate is
<code>(1 &minus; &Sigma;n(arm)/&Sigma;n(random)) &times; 100</code>, positive
meaning fewer than random. The target in each cell is the median final value
the random arm reaches there, so it is attainable everywhere and means the
same thing in every cell. <em>Quality wins</em> counts cells where the arm is
significantly better than random at generation {gen + 1}, which removes every
accounting choice from the comparison.</p>
<p class="prose"><strong>init s</strong> is the median wall-clock time to
build one population. Acceleration is counted in iterations and so charges
nothing for it, which is the right unit when an objective evaluation is
expensive &mdash; but the engines differ by several-fold there, and reading
the two columns together is what separates &ldquo;reaches the target in fewer
iterations&rdquo; from &ldquo;reaches it sooner&rdquo;.</p>
<div class="scroll"><table><thead><tr><th class="l">arm</th>
<th>AR % (iterations)</th><th>AR % (calls)</th><th>init s</th>
<th>success</th><th>quality wins</th></tr></thead><tbody>""")
    slowest = max(init_med.values()) if init_med else 0.0
    for label, group in GROUPS:
        P.append(f'<tr><td class="l dim" colspan="6">{esc(label)}</td></tr>')
        for a in group:
            i_, n_ = ar(a, "it"), ar(a, "nfc")
            c1 = "good" if i_ > 1 else ("bad" if i_ < -1 else "dim")
            c2 = "good" if n_ > 1 else ("bad" if n_ < -1 else "dim")
            t_ = init_med.get(a, 0.0)
            c3 = "bad" if slowest and t_ > 0.5 * slowest else "dim"
            P.append(f'<tr><td class="l">{esc(a)}</td>'
                     f'<td class="{c1}">{i_:+.2f}</td>'
                     f'<td class="{c2}">{n_:+.2f}</td>'
                     f'<td class="{c3}">{t_:.3f}</td>'
                     f'<td>{sr[a]:.1f}%</td>'
                     f'<td class="dim">{wins[a]}/{len(cells)}</td></tr>')
    P.append("</tbody></table></div>")

    # ---- break-even ------------------------------------------------------
    # Initialization is negligible exactly when the optimizer's own evaluation
    # bill dwarfs it. That crossover is a property of the objective, not of
    # the initializer, so it is stated as one number the reader can compare
    # their own problem against.
    evals = n_pop * (gen + 1)
    # Only arms that actually build something. The samplers and the opposition
    # arms initialise in microseconds, so including them makes the "cheapest"
    # end of the range 0.00 ms, which is true and says nothing.
    be = sorted(((t / evals, a) for a, t in init_med.items() if t >= 1e-3),
                reverse=True)
    if be:
        worst_ms, worst_arm = be[0][0] * 1e3, be[0][1]
        best_ms, best_arm = be[-1][0] * 1e3, be[-1][1]
        P.append(f"""<section><h2>When the initialization is free</h2>
<p class="prose">A run here spends {evals:,} objective evaluations after
initialization. So the slowest arm ({esc(worst_arm)}, {init_med[worst_arm]:.3f}s)
stops being more than half the wall clock once one evaluation costs about
<strong>{worst_ms:.2f} ms</strong>; the cheapest engine that builds anything
({esc(best_arm)}, {init_med[best_arm]:.3f}s) needs
<strong>{best_ms:.2f} ms</strong>.</p>
<p class="prose">Above those thresholds the iteration-based acceleration rate
is the honest measure and the initialization is a rounding error &mdash; which
is the case pyBlindOpt is built for, where an evaluation trains an agent.
Below them the ranking by acceleration and the ranking by time apart, and an
arm can win every cell here while finishing last in real time.</p></section>""")
    P.append("</section>")

    P.append("""<section><h2>Does the advantage grow with dimension?</h2>
<p class="prose">The claim motivating the whole index effort is that
empty-space search should matter <em>more</em> as dimension rises, because
that is where random coverage degrades fastest. If ESS is working, the
OBLESA arms should pull away from the cost controls to the right of this
table. Acceleration rate in iterations, by dimension.</p>
<div class="scroll"><table><thead><tr><th class="l">arm</th>""")
    for d in dims:
        P.append(f"<th>d={d}</th>")
    P.append("</tr></thead><tbody>")
    for label, group in GROUPS:
        P.append(f'<tr><td class="l dim" colspan="{len(dims)+1}">{esc(label)}</td></tr>')
        for a in group:
            P.append(f'<tr><td class="l">{esc(a)}</td>')
            for d in dims:
                v = ar(a, "it", by_dim[d])
                c = "good" if v > 1 else ("bad" if v < -1 else "dim")
                P.append(f'<td class="{c}">{v:+.1f}</td>')
            P.append("</tr>")
    P.append("</tbody></table></div>")

    # Cost by dimension, next to acceleration by dimension. The two move in
    # opposite directions for the search-based engines -- they buy the most
    # iterations exactly where they cost the most seconds -- and that trade is
    # invisible unless the columns sit side by side.
    P.append("""<p class="prose">And what each costs to build, in seconds,
at the same dimensions. The search-based engines buy the most iterations
where they also cost the most, so the two tables are read together.</p>
<div class="scroll"><table><thead><tr><th class="l">arm</th>""")
    for d in dims:
        P.append(f"<th>d={d}</th>")
    P.append("</tr></thead><tbody>")
    for label, group in GROUPS:
        P.append(f'<tr><td class="l dim" colspan="{len(dims)+1}">'
                 f'{esc(label)}</td></tr>')
        for a in group:
            P.append(f'<tr><td class="l">{esc(a)}</td>')
            for d in dims:
                t_ = init_by_dim[d].get(a, 0.0)
                worst = max(init_by_dim[d].values()) if init_by_dim[d] else 0.0
                c = "bad" if worst and t_ > 0.5 * worst else "dim"
                P.append(f'<td class="{c}">{t_:.3f}</td>')
            P.append("</tr>")
    P.append("</tbody></table></div></section>")

    P.append(f"""<section><h2>Quality at equal generations</h2>
<p class="prose">Every arm read at generation {gen + 1}; median over
{seeds} paired seeds, lower better. Bold marks a significant improvement on
random (paired Wilcoxon, p&lt;0.05). This is the accounting-free view: the
only difference between arms is the population they started from.</p>
<div class="scroll"><table><thead><tr><th class="l">function</th><th>d</th>""")
    for a in ARMS:
        P.append(f"<th>{esc(a)}</th>")
    P.append("</tr></thead><tbody>")
    for (f, d, row) in qual_rows:
        P.append(f'<tr><td class="l">{esc(f)}</td><td>{d}</td>')
        for a in ARMS:
            med, p_, better = row[a]
            cls = "good" if better else ""
            P.append(f'<td class="{cls}">{fmt(med)}</td>')
        P.append("</tr>")
    P.append("</tbody></table></div></section>")

    P.append(f"""<section><h2>Head to head, on quality</h2>
<p class="prose">Wins and losses against each rival at generation
{gen + 1}, over {len(cells)} cells (paired Wilcoxon, p&lt;0.05). This is the
comparison the acceleration table cannot make: at equal <em>generations</em>
an arm that paid 4N to initialise has spent more calls than one that paid
2N, so reading it beside the per-call acceleration rate is what separates
&ldquo;ESS helps&rdquo; from &ldquo;ESS helps enough to pay for
itself&rdquo;.</p>
<div class="scroll"><table><thead><tr><th class="l">arm</th>""")
    RIVALS = tuple(r_ for r_ in ("random", "random4x", "obl2x", "obl",
                                "qobl") if r_ in ARMS)
    for r_ in RIVALS:
        P.append(f"<th>vs {esc(r_)}</th>")
    P.append("</tr></thead><tbody>")
    for label, group in GROUPS:
        P.append(f'<tr><td class="l dim" colspan="{len(RIVALS)+1}">{esc(label)}</td></tr>')
        for a in group:
            P.append(f'<tr><td class="l">{esc(a)}</td>')
            for r_ in RIVALS:
                if r_ == a:
                    P.append('<td class="dim">&mdash;</td>')
                    continue
                w = l = 0
                for (f, d) in cells:
                    v, o = at(f, d, a), at(f, d, r_)
                    if wilcoxon_signed_rank(v, o) < 0.05:
                        if np.median(v) < np.median(o):
                            w += 1
                        else:
                            l += 1
                cls = "good" if w > l else ("bad" if l > w else "dim")
                P.append(f'<td class="{cls}">{w}W&ndash;{l}L</td>')
            P.append("</tr>")
    P.append("</tbody></table></div></section>")

    P.append("""<section><h2>What this benchmark got wrong, three times</h2>
<p class="prose">Recorded because each defect produced a confident,
publishable-looking table that was noise, and because the same structural
invariant found all three.</p><ul>
<li><strong>Unpaired seeds.</strong> ESS consumes far more randomness than
plain sampling, so passing the optimizer whatever generator state the
initializer left behind gave each arm a different search trajectory.
Re-running six cells with the optimizer seeded identically across arms
<strong>flipped five of the six winners</strong>.</li>
<li><strong>Measuring the endpoint.</strong> The initializer acts on
generation zero and the optimizer washes it out long before the budget ends,
so a final fitness value is mostly optimizer variance. Whole convergence
curves are now recorded and read at matched points.</li>
<li><strong>An absolute stopping tolerance.</strong> A fixed
<code>alpha = 0.1</code> is unreachable for rastrigin at d=32 and passed in
two generations by sphere at d=2 — it censored the hard cells and saturated
the easy ones. The target is now the median final value the baseline
reaches in that same cell.</li>
</ul></section>""")

    P.append(f"""<section><h2>Known limitations</h2><ul>
<li><strong>The index is never exercised.</strong> At n_pop={n_pop}, ESS runs
on {2 * n_pop} points — below torann's brute-force crossover of 512 — so the
LSH path does no work in any of these {len(rows):,} runs. Everything torann
contributes is untested here.</li>
<li><strong>ESS above d&nbsp;=&nbsp;8 is known to be weak.</strong> Its force
law stops discriminating: at d=32 the 64th neighbour pushes 86% as hard as
the nearest, so a wrongly-returned far neighbour votes at nearly full
strength. Any dimensional trend in this report is measured on that ESS, not
on a fixed one.</li>
<li><strong>One optimizer, one suite.</strong> DE only, on five analytic
functions. The published protocol uses five optimizers over COCO/bbob, and
the effect is known to be optimizer-dependent.</li>
<li><strong>Bugs remain likely.</strong> Three were found in this harness
already. The invariant check is a guard, not a proof.</li>
</ul></section>""")

    P.append(f"""<hr class="rule"><p class="foot">Generated from
<code>{esc(os.path.basename(path))}</code>. Budget {budget:,} function calls
per run, population {n_pop}, target = median final value of the
<code>{BASELINE}</code> arm per cell. Paired Wilcoxon signed-rank, normal
approximation. Lower objective values are better throughout.</p>
</div></body></html>""")
    return "".join(P)


def load_sweep_dir(path, optimizer=None):
    """Reassemble the per-arm JSONL an `--arm-index` sweep leaves behind.

    Only ``*.jsonl`` is read. A task in flight is writing ``<arm>.jsonl.tmp``
    and renames it on completion, so restricting the glob to the final suffix
    is what makes a partial sweep safe to report on: incomplete arms are absent
    rather than truncated, and `organise` already drops any cell that is not
    present for every arm.

    The sweep crosses initializers with optimizers and names an arm
    ``<init>@<optimizer>``, but every statistic in this report is relative to a
    single ``random`` baseline. Comparing a JADE arm against a DE baseline
    would report the optimizer's effect as the initializer's, so `optimizer`
    selects one and strips the suffix: the report is then produced once per
    optimizer, each internally consistent.

    Args:
        path (str): Directory holding the per-arm files.
        optimizer (str | None): Keep only arms driven by this optimizer and
            drop the ``@`` suffix from their names. Required when the files
            hold more than one.

    Returns:
        dict: ``{"config": ..., "rows": [...]}``, matching the serial JSON.

    Raises:
        FileNotFoundError: If no completed arm files are present.
        ValueError: If several optimizers are present and none was chosen.
    """
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"no completed *.jsonl in {path!r}. A sweep still running leaves "
            f"only *.jsonl.tmp, which is deliberately not read."
        )
    rows = []
    for f in files:
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    opts = {r["arm"].partition("@")[2] for r in rows}
    opts.discard("")
    if optimizer is not None:
        rows = [r for r in rows if r["arm"].partition("@")[2] == optimizer]
        if not rows:
            raise ValueError(
                f"no rows for optimizer {optimizer!r}; present: {sorted(opts)}")
        for r in rows:
            r["arm"] = r["arm"].partition("@")[0]
    elif len(opts) > 1:
        raise ValueError(
            f"{path!r} holds several optimizers ({sorted(opts)}). Pass "
            f"--optimizer to pick one: every statistic here is measured "
            f"against a single 'random' baseline, so mixing them would "
            f"attribute the optimizer's effect to the initializer."
        )

    arms = {r["arm"] for r in rows}
    print(f"merged {len(files)} arm files -> {len(rows)} rows, "
          f"{len(arms)} distinct arms"
          + (f", optimizer={optimizer}" if optimizer else ""))
    # `build` needs the three run parameters that a serial run stores in its
    # config block. An array sweep has no single config file -- each task wrote
    # its own -- so they are recovered from the rows, which carry all three.
    cfg = {
        "source": path,
        "arm_files": len(files),
        "n_pop": sorted({r["n_pop"] for r in rows}),
        "iters": max(r["n_iter"] for r in rows),
        "seeds": len({r["seed"] for r in rows}),
    }
    return {"config": cfg, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=os.path.join(OUT, "bench_knobs.json"))
    ap.add_argument("--sweep-dir",
                    help="read per-arm JSONL from a Slurm array sweep instead "
                         "of a single --json file")
    ap.add_argument("--optimizer",
                    help="with --sweep-dir, restrict to one optimizer. "
                         "Required when the sweep crossed more than one")
    ap.add_argument("--out", default=os.path.join(OUT, "bench_init_oblesa.html"))
    args = ap.parse_args()

    if args.sweep_dir:
        data = load_sweep_dir(args.sweep_dir, args.optimizer)
    else:
        with open(args.json) as fh:
            data = json.load(fh)

    # Point the module's arm table at what the data actually contains. Left as
    # globals because `build` and `organise` read them from module scope; the
    # alternative is threading two more arguments through both.
    global GROUPS, ARMS, OBLESA_REF, OPP_REF
    present = sorted({r["arm"] for r in data["rows"]})
    GROUPS = derive_groups(present)
    ARMS = tuple(a for _, g in GROUPS for a in g)

    # The invariant compares OBLESA against the opposition arm whose candidate
    # pool it extends, so the partner has to match the opposition mode the
    # OBLESA arms were built with -- `qobl` for a sweep fixed at quasi, not
    # `obl`. Picking greedy-on-fitness (`s00`) as the OBLESA reference keeps
    # the comparison one of pools rather than of selection rules.
    if OBLESA_REF not in present:
        pref = [a for a in present
                if (p := parse_arm(a)) and p["sel"] == "s00"
                and p["engine"] not in ("null",)]
        OBLESA_REF = pref[0] if pref else None
    if OPP_REF not in present:
        OPP_REF = "qobl" if "qobl" in present else None
    doc = build(data, args.json)
    with open(args.out, "w") as fh:
        fh.write(doc)
    print(f"wrote {args.out}  ({len(doc) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
