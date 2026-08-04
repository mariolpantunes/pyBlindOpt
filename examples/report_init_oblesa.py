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
"""

import argparse
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
    ("OBLESA, largest-empty-sphere engine (3N / 4N pool)",
     ("oblesa-dart", "oblesa-quasi-dart",
      "oblesa-dart-4n", "oblesa-quasi-dart-4n")),
    ("OBLESA, selection knobs on the 4N pool",
     ("oblesa-dart-4n-div25", "oblesa-quasi-dart-4n-div25",
      "oblesa-dart-4n-prob", "oblesa-quasi-dart-4n-prob",
      "oblesa-dart-4n-mm25", "oblesa-quasi-dart-4n-mm25")),
    ("uniform-noise null, matched settings",
     ("oblesa-rand", "oblesa-quasi-rand", "oblesa-rand-4n",
      "oblesa-quasi-rand-4n", "oblesa-rand-4n-mm25",
      "oblesa-quasi-rand-4n-mm25")),
)
ARMS = tuple(a for _, g in GROUPS for a in g)
BASELINE = "random"
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
    "oblesa-dart": "OBLESA with the empty-space stage replaced by explicit "
                   "largest-empty-sphere filling: each point goes at the "
                   "centre of the biggest empty sphere, found by search over a "
                   "candidate cloud. Same objective as a Voronoi/Delaunay "
                   "construction, no torus and no force kernel — it tests the "
                   "*idea* rather than this implementation of it.",
    "oblesa-quasi-dart": "The same with quasi-opposition.",
    "oblesa-dart-4n": "The largest-empty-sphere engine sized to the static "
                      "pool it was placed against: n_ess = 2 * n_pop, so the "
                      "candidate pool is 4N rather than the paper's 3N.",
    "oblesa-quasi-dart-4n": "Quasi-opposition on the 4N pool.",
    "oblesa-dart-4n-div25": "4N pool, fitness blended with NSGA-II crowding "
                            "distance at weight 0.25.",
    "oblesa-dart-4n-prob": "4N pool, probabilistic selection over that blended "
                           "score rather than greedy.",
    "oblesa-dart-4n-mm25": "4N pool, sequential maximin over the fittest "
                           "1.75x n_pop.",
    "oblesa-quasi-dart-4n-div25": "Quasi-opposition, crowding blend.",
    "oblesa-quasi-dart-4n-prob": "Quasi-opposition, probabilistic selection.",
    "oblesa-quasi-dart-4n-mm25": "Quasi-opposition, maximin selection.",
    "oblesa-rand-4n": "The null at the 4N pool size.",
    "oblesa-quasi-rand-4n": "Quasi-opposition null at the 4N pool size.",
    "oblesa-rand-4n-mm25": "Null with maximin selection, so the dart arm at "
                           "the same setting is compared like for like.",
    "oblesa-quasi-rand-4n-mm25": "Quasi-opposition null, maximin selection.",
    "oblesa-quasi-rand": "Quasi-opposition null at the 3N pool size.",
}


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
    for k in by:
        by[k].sort(key=lambda r: r["seed"])
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
        # invariant: OBLESA must not be significantly worse than OBL
        e, o = at(f, d, "oblesa"), at(f, d, "obl")
        if np.median(e) > np.median(o) and wilcoxon_signed_rank(e, o) < 0.05:
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

    r4 = ar("random4x", "it"); o2 = ar("obl2x", "it")
    obe = ar("oblesa", "it"); best = ar(best_arm, "it")
    ctrl = max(r4, o2)
    verdict = ("clears" if best > ctrl else "does not clear")
    P.append(f"""<section><h2>Verdict</h2><div class="verdict">
<p><strong>The best OBLESA configuration ({esc(best_arm)}) reaches
{best:.2f}% acceleration; the strongest ESS-free control at the same 4N
budget reaches {ctrl:.2f}%.</strong> On this evidence OBLESA
<strong>{verdict}</strong> its own cost controls. Default OBLESA sits at
{obe:.2f}%, so configuration matters more than the method label.</p>
<p><strong>Controls:</strong> <code>random4x</code> {r4:.2f}%,
<code>obl2x</code> {o2:.2f}% &mdash; both spend exactly what OBLESA spends,
neither uses ESS. The gap between the best OBLESA arm and the best of these
is the part attributable to empty-space search rather than to a larger
selection pool.</p>
<p><strong>Structural invariant:</strong> {viol} of {len(cells)} cells where
OBLESA is significantly worse than OBL at equal generations. OBLESA selects
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
                     f"{esc(ARM_GLOSS[a])}</li>")
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
<div class="scroll"><table><thead><tr><th class="l">arm</th>
<th>AR % (iterations)</th><th>AR % (calls)</th><th>success</th>
<th>quality wins</th></tr></thead><tbody>""")
    for label, group in GROUPS:
        P.append(f'<tr><td class="l dim" colspan="5">{esc(label)}</td></tr>')
        for a in group:
            i_, n_ = ar(a, "it"), ar(a, "nfc")
            c1 = "good" if i_ > 1 else ("bad" if i_ < -1 else "dim")
            c2 = "good" if n_ > 1 else ("bad" if n_ < -1 else "dim")
            P.append(f'<tr><td class="l">{esc(a)}</td>'
                     f'<td class="{c1}">{i_:+.2f}</td>'
                     f'<td class="{c2}">{n_:+.2f}</td>'
                     f'<td>{sr[a]:.1f}%</td>'
                     f'<td class="dim">{wins[a]}/{len(cells)}</td></tr>')
    P.append("</tbody></table></div></section>")

    P.append(f"""<section><h2>Does the advantage grow with dimension?</h2>
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
    RIVALS = ("random", "random4x", "obl2x", "obl", "qobl")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=os.path.join(OUT, "bench_knobs.json"))
    ap.add_argument("--out", default=os.path.join(OUT, "bench_init_oblesa.html"))
    args = ap.parse_args()

    with open(args.json) as fh:
        data = json.load(fh)
    doc = build(data, args.json)
    with open(args.out, "w") as fh:
        fh.write(doc)
    print(f"wrote {args.out}  ({len(doc) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
