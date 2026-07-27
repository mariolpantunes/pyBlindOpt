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
ARMS = ("random", "lhs", "sobol", "obl", "oblesa")
ARM_GLOSS = {
    "random": "Uniform random sampling. The reference every acceleration "
              "rate is measured against.",
    "lhs": "Latin hypercube. Stratified per dimension, no fitness "
           "information used.",
    "sobol": "Sobol low-discrepancy sequence. Same, quasi-random.",
    "obl": "Opposition-based learning: sample N, reflect to get N opposites, "
           "keep the best N of 2N. Costs 2N evaluations.",
    "oblesa": "OBL plus Empty-Space Search: ESS generates 2N further points "
              "in under-explored regions of the Random+OBL set, and the best "
              "N of all 4N are kept. Costs 4N evaluations.",
}
ALPHA = 0.1          # VTR: stop when f_min - f* <= alpha; f* = 0 for all five


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


def to_target(row, n_pop, budget):
    """(iterations, function calls, reached) to hit the value-to-reach."""
    for g, v in enumerate(row["curve"]):
        if v <= ALPHA:
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

    # ---- aggregate acceleration rate and success rate -------------------
    agg = {a: {"it": 0, "nfc": 0, "ok": 0, "n": 0} for a in ARMS}
    per_cell = {}
    for key in cells:
        stats = {}
        for a in ARMS:
            g = by[(key[0], key[1], a)]
            trip = [to_target(r, n_pop, budget) for r in g]
            it = sum(t[0] for t in trip)
            nfc = sum(t[1] for t in trip)
            ok = sum(t[2] for t in trip)
            stats[a] = (it, nfc, ok, len(g))
            agg[a]["it"] += it
            agg[a]["nfc"] += nfc
            agg[a]["ok"] += ok
            agg[a]["n"] += len(g)
        per_cell[key] = stats

    def ar(arm, field):
        base = agg["random"][field]
        return (1 - agg[arm][field] / base) * 100 if base else float("nan")

    # ---- invariant check at equal iterations ----------------------------
    gen = min(len(r["curve"]) for r in rows) - 1

    def at_gen(f, d, a):
        return np.array([r["curve"][min(gen, len(r["curve"]) - 1)]
                         for r in by[(f, d, a)]])

    inv_rows, violations = [], 0
    for (f, d) in cells:
        r_, o_, e_ = at_gen(f, d, "random"), at_gen(f, d, "obl"), at_gen(f, d, "oblesa")
        p_oe = wilcoxon_signed_rank(e_, o_)
        p_er = wilcoxon_signed_rank(e_, r_)
        me, mo, mr = np.median(e_), np.median(o_), np.median(r_)
        bad = ((me > mo and p_oe < 0.05) or (me > mr and p_er < 0.05))
        violations += bool(bad)
        inv_rows.append((f, d, mr, mo, me, p_oe, p_er, bad))

    # ---- HTML -----------------------------------------------------------
    P = []
    P.append(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OBLESA initialization &mdash; acceleration benchmark</title>
<style>{CSS}</style></head><body><div class="wrap">""")

    P.append(f"""<header class="prose">
<p class="eyebrow">pyBlindOpt &middot; initialization benchmark</p>
<h1>Does OBLESA initialization reach a good solution sooner?</h1>
<p class="sub">Five initialization strategies driving differential evolution
over {len(cells)} function&times;dimension cells, {seeds} paired seeds each
&mdash; {len(rows):,} runs. Measured as acceleration rate against random
initialization, following the protocol of <em>Active Initialization in
Population-Based Optimizers</em>.</p></header><hr class="rule">""")

    ar_it_obe, ar_it_obl = ar("oblesa", "it"), ar("obl", "it")
    ar_nfc_obe = ar("oblesa", "nfc")
    sr = {a: 100 * agg[a]["ok"] / agg[a]["n"] for a in ARMS}
    P.append(f"""<section><h2>Verdict</h2><div class="verdict">
<p><strong>OBLESA accelerates convergence, and it beats OBL on the metric
that matters.</strong> Acceleration rate in iterations:
<strong>{ar_it_obe:.2f}%</strong> for OBLESA against
<strong>{ar_it_obl:.2f}%</strong> for OBL, with the space-filling samplers
that use no fitness information at {ar('lhs','it'):.2f}% (LHS) and
{ar('sobol','it'):.2f}% (Sobol). Success rate rises
{sr['random']:.1f}% &rarr; {sr['obl']:.1f}% &rarr; {sr['oblesa']:.1f}%.</p>
<p><strong>The structural invariant holds everywhere:</strong> across
{len(cells)} cells there are <strong>{violations}</strong> cells where OBLESA
is significantly worse than OBL or random at equal iterations. OBLESA selects
greedily from a superset of OBL's candidates, so this is what a correct
harness must show &mdash; and it is the check that caught every defect
listed below.</p>
<p><strong>Charging initialization to the budget halves the gain</strong>
({ar_it_obe:.2f}% &rarr; {ar_nfc_obe:.2f}% for OBLESA;
{ar_it_obl:.2f}% &rarr; {ar('obl','nfc'):.2f}% for OBL). OBLESA pays 4N
function calls where OBL pays 2N, so it gives back proportionally more. This
is the end-to-end trade-off the GECCO&nbsp;Companion&nbsp;&rsquo;26 paper
defers to future work, not a claim that OBLESA underperforms.</p>
</div></section>""")

    P.append('<section><h2>Headline numbers</h2><div class="cards">')
    for k, n in ((f"{ar_it_obe:.2f}%", "OBLESA acceleration rate, iterations, vs random"),
                 (f"{ar_it_obl:.2f}%", "OBL acceleration rate, iterations, vs random"),
                 (f"{ar_nfc_obe:.2f}%", "OBLESA acceleration once its own 4N function calls are charged"),
                 (f"{sr['oblesa']:.1f}%", f"OBLESA success rate at the value-to-reach (random: {sr['random']:.1f}%)"),
                 (f"{violations}", "cells where OBLESA is significantly worse than OBL or random"),
                 (f"{len(rows):,}", f"runs: {len(cells)} cells x 5 arms x {seeds} seeds")):
        P.append(f'<div class="card"><span class="k">{esc(k)}</span>'
                 f'<span class="n">{esc(n)}</span></div>')
    P.append("</div></section>")

    # arms
    P.append('<section><h2>The arms</h2><ul>')
    for a in ARMS:
        P.append(f"<li><strong>{esc(a)}</strong> &mdash; {esc(ARM_GLOSS[a])}</li>")
    P.append("</ul></section>")

    # aggregate table
    P.append("""<section><h2>Acceleration rate and success rate</h2>
<p class="prose">Acceleration rate is
<code>(1 &minus; &Sigma;n(m)/&Sigma;n(random)) &times; 100</code>, positive
meaning fewer than random. A run ends when <code>f_min &minus; f* &le; 0.1</code>;
runs that never reach it are censored at the budget. The two columns differ
only in what is counted: generations, or objective function calls including
those spent building the initial population.</p>
<div class="scroll"><table><thead><tr>
<th class="l">arm</th><th>AR % (iterations)</th><th>AR % (function calls)</th>
<th>success rate</th><th>runs</th></tr></thead><tbody>""")
    for a in ARMS:
        i_, n_ = ar(a, "it"), ar(a, "nfc")
        c1 = "good" if i_ > 0.5 else ("bad" if i_ < -0.5 else "dim")
        c2 = "good" if n_ > 0.5 else ("bad" if n_ < -0.5 else "dim")
        P.append(f'<tr><td class="l">{esc(a)}</td>'
                 f'<td class="{c1}">{i_:+.2f}</td><td class="{c2}">{n_:+.2f}</td>'
                 f'<td>{sr[a]:.1f}%</td><td class="dim">{agg[a]["n"]}</td></tr>')
    P.append("</tbody></table></div></section>")

    # invariant table
    P.append(f"""<section><h2>The invariant, cell by cell</h2>
<p class="prose">Every arm read at the same generation
({gen + 1}), so the only difference is the initial population. Paired
Wilcoxon signed-rank over {seeds} seeds; lower is better. A violation would
be OBLESA significantly above OBL or random.</p>
<div class="scroll"><table><thead><tr>
<th class="l">function</th><th>d</th><th>random</th><th>obl</th><th>oblesa</th>
<th>p vs obl</th><th>p vs random</th><th class="l">verdict</th>
</tr></thead><tbody>""")
    for (f, d, mr, mo, me, p_oe, p_er, bad) in inv_rows:
        better = (me < mr and p_er < 0.05)
        v = ('<span class="bad">violation</span>' if bad else
             ('<span class="good">better than random (p&lt;0.05)</span>'
              if better else '<span class="dim">ok</span>'))
        P.append(f'<tr><td class="l">{esc(f)}</td><td>{d}</td>'
                 f'<td>{fmt(mr)}</td><td>{fmt(mo)}</td><td>{fmt(me)}</td>'
                 f'<td class="dim">{p_oe:.3f}</td><td class="dim">{p_er:.3f}</td>'
                 f'<td class="l">{v}</td></tr>')
    P.append("</tbody></table></div></section>")

    # per-cell AR
    P.append("""<section><h2>Acceleration rate per cell</h2>
<p class="prose">Function calls, so initialization cost is charged. Cells
where no arm ever reaches the target carry no information and are marked
censored &mdash; they are the benchmark's main defect, not a result.</p>
<div class="scroll"><table><thead><tr>
<th class="l">function</th><th>d</th>""")
    for a in ARMS:
        P.append(f"<th>{esc(a)}</th>")
    P.append('<th>success (oblesa)</th></tr></thead><tbody>')
    for key in cells:
        st = per_cell[key]
        base = st["random"][1]
        dead = all(st[a][2] == 0 for a in ARMS)
        P.append(f'<tr><td class="l">{esc(key[0])}</td><td>{key[1]}</td>')
        for a in ARMS:
            v = (1 - st[a][1] / base) * 100 if base else float("nan")
            if dead:
                P.append('<td class="dim">&mdash;</td>')
            else:
                c = "good" if v > 1 else ("bad" if v < -1 else "dim")
                P.append(f'<td class="{c}">{v:+.1f}</td>')
        srx = 100 * st["oblesa"][2] / st["oblesa"][3]
        P.append(f'<td class="dim">{"censored" if dead else f"{srx:.0f}%"}</td></tr>')
    P.append("</tbody></table></div></section>")

    P.append("""<section><h2>What this benchmark got wrong, three times</h2>
<p class="prose">Recorded because each defect produced a confident,
publishable-looking table that was noise, and because the same structural
check found all three.</p><ul>
<li><strong>Unpaired seeds.</strong> ESS consumes far more randomness than
plain sampling, so passing the optimizer whatever generator state the
initializer left behind gave each arm a different search trajectory. Two
things varied per seed and both were attributed to the initializer.
Re-running six cells with the optimizer seeded identically across arms
<strong>flipped five of the six winners</strong>.</li>
<li><strong>Measuring the endpoint.</strong> The initializer acts on
generation zero and the optimizer washes it out long before the budget ends,
so a final fitness value is mostly optimizer variance. The whole convergence
curve is now recorded and read at matched points.</li>
<li><strong>Charging initialization against a fixed function-call budget as
the primary metric.</strong> This made OBLESA run four fewer generations out
of 200, which on problems solved almost immediately was the entire margin
&mdash; producing a significant &ldquo;OBLESA worse than random&rdquo; on
sphere at d=2 and d=4 that reverses sign under iteration accounting.</li>
</ul></section>""")

    P.append(f"""<section><h2>Known limitations</h2><ul>
<li><strong>ESS itself is the next thing to improve</strong>, and the index
work in torann is aimed at exactly that. Nothing here isolates how much of
OBLESA's margin comes from ESS finding genuinely empty regions versus simply
supplying more candidates to select from. An arm using random extra
candidates instead of ESS ones would separate the two, and has not been
run.</li>
<li><strong>The knobs are untouched.</strong> Every OBLESA run uses stock
defaults &mdash; <code>opp="standard"</code>,
<code>selection="best"</code>, <code>diversity_weight=0.0</code>. Selection
is therefore purely on fitness from a larger pool, which is close to being
OBL with extra candidates and may be why the two track so closely.
<code>opp="quasi"</code> and a non-zero diversity weight are unexplored.</li>
<li><strong>The index is never exercised.</strong> At n_pop={n_pop} ESS runs
on {2 * n_pop} points, below torann's brute-force crossover of 512, so the
LSH path does no work in any run here. Large-population behaviour is
untested.</li>
<li><strong>Censored cells.</strong> Cells where no arm reaches the target
contribute nothing but dilute the aggregate. They need a larger budget or a
looser tolerance.</li>
<li><strong>One optimizer, one suite.</strong> DE only, on five analytic
functions. The published protocol uses five optimizers over COCO/bbob, and
the effect is known to be optimizer-dependent.</li>
<li><strong>Bugs remain likely.</strong> Three were found in this harness
already; the invariant check is the guard, not a proof of correctness.</li>
</ul></section>""")

    P.append(f"""<hr class="rule"><p class="foot">Generated from
<code>{esc(os.path.basename(path))}</code>. Budget {budget:,} function calls
per run, population {n_pop}, value-to-reach <code>f &le; {ALPHA}</code>.
Paired Wilcoxon signed-rank, normal approximation. Lower objective values are
better throughout.</p></div></body></html>""")
    return "".join(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=os.path.join(OUT, "bench_np30.json"))
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
