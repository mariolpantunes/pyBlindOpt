# TODO — the OBLESA tuning sweep

Deferred deliberately. The library is released and correct; what is *not* done
is measuring OBLESA's knobs on the engine it actually ships with. Everything
below is a debt against the numbers, not against the code.

## Why every OBLESA number on record needs re-taking

All 620,000 runs of the earlier sweep used a dart stand-in
(`dart-fitness-guided-lambda*`), built to answer "is OBLESA worth pursuing"
before ESS existed. It is a different engine on a different scale:

- `force_weight` was a scalar on a z-scored surrogate over a candidate cloud,
  useful into the tens. It is now ESS's `attraction_weight`, a scale on a
  pairwise force, refused at or above 2.5. **There is no conversion between
  them.**
- Dart has no relaxation step at all — the placement *is* the answer. ESS
  places and then relaxes under repulsion and attraction together.

The baselines and the optimizer axis survive (they never touched the engine).
Every OBLESA-specific result does not.

## 1. Return `opp` to the swept axis — do this first

`examples/bench_init_oblesa.py::_FIXED` pins `opp="quasi"` on dart-era
evidence (31.6 against 21.0 for standard). A later 12-seed × 8-function probe
under `ga` measured the opposite:

| d | standard | none | quasi |
|---|---|---|---|
| 32 | **+39.09** | +29.91 | +2.39 |

37 AR points, on the optimizer least able to recover from a bad start. It is
the single most damaging knob currently held constant, and it is held at the
wrong value for at least one optimizer. Sweep it; if the answer is
optimizer-dependent, that is the finding and it belongs in the paper.

## 2. Re-run the quality grid against the 3N controls

`BASELINE_ARMS` moved from `random4x, obl2x` (4N) to `random3x, obl15x` (3N)
so the comparison matches the published three-stage pipeline. The quality grid
on record was taken against the 4N controls, so it is not comparable to
anything the current harness produces. Nothing from it should be quoted.

## 3. Confirm `att_model="auto"` end to end

ESS 0.5.1 made `auto` the default and pyBlindOpt now passes it. It was
validated *inside* ESS on synthetic truths, where it correctly refuses the
ridge fit on a non-additive objective (which scores 1.29–1.49 held-out, worse
than predicting the mean). It has never been measured through OBLESA on the
benchmark objectives — four of the eight are non-separable, which is exactly
the case `auto` exists for. `_ATT_MODELS` crosses `fourier`/`idw`/`detrended`;
add `auto` and check it is at least as good as the best fixed choice, since
that is the whole claim.

## 4. Then the sweep proper

The grid is 42 OBLESA arms + 7 baselines, × 5 optimizers, × 8 functions ×
the dimension ladder. `slurm/` has the array harness; `--arm-index` gives each
arm its own JSONL so it resumes for free, and `report_init_oblesa.py
--sweep-dir` reassembles them.

Two things to hold fixed while sweeping, both already argued in the harness
docstring: initialization is preprocessing and is never charged against the
iteration budget (GECCO Companion '26 Eq. 3 is a ratio of *iterations*), and
every arm and optimizer sees the same reseeded `rng_opt`, so the initial
population is the only thing that differs.

## Not on this list

- **Dart.** It is gone from pyBlindOpt and stays inside ESS as test
  scaffolding. Do not add a dart arm back to compare against; ESS beats it on
  placement and attractiveness and is faster, so the comparison measures
  nothing OBLESA is made of.
- **`profile`/`sweep` scripts.** Debug and profiling code stays out of the
  index.
