# Changelog

Notable changes per release. Versions follow [semantic versioning][semver],
with the pre-1.0 caveat that the minor number carries breaking changes.

[semver]: https://semver.org/spec/v2.0.0.html

## 0.4.0 — 2026-08-06

Maintenance: the packaging metadata is now checked rather than hand-kept, the
lint gate covers the whole repository, and the empty-space stage is ESS alone.

### Breaking

- **`pyBlindOpt.emptyspace` removed.** `dart_esa` and `random_esa` were a
  stand-in used to test whether OBLESA was worth pursuing before ESS was
  ready. ESS beats dart on both placement and attractiveness and is several
  times faster, so they measured nothing OBLESA is made of. Dart remains
  inside EmptySpaceSearch as test scaffolding.
- **`force` accepts only `'guided'` / `'ess'`.** The `'repulsive'` and
  `'uniform'` levels went with the module above. A control for an experiment
  belongs to the harness running it and goes in through `engine=`.
- **`att_model` defaults to `'auto'`**, following EmptySpaceSearch 0.5.1. It
  cross-validates the attractiveness models on the pool that was already
  evaluated, so it costs no objective calls. Which one wins depends on whether
  the objective is separable — not something the dimension can predict.
- **Requires `EmptySpaceSearch>=0.5.1`**, and states `numpy>=2.0.0`, which is
  what ESS has required all along.
- **Per-module `__version__`, `__email__`, `__url__` and `__status__` are
  gone.** They live in `pyBlindOpt/__init__.py` only.

### Fixed

- **`pyBlindOpt.__version__` reported the wrong version.** 0.3.0 shipped
  saying `0.2.0`; eighteen modules said `0.2.0`, one `0.1.0`, one `0.3.0`. It
  is now read from the installed distribution, so there is one copy and it
  cannot drift. `test_packaging.py` asserts it against `setup.cfg`.
- **`__email__` was `mario.antunes@ua.com`**, a domain that does not exist,
  in all twenty modules. It is `mario.antunes@ua.pt`, and the test pins it to
  what `setup.cfg` publishes.
- **`requirements.txt` said `numpy>=1.6.4`** against `setup.cfg`'s
  `numpy>=1.26.4` — two decades apart — and the `tqdm` floors also disagreed.
  The two files are now asserted equal.
- **`examples/tune_oblesa.py` passed `callback=stopper.callback`**, which
  `EarlyStopping` does not have; it defines `__call__`. The tuner raised
  `AttributeError` on every run. Caught by widening the type gate.
- `CITATION.cff` was titled `optimization` and carried no version,
  repository, license or release date, so the released version could not be
  cited from it.

### Changed

- **CI lints `src test examples`, not `src`.** The narrower gate reported
  green over 33 errors in tracked files; `basedpyright` now covers the same
  three, which is what found the callback bug above.
- The lint toolchain is pinned in `requirements.txt` at the versions CI
  installs, so the local gate and CI cannot disagree about what passes.
- `Optimizer.optimize` and the four `Sampler.sample` implementations are
  documented; pdoc rendered them blank before.
- README documents the class interface — roughly half the public API, and
  previously absent.

## 0.3.0 — OBLESA on the ESS backend

Brought the OBLESA initializer onto the production EmptySpaceSearch backend
and fixed two defects that were distorting every measurement taken through it.

### Breaking

- **`force_weight` changed units**, `8.0` → `0.5`: EmptySpaceSearch's
  `attraction_weight`, which scales a pairwise force bounded by a collapse
  condition and is refused at or above 2.5. No value tuned against the retired
  dart ladder transfers.
- **`k_cand` `2048` → `64`**, ESS's own `init_pool`. At `n_pop=30` on
  rastrigin: 4.5–7.1× faster, bit-identical population at d=32, 64 and 100.
- `emptyspace.fitness_dart_esa` removed; `force='guided'` resolves to
  `ess.esa`.
- `compute_crowding_distance` returns different values at every dimension, so
  any tuning of `diversity_weight` against the old function needs redoing.

### Fixed

- **Crowding distance saturated with dimension.** The `np.inf` endpoint rule
  is NSGA-II's, for a 2–3 objective front; across a `D`-dimensional decision
  space it marks up to `2*D` points — 88% of a 120-candidate pool at `D=100`,
  which then collapsed to one shared constant. All 120 now separate. The
  assignment was also overwriting its own accumulator.
- **The guided engine carried a toroidal metric into its fitness surrogate**,
  asserting the objective is periodic: a candidate at `x=4.9` took a heavily
  weighted contribution from a sample at `x=-4.9`. Removed with the engine.
