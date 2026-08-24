"""The benchmark landscape is reproducible, and it varies with the seed.

Two defects in `examples/bench_init_oblesa.py` invalidated a 980,000-row
sweep, and neither raised anything.

`shifted()` drew its offset from `np.random.default_rng(abs(hash(name)) ...)`.
CPython salts `hash(str)` per process unless `PYTHONHASHSEED` is set, and the
sweep sets it nowhere. Every SLURM array task is its own process and writes
one arm, so **each arm was scored on a different landscape** -- three
processes gave `f(0) = 30.54 / 71.23 / 56.57` for sphere at `d=8`. The whole
analysis was built on pairing arms at equal `(function, dimension, seed)`,
and that pairing never existed.

`objective_for()` took no `seed`, so all 100 seeds shared one instance. The
seeds bought precision about a single landscape instead of evidence about
landscapes, and `shifted`'s own `frac` guard -- there so a near-central
optimum cannot flatter quasi-opposition's centre bias -- had nothing to
average over.

Both are invisible in a single-process run, which is why nothing caught
them: the first test below fails only across processes, so it runs the check
in a subprocess rather than trusting this one.
"""

import os
import subprocess
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXAMPLES = os.path.join(ROOT, "examples")
sys.path.insert(0, EXAMPLES)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Deliberately run with a *different* PYTHONHASHSEED in each child: that is
# exactly what the cluster does across array tasks, and the old code changed
# its answer under it.
_PROBE = (
    f"import sys; sys.path.insert(0, {EXAMPLES!r}); "
    f"sys.path.insert(0, {os.path.join(ROOT, 'src')!r})\n"
    "import numpy as np, bench_init_oblesa as B\n"
    "print(repr(float(B.objective_for('sphere', 8, 0)(np.zeros(8)))))\n"
)


class TestLandscapeIsReproducible(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import bench_init_oblesa  # type: ignore[reportMissingImports]
        except ImportError as exc:            # pragma: no cover
            raise unittest.SkipTest(f"benchmark not importable: {exc}") from exc
        cls.bench = bench_init_oblesa

    def test_same_key_gives_same_landscape_in_a_different_process(self):
        """The bug that cost the sweep: this passes in-process either way."""
        out = []
        for hashseed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hashseed)
            proc = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                                  capture_output=True, text=True,
                                  timeout=300, check=False)
            self.assertEqual(proc.returncode, 0,
                             f"probe failed:\n{proc.stderr[-2000:]}")
            out.append(proc.stdout.strip())
        self.assertEqual(
            len(set(out)), 1,
            "the landscape depends on something volatile across processes; "
            f"got {out}. Arms are compared across processes, so this makes "
            "every paired comparison in the sweep meaningless.")

    def test_the_landscape_varies_with_the_seed(self):
        """Otherwise `--seeds N` reruns one instance N times."""
        import numpy as np
        x = np.zeros(8)
        vals = {float(self.bench.objective_for("sphere", 8, s)(x))
                for s in range(8)}
        self.assertGreater(
            len(vals), 1,
            "every seed produced the same landscape — the seed axis is "
            "repetition, not independent problem instances")

    def test_every_attraction_model_is_on_the_ladder(self):
        """`projection` and `auto` are the ones for M < 2d+1.

        OBLESA gives ESS `2 * n_pop` measured sources at every dimension,
        while `fourier`/`detrended` need `2d + 1` coefficients, so from d=32
        they are underdetermined. A ladder without the two models built for
        that regime cannot say anything about high `d` — and `auto` is ESS's
        own default, which naming a model here overrides.
        """
        for want in ("projection", "auto"):
            self.assertIn(want, self.bench._ATT_MODELS)
            self.assertIn(want, self.bench._MODEL_SUFFIX)
        self.assertEqual(
            len(set(self.bench._MODEL_SUFFIX.values())),
            len(self.bench._MODEL_SUFFIX),
            "two attraction models share a suffix, so their arms collide")


if __name__ == "__main__":
    unittest.main()


class TestLandscapeCoverage(unittest.TestCase):
    """The benchmark set has to *vary* the properties it claims to test.

    A set that is uniformly separable cannot show whether joint-space coverage
    helps, because on a separable landscape a coordinate-marginal design is
    already near-optimal. Seven of the original eight landscapes were
    separable at D=32, so the sweep could not have answered the question it
    was built for -- and no amount of extra seeds would have helped, because
    the missing thing was variance across landscapes, not precision within
    one.

    These assert the set still spans both axes, so trimming it later cannot
    silently collapse it back.
    """

    @staticmethod
    def _bench():
        sys.path.insert(0, EXAMPLES)
        sys.path.insert(0, os.path.join(ROOT, "src"))
        import bench_init_oblesa  # type: ignore[reportMissingImports]
        return bench_init_oblesa

    @staticmethod
    def _coupling(fn, d, rng, n_pairs=40, n_pts=8, h=1e-4):
        """Fraction of coordinate pairs with a non-zero mixed second
        difference, which is zero for every pair iff f is additively
        separable."""
        import numpy as np
        hits = []
        for _ in range(n_pts):
            x = rng.uniform(-5.0, 5.0, d)
            base = float(fn(x))
            for _ in range(n_pairs):
                i, j = rng.choice(d, 2, replace=False)
                xi, xj, xij = x.copy(), x.copy(), x.copy()
                xi[i] += h; xj[j] += h; xij[i] += h; xij[j] += h
                mixed = float(fn(xij)) - float(fn(xi)) - float(fn(xj)) + base
                hits.append(abs(mixed) / (h * h * max(abs(base), 1e-12)))
        return float((np.array(hits) > 1e-3).mean())

    def test_the_default_set_contains_coupled_landscapes(self):
        import numpy as np
        b = self._bench()
        rng = np.random.default_rng(0)
        coupled = [name for name in b.DEFAULT_FUNCTIONS
                   if self._coupling(b.shifted(name, 8, 0), 8, rng) > 0.5]
        self.assertGreaterEqual(
            len(coupled), 4,
            f"only {len(coupled)} densely coupled landscapes in the default "
            f"set; joint-space coverage cannot be distinguished from "
            f"coordinate-marginal coverage on separable problems")

    def test_rotation_actually_couples(self):
        """A rotated landscape must be coupled where its unrotated twin is
        not -- otherwise `rot_` is a label, not a transformation."""
        import numpy as np
        b = self._bench()
        rng = np.random.default_rng(1)
        for name in ("rastrigin", "levy", "dixon"):
            plain = self._coupling(b.shifted(name, 8, 0), 8, rng)
            rot = self._coupling(b.shifted(f"rot_{name}", 8, 0), 8, rng)
            self.assertLess(plain, 0.5, f"{name} was already coupled")
            self.assertGreater(rot, 0.5, f"rot_{name} did not couple")

    def test_every_landscape_attains_zero_at_its_optimum(self):
        """`f* = 0` is what the alpha success criterion measures against. A
        floor that is not reachable is a target no run can ever hit."""
        import zlib

        import numpy as np
        b = self._bench()
        for name in b.DEFAULT_FUNCTIONS:
            for d in (8, 32):
                g = b.shifted(name, d, 0)
                rng = np.random.default_rng(np.random.SeedSequence(
                    [zlib.crc32(name.encode()), d, 0]))
                target = rng.uniform(-0.8, 0.8, d) * 5.0
                if name in b.SIGN_FLIP_ONLY:
                    sign = rng.choice((-1.0, 1.0), size=d)
                    target = sign * b._SCHWEFEL_X
                self.assertLess(
                    abs(float(g(target))), 1e-8,
                    f"{name} d={d} does not reach 0 at its stated optimum")

    def test_no_landscape_goes_below_its_stated_floor(self):
        """The other half: nothing in the searched box may beat f* = 0.

        Schwefel breaches this without a boundary penalty -- its sine term
        keeps growing past the native domain, so a D=32 point outside it
        scores about -15160 against a claimed optimum of 0, and every
        'improvement' past zero is the benchmark leaking rather than the
        optimizer working.
        """
        import numpy as np
        b = self._bench()
        rng = np.random.default_rng(3)
        for name in b.DEFAULT_FUNCTIONS:
            for d in (8, 32):
                g = b.shifted(name, d, 0)
                worst = float(np.min([float(g(x)) for x in
                                      rng.uniform(-5.0, 5.0, (400, d))]))
                self.assertGreater(
                    worst, -1e-6,
                    f"{name} d={d} reaches {worst} inside the box, below its "
                    f"stated floor of 0")

    def test_every_landscape_evaluates_a_whole_population_at_once(self):
        """A landscape must accept an (n, d) matrix, not just one point.

        `utils.compute_objective` tries the vectorised call and falls back to
        row-by-row on ValueError/TypeError, which is a deliberate courtesy to
        objectives that are not vectorised. For a *benchmark* landscape it is
        a trap: the fallback is silent, so a landscape that raises on a batch
        still produces correct scores while quietly costing one counted
        evaluation per row. The rotated landscapes did exactly that -- written
        as `q @ (x - target)`, which is only valid for a single point -- and
        every rotated arm reported 120 initial evaluations where its unrotated
        twin reported 60. The scores were right and the budget was double.

        Worse at n == d, where `q @ X` is a legal matmul that rotates *across
        individuals* instead of within one. No exception, no fallback, wrong
        landscape.
        """
        import numpy as np
        b = self._bench()
        rng = np.random.default_rng(7)
        for name in b.DEFAULT_FUNCTIONS:
            for d in (4, 8):
                g = b.shifted(name, d, 0)
                for n in (1, 5, d):          # d included on purpose
                    X = rng.uniform(-5.0, 5.0, (n, d))
                    out = np.asarray(g(X))
                    self.assertEqual(
                        out.shape, (n,),
                        f"{name} d={d} returned {out.shape} for a ({n}, {d}) "
                        f"population")
                    rows = np.array([float(g(x)) for x in X])
                    np.testing.assert_allclose(
                        out, rows, rtol=1e-9, atol=1e-9,
                        err_msg=f"{name} d={d} n={n}: batched != row-wise")
