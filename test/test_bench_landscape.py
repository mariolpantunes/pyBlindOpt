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
