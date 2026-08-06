# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.1"
__email__ = "mario.antunes@av.it.pt"
__status__ = "Development"


import importlib.util
import os
import unittest

import numpy as np

import pyBlindOpt.emptyspace as emptyspace
import pyBlindOpt.utils as utils


def _load_reference():
    """`examples/emptyspace_reference.py`, or None if it or SciPy is missing.

    The exact Delaunay engine deliberately lives outside the library because
    it needs SciPy and pyBlindOpt is NumPy-only. Loading it by path rather than
    by import keeps `examples/` off the package path.
    """
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "examples", "emptyspace_reference.py")
    if not os.path.exists(path):
        return None
    try:
        import scipy.spatial  # noqa: F401
    except ImportError:
        return None
    spec = importlib.util.spec_from_file_location("emptyspace_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = _load_reference()


def _empty_radii(placed, static):
    """Empty-sphere radius each placed point achieved, in placement order."""
    current = static.copy()
    out = []
    for p in placed:
        out.append(float(np.sqrt(emptyspace._nearest_sq(p[None, :], current)[0])))
        current = np.vstack([current, p[None, :]])
    return np.array(out)


class TestEmptySpace(unittest.TestCase):
    def setUp(self):
        self.bounds = np.asarray([[-5.0, 5.0]] * 3)
        self.static = utils.RandomSampler(np.random.default_rng(0)).sample(
            40, self.bounds
        )

    # --- dart_esa ---
    def test_dart_shape_and_bounds(self):
        placed = emptyspace.dart_esa(
            self.static, self.bounds, n=12, seed=1, k_cand=256)

        self.assertEqual(placed.shape, (12, 3))
        self.assertTrue(utils.assert_bounds(placed, self.bounds))

    def test_dart_deterministic(self):
        kw = dict(n=8, seed=7, k_cand=128)
        a = emptyspace.dart_esa(self.static, self.bounds, **kw)
        b = emptyspace.dart_esa(self.static, self.bounds, **kw)

        np.testing.assert_array_equal(a, b)

    def test_dart_zero_points(self):
        placed = emptyspace.dart_esa(self.static, self.bounds, n=0, seed=1)
        self.assertEqual(placed.shape, (0, 3))

    def test_dart_from_empty_static_set(self):
        """No existing points is not a special case the caller has to avoid."""
        placed = emptyspace.dart_esa(
            np.empty((0, 3)), self.bounds, n=5, seed=1, k_cand=64)

        self.assertEqual(placed.shape, (5, 3))
        self.assertTrue(utils.assert_bounds(placed, self.bounds))

    def test_dart_beats_random_on_emptiness(self):
        """
        The whole claim of an empty-space engine: it must place points further
        from what is already there than uniform sampling does.
        """
        kw = dict(n=20, seed=3)
        dart = emptyspace.dart_esa(self.static, self.bounds, k_cand=2048, **kw)
        rand = emptyspace.random_esa(self.static, self.bounds, **kw)

        self.assertGreater(
            _empty_radii(dart, self.static).min(),
            _empty_radii(rand, self.static).min(),
        )

    def test_dart_improves_with_candidate_count(self):
        """
        `k_cand` is the accuracy knob, so more candidates must not do worse.

        This is the property that makes the approximation auditable: the
        objective is fixed and only the search over it gets finer.
        """
        kw = dict(n=15, seed=5)
        coarse = _empty_radii(
            emptyspace.dart_esa(self.static, self.bounds, k_cand=64, **kw), self.static
        ).mean()
        fine = _empty_radii(
            emptyspace.dart_esa(self.static, self.bounds, k_cand=8192, **kw), self.static
        ).mean()

        self.assertGreater(fine, coarse)

    def test_torus_metric_removes_the_boundary_bias(self):
        """
        The default toroidal metric must not push probes against the walls.

        With the plain box metric the point furthest from everything is almost
        always on a face -- a wall has no neighbours behind it, so it reads as
        infinitely empty -- which sends every probe outward regardless of where
        the objective is good. Wrapping the axes removes the boundary, and the
        resulting spread should be indistinguishable from uniform in how far
        out it sits, while still being *placed* rather than drawn.
        """
        bounds = np.asarray([[-5.0, 5.0]] * 10)
        static = utils.RandomSampler(np.random.default_rng(0)).sample(60, bounds)
        kw = dict(n=30, seed=1, k_cand=1024)

        def outwardness(p):
            return float(np.mean(np.abs(p)) / 5.0)

        box = outwardness(emptyspace.dart_esa(static, bounds, _torus=False, **kw))
        tor = outwardness(emptyspace.dart_esa(static, bounds, _torus=True, **kw))
        rnd = outwardness(emptyspace.random_esa(static, bounds, n=30, seed=1))

        self.assertGreater(box, rnd * 1.15, "box metric should hug the walls")
        self.assertAlmostEqual(tor, rnd, delta=0.05)
        # ...and it must still be a placement, not a redraw.
        self.assertGreater(
            _empty_radii(
                emptyspace.dart_esa(static, bounds, _torus=True, **kw), static).min(),
            _empty_radii(
                emptyspace.random_esa(static, bounds, n=30, seed=1), static).min(),
        )

    def test_dart_accepts_esa_kwargs(self):
        """Must be substitutable for `ess.esa`, whose kwargs it does not share."""
        placed = emptyspace.dart_esa(
            self.static, self.bounds, n=4, seed=1, k_cand=64,
            epochs=1024, lr=0.01, metric="softened_inverse", border_strategy="clip",
        )
        self.assertEqual(placed.shape, (4, 3))

    # --- random_esa ---
    def test_random_esa_shape_and_bounds(self):
        placed = emptyspace.random_esa(self.static, self.bounds, n=9, seed=2)

        self.assertEqual(placed.shape, (9, 3))
        self.assertTrue(utils.assert_bounds(placed, self.bounds))

    # --- delaunay_esa ---
    def test_delaunay_matches_dart(self):
        """
        The exact largest-empty-sphere must agree with the sampled one.

        The reference is a *box* construction, so `_torus=False` here: the two
        only measure the same quantity under the same metric. The reference
        solves for the Voronoi vertex; `dart_esa` finds it by search. If the two reach comparable radii then `dart_esa` is
        approximating the right quantity rather than merely something spread
        out -- which is the only reason to trust it in the dimensions where an
        exact triangulation cannot run.
        """
        if REFERENCE is None:
            self.skipTest("SciPy or examples/emptyspace_reference.py unavailable")

        for d in (2, 3, 4):
            bounds = np.asarray([[-5.0, 5.0]] * d)
            static = utils.RandomSampler(np.random.default_rng(0)).sample(40, bounds)

            exact = _empty_radii(
                REFERENCE.delaunay_esa(static, bounds, n=15), static).mean()
            dart = _empty_radii(
                emptyspace.dart_esa(static, bounds, n=15, seed=1, k_cand=16384,
                                    _torus=False),
                static).mean()

            # The reference is the optimum, so a sampled search cannot beat it
            # by more than floating-point slack. This is the assertion that
            # caught the reference being wrong: restricted to interior Voronoi
            # vertices it was beaten by up to 41%, because in a box the largest
            # empty sphere usually sits on a face.
            self.assertLessEqual(dart, exact * 1.001, f"dart beat exact at d={d}")
            # And it has to get close, or it is optimizing something else.
            self.assertGreater(dart, exact * 0.85, f"dart lags exact at d={d}")

    def test_dart_converges_towards_exact(self):
        """More candidates must close the gap to the exact answer."""
        if REFERENCE is None:
            self.skipTest("SciPy or examples/emptyspace_reference.py unavailable")

        bounds = np.asarray([[-5.0, 5.0]] * 3)
        static = utils.RandomSampler(np.random.default_rng(0)).sample(40, bounds)
        exact = _empty_radii(
            REFERENCE.delaunay_esa(static, bounds, n=15), static).mean()

        fractions = [
            _empty_radii(
                emptyspace.dart_esa(static, bounds, n=15, seed=1, k_cand=k,
                                    _torus=False),
                static).mean() / exact
            for k in (64, 1024, 16384)
        ]

        self.assertEqual(fractions, sorted(fractions), "k_cand did not converge")
        self.assertGreater(fractions[-1], 0.9)

    def test_delaunay_refuses_high_dimension(self):
        if REFERENCE is None:
            self.skipTest("SciPy or examples/emptyspace_reference.py unavailable")

        with self.assertRaises(ValueError):
            REFERENCE.delaunay_esa(
                self.static, np.asarray([[0.0, 1.0]] * 9), n=2, max_dim=6)

    def test_delaunay_stays_in_bounds(self):
        if REFERENCE is None:
            self.skipTest("SciPy or examples/emptyspace_reference.py unavailable")

        bounds = np.asarray([[-2.0, 8.0], [0.0, 3.0]])
        static = utils.RandomSampler(np.random.default_rng(2)).sample(25, bounds)

        placed = REFERENCE.delaunay_esa(static, bounds, n=10)

        self.assertEqual(placed.shape, (10, 2))
        self.assertTrue(utils.assert_bounds(placed, bounds))
        # No duplicates: a circumcentre must never land on an existing point.
        self.assertTrue(np.all(_empty_radii(placed, static) > 0))


if __name__ == "__main__":
    unittest.main()
