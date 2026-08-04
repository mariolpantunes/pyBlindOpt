# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"

import unittest

import numpy as np

import pyBlindOpt.functions as func
import pyBlindOpt.utils as utils


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Shared RNG for reproducible tests
        self.rng = np.random.default_rng(42)

    def test_inherit_docs(self):
        """Test docstring inheritance decorator"""

        class Parent:
            """Parent documentation."""

            pass

        @utils.inherit_docs(Parent)
        def child_func():
            """Child documentation."""
            pass

        doc = child_func.__doc__

        self.assertIsNotNone(doc, "Docstring should not be None")
        self.assertIn("Parent documentation", doc or "")
        self.assertIn("Child documentation", doc or "")

    def test_check_bounds_00(self):
        """Test clipping single dimension"""
        bounds = np.asarray([(-5.0, 5.0)])
        solution = np.asarray([[10.0]])  # Shape (1, 1)

        result = utils.check_bounds(solution, bounds)
        desired = np.asarray([[5.0]])

        np.testing.assert_array_almost_equal(result, desired)

    def test_check_bounds_01(self):
        """Test clipping multiple dimensions"""
        bounds = np.asarray([(-5.0, 5.0), (-1.0, 1.0), (-10.0, 10.0)])
        # Input violates min in dim 1, max in dim 0, valid in dim 2
        solution = np.asarray([[10.0, -2.0, 7.0]])

        result = utils.check_bounds(solution, bounds)
        desired = np.asarray([[5.0, -1.0, 7.0]])

        np.testing.assert_array_almost_equal(result, desired)

    def test_assert_bounds(self):
        """Test boolean bound verification"""
        bounds = np.asarray([[-5.0, 5.0]])
        valid = np.asarray([[0.0], [-5.0], [4.99]])
        invalid = np.asarray([[5.1], [-6.0]])

        self.assertTrue(utils.assert_bounds(valid, bounds))
        self.assertFalse(utils.assert_bounds(invalid, bounds))

    def test_get_random_solution(self):
        """Test single random solution generation"""
        bounds = np.asarray([(-5.0, 5.0), (-1.0, 1.0), (-10.0, 10.0)])

        # Now requires RNG
        result = utils.get_random_solution(bounds, self.rng)

        # Implementation returns single vector (D,)
        self.assertEqual(result.shape, (3,))

        # Valid check (using clipping to verify it doesn't change)
        clipped = utils.check_bounds(result[np.newaxis, :], bounds)
        np.testing.assert_array_equal(result, clipped.flatten())

    def test_random_sampler(self):
        bounds = np.asarray([[-5.0, 5.0], [0.0, 10.0]])
        sampler = utils.RandomSampler(self.rng)

        pop = sampler.sample(100, bounds)
        self.assertEqual(pop.shape, (100, 2))
        self.assertTrue(utils.assert_bounds(pop, bounds))

    def test_hlc_sampler(self):
        """Latin Hypercube Sampler test"""
        bounds = np.asarray([[-5.0, 5.0], [0.0, 10.0]])
        # Note: Class name changed from HLCSampler to LatinHypercubeSampler in recommended code
        # Adjust based on your specific class name
        sampler = utils.HLCSampler(self.rng)

        pop = sampler.sample(50, bounds)
        self.assertEqual(pop.shape, (50, 2))
        self.assertTrue(utils.assert_bounds(pop, bounds))

    def test_sobol_sampler(self):
        """Sobol Sequence Sampler test (Pure NumPy)"""
        # Test standard dimensions
        bounds = np.asarray([[0, 1]] * 5)
        sampler = utils.SobolSampler(self.rng)

        pop = sampler.sample(32, bounds)
        self.assertEqual(pop.shape, (32, 5))
        self.assertTrue(utils.assert_bounds(pop, bounds))

        # Test High Dimensions (Supported up to 40)
        bounds_high = np.asarray([[0, 1]] * 40)
        pop_high = sampler.sample(10, bounds_high)
        self.assertEqual(pop_high.shape, (10, 40))

    def test_sobol_no_dimension_limit(self):
        """Sobol has no dimension ceiling: direction numbers are generated."""
        bounds = np.zeros((64, 2))
        bounds[:, 1] = 1.0

        pop = utils.SobolSampler(self.rng).sample(16, bounds)

        self.assertEqual(pop.shape, (16, 64))
        self.assertTrue(utils.assert_bounds(pop, bounds))

    def test_sobol_primitive_polynomials(self):
        """
        Every generated polynomial must be primitive over GF(2) and distinct.

        This is the property the old hard-coded table violated: five of its
        rows were not primitive and six polynomials were reused across up to
        four dimensions, which is why dimensions above 19 did not describe a
        Sobol sequence at all.
        """
        polys = utils._primitive_polynomials(80)

        self.assertEqual(len(polys), 80)
        self.assertEqual(len(set(polys)), 80, "polynomials must be distinct")
        for s, a in polys:
            self.assertTrue(
                utils._is_primitive((1 << s) | (a << 1) | 1, s),
                f"({s}, {a}) is not primitive over GF(2)",
            )
        # Degree must be non-decreasing: the enumeration goes by degree.
        degrees = [s for s, _ in polys]
        self.assertEqual(degrees, sorted(degrees))

    def test_sobol_stratification(self):
        """
        Each coordinate of a 2^k-point design must be a (0, k, 1)-net.

        With `n = 2**k` points every dimension has to hit each of the `n`
        equal strata exactly once. The previous Gray-code driver derived its
        direction index from `i` instead of `i - 1`, shifting the sequence by
        one point and dropping the origin, so this failed in every dimension.
        """
        n = 128
        bounds = np.zeros((24, 2))
        bounds[:, 1] = 1.0

        pop = utils.SobolSampler(self.rng).sample(n, bounds)
        strata = np.floor(pop * n).astype(int)

        for d in range(bounds.shape[0]):
            self.assertEqual(
                len(np.unique(strata[:, d])), n, f"dimension {d} does not stratify"
            )

    def test_sobol_projections_uncorrelated(self):
        """
        No coordinate pair may correlate strongly when points outnumber dims.

        Correlated projections are the failure mode of unoptimized direction
        numbers, and the reason `_sobol_extend` searches for them rather than
        defaulting to `m_i = 1`. Only asserted for `n >= 4 * d`: with fewer
        points than dimensions some pair must correlate whatever the design.
        """
        for d, n in ((16, 256), (40, 256), (64, 1024)):
            bounds = np.zeros((d, 2))
            bounds[:, 1] = 1.0
            pop = utils.SobolSampler(np.random.default_rng(0)).sample(n, bounds)

            corr = np.corrcoef(pop.T)
            np.fill_diagonal(corr, 0.0)

            self.assertLess(
                float(np.abs(corr).max()), 0.5, f"correlated projection at d={d}"
            )

    def test_sobol_prefix_stable(self):
        """
        The direction numbers for `d` dimensions must be a prefix of those for
        any larger `d`, so the cache can be grown and sliced.
        """
        wide = utils._sobol_extend(50)
        narrow = utils._sobol_extend(12)

        np.testing.assert_array_equal(wide[:12], narrow)

    def test_chaotic_sampler(self):
        """Test Chaotic Map Sampler (Logistic Map)"""
        bounds = np.asarray([[-5.0, 5.0], [0.0, 10.0]])
        # Assuming ChaoticSampler is added to utils
        if not hasattr(utils, "ChaoticSampler"):
            return  # Skip if not implemented yet

        sampler = utils.ChaoticSampler(self.rng)

        # 1. Check Shapes
        pop = sampler.sample(50, bounds)
        self.assertEqual(pop.shape, (50, 2))

        # 2. Check Bounds
        self.assertTrue(utils.assert_bounds(pop, bounds))

        # 3. Check Determinism
        # Chaotic maps are sensitive to initial conditions.
        # Since the initial 'x' is drawn from self.rng, resetting self.rng
        # should produce the exact same chaotic sequence.
        rng_replay = np.random.default_rng(42)
        sampler_replay = utils.ChaoticSampler(rng_replay)
        pop_replay = sampler_replay.sample(50, bounds)

        # Note: self.rng was initialized with 42 in setUp
        # We need to re-initialize a fresh one to compare against 'replay'
        # because self.rng has been advanced by other tests.
        rng_fresh = np.random.default_rng(42)
        sampler_fresh = utils.ChaoticSampler(rng_fresh)
        pop_fresh = sampler_fresh.sample(50, bounds)

        np.testing.assert_array_almost_equal(pop_fresh, pop_replay)

    def test_scale_inv_scale(self):
        """Test Normalization and Denormalization cycle"""
        original = np.array([[10.0], [20.0], [30.0]])

        # Scale to [0, 1]
        scaled, min_v, max_v = utils.scale(original)
        expected_scaled = np.array([[0.0], [0.5], [1.0]])

        np.testing.assert_array_almost_equal(scaled, expected_scaled)

        # Inverse Scale back
        restored = utils.inv_scale(scaled, min_v, max_v)
        np.testing.assert_array_almost_equal(restored, original)

    def test_scale_zero_variance(self):
        """Test scaling when all values are identical (div by zero protection)"""
        # Input: [10, 10, 10] -> Max=10, Min=10 -> Denom=0
        arr = np.array([[10.0], [10.0], [10.0]])

        scaled, min_v, max_v = utils.scale(arr)

        # Expect: All zeros (not NaNs or Infs)
        expected = np.zeros_like(arr)

        np.testing.assert_array_equal(scaled, expected)
        self.assertEqual(min_v, 10.0)
        self.assertEqual(max_v, 10.0)

    def test_score_2_probs_softmax(self):
        """Test Softmax probability conversion"""
        # Minimization problem: -10 is better than 10
        scores = np.array([-10.0, 0.0, 10.0])

        # 1. Standard Temperature (1.0)
        probs = utils.score_2_probs(scores, temperature=1.0)

        self.assertAlmostEqual(np.sum(probs), 1.0)
        # Best score (-10) must have highest probability
        self.assertTrue(probs[0] > probs[1] > probs[2])
        # Softmax ensures no probability is exactly zero
        self.assertTrue(np.all(probs > 0.0))

        # 2. High Temperature (Exploration/Random)
        probs_high = utils.score_2_probs(scores, temperature=100.0)
        # Probabilities should be nearly uniform (approx 0.33 each)
        self.assertTrue(np.allclose(probs_high, 0.333, atol=0.1))

        # 3. Low Temperature (Greedy)
        probs_low = utils.score_2_probs(scores, temperature=0.1)
        # The best score should take almost all probability mass
        self.assertTrue(probs_low[0] > 0.99)

    def test_score_2_probs_flat(self):
        """Test probability distribution when all scores are equal"""
        scores = np.array([5.0, 5.0, 5.0, 5.0])

        # Should return uniform distribution [0.25, 0.25, 0.25, 0.25]
        probs = utils.score_2_probs(scores)

        expected = np.full_like(scores, 0.25)
        np.testing.assert_array_almost_equal(probs, expected)

    def test_global_distances(self):
        """Test vectorized sum of distances"""
        samples = np.array([[0.0], [1.0], [3.0]])
        dists = utils.global_distances(samples)
        expected = np.array([4.0, 3.0, 5.0])
        np.testing.assert_array_almost_equal(dists, expected)

    def test_crowding_distance(self):
        """Test NSGA-II Crowding Distance"""
        samples = np.array([[0.0], [1.0], [2.0], [5.0]])
        crowding = utils.compute_crowding_distance(samples)

        # Internal points check
        # Range=5. P1 (1.0) -> (2-0)/5 = 0.4
        self.assertAlmostEqual(crowding[1], 0.4)

        # Boundaries check (Boosted Finite Max)
        # Max finite is 0.8. Boundary logic is max_dist * 2.0 -> 1.6
        self.assertTrue(crowding[0] > 0.8)
        self.assertTrue(crowding[3] > 0.8)

    def test_compute_objective_vectorized(self):
        """
        Test vectorized evaluation
        """
        pop = np.array([[1, 1], [2, 2], [3, 3]])

        scores = utils.compute_objective(pop, func.sphere, n_jobs=1)
        expected = np.array([2, 8, 18])
        np.testing.assert_array_equal(scores, expected)

    def test_compute_objective_parallel(self):
        """
        Test parallel execution via joblib
        """
        pop = np.array([[1], [2], [3]])

        scores = utils.compute_objective(pop, func.sphere, n_jobs=2)
        expected = np.array([1, 4, 9])
        np.testing.assert_array_equal(scores, expected)

    def test_compute_objective_fallback(self):
        """
        Test fallback to row-by-row when vectorization fails
        """
        pop = np.array([[1, 2], [3, 4]])

        # Define a function that CRASHES if given a 2D matrix
        # This simulates a user function that doesn't support vectorization
        def strict_vector_func(x):
            if x.ndim != 1:
                raise ValueError("I only accept 1D vectors!")
            return np.sum(x)

        # This should catch the ValueError inside utils and fallback to apply_along_axis
        scores = utils.compute_objective(pop, strict_vector_func)

        expected = np.array([3, 7])
        np.testing.assert_array_equal(scores, expected)

    def test_shape_and_type(self):
        """
        Verify the output dimensions and data type.
        """
        n_pop, dim = 50, 3
        steps = utils.levy_flight(n_pop, dim)
        self.assertEqual(steps.shape, (n_pop, dim))
        self.assertTrue(np.issubdtype(steps.dtype, np.floating))

    def test_reproducibility(self):
        """
        Verify that passing a seeded generator produces identical results.
        """
        seed = 42

        # Run 1
        rng1 = np.random.default_rng(seed)
        step1 = utils.levy_flight(100, 2, beta=1.5, rng=rng1)

        # Run 2
        rng2 = np.random.default_rng(seed)
        step2 = utils.levy_flight(100, 2, beta=1.5, rng=rng2)

        np.testing.assert_array_equal(step1, step2)

    def test_valid_values(self):
        """
        Ensure no NaNs or Infs are generated (guard against division by zero).
        """
        # Generate a large sample to increase odds of hitting edge cases
        steps = utils.levy_flight(1000, 10)

        self.assertFalse(np.any(np.isnan(steps)), "Lévy flight produced NaNs")
        self.assertFalse(np.any(np.isinf(steps)), "Lévy flight produced Infs")

    def test_heavy_tail_property(self):
        """
        Statistical Sanity Check:
        Lévy flights (beta < 2) are heavy-tailed.
        They should produce large outlier values ('jumps') much more frequently
        than a standard Normal distribution.
        """
        rng = np.random.default_rng(42)

        # Generate a sample of Lévy steps
        levy_steps = utils.levy_flight(2000, 1, beta=1.5, rng=rng)

        # Generate a sample of Standard Normal steps (approx beta=2.0 behavior)
        normal_steps = rng.standard_normal(2000)

        # Calculate the maximum absolute jump in both
        max_levy = np.max(np.abs(levy_steps))
        max_normal = np.max(np.abs(normal_steps))

        # With beta=1.5, we expect the Levy max to be significantly larger
        # than the Normal max (which rarely exceeds 4 or 5).
        # We assert it is at least 2x larger to be safe but statistically valid.
        self.assertGreater(
            max_levy,
            max_normal * 2,
            f"Lévy tail not heavy enough: Levy Max={max_levy:.2f}, Normal Max={max_normal:.2f}",
        )

    def test_beta_sensitivity(self):
        """
        Verify that changing beta changes the scale/behavior.
        Beta -> 2.0 approaches Gaussian (smaller jumps).
        Beta -> 1.0 approaches Cauchy (massive jumps).
        """
        rng = np.random.default_rng(123)

        # Low beta = Heavy tails (Exploration)
        steps_low_beta = utils.levy_flight(1000, 1, beta=1.1, rng=rng)

        # High beta = Light tails (Exploitation/Gaussian-like)
        steps_high_beta = utils.levy_flight(1000, 1, beta=1.9, rng=rng)

        # The range of values for low beta should be much larger
        range_low = np.ptp(steps_low_beta)  # Peak-to-peak (max - min)
        range_high = np.ptp(steps_high_beta)

        self.assertGreater(range_low, range_high)

    def test_select_population_best(self):
        """Test that pure greedy selection accurately picks lowest scores."""
        pop = np.array([[1.0], [2.0], [3.0], [4.0]])
        scores = np.array([10.0, 1.0, 5.0, 0.5])  # best indices are 3, 1

        selected = utils.select_population(pop, scores, n_pop=2, selection="best")

        # Expected to pick [4.0] and [2.0]
        self.assertEqual(selected.shape, (2, 1))
        self.assertTrue(4.0 in selected)
        self.assertTrue(2.0 in selected)

    def test_select_population_random(self):
        """Test roulette selection executes without shape errors."""
        pop = np.array([[1.0], [2.0], [3.0], [4.0]])
        scores = np.array([10.0, 1.0, 5.0, 0.5])

        # We just verify it executes, returns right shape, and doesn't crash
        selected = utils.select_population(pop, scores, n_pop=2, selection="random")
        self.assertEqual(selected.shape, (2, 1))

    def test_select_population_diversity(self):
        """Test selection when diversity weight is heavily favored."""
        pop = np.array([[1.0], [1.1], [1.2], [9.0]])
        scores = np.array(
            [0.1, 0.1, 0.1, 10.0]
        )  # 9.0 has terrible fitness but great diversity

        selected = utils.select_population(
            pop, scores, n_pop=2, selection="best", diversity_weight=1.0
        )
        self.assertEqual(selected.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
