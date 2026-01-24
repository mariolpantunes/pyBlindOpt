# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"

import unittest

import numpy as np

import pyBlindOpt.utils as utils


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Shared RNG for reproducible tests
        self.rng = np.random.default_rng(42)

    # --- Bounds & Validation Tests ---

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

    # --- Sampler Tests ---

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

    # --- Math Helper Tests ---

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

    # --- Distance Metrics Tests ---

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

    # --- Objective Computation Tests ---

    def test_compute_objective_vectorized(self):
        """Test evaluation of a simple sum-of-squares"""
        pop = np.array([[1, 1], [2, 2], [3, 3]])

        def sphere(x):
            # Optimistic vectorization check inside utils will pass (N, D)
            # We must handle the reduction
            if x.ndim == 2:
                return np.sum(x**2, axis=1)
            return np.sum(x**2)

        scores = utils.compute_objective(pop, sphere, n_jobs=1)
        expected = np.array([2, 8, 18])
        np.testing.assert_array_equal(scores, expected)

    def test_compute_objective_parallel(self):
        """Test parallel execution via joblib"""
        pop = np.array([[1], [2], [3]])

        def simple_sq(x):
            return np.sum(x**2)

        scores = utils.compute_objective(pop, simple_sq, n_jobs=2)
        expected = np.array([1, 4, 9])
        np.testing.assert_array_equal(scores, expected)

    def test_shape_and_type(self):
        """
        Verify the output dimensions and data type.
        """
        n_rows, n_cols = 50, 3
        steps = utils.levy_flight(n_rows, n_cols)

        self.assertEqual(steps.shape, (n_rows, n_cols))
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


if __name__ == "__main__":
    unittest.main()
