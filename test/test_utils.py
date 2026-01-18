# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


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
        solution = np.asarray([[10.0]]) # Shape (1, 1)
        
        result = utils.check_bounds(solution, bounds)
        desired = np.asarray([[5.0]])
        
        np.testing.assert_array_almost_equal_nulp(result, desired)
    
    def test_check_bounds_01(self):
        """Test clipping multiple dimensions"""
        bounds = np.asarray([(-5.0, 5.0), (-1.0, 1.0), (-10.0, 10.0)])
        # Input violates min in dim 1, max in dim 0, valid in dim 2
        solution = np.asarray([[10.0, -2.0, 7.0]]) 
        
        result = utils.check_bounds(solution, bounds)
        desired = np.asarray([[5.0, -1.0, 7.0]])
        
        np.testing.assert_array_almost_equal_nulp(result, desired)
    
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
        
        # Shape check (should be 1D array of size D, or match implementation)
        # The util implementation returns rng.uniform(low, high) which implies (D,)
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
        """Hyper-Latin Cube Sampler test"""
        bounds = np.asarray([[-5.0, 5.0], [0.0, 10.0]])
        sampler = utils.HLCSampler(self.rng)
        
        pop = sampler.sample(50, bounds)
        self.assertEqual(pop.shape, (50, 2))
        self.assertTrue(utils.assert_bounds(pop, bounds))

    def test_sobol_sampler(self):
        """Sobol Sequence Sampler test"""
        # Sobol works best with 2^k samples
        bounds = np.asarray([[0, 1]] * 5) # 5 Dimensions
        sampler = utils.SobolSampler(self.rng)
        
        pop = sampler.sample(32, bounds)
        self.assertEqual(pop.shape, (32, 5))
        self.assertTrue(utils.assert_bounds(pop, bounds))
        
        # Verify it handled bounds correctly (0-1)
        self.assertTrue(np.all(pop >= 0) and np.all(pop <= 1))

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

    def test_score_2_probs(self):
        """Test probability conversion with negative scores"""
        # Minimization problem: -10 is better than 10
        scores = np.array([-10.0, 0.0, 10.0])
        
        probs = utils.score_2_probs(scores)
        
        # Sum must be 1.0
        self.assertAlmostEqual(np.sum(probs), 1.0)
        
        # Best score (-10) should have highest probability
        self.assertTrue(probs[0] > probs[1])
        self.assertTrue(probs[1] > probs[2])

    # --- Distance Metrics Tests ---

    def test_global_distances(self):
        """Test vectorized sum of distances"""
        # Points on a line: 0, 1, 3
        # Dist(0): |0-1| + |0-3| = 1 + 3 = 4
        # Dist(1): |1-0| + |1-3| = 1 + 2 = 3
        # Dist(3): |3-0| + |3-1| = 3 + 2 = 5
        samples = np.array([[0.0], [1.0], [3.0]])
        
        dists = utils.global_distances(samples)
        expected = np.array([4.0, 3.0, 5.0])
        
        np.testing.assert_array_almost_equal(dists, expected)

    def test_crowding_distance(self):
        """Test NSGA-II Crowding Distance"""
        # 1D points: 0, 1, 2, 5
        # Range = 5 - 0 = 5
        # P0 (0): Boundary -> Inf
        # P3 (5): Boundary -> Inf
        # P1 (1): Neighbors (0, 2). Dist = (2 - 0) / 5 = 0.4
        # P2 (2): Neighbors (1, 5). Dist = (5 - 1) / 5 = 0.8
        
        samples = np.array([[0.0], [1.0], [2.0], [5.0]])
        crowding = utils.compute_crowding_distance(samples)
        
        # Check internal points logic
        self.assertAlmostEqual(crowding[1], 0.4)
        self.assertAlmostEqual(crowding[2], 0.8)
        
        # Check boundaries are boosted (largest finite * 2)
        # Max finite is 0.8 -> Boundary should be 1.6
        self.assertAlmostEqual(crowding[0], 1.6)
        self.assertAlmostEqual(crowding[3], 1.6)

    # --- Objective Computation Tests ---

    def test_compute_objective_vectorized(self):
        """Test evaluation of a simple sum-of-squares"""
        # 3 points, 2 dims
        pop = np.array([[1, 1], [2, 2], [3, 3]])
        
        # Function handles vectors or matrices
        def sphere(x):
            # If x is (N, D), sum axis 1 -> (N,)
            if x.ndim == 2:
                return np.sum(x**2, axis=1)
            return np.sum(x**2)

        scores = utils.compute_objective(pop, sphere, n_jobs=1)
        expected = np.array([2, 8, 18])
        
        np.testing.assert_array_equal(scores, expected)

    def test_compute_objective_parallel(self):
        """Test parallel execution via joblib"""
        pop = np.array([[1], [2], [3]])
        
        # Slow function simulation not needed, just logic check
        def simple_sq(x):
            return np.sum(x**2)
            
        # n_jobs=2 forces parallel path
        scores = utils.compute_objective(pop, simple_sq, n_jobs=2)
        expected = np.array([1, 4, 9])
        
        np.testing.assert_array_equal(scores, expected)


if __name__ == '__main__':
    unittest.main()