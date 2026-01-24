# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.1"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import unittest

import numpy as np

import pyBlindOpt.functions as functions
import pyBlindOpt.init as init
import pyBlindOpt.utils as utils


class TestInit(unittest.TestCase):
    def setUp(self):
        # Create a shared random generator and sampler for tests requiring them
        self.rng = np.random.default_rng(42)
        self.sampler = utils.RandomSampler(self.rng)

    # --- Random Initialization Tests ---
    def test_random_00(self):
        bounds = np.asarray([[-3.0, 5.0]])
        # Use the explicit get_initial_population wrapper or the sampler directly
        # Assuming init.random maps to get_initial_population or similar
        population = init.get_initial_population(10, bounds, self.sampler)
        self.assertTrue(utils.assert_bounds(population, bounds))
        self.assertEqual(population.shape, (10, 1))

    def test_random_01(self):
        bounds = np.asarray([[-3.0, 5.0], [-5.0, 3.0]])
        population = init.get_initial_population(10, bounds, self.sampler)
        self.assertTrue(utils.assert_bounds(population, bounds))
        self.assertEqual(population.shape, (10, 2))

    # --- Opposition Based Tests ---
    def test_opposition_list_input(self):
        """Test with explicit list/array input to verify mathematical correctness"""
        bounds = np.asarray([[-3.0, 5.0]])
        # Input: [-2], [4.7]
        # Opposites: (-3+5) - (-2) = 4; (-3+5) - 4.7 = -2.7
        # Sphere function: 0 is best.
        # Scores:
        # P1: -2 (sq=4), P2: 4.7 (sq=22.09)
        # O1: 4 (sq=16), O2: -2.7 (sq=7.29)
        # Sorted Best 2: P1 (-2), O2 (-2.7) -> Wait, Sphere minimizes.
        # Best fitness: 4 (P1), 7.29 (O2).

        population = np.array([[-2.0], [4.7]])
        result = init.opposition_based(
            functions.sphere, bounds, population=population, n_pop=2
        )

        # We expect the algorithm to pick the best 2 from the pool of 4
        # Pool: [-2.0], [4.7], [4.0], [-2.7]
        # Scores: 4.0, 22.09, 16.0, 7.29
        # Best two: -2.0 and -2.7

        # Note: Order might vary depending on implementation (sort vs argpartition)
        # We verify membership
        expected_values = {-2.0, -2.7}
        result_values = set(result.flatten())

        self.assertTrue(expected_values.issubset(result_values))

    def test_opposition_sampler_input(self):
        """Test passing a Sampler object"""
        bounds = np.asarray([[-3.0, 5.0], [-5.0, 3.0]])
        result = init.opposition_based(
            functions.sphere, bounds, population=self.sampler, n_pop=10, seed=42
        )
        self.assertTrue(utils.assert_bounds(result, bounds))
        self.assertEqual(result.shape, (10, 2))

    # --- Round Init Tests ---
    def test_round_init_00(self):
        bounds = np.asarray([[-3.0, 5.0]])
        # Updated to pass the required 'sampler' argument
        population = init.round_init(
            functions.sphere, bounds, sampler=self.sampler, n_pop=10, n_rounds=3
        )
        self.assertTrue(utils.assert_bounds(population, bounds))
        self.assertEqual(population.shape, (10, 1))

    def test_round_init_diversity(self):
        """Ensure diversity weighting doesn't crash execution"""
        bounds = np.asarray([[-10, 10], [-10, 10]])
        # High diversity weight
        population = init.round_init(
            functions.sphere,
            bounds,
            sampler=self.sampler,
            n_pop=10,
            n_rounds=5,
            diversity_weight=0.9,
        )
        self.assertEqual(population.shape, (10, 2))

    # --- OBLESA Tests ---
    def test_oblesa_shape_and_bounds(self):
        """Basic sanity check for OBLESA output"""
        bounds = np.asarray([[-3.0, 5.0], [-5.0, 3.0]])
        population = init.oblesa(functions.sphere, bounds, n_pop=10, seed=42)

        self.assertTrue(utils.assert_bounds(population, bounds))
        self.assertEqual(population.shape, (10, 2))

    def test_oblesa_determinism(self):
        """Verify that passing a seed produces identical results"""
        bounds = np.asarray([[-10, 10], [-10, 10]])

        # Run 1
        pop1 = init.oblesa(functions.sphere, bounds, n_pop=5, seed=12345, epochs=10)
        # Run 2
        pop2 = init.oblesa(functions.sphere, bounds, n_pop=5, seed=12345, epochs=10)

        np.testing.assert_array_almost_equal(
            pop1,
            pop2,
            decimal=6,
            err_msg="OBLESA should be deterministic when seed is provided",
        )

    def test_oblesa_polymorphism_sampler(self):
        """Test OBLESA with a Sampler instance passed as population"""
        bounds = np.asarray([[-5, 5]])

        # Create a specific sampler
        my_sampler = utils.RandomSampler(np.random.default_rng(99))

        # Pass it to OBLESA
        pop = init.oblesa(
            functions.sphere, bounds, population=my_sampler, n_pop=8, epochs=5
        )

        self.assertEqual(pop.shape, (8, 1))
        self.assertTrue(utils.assert_bounds(pop, bounds))

    def test_oblesa_polymorphism_array(self):
        """Test OBLESA with an existing ndarray passed as population"""
        bounds = np.asarray([[-10, 10]])
        # User provides specific starting guesses
        initial_guess = np.array([[0.5], [-0.5], [8.0]])

        # n_pop should adapt to input size (3)
        pop = init.oblesa(functions.sphere, bounds, population=initial_guess, epochs=5)

        self.assertEqual(pop.shape, (3, 1))
        self.assertTrue(utils.assert_bounds(pop, bounds))

    def test_oblesa_metric_kwargs(self):
        """Test passing custom metric parameters (e.g., sigma for gaussian)"""
        bounds = np.asarray([[-5, 5], [-5, 5]])

        # Should run without error using gaussian metric and custom sigma
        pop = init.oblesa(
            functions.sphere, bounds, n_pop=5, epochs=5, metric="gaussian", sigma=0.5
        )
        self.assertEqual(pop.shape, (5, 2))

    def test_quasi_opposition_execution(self):
        """Test Quasi-Opposition Based Learning execution"""

        bounds = np.asarray([[-5.0, 5.0], [-5.0, 5.0]])

        # QOBL should return n_pop individuals
        population = init.quasi_opposition_based(
            functions.sphere, bounds, population=self.sampler, n_pop=10, seed=42
        )

        self.assertEqual(population.shape, (10, 2))
        self.assertTrue(utils.assert_bounds(population, bounds))

    def test_quasi_opposition_logic(self):
        """
        Verify QOBL logic: It should sample between Center and Opposite.
        """

        # 1D Bound: [0, 10]. Center = 5.
        bounds = np.asarray([[0.0, 10.0]])

        # Specific Population Input: [1.0] (Fitness 1.0 using Sphere)
        # Opposite = 0 + 10 - 1 = 9.
        # Center = 5.
        # QOBL range for this point: [5, 9] (since 5 < 9)

        # We mock the RNG to control the "Uniform" sample inside QOBL
        # We want to ensure the generated point is indeed within [5, 9]
        # However, since we can't easily mock the internal RNG call without patching,
        # we check the bounds of the output over multiple runs or simply check constraints.

        population_in = np.array([[1.0]])

        # Run QOBL
        result = init.quasi_opposition_based(
            functions.sphere, bounds, population=population_in, n_pop=1, seed=42
        )

        res_val = result[0, 0]

        # The result must be either the original (1.0) or the quasi-opposite.
        # If it selected the quasi-opposite, it MUST be in [5, 9].
        # Sphere minimizes:
        # Orig (1.0) -> Cost 1.0
        # Quasi in [5, 9] -> Cost > 25.0
        # Therefore, QOBL should strictly prefer the Original (1.0) because it's better.

        self.assertAlmostEqual(res_val, 1.0)

        # Now let's try a case where Quasi is better.
        # Point = 9.0 (Cost 81). Opposite = 1.0. Center = 5.
        # Quasi Range: [1, 5].
        # Any point in [1, 5] has Cost < 25, which is better than 81.
        # So it should ALWAYS pick the Quasi point.

        population_bad = np.array([[9.0]])
        result_quasi = init.quasi_opposition_based(
            functions.sphere, bounds, population=population_bad, n_pop=1, seed=42
        )

        # The result must NOT be 9.0
        self.assertNotAlmostEqual(result_quasi[0, 0], 9.0)
        # It must be within [1, 5]
        self.assertTrue(1.0 <= result_quasi[0, 0] <= 5.0)


if __name__ == "__main__":
    unittest.main()
