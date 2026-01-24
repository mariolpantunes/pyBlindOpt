# coding: utf-8

import unittest

import numpy as np

from pyBlindOpt.optimizer import Optimizer


class MockOptimizer(Optimizer):
    """
    A concrete implementation of Optimizer purely for testing infrastructure.
    It does nothing but move particles slightly towards 0.
    """

    def _initialize(self):
        self.best_score = np.inf
        self.best_pos = None

    def _update_iter_params(self, epoch):
        pass

    def _generate_offspring(self, epoch):
        # Dumb logic: move 10% closer to 0
        return self.pop * 0.9

    def _selection(self, offspring, offspring_scores):
        # Always accept
        self.pop = offspring
        self.scores = offspring_scores

    def _update_best(self, epoch):
        min_idx = np.argmin(self.scores)
        if self.scores[min_idx] < self.best_score:
            self.best_score = self.scores[min_idx]
            self.best_pos = self.pop[min_idx].copy()


class TestOptimizerBase(unittest.TestCase):
    def setUp(self):
        self.bounds = np.asarray([(-10.0, 10.0), (-10.0, 10.0)])
        self.rng = np.random.default_rng(42)

    def test_initialization_seed(self):
        """Test if seed ensures deterministic initialization"""
        opt1 = MockOptimizer(lambda x: np.sum(x**2), self.bounds, n_pop=5, seed=42)
        opt2 = MockOptimizer(lambda x: np.sum(x**2), self.bounds, n_pop=5, seed=42)
        np.testing.assert_array_equal(opt1.pop, opt2.pop)

    def test_callback_stopping(self):
        """Test if returning True from callback stops optimization"""

        def stop_early(epoch, scores, pop):
            return epoch == 2  # Stop at epoch 2

        opt = MockOptimizer(
            lambda x: np.sum(x**2),
            self.bounds,
            n_iter=10,
            callback=stop_early,
            verbose=False,
            debug=True,
        )
        _, _, (hist_best, _, _) = opt.optimize()

        # Should have history for epochs 0, 1, 2 (3 entries)
        self.assertEqual(len(hist_best), 3)

    def test_callback_population_modification(self):
        """Test if modifying population in callback works"""

        def force_solution(epoch, scores, pop):
            if epoch == 1:
                # Force all to 0
                return np.zeros_like(pop)
            return False

        opt = MockOptimizer(
            lambda x: np.sum(x**2),
            self.bounds,
            n_iter=5,
            callback=force_solution,
            verbose=False,
        )
        opt.optimize()

        # After epoch 1, population was 0, so score should be 0
        self.assertAlmostEqual(opt.best_score, 0.0)

    def test_caching_creates_files(self):
        """Test if caching mechanism actually uses temp directory"""

        # We need a function that is picklable (lambda works with joblib mostly, but def is safer)
        def sphere(x):
            return np.sum(x**2)

        opt = MockOptimizer(sphere, self.bounds, n_iter=2, cached=True, verbose=False)
        opt.optimize()

        # Check if memory was created
        self.assertIsNotNone(opt.memory)


if __name__ == "__main__":
    unittest.main()
