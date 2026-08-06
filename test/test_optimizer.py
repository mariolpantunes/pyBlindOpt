# coding: utf-8

import importlib
import unittest
import unittest.mock as mock

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

    @mock.patch("pyBlindOpt.optimizer.tqdm.tqdm")
    def test_verbose_progress_bar_postfix(self, mock_tqdm):
        """Test if the progress bar updates with the best_score when verbose is True."""

        # 1. Setup the mock so that tqdm acts like a normal iterator for our loop
        mock_pbar = mock.MagicMock()
        mock_pbar.__iter__.return_value = [0, 1, 2]  # Simulate 3 epochs
        mock_tqdm.return_value = mock_pbar

        # 2. Run the optimizer with verbose=True
        opt = MockOptimizer(lambda x: np.sum(x**2), self.bounds, n_iter=3, verbose=True)
        opt.optimize()

        # 3. Assertions
        # Check that tqdm was actually initialized
        self.assertTrue(mock_tqdm.called)

        # Check that set_postfix was called during the loop
        self.assertTrue(mock_pbar.set_postfix.called)

        # Check that the final call to set_postfix matches the expected scientific notation format
        expected_format = f"{opt.best_score:.3e}"
        mock_pbar.set_postfix.assert_called_with(best_score=expected_format)


#: Every functional entry point in the package, as `(module, function)`.
ENTRY_POINTS = (
    ("abc_opt", "artificial_bee_colony"),
    ("cs", "cuckoo_search"),
    ("de", "differential_evolution"),
    ("egwo", "enhanced_grey_wolf_optimization"),
    ("fa", "firefly_algorithm"),
    ("ga", "genetic_algorithm"),
    ("gwo", "grey_wolf_optimization"),
    ("hba", "honey_badger_algorithm"),
    ("hc", "hill_climbing"),
    ("hho", "harris_hawks_optimization"),
    ("pso", "particle_swarm_optimization"),
    ("rs", "random_search"),
    ("sa", "simulated_annealing"),
)


class TestSuppliedPopulation(unittest.TestCase):
    """Every optimizer must actually start from the population it is given.

    This is the invariant the whole initialization programme rests on. An
    optimizer that quietly resamples, or that keeps the supplied rows but
    reorders or replaces some of them before the first evaluation, would make
    every acceleration-rate measurement a comparison of initializers that were
    never used. It is also the kind of defect that leaves no trace in the
    output: the run still converges, just from somewhere else.

    Pinning it needs no tolerance and no statistics. The population contains
    the exact optimum of a bowl, so an optimizer that received it reports 0.0
    and one that resampled cannot: the probability of drawing the optimum of a
    continuous function is zero.
    """

    def setUp(self):
        self.bounds = np.asarray([(-10.0, 10.0)] * 3)
        rng = np.random.default_rng(11)
        self.pop = rng.uniform(-10.0, 10.0, (12, 3))
        self.pop[7] = 0.0            # the exact argmin of `sphere`

    @staticmethod
    def sphere(x):
        return float(np.sum(np.asarray(x, float) ** 2))

    def _entry(self, module, name):
        return getattr(importlib.import_module(f"pyBlindOpt.{module}"), name)

    def test_supplied_population_is_used(self):
        for module, name in ENTRY_POINTS:
            with self.subTest(optimizer=module):
                _, score = self._entry(module, name)(
                    self.sphere, self.bounds, population=self.pop.copy(),
                    n_pop=len(self.pop), n_iter=3, seed=7)
                self.assertEqual(
                    score, 0.0,
                    f"{module} did not start from the supplied population")

    def test_supplied_population_is_not_written_through(self):
        """The caller's array survives the run unchanged.

        The benchmark hands one population to several optimizers in turn, so
        an optimizer that evolves the caller's array in place would silently
        hand every later optimizer a different, already-improved start.
        """
        for module, name in ENTRY_POINTS:
            with self.subTest(optimizer=module):
                given = self.pop.copy()
                self._entry(module, name)(
                    self.sphere, self.bounds, population=given,
                    n_pop=len(given), n_iter=3, seed=7)
                np.testing.assert_array_equal(given, self.pop)


if __name__ == "__main__":
    unittest.main()
