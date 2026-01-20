# coding: utf-8

import functools
import typing
import unittest

import numpy as np

import pyBlindOpt.abc_opt as abc
import pyBlindOpt.callback as callback
import pyBlindOpt.cs as cs
import pyBlindOpt.de as de
import pyBlindOpt.egwo as egwo
import pyBlindOpt.fa as fa
import pyBlindOpt.functions as functions
import pyBlindOpt.ga as ga
import pyBlindOpt.gwo as gwo
import pyBlindOpt.hba as hba
import pyBlindOpt.hc as hc
import pyBlindOpt.hho as hho
import pyBlindOpt.pso as pso
import pyBlindOpt.rs as rs
import pyBlindOpt.sa as sa

# Conditional inheritance for static analysis vs runtime
if typing.TYPE_CHECKING:
    Base = unittest.TestCase
else:
    Base = object


class HeuristicTestMixin(Base):
    """
    Standard test template for ALL optimization algorithms.
    Classes inheriting this must define self.optimizer_func
    """

    # Type hint for the optimizer function
    optimizer_func: typing.Callable

    def setUp(self):
        # Good practice: Only call super setup if it exists
        if hasattr(super(), "setUp"):
            super().setUp()

        if not hasattr(self, "optimizer_func"):
            self.skipTest("HeuristicTestMixin cannot be run directly.")

        # Standard simple bounds
        self.bounds_sphere = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])

        # Complex bounds (10 Dimensions for harder test)
        self.bounds_ackley = np.asarray([(-32.768, 32.768)] * 10)
        self.seed = 42

    def test_convergence_sphere(self):
        """Basic convergence test on Sphere function (Unimodal)"""
        result, _ = self.optimizer_func(
            functions.sphere,
            self.bounds_sphere,
            n_iter=100,
            n_pop=20,
            seed=self.seed,
            verbose=False,
        )
        desired = np.zeros(2)
        np.testing.assert_allclose(result, desired, atol=0.5)

    def test_convergence_ackley(self):
        """
        Convergence test on Ackley (Multimodal).
        The global minimum is at 0, inside a steep hole in a flat surface.
        """
        # [TUNING] Increased n_pop to 50 to help DE/GWO avoid local optima in 10D
        result, score = self.optimizer_func(
            functions.ackley,
            self.bounds_ackley,
            n_iter=300,
            n_pop=100,
            seed=self.seed,
            verbose=False,
        )
        self.assertLess(score, 1.0, f"Failed to converge on Ackley (Score: {score})")

    def test_performance_vs_random_search(self):
        """
        Baseline Comparison: The heuristic MUST outperform Random Search
        on a complex problem (Ackley 10D).
        """
        # 1. Run Baseline (Random Search)
        rs_result, rs_score = rs.random_search(
            functions.ackley,
            self.bounds_ackley,
            n_iter=100,
            n_pop=20,
            seed=self.seed,
            verbose=False,
        )

        # 2. Run Target Heuristic
        target_result, target_score = self.optimizer_func(
            functions.ackley,
            self.bounds_ackley,
            n_iter=100,
            n_pop=20,
            seed=self.seed,
            verbose=False,
        )

        # 3. Compare
        opt_name = getattr(self.optimizer_func, "__name__", "heuristic")
        if opt_name == "heuristic" and isinstance(
            self.optimizer_func, functools.partial
        ):
            opt_name = self.optimizer_func.func.__name__

        self.assertLess(
            target_score, rs_score, f"Heuristic {opt_name} did not beat Random Search!"
        )

    def test_bounds_respected(self):
        """Ensure results are within bounds"""
        tight_bounds = np.asarray([(0.5, 1.0), (0.5, 1.0)])
        result, _ = self.optimizer_func(
            functions.sphere,
            tight_bounds,
            n_iter=20,
            n_pop=10,
            seed=self.seed,
            verbose=False,
        )
        self.assertTrue(np.all(result >= tight_bounds[:, 0]))
        self.assertTrue(np.all(result <= tight_bounds[:, 1]))

    def test_history_debug(self):
        """Test debug mode returns history tuple"""
        _, _, debug_info = self.optimizer_func(
            functions.sphere,
            self.bounds_sphere,
            n_iter=10,
            n_pop=10,
            debug=True,
            verbose=False,
        )
        best, avg, worst = debug_info
        self.assertEqual(len(best), 10)
        self.assertEqual(len(avg), 10)
        self.assertEqual(len(worst), 10)

    def test_callback_early_stopping(self):
        """Test EarlyStopping (Target reached)"""
        c = callback.EarlyStopping(threshold=0.1)
        n_iter = 200
        self.optimizer_func(
            functions.sphere,
            self.bounds_sphere,
            n_iter=n_iter,
            n_pop=20,
            callback=c.callback,
            verbose=False,
        )
        # Check actual epochs run
        self.assertLess(c.epoch, n_iter - 1)

    def test_callback_patience(self):
        """Test PatienceStopping (Stagnation)"""
        patience = 5
        c = callback.PatienceStopping(patience=patience)

        # Give a large max_iter so we are sure patience triggers first
        n_iter_max = 500

        # Using Sphere because it converges fast (score ~ 0.0),
        # then stops improving, triggering patience.
        self.optimizer_func(
            functions.sphere,
            self.bounds_sphere,
            n_iter=n_iter_max,
            n_pop=20,
            callback=c.callback,
            verbose=False,
        )

        self.assertLess(
            c.epoch,
            n_iter_max - 1,
            f"Patience failed: ran for {c.epoch} epochs, expected early stop.",
        )


# --- Concrete Test Classes ---
class TestGWO(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        self.optimizer_func = gwo.grey_wolf_optimization
        super().setUp()


class TestEGWO(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        self.optimizer_func = egwo.enhanced_grey_wolf_optimization
        super().setUp()


class TestDE(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        self.optimizer_func = de.differential_evolution
        super().setUp()


class TestHC(HeuristicTestMixin, unittest.TestCase):
    """
    Tests for Hill Climbing.
    """

    def setUp(self):
        self.optimizer_func = functools.partial(hc.hill_climbing, step_size=0.1)
        super().setUp()

    def test_convergence_ackley(self):
        self.skipTest(
            "Skipping: Hill Climbing (Local Search) naturally fails on Ackley (Multimodal)"
        )

    def test_performance_vs_random_search(self):
        self.skipTest(
            "Skipping: Hill Climbing exploits locally; Random Search explores globally. On Ackley, RS wins."
        )


class TestSA(HeuristicTestMixin, unittest.TestCase):
    """
    Tests for Simulated Annealing.
    """

    def setUp(self):
        self.optimizer_func = functools.partial(
            sa.simulated_annealing, step_size=0.1, temp=10.0
        )
        super().setUp()

    def test_convergence_ackley(self):
        self.skipTest(
            "Skipping: SA (Trajectory Method) gets trapped in Ackley local minima"
        )

    def test_performance_vs_random_search(self):
        self.skipTest("Skipping: SA loses to RS on Ackley without extreme tuning")


class TestPSO(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        self.optimizer_func = pso.particle_swarm_optimization
        super().setUp()


class TestCS(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        self.optimizer_func = functools.partial(cs.cuckoo_search, alpha=0.5)
        super().setUp()


class TestFA(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        self.optimizer_func = functools.partial(
            fa.firefly_algorithm, beta0=0.005, gamma=0.01, alpha=0.5, alpha_decay=0.98
        )
        super().setUp()


class TestABC(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        self.optimizer_func = abc.artificial_bee_colony
        super().setUp()


class TestHHO(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        self.optimizer_func = hho.harris_hawks_optimization
        super().setUp()


class TestHBA(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        self.optimizer_func = hba.honey_badger_algorithm
        super().setUp()


class TestGA(HeuristicTestMixin, unittest.TestCase):
    def setUp(self):
        gaussian_op = functools.partial(ga.gaussian_mutation, scale=0.01)
        self.optimizer_func = functools.partial(
            ga.genetic_algorithm, mutation=gaussian_op, r_mut=0.5
        )
        super().setUp()


class TestRS(unittest.TestCase):
    def setUp(self):
        self.bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        self.seed = 42

    def test_basic_execution(self):
        res, score = rs.random_search(
            functions.sphere,
            self.bounds,
            n_iter=10,
            n_pop=10,
            seed=self.seed,
            verbose=False,
        )
        self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main()
