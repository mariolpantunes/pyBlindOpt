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
        n_iter = 200
        c = callback.EarlyStopping(threshold=0.1)
        self.optimizer_func(
            functions.sphere,
            self.bounds_sphere,
            n_iter=n_iter,
            n_pop=20,
            callback=c,
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
            callback=c,
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


class TestDEPolicyContract(unittest.TestCase):
    """The policy layer must be invisible until something opts in.

    This library runs live in classes and trains agents against a game
    engine, so the contract is stricter than "still converges": seeded runs
    must not move at all, and every generation must still be exactly one
    batch of exactly `n_pop`.
    """

    BOUNDS = np.array([[-5.12, 5.12]] * 4)
    VARIANTS = [f"{b}/{c}"
                for b in ("rand/1", "best/1", "rand/2", "best/2",
                          "current-to-best/1", "current-to-pbest/1",
                          "current-to-rand/1")
                for c in ("bin", "exp")]

    def run_de(self, **kw):
        kw.setdefault("n_pop", 15)
        kw.setdefault("n_iter", 25)
        kw.setdefault("verbose", False)
        return de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)

    def test_default_policy_is_fixed(self):
        opt = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS, verbose=False)
        self.assertIsInstance(opt.policy, de.FixedPolicy)
        self.assertEqual(opt.policy.n_trials, 1)

    def test_every_variant_still_runs_and_is_finite(self):
        for v in self.VARIANTS:
            for sel in ("rand", "tournament"):
                pos, score = self.run_de(variant=v, parent_selection=sel, seed=0)
                self.assertTrue(np.isfinite(pos).all(), f"{v}/{sel}")
                self.assertTrue(np.isfinite(score), f"{v}/{sel}")

    def test_seeded_runs_are_reproducible(self):
        """The property every exercise built on this library depends on."""
        for v in ("best/1/bin", "current-to-pbest/1/exp", "rand/2/bin"):
            a = self.run_de(variant=v, seed=42)
            b = self.run_de(variant=v, seed=42)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1], b[1])

    def test_one_batch_of_exactly_n_pop_per_generation(self):
        """The connected-agent contract: the engine binds `n_pop` slots, so
        a generation may never evaluate more or fewer at once."""
        shapes = []
        opt = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS, n_pop=13, n_iter=6, seed=1,
            verbose=False)
        original = opt.evaluate

        def spy(population):
            shapes.append(np.asarray(population).shape)
            return original(population)

        opt.evaluate = spy
        opt.optimize()
        # one initial evaluation, then one per generation
        self.assertEqual({sh[0] for sh in shapes}, {13})
        self.assertGreaterEqual(len(shapes), 6)

    def test_population_size_is_constant(self):
        opt = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS, n_pop=11, n_iter=10, seed=1,
            verbose=False)
        opt.optimize()
        self.assertEqual(opt.n_pop, 11)
        self.assertEqual(len(opt.pop), 11)

    def test_the_policy_is_told_what_survived(self):
        """Everything adaptive is built on this report; before the policy
        layer the mask was computed and discarded."""
        calls = []

        class Spy(de.FixedPolicy):
            def observe(self, improved, proposal, delta, replaced):
                calls.append((improved.copy(), proposal.F.copy(), delta.copy()))

        opt = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS, n_pop=12, n_iter=5, seed=2,
            verbose=False)
        opt.policy = Spy(opt.F, opt.cr, opt.mutation_op, opt.samples_needed)
        opt.optimize()

        self.assertEqual(len(calls), 5)
        for improved, F_used, delta in calls:
            self.assertEqual(improved.shape, (12,))
            self.assertEqual(F_used.shape, (12,))
            self.assertEqual(delta.dtype, np.float64)
            # a survivor is exactly an individual that did not get worse
            self.assertTrue((delta[improved] >= 0).all())

    def test_fixed_policy_draws_no_randomness_of_its_own(self):
        """If it did, every seeded run in every exercise would shift."""
        rng = np.random.default_rng(5)
        before = rng.bit_generator.state
        de.FixedPolicy(0.5, 0.7, de.mutation_best_1, 2).begin(
            np.zeros((10, 3)), np.zeros(10), rng, 0)
        self.assertEqual(rng.bit_generator.state, before)


class TestDEArchive(unittest.TestCase):
    """JADE's optional external archive.

    Defeated parents are kept and used as the *subtracted* difference vector,
    so it points along the direction of recent progress. The tests pin the
    bookkeeping, since the quality effect is landscape-dependent by the
    paper's own account and is not something a unit test should assert.
    """

    BOUNDS = np.array([[-5.12, 5.12]] * 4)

    def make(self, **kw):
        kw.setdefault("variant", "current-to-pbest/1/bin")
        kw.setdefault("n_pop", 20)
        kw.setdefault("n_iter", 25)
        kw.setdefault("seed", 1)
        pol = de.ArchivePolicy(0.5, 0.7, de.mutation_current_to_pbest_1, 2)
        opt = de.DifferentialEvolution(
            functions.rastrigin, self.BOUNDS, policy=pol, verbose=False, **kw)
        return opt, pol

    def test_selectable_by_name(self):
        opt = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS, policy="archive", verbose=False)
        self.assertIsInstance(opt.policy, de.ArchivePolicy)

    def test_it_fills_and_respects_the_cap(self):
        opt, pol = self.make()
        opt.optimize()
        self.assertGreater(len(pol.archive), 0)
        self.assertLessEqual(len(pol.archive), 20)

    def test_an_explicit_cap_is_honoured(self):
        pol = de.ArchivePolicy(0.5, 0.7, de.mutation_best_1, 2, cap=5)
        de.DifferentialEvolution(
            functions.rastrigin, self.BOUNDS, policy=pol, n_pop=20,
            n_iter=15, seed=2, verbose=False).optimize()
        self.assertLessEqual(len(pol.archive), 5)

    def test_it_holds_no_live_population_member(self):
        """It stores parents that were *replaced*; a row still in the
        population would mean a survivor was archived by mistake."""
        opt, pol = self.make()
        opt.optimize()
        live = {row.tobytes() for row in opt.pop}
        self.assertEqual([a for a in pol.archive if a.tobytes() in live], [])

    def test_the_archive_reaches_the_difference_vectors(self):
        """`augment` must actually substitute, or the archive is dead weight."""
        pol = de.ArchivePolicy(0.5, 0.7, de.mutation_best_1, 2)
        pol._n_pop = 1                     # union is almost all archive
        pol.archive = np.full((4, 3), 99.0)
        rng = np.random.default_rng(0)
        cands = np.zeros((2, 3))
        swapped = sum(bool((pol.augment(cands, rng)[-1] == 99.0).all())
                      for _ in range(200))
        self.assertGreater(swapped, 100)

    def test_an_empty_archive_changes_nothing(self):
        pol = de.ArchivePolicy(0.5, 0.7, de.mutation_best_1, 2)
        pol.archive = np.empty((0, 3))
        cands = np.arange(6, dtype=float).reshape(2, 3)
        np.testing.assert_array_equal(
            pol.augment(cands, np.random.default_rng(0)), cands)

    def test_runs_are_reproducible(self):
        outs = []
        for _ in range(2):
            opt, _ = self.make()
            opt.optimize()
            outs.append(opt.best_pos.copy())
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_the_default_path_still_receives_no_archive(self):
        """`FixedPolicy.augment` must not touch the candidates, or every
        seeded run in every exercise moves."""
        pol = de.FixedPolicy(0.5, 0.7, de.mutation_best_1, 2)
        rng = np.random.default_rng(5)
        before = rng.bit_generator.state
        cands = np.arange(6, dtype=float).reshape(2, 3)
        np.testing.assert_array_equal(pol.augment(cands, rng), cands)
        self.assertEqual(rng.bit_generator.state, before)


class TestDEJade(unittest.TestCase):
    """JADE: p-best mutation, archive, and `F`/`cr` learned from survivors.

    The adaptation is tested by driving `observe` directly with known values,
    rather than by running an optimizer and hoping -- a directional assertion
    on a stochastic run is how a suite becomes flaky.
    """

    BOUNDS = np.array([[-5.12, 5.12]] * 6)

    def test_selectable_by_name(self):
        opt = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS, variant="current-to-pbest/1/bin",
            policy="jade", verbose=False)
        self.assertIsInstance(opt.policy, de.JadePolicy)

    def test_it_warns_when_paired_with_another_mutation(self):
        """JADE is defined on current-to-pbest/1; anything else still runs
        but is not the published algorithm, and should say so."""
        with self.assertLogs("pyBlindOpt.de", level="WARNING") as log:
            de.DifferentialEvolution(
                functions.sphere, self.BOUNDS, variant="best/1/bin",
                policy="jade", verbose=False)
        self.assertIn("current-to-pbest/1", "".join(log.output))

    def test_mu_moves_toward_the_values_that_won(self):
        pol = de.JadePolicy(de.mutation_current_to_pbest_1, 2, mu_F=0.5,
                            mu_cr=0.5, c=0.5)
        pr = de.Proposal(F=np.array([0.9, 0.9, 0.1, 0.1]),
                         cr=np.array([0.2, 0.2, 0.8, 0.8]),
                         ops=[None] * 4, samples=[2] * 4)
        won = np.array([True, True, False, False])
        pol.observe(won, pr, np.ones(4), np.zeros((2, 6)))
        self.assertGreater(pol.mu_F, 0.5)     # 0.9 won -> F rises
        self.assertLess(pol.mu_cr, 0.5)       # 0.2 won -> cr falls

    def test_F_uses_the_lehmer_mean_not_the_arithmetic_one(self):
        """The Lehmer mean weights large values more, which is what stops
        `mu_F` decaying toward small steps that succeed often precisely
        because they barely move."""
        pol = de.JadePolicy(de.mutation_current_to_pbest_1, 2, mu_F=0.5, c=1.0)
        F_won = np.array([0.2, 0.9])
        pr = de.Proposal(F=F_won, cr=np.array([0.5, 0.5]),
                         ops=[None] * 2, samples=[2] * 2)
        pol.observe(np.array([True, True]), pr, np.ones(2), np.zeros((1, 6)))
        lehmer = (F_won ** 2).sum() / F_won.sum()
        self.assertAlmostEqual(pol.mu_F, lehmer)
        self.assertGreater(lehmer, F_won.mean())

    def test_nothing_adapts_when_nothing_survives(self):
        pol = de.JadePolicy(de.mutation_current_to_pbest_1, 2)
        before = (pol.mu_F, pol.mu_cr)
        pr = de.Proposal(F=np.full(3, 0.9), cr=np.full(3, 0.9),
                         ops=[None] * 3, samples=[2] * 3)
        pol.observe(np.zeros(3, dtype=bool), pr, np.zeros(3), np.zeros((0, 6)))
        self.assertEqual((pol.mu_F, pol.mu_cr), before)

    def test_drawn_parameters_stay_in_range(self):
        """`F` is Cauchy, so its tails are heavy; a clamp at zero would pile
        probability on the boundary, hence resampling."""
        for mu in (0.05, 0.5, 0.95):
            pol = de.JadePolicy(de.mutation_current_to_pbest_1, 2, mu_F=mu,
                                mu_cr=mu)
            pr = pol.begin(np.zeros((5000, 4)), np.zeros(5000),
                           np.random.default_rng(0), 0)
            self.assertTrue((pr.F > 0).all() and (pr.F <= 1).all(), mu)
            self.assertTrue((pr.cr >= 0).all() and (pr.cr <= 1).all(), mu)

    def test_adaptation_rate_is_validated(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                de.JadePolicy(de.mutation_current_to_pbest_1, 2, c=bad)

    def test_runs_are_reproducible(self):
        kw = dict(variant="current-to-pbest/1/bin", policy="jade", n_pop=20,
                  n_iter=30, seed=4, verbose=False)
        a = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        b = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        np.testing.assert_array_equal(a[0], b[0])

    def test_it_beats_the_default_where_the_default_fails(self):
        """Margins from a measured run: at d=10 over 25 seeds JADE reached
        0.0031 on rastrigin where best/1/bin managed 7.96. Asserted with
        generous headroom on a smaller budget."""
        def median(**kw):
            return float(np.median([
                de.differential_evolution(
                    functions.rastrigin, np.array([[-5.12, 5.12]] * 10),
                    n_pop=40, n_iter=300, seed=s, verbose=False, **kw)[1]
                for s in range(6)]))

        jade = median(variant="current-to-pbest/1/bin", policy="jade")
        default = median(variant="best/1/bin")
        self.assertLess(jade, default * 0.25)


class TestDEShade(unittest.TestCase):
    """SHADE: a memory of `h` settings instead of JADE's single running mean.

    Driven through `observe` with known values for the same reason as the JADE
    suite -- a directional assertion on a stochastic run is how a suite becomes
    flaky.
    """

    BOUNDS = np.array([[-5.12, 5.12]] * 6)

    def _proposal(self, F, cr):
        n = len(F)
        return de.Proposal(F=np.asarray(F, float), cr=np.asarray(cr, float),
                           ops=[None] * n, samples=[2] * n)

    def test_selectable_by_name(self):
        opt = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS, variant="current-to-pbest/1/bin",
            policy="shade", verbose=False)
        self.assertIsInstance(opt.policy, de.ShadePolicy)

    def test_it_warns_when_paired_with_another_mutation(self):
        with self.assertLogs("pyBlindOpt.de", level="WARNING") as log:
            de.DifferentialEvolution(
                functions.sphere, self.BOUNDS, variant="best/1/bin",
                policy="shade", verbose=False)
        self.assertIn("current-to-pbest/1", "".join(log.output))

    def test_one_slot_is_written_per_generation(self):
        """The point of the memory: a bad generation costs one slot, not the
        whole distribution, and the index cycles."""
        pol = de.ShadePolicy(de.mutation_current_to_pbest_1, 2, h=4)
        pr = self._proposal([0.9, 0.9], [0.3, 0.3])
        for expected_k in (1, 2, 3, 0):
            pol.observe(np.array([True, True]), pr, np.ones(2),
                        np.zeros((2, 6)))
            self.assertEqual(pol.k, expected_k)
        # exactly the four slots touched, none left at the 0.5 default
        self.assertTrue((pol.M_F != 0.5).all())

    def test_the_means_are_weighted_by_improvement(self):
        """A trial that barely improved must not count as much as one that
        halved the objective; an unweighted mean chases frequent settings
        rather than effective ones."""
        pol = de.ShadePolicy(de.mutation_current_to_pbest_1, 2, h=1)
        pr = self._proposal([0.2, 0.9], [0.2, 0.9])
        # second trial improved 99x more, so both means must sit near 0.9
        pol.observe(np.array([True, True]), pr, np.array([0.01, 0.99]),
                    np.zeros((2, 6)))
        self.assertGreater(pol.M_cr[0], 0.85)
        self.assertGreater(pol.M_F[0], 0.85)

    def test_F_uses_the_weighted_lehmer_mean(self):
        pol = de.ShadePolicy(de.mutation_current_to_pbest_1, 2, h=1)
        F = np.array([0.2, 0.9])
        w = np.array([0.5, 0.5])
        pol.observe(np.array([True, True]), self._proposal(F, [0.5, 0.5]),
                    np.array([1.0, 1.0]), np.zeros((2, 6)))
        lehmer = (w * F ** 2).sum() / (w * F).sum()
        self.assertAlmostEqual(pol.M_F[0], lehmer)
        self.assertGreater(lehmer, float((w * F).sum()))

    def test_nothing_is_written_when_nothing_survives(self):
        pol = de.ShadePolicy(de.mutation_current_to_pbest_1, 2, h=3)
        before = (pol.M_F.copy(), pol.M_cr.copy(), pol.k)
        pol.observe(np.zeros(3, dtype=bool), self._proposal([0.9] * 3,
                    [0.9] * 3), np.zeros(3), np.zeros((0, 6)))
        np.testing.assert_array_equal(pol.M_F, before[0])
        np.testing.assert_array_equal(pol.M_cr, before[1])
        self.assertEqual(pol.k, before[2])

    def test_a_generation_of_exact_ties_leaves_the_memory_alone(self):
        """Survivors that only tied their parents are not evidence that their
        settings were better, so writing a slot would be writing noise."""
        pol = de.ShadePolicy(de.mutation_current_to_pbest_1, 2, h=3)
        before = pol.M_F.copy()
        pol.observe(np.array([True, True]), self._proposal([0.9, 0.1],
                    [0.9, 0.1]), np.zeros(2), np.zeros((2, 6)))
        np.testing.assert_array_equal(pol.M_F, before)
        self.assertEqual(pol.k, 0)

    def test_drawn_parameters_stay_in_range(self):
        pol = de.ShadePolicy(de.mutation_current_to_pbest_1, 2, h=6)
        pr = pol.begin(np.zeros((5000, 4)), np.zeros(5000),
                       np.random.default_rng(0), 0)
        self.assertTrue((pr.F > 0).all() and (pr.F <= 1).all())
        self.assertTrue((pr.cr >= 0).all() and (pr.cr <= 1).all())

    def test_p_is_drawn_per_individual(self):
        """SHADE's greediness scales with population size on its own; a single
        shared `p` would reintroduce the constant it exists to remove."""
        pol = de.ShadePolicy(de.mutation_current_to_pbest_1, 2)
        pr = pol.begin(np.zeros((200, 4)), np.zeros(200),
                       np.random.default_rng(1), 0)
        self.assertIsNotNone(pr.p)
        assert pr.p is not None
        self.assertEqual(pr.p.shape, (200,))
        self.assertGreater(len(np.unique(pr.p)), 100)
        self.assertTrue((pr.p >= 2.0 / 200).all() and (pr.p <= 0.2).all())

    def test_memory_size_is_validated(self):
        with self.assertRaises(ValueError):
            de.ShadePolicy(de.mutation_current_to_pbest_1, 2, h=0)

    def test_runs_are_reproducible(self):
        kw = dict(variant="current-to-pbest/1/bin", policy="shade", n_pop=20,
                  n_iter=30, seed=4, verbose=False)
        a = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        b = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        np.testing.assert_array_equal(a[0], b[0])

    def test_it_improves_on_jade(self):
        """Measured on rastrigin at d=10, n_pop=30, 150 iterations over 12
        seeds: SHADE 2.87 median against JADE 8.11 and best/1/bin 9.45.
        Asserted with headroom, since the ordering is the claim, not the gap.
        """
        def median(**kw):
            return float(np.median([
                de.differential_evolution(
                    functions.rastrigin, np.array([[-5.12, 5.12]] * 10),
                    n_pop=30, n_iter=150, seed=s, verbose=False, **kw)[1]
                for s in range(8)]))

        shade = median(variant="current-to-pbest/1/bin", policy="shade")
        jade = median(variant="current-to-pbest/1/bin", policy="jade")
        self.assertLess(shade, jade)


class TestDEEnsemble(unittest.TestCase):
    """`EnsemblePolicy` -- a pool of strategies and parameters, per individual.

    The learning rule is the whole algorithm: a triple that produced a
    surviving trial is kept, one that failed is resampled. Random switching
    with no memory would be a different and weaker method, so that invariant
    is what these tests pin.
    """

    BOUNDS = np.array([[-5.12, 5.12]] * 5)

    def test_selectable_by_name_and_by_instance(self):
        a = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS, policy="ensemble", verbose=False)
        self.assertIsInstance(a.policy, de.EnsemblePolicy)
        b = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS,
            policy=de.EnsemblePolicy(strategies=["rand/1"]), verbose=False)
        self.assertEqual(b.policy.strategy_keys, ["rand/1"])

    def test_unknown_policy_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            de.DifferentialEvolution(
                functions.sphere, self.BOUNDS, policy="nope", verbose=False)
        self.assertIn("ensemble", str(cm.exception))

    def test_every_survivor_keeps_its_triple(self):
        """The learning rule. Note the converse does *not* hold: a resampled
        triple can coincidentally redraw the same values out of a finite
        pool, so this asserts survivors are kept, not that only they are."""
        pol = de.EnsemblePolicy()
        ok = []
        original = pol.observe

        def spy(improved, proposal, delta, replaced):
            before = pol._held.copy()
            original(improved, proposal, delta, replaced)
            kept = (before == pol._held).all(axis=1)
            ok.append(bool(kept[improved].all()))

        pol.observe = spy
        de.DifferentialEvolution(
            functions.rastrigin, self.BOUNDS, policy=pol, n_pop=24,
            n_iter=15, seed=1, verbose=False).optimize()
        self.assertTrue(ok and all(ok))

    def test_failures_are_resampled(self):
        """Otherwise a bad triple would be held forever."""
        pol = de.EnsemblePolicy()
        rng = np.random.default_rng(0)
        pol.begin(np.zeros((6, 3)), np.zeros(6), rng, 0)
        before = pol._held.copy()
        pol.observe(np.zeros(6, dtype=bool), None, np.zeros(6), np.zeros((6, 3)))
        self.assertFalse(np.array_equal(before, pol._held))

    def test_the_whole_pool_is_reachable(self):
        pol = de.EnsemblePolicy()
        rng = np.random.default_rng(2)
        drawn = pol._draw(rng, 3000)
        self.assertEqual(sorted(set(drawn[:, 0].astype(int))),
                         list(range(len(pol._ops))))
        self.assertEqual(sorted(set(drawn[:, 1])), sorted(pol.F_pool))
        self.assertEqual(sorted(set(drawn[:, 2])), sorted(pol.cr_pool))

    def test_custom_pools_are_honoured(self):
        pol = de.EnsemblePolicy(strategies=["best/1"], F_pool=[0.3],
                                cr_pool=[0.9])
        rng = np.random.default_rng(0)
        pr = pol.begin(np.zeros((8, 3)), np.zeros(8), rng, 0)
        np.testing.assert_array_equal(pr.F, np.full(8, 0.3))
        np.testing.assert_array_equal(pr.cr, np.full(8, 0.9))

    def test_runs_are_reproducible(self):
        a = de.differential_evolution(
            functions.rastrigin, self.BOUNDS, policy="ensemble", n_pop=20,
            n_iter=30, seed=9, verbose=False)
        b = de.differential_evolution(
            functions.rastrigin, self.BOUNDS, policy="ensemble", n_pop=20,
            n_iter=30, seed=9, verbose=False)
        np.testing.assert_array_equal(a[0], b[0])

    def test_it_beats_a_single_greedy_strategy_on_a_multimodal_landscape(self):
        """The reason the ensemble exists. Margins come from a measured run,
        not a guess: over 8 seeds at this budget the ensemble solved
        rastrigin on every one (median 0.0) where best/1/bin had median 1.99
        and a worst case of 5.97. Asserted with generous headroom, because a
        threshold picked by intuition is how a suite becomes flaky.
        """
        def median_of(**kw):
            return float(np.median([
                de.differential_evolution(
                    functions.rastrigin, self.BOUNDS, n_pop=40, n_iter=250,
                    seed=s, verbose=False, **kw)[1]
                for s in range(8)
            ]))

        self.assertLess(median_of(policy="ensemble"),
                        median_of(variant="best/1/bin") * 0.5)


class TestDEPBest(unittest.TestCase):
    """`current-to-pbest/1` -- JADE's mutation.

    Every individual is pulled toward its own draw from the fittest `p`
    fraction, instead of all of them toward one global best. The tests that
    matter are the two limits, because they pin the drawing rather than
    merely exercising it: the variant is bracketed by two others already in
    the registry and must reduce to each.
    """

    def setUp(self):
        self.bounds = np.array([[-5.0, 5.0]] * 4)

    def run_de(self, **kw):
        kw.setdefault("n_pop", 20)
        kw.setdefault("n_iter", 40)
        kw.setdefault("seed", 11)
        kw.setdefault("verbose", False)
        return de.differential_evolution(functions.sphere, self.bounds, **kw)

    def test_it_is_registered_and_runs(self):
        self.assertIn("current-to-pbest/1", de.DifferentialEvolution._STRATEGIES)
        pos, score = self.run_de(variant="current-to-pbest/1/bin")
        self.assertEqual(pos.shape, (4,))
        self.assertLess(score, 1e-3)

    def test_small_p_reduces_to_current_to_best(self):
        """The pool floors at one individual, which is the global best."""
        a = self.run_de(variant="current-to-pbest/1/bin", p=1e-9)
        b = self.run_de(variant="current-to-best/1/bin")
        np.testing.assert_allclose(a[0], b[0])
        self.assertEqual(a[1], b[1])

    def test_p_of_one_draws_from_the_whole_population(self):
        """Not identical to current-to-rand/1 -- that draws from the pool of
        difference vectors -- but it must stop behaving like current-to-best,
        which is the property that matters."""
        wide = self.run_de(variant="current-to-pbest/1/bin", p=1.0)
        greedy = self.run_de(variant="current-to-best/1/bin")
        self.assertFalse(np.allclose(wide[0], greedy[0]))

    def make(self, **kw):
        """An optimizer with a known population, for testing `_base_vector`
        directly rather than inferring it from a whole run."""
        kw.setdefault("variant", "current-to-pbest/1/bin")
        kw.setdefault("n_pop", 20)
        opt = de.DifferentialEvolution(
            functions.sphere, self.bounds, seed=3, verbose=False, **kw)
        rng = np.random.default_rng(0)
        opt.pop = rng.random((20, 4))
        opt.scores = np.arange(20, dtype=float)   # individual i has score i
        opt.n_pop = 20
        opt.rng = rng
        opt.best_pos = opt.pop[0]
        return opt

    def test_the_draw_is_per_individual_not_per_generation(self):
        """One draw shared across the population would reintroduce the single
        attractor the variant exists to remove, so the same call must be able
        to return different vectors."""
        opt = self.make(p=0.5)                      # pool of 10
        seen = {tuple(opt._base_vector(j)) for j in range(20)}
        self.assertGreater(len(seen), 1)

    def test_the_draw_only_ever_comes_from_the_fittest_fraction(self):
        """Scores are 0..19, so p=0.25 must only ever return one of the five
        lowest-scoring individuals."""
        opt = self.make(p=0.25)
        allowed = {tuple(row) for row in opt.pop[:5]}
        for _ in range(200):
            self.assertIn(tuple(opt._base_vector(0)), allowed)

    def test_the_pool_never_empties_on_a_small_population(self):
        """floor(p*N) would be 0 for p=0.05 at N=20; it must floor at one."""
        opt = self.make(p=0.001)
        self.assertEqual(tuple(opt._base_vector(0)), tuple(opt.pop[0]))

    def test_p_must_be_a_fraction(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                self.run_de(variant="current-to-pbest/1/bin", p=bad)

    def test_p_is_ignored_by_other_variants(self):
        a = self.run_de(variant="best/1/bin", p=0.05)
        b = self.run_de(variant="best/1/bin", p=0.9)
        np.testing.assert_array_equal(a[0], b[0])

    def test_existing_variants_are_untouched(self):
        """The p-best work must not shift any seeded run already in use."""
        for v in ("rand/1/bin", "best/1/bin", "best/2/exp",
                  "current-to-best/1/bin", "current-to-rand/1/bin"):
            pos, score = self.run_de(variant=v)
            self.assertTrue(np.isfinite(pos).all(), v)
            self.assertTrue(np.isfinite(score), v)


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
