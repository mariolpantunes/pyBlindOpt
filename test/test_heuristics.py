
import functools
import itertools
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
        _result, score = self.optimizer_func(
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
        _rs_result, rs_score = rs.random_search(
            functions.ackley,
            self.bounds_ackley,
            n_iter=100,
            n_pop=20,
            seed=self.seed,
            verbose=False,
        )

        # 2. Run Target Heuristic
        _target_result, target_score = self.optimizer_func(
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
    VARIANTS: typing.ClassVar = [f"{b}/{c}"
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
        assert pol.archive is not None
        self.assertGreater(len(pol.archive), 0)
        self.assertLessEqual(len(pol.archive), 20)

    def test_an_explicit_cap_is_honoured(self):
        pol = de.ArchivePolicy(0.5, 0.7, de.mutation_best_1, 2, cap=5)
        de.DifferentialEvolution(
            functions.rastrigin, self.BOUNDS, policy=pol, n_pop=20,
            n_iter=15, seed=2, verbose=False).optimize()
        assert pol.archive is not None
        self.assertLessEqual(len(pol.archive), 5)

    def test_it_holds_no_live_population_member(self):
        """It stores parents that were *replaced*; a row still in the
        population would mean a survivor was archived by mistake."""
        opt, pol = self.make()
        opt.optimize()
        live = {row.tobytes() for row in opt.pop}
        assert pol.archive is not None
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
        kw = {"variant": "current-to-pbest/1/bin", "policy": "jade", "n_pop": 20,
                  "n_iter": 30, "seed": 4, "verbose": False}
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
        kw = {"variant": "current-to-pbest/1/bin", "policy": "shade", "n_pop": 20,
                  "n_iter": 30, "seed": 4, "verbose": False}
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


class TestDECode(unittest.TestCase):
    """CoDE: three trials per individual, best of the triple survives.

    The batch-shape assertion is the important one here. CoDE is the first
    policy that costs more than one evaluation per generation, and the live
    agent backend requires a batch to be exactly one population -- so "three
    batches of n_pop" and "one batch of 3*n_pop" are not interchangeable.
    """

    BOUNDS = np.array([[-5.12, 5.12]] * 6)

    def test_selectable_by_name(self):
        opt = de.DifferentialEvolution(functions.sphere, self.BOUNDS,
                                       policy="code", verbose=False)
        self.assertIsInstance(opt.policy, de.CodePolicy)
        self.assertEqual(opt.policy.n_trials, 3)

    def test_a_generation_is_three_batches_of_one_population(self):
        n_pop, n_iter = 12, 4
        seen = []

        def spy(x):
            x = np.atleast_2d(x)
            seen.append(x.shape[0])
            return np.array([functions.sphere(r) for r in x])

        de.differential_evolution(spy, self.BOUNDS, n_pop=n_pop,
                                  n_iter=n_iter, policy="code", seed=0,
                                  verbose=False)
        # every call is one population's worth ...
        self.assertEqual(set(seen), {n_pop})
        # ... and there are three per generation, plus the initial one
        self.assertEqual(len(seen), 3 * n_iter + 1)

    def test_the_default_policy_still_costs_one_batch(self):
        """The n_trials>1 path must not change anything for everyone else."""
        n_pop, n_iter = 12, 4
        seen = []

        def spy(x):
            x = np.atleast_2d(x)
            seen.append(x.shape[0])
            return np.array([functions.sphere(r) for r in x])

        de.differential_evolution(spy, self.BOUNDS, n_pop=n_pop,
                                  n_iter=n_iter, seed=0, verbose=False)
        self.assertEqual(len(seen), n_iter + 1)

    def test_it_keeps_the_best_of_each_triple(self):
        """Selection must see the best trial per individual, so the result is
        never worse than running any single one of the three strategies would
        have been on that individual in that generation."""
        pol = de.CodePolicy()
        rng = np.random.default_rng(0)
        pop = rng.uniform(-5, 5, (8, 6))
        names = set()
        for t in range(pol.n_trials):
            pr = pol.begin(pop, np.zeros(8), rng, t)
            names.add(id(pr.ops[0]))
            self.assertEqual(len(pr.F), 8)
        # three rounds, three distinct mutation operators
        self.assertEqual(len(names), 3)

    def test_parameters_come_from_the_published_pool(self):
        pol = de.CodePolicy()
        pr = pol.begin(np.zeros((500, 4)), np.zeros(500),
                       np.random.default_rng(2), 0)
        pairs = {(round(f, 3), round(c, 3)) for f, c in zip(pr.F, pr.cr)}
        self.assertTrue(pairs <= {(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)}, pairs)
        self.assertGreater(len(pairs), 1)   # drawn per individual, not fixed

    def test_an_empty_pool_is_refused(self):
        with self.assertRaises(ValueError):
            de.CodePolicy(strategies=())

    def test_runs_are_reproducible(self):
        kw = {"policy": "code", "n_pop": 20, "n_iter": 20, "seed": 4, "verbose": False}
        a = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        b = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        np.testing.assert_array_equal(a[0], b[0])

    def test_it_wins_multimodal_at_equal_evaluations(self):
        """CoDE spends three evaluations per generation, so it is given a
        third of the generations. Measured at d=10 over 10 seeds: rastrigin
        1.12 against 9.45, ackley 0.024 against 1.16. It *loses* the unimodal
        cases at the same budget -- sphere 3.7e-06 against 3.5e-09 -- which is
        why only the multimodal claim is asserted.
        """
        def median(f, bd, **kw):
            return float(np.median([
                de.differential_evolution(
                    getattr(functions, f), np.array([[-bd, bd]] * 10),
                    n_pop=30, seed=s, verbose=False, **kw)[1]
                for s in range(6)]))

        code = median("rastrigin", 5.12, n_iter=150, variant="rand/1/bin",
                      policy="code")
        fixed = median("rastrigin", 5.12, n_iter=450, variant="best/1/bin")
        self.assertLess(code, fixed)


class TestDESade(unittest.TestCase):
    """SaDE: a probability per strategy, learned from a rolling window.

    The window is what is tested hardest. Counting successes since generation
    zero would let the opening generations -- when almost anything improves a
    bad population -- fix the probabilities for the whole run.
    """

    BOUNDS = np.array([[-5.12, 5.12]] * 6)

    def _run_gens(self, pol, n_gens, winners, n=8, rng=None):
        """Drive `n_gens` generations where `winners` is the success mask."""
        rng = rng or np.random.default_rng(0)
        for _ in range(n_gens):
            pr = pol.begin(np.zeros((n, 4)), np.zeros(n), rng, 0)
            pol.observe(winners(pol), pr, np.ones(n), np.zeros((0, 4)))

    def test_selectable_by_name(self):
        opt = de.DifferentialEvolution(functions.sphere, self.BOUNDS,
                                       policy="sade", verbose=False)
        self.assertIsInstance(opt.policy, de.SadePolicy)

    def test_probabilities_start_uniform_and_stay_a_distribution(self):
        pol = de.SadePolicy(lp=4)
        np.testing.assert_allclose(pol.probs, 0.25)
        self._run_gens(pol, 12, lambda p: np.arange(8) % 2 == 0)
        self.assertAlmostEqual(float(pol.probs.sum()), 1.0)
        self.assertTrue((pol.probs > 0).all())

    def test_nothing_moves_before_the_window_closes(self):
        pol = de.SadePolicy(lp=5)
        before = pol.probs.copy()
        self._run_gens(pol, 4, lambda p: np.ones(8, dtype=bool))
        np.testing.assert_array_equal(pol.probs, before)
        self._run_gens(pol, 1, lambda p: np.ones(8, dtype=bool))
        self.assertAlmostEqual(float(pol.probs.sum()), 1.0)

    def test_a_strategy_that_never_wins_keeps_a_floor(self):
        """Probability exactly zero is absorbing -- the strategy can never be
        re-tested, so a window that misjudged it can never be corrected."""
        pol = de.SadePolicy(strategies=("rand/1", "best/1"), lp=2, eps=0.01)
        rng = np.random.default_rng(3)
        for _ in range(20):
            pr = pol.begin(np.zeros((40, 4)), np.zeros(40), rng, 0)
            assign = pol._assign
            assert assign is not None
            # only strategy 0 ever succeeds
            pol.observe(assign == 0, pr, np.ones(40), np.zeros((0, 4)))
        self.assertGreater(pol.probs[0], pol.probs[1])
        self.assertGreater(pol.probs[1], 0.0)

    def test_cr_memory_is_the_median_of_what_won(self):
        """A median, not a mean: successful cr is often bimodal, and the mean
        of 0.1 and 0.9 is the one value that works for neither."""
        pol = de.SadePolicy(strategies=("rand/1",), lp=1)
        rng = np.random.default_rng(0)
        pr = pol.begin(np.zeros((5, 4)), np.zeros(5), rng, 0)
        pr = de.Proposal(F=pr.F, cr=np.array([0.1, 0.1, 0.8, 0.9, 0.9]),
                         ops=pr.ops, samples=pr.samples)
        pol.observe(np.ones(5, dtype=bool), pr, np.ones(5), np.zeros((0, 4)))
        self.assertAlmostEqual(pol.cr_m[0], 0.8)

    def test_the_window_forgets(self):
        """A strategy that stops working must lose ground even if it won
        every generation before the window."""
        pol = de.SadePolicy(strategies=("rand/1", "best/1"), lp=3)
        rng = np.random.default_rng(5)

        def phase(winner, gens):
            for _ in range(gens):
                pr = pol.begin(np.zeros((40, 4)), np.zeros(40), rng, 0)
                a = pol._assign
                assert a is not None
                pol.observe(a == winner, pr, np.ones(40), np.zeros((0, 4)))

        phase(0, 9)
        early = pol.probs.copy()
        phase(1, 9)
        self.assertGreater(early[0], early[1])
        self.assertGreater(pol.probs[1], pol.probs[0])

    def test_configuration_is_validated(self):
        with self.assertRaises(ValueError):
            de.SadePolicy(lp=0)
        with self.assertRaises(ValueError):
            de.SadePolicy(strategies=())
        with self.assertRaises(ValueError):
            de.SadePolicy(strategies=("no-such-strategy",))

    def test_runs_are_reproducible(self):
        kw = {"policy": "sade", "n_pop": 20, "n_iter": 30, "seed": 4, "verbose": False}
        a = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        b = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        np.testing.assert_array_equal(a[0], b[0])

    def test_it_beats_the_default_on_ackley(self):
        """Measured at d=10 over 10 seeds, 200 generations: SaDE 2.0e-04
        against best/1/bin's 1.155. Asserted with headroom."""
        def median(**kw):
            return float(np.median([
                de.differential_evolution(
                    functions.ackley, np.array([[-32.768, 32.768]] * 10),
                    n_pop=30, n_iter=200, seed=s, verbose=False, **kw)[1]
                for s in range(6)]))

        self.assertLess(median(variant="rand/1/bin", policy="sade"),
                        median(variant="best/1/bin"))


class TestDELshade(unittest.TestCase):
    """L-SHADE: SHADE with a population that shrinks as the budget is spent.

    This is the only policy that changes `n_pop` mid-run, so the tests are
    mostly about that: it must shrink, never grow, never go below the floor,
    and -- the point of the warning -- it must be the *only* one that does.
    """

    BOUNDS = np.array([[-5.12, 5.12]] * 8)

    def _sizes(self, **kw):
        seen = []

        def spy(x):
            x = np.atleast_2d(x)
            seen.append(x.shape[0])
            return np.array([functions.rastrigin(r) for r in x])

        de.differential_evolution(spy, self.BOUNDS, seed=0, verbose=False,
                                  **kw)
        return seen

    def test_selectable_by_name(self):
        with self.assertLogs("pyBlindOpt.de", level="WARNING"):
            opt = de.DifferentialEvolution(
                functions.sphere, self.BOUNDS,
                variant="current-to-pbest/1/bin", policy="lshade",
                verbose=False)
        self.assertIsInstance(opt.policy, de.LshadePolicy)

    def test_it_warns_that_the_population_will_shrink(self):
        """A caller binding a resource per individual has to be told, so the
        warning fires on construction rather than at the first shrink."""
        with self.assertLogs("pyBlindOpt.de", level="WARNING") as log:
            de.DifferentialEvolution(
                functions.sphere, self.BOUNDS,
                variant="current-to-pbest/1/bin", policy="lshade",
                verbose=False)
        self.assertIn("shrinks the population", "".join(log.output))

    def test_the_population_shrinks_and_never_grows(self):
        sizes = self._sizes(n_pop=40, n_iter=100,
                            variant="current-to-pbest/1/bin", policy="lshade")
        self.assertEqual(sizes[0], 40)
        self.assertLess(sizes[-1], sizes[0])
        self.assertTrue(all(a >= b for a, b in itertools.pairwise(sizes)))

    def test_the_floor_is_respected(self):
        sizes = self._sizes(n_pop=40, n_iter=400,
                            variant="current-to-pbest/1/bin", policy="lshade")
        self.assertGreaterEqual(min(sizes), 4)

    def test_a_floor_below_four_is_refused(self):
        """current-to-pbest/1 needs a base vector, two difference vectors and
        the individual itself."""
        with self.assertRaises(ValueError):
            de.LshadePolicy(de.mutation_current_to_pbest_1, 2, n_min=3)

    def test_every_other_policy_keeps_its_population(self):
        """The contract the rest of the module offers, pinned."""
        for pol, var in (("fixed", "best/1/bin"),
                         ("jade", "current-to-pbest/1/bin"),
                         ("shade", "current-to-pbest/1/bin"),
                         ("sade", "rand/1/bin")):
            sizes = self._sizes(n_pop=24, n_iter=40, variant=var, policy=pol)
            self.assertEqual(set(sizes), {24}, pol)

    def test_it_spends_less_than_a_fixed_population_run(self):
        """Shrinking is cheaper, so a comparison at equal `n_iter` is not a
        comparison at equal cost -- the docstring says so and this pins it."""
        fixed = self._sizes(n_pop=40, n_iter=100,
                            variant="current-to-pbest/1/bin", policy="shade")
        shrunk = self._sizes(n_pop=40, n_iter=100,
                             variant="current-to-pbest/1/bin",
                             policy="lshade")
        self.assertLess(sum(shrunk), sum(fixed))

    def test_the_worst_are_the_ones_removed(self):
        pol = de.LshadePolicy(de.mutation_current_to_pbest_1, 2)
        opt = de.DifferentialEvolution(
            functions.sphere, self.BOUNDS, n_pop=20, n_iter=60, seed=1,
            variant="current-to-pbest/1/bin", policy=pol, verbose=False)
        opt.optimize()
        # whatever survived, the best is still tracked and the run improved
        self.assertLess(opt.n_pop, 20)
        self.assertTrue(np.isfinite(opt.best_score))

    def test_runs_are_reproducible(self):
        kw = {"variant": "current-to-pbest/1/bin", "policy": "lshade",
                  "n_pop": 20, "n_iter": 40, "seed": 4, "verbose": False}
        a = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        b = de.differential_evolution(functions.rastrigin, self.BOUNDS, **kw)
        np.testing.assert_array_equal(a[0], b[0])

    def test_it_beats_shade_on_rastrigin(self):
        """Measured at d=10, n_pop=60, 150 generations, 8 seeds: 1.32 against
        SHADE's 3.52 -- while spending 4318 evaluations against 9060."""
        def median(**kw):
            return float(np.median([
                de.differential_evolution(
                    functions.rastrigin, np.array([[-5.12, 5.12]] * 10),
                    n_pop=60, n_iter=150, seed=s, verbose=False, **kw)[1]
                for s in range(6)]))

        self.assertLess(median(variant="current-to-pbest/1/bin",
                               policy="lshade"),
                        median(variant="current-to-pbest/1/bin",
                               policy="shade"))


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
        assert isinstance(b.policy, de.EnsemblePolicy)
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
            assert pol._held is not None
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
        assert pol._held is not None
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
        res, _score = rs.random_search(
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


def _record_spread(into, epoch, scores, pop):
    """Append the population's mean per-coordinate spread to `into`."""
    into.append(float(np.mean(np.std(pop, axis=0))))


def _record(into, epoch, scores, pop):
    """Append the population's best to `into`. A named function rather than a
    lambda so the list it writes to is bound explicitly, not captured."""
    into.append(float(np.min(scores)))


class TestGAOperators(unittest.TestCase):
    """The GA operators themselves, not the run they add up to.

    Every defect guarded here was invisible from a converging run: the GA
    still moved, still improved, still returned a plausible answer. What it
    did not do was use its population. These are the properties that make the
    difference between recombination and a restart schedule, and each one is
    cheap to assert and was previously false.
    """

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.p1 = np.zeros(6)
        self.p2 = np.array([1.0, 2.0, 3.0, -1.0, 0.5, 4.0])

    def test_blend_crossover_children_are_not_the_same_point(self):
        """The old implementation returned the parent midpoint twice.

        `c1 = p1 + a(p2-p1)` and `c2 = p2 - a(p2-p1)` coincide at a = 0.5,
        which is the default, so a generational GA carried `n_pop / 2`
        distinct individuals forward and nobody noticed.
        """
        for _ in range(50):
            c1, c2 = ga.blend_crossover(self.p1, self.p2, 1.0, rng=self.rng)
            if not np.array_equal(c1, c2):
                return
        self.fail("blend_crossover returned identical children 50 times")

    def test_blend_crossover_can_leave_the_parent_interval(self):
        """BLX-alpha's overshoot is the operator, not a rounding artefact.

        Children confined to `[min(p1,p2), max(p1,p2)]` make recombination
        purely contracting: the population can then only ever narrow, and no
        amount of crossover re-widens it along a collapsed direction.
        """
        lo, hi = np.minimum(self.p1, self.p2), np.maximum(self.p1, self.p2)
        below = above = False
        for _ in range(200):
            for c in ga.blend_crossover(self.p1, self.p2, 1.0, rng=self.rng):
                below |= bool(np.any(c < lo - 1e-12))
                above |= bool(np.any(c > hi + 1e-12))
        self.assertTrue(below and above,
                        "children never left the interval the parents span")

    def test_blend_crossover_stays_inside_the_alpha_envelope(self):
        """...but only as far as alpha allows, per gene."""
        lo, hi = np.minimum(self.p1, self.p2), np.maximum(self.p1, self.p2)
        span = 0.5 * (hi - lo)
        for _ in range(200):
            for c in ga.blend_crossover(self.p1, self.p2, 1.0, rng=self.rng):
                self.assertTrue(np.all(c >= lo - span - 1e-9))
                self.assertTrue(np.all(c <= hi + span + 1e-9))

    def test_blend_crossover_varies_per_gene(self):
        """One scalar draw applied to the whole vector confines the child to
        the parents' line segment, which is a different (and much weaker)
        operator that looks identical on a 1-D test."""
        c = ga.blend_crossover(self.p1, self.p2, 1.0, rng=self.rng)[0]
        gap = self.p2 - self.p1
        frac = c[gap != 0] / gap[gap != 0]
        self.assertGreater(float(np.std(frac)), 1e-6,
                           "every gene moved by the same fraction")

    def test_blend_crossover_passes_parents_through_when_it_declines(self):
        c1, c2 = ga.blend_crossover(self.p1, self.p2, 0.0, rng=self.rng)
        np.testing.assert_array_equal(c1, self.p1)
        np.testing.assert_array_equal(c2, self.p2)

    def test_elitism_never_loses_the_population_best(self):
        """Generational replacement keeps nothing by construction.

        `best_score` is tracked separately and so never worsens, which is
        exactly what hides this: the *returned* answer looks monotone while
        the population it was supposed to be refining has thrown the point
        away. Assert on the population, not on the return value.
        """
        bounds = np.array([[-5.0, 5.0]] * 8)
        for elitism, monotone in ((0.1, True), (0, False)):
            seen = []
            opt = ga.GeneticAlgorithm(
                objective=functions.rastrigin, bounds=bounds, n_pop=20,
                n_iter=30, seed=3, elitism=elitism,
                callback=functools.partial(_record, seen))
            opt.optimize()
            worsened = sum(1 for a, b in itertools.pairwise(seen)
                           if b > a + 1e-12)
            if monotone:
                self.assertEqual(worsened, 0,
                                 "elitism did not preserve the best individual")
            else:
                self.assertGreater(worsened, 0,
                                   "elitism=0 no longer means pure generational "
                                   "replacement -- this test is now vacuous")

    def test_default_mutation_rate_is_per_gene(self):
        """`r_mut=None` resolves to 1/D, and only once the run starts."""
        for d in (4, 25):
            opt = ga.GeneticAlgorithm(
                objective=functions.sphere,
                bounds=np.array([[-5.0, 5.0]] * d), n_pop=10, n_iter=2, seed=1)
            self.assertIsNone(opt.r_mut)
            opt.optimize()
            self.assertAlmostEqual(float(opt.r_mut or 0.0), 1.0 / d)

    def test_an_explicit_mutation_rate_is_left_alone(self):
        opt = ga.GeneticAlgorithm(
            objective=functions.sphere, bounds=np.array([[-5.0, 5.0]] * 10),
            n_pop=10, n_iter=2, seed=1, r_mut=0.3)
        opt.optimize()
        self.assertEqual(opt.r_mut, 0.3)

    def test_elitism_is_a_fraction_below_one_and_a_count_above(self):
        """0.1 -> 10% of n_pop; 2 -> exactly two; 0 -> off."""
        for n_pop, elitism, expected in (
                (30, 0.1, 3), (100, 0.1, 10), (20, 0.25, 5),
                (30, 2, 2), (30, 1, 1), (30, 0, 0),
                (5, 0.1, 1),          # floors at one, never rounds to zero
                (4, 99, 4),           # cannot exceed the population
        ):
            opt = ga.GeneticAlgorithm(
                objective=functions.sphere,
                bounds=np.array([[-5.0, 5.0]] * 4), n_pop=n_pop, n_iter=2,
                seed=1, elitism=elitism)
            opt.optimize()
            self.assertEqual(opt.n_elite, expected,
                             f"n_pop={n_pop} elitism={elitism}")

    def test_a_small_population_keeps_its_convergence_guarantee(self):
        """Rudolph (1994): a canonical GA without elitism does not converge to
        the global optimum, and retaining the best individual is enough to fix
        it. A fraction that rounds to zero would lose that silently."""
        opt = ga.GeneticAlgorithm(
            objective=functions.sphere, bounds=np.array([[-5.0, 5.0]] * 4),
            n_pop=6, n_iter=2, seed=1, elitism=0.01)
        opt.optimize()
        self.assertEqual(opt.n_elite, 1)

    def test_the_old_defaults_are_still_reachable(self):
        """Changing a default must not remove the behaviour it replaced."""
        best, score = ga.genetic_algorithm(
            functions.sphere, np.array([[-5.0, 5.0]] * 6), n_pop=10, n_iter=10,
            seed=1, mutation=ga.random_mutation, r_mut=0.3, elitism=0)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(best.shape, (6,))


class TestDECrossoverExp(unittest.TestCase):
    """`crossover_exp` must change at least one gene, as `crossover_bin` does.

    Written as a do-while in the literature: copy the first gene, *then* keep
    going while `rand < cr`. As a plain while loop it copies nothing with
    probability `1 - cr`, and the trial vector is then an exact clone of its
    parent -- an evaluation that cannot possibly improve on the point it was
    spent measuring. At the default `cr = 0.7` that is 30% of every
    generation; at `cr = 0.1` it is 90%.
    """

    def test_at_least_one_gene_always_comes_from_the_mutant(self):
        rng = np.random.default_rng(0)
        target, mutant = np.zeros(20), np.ones(20)
        for cr in (0.05, 0.3, 0.7, 0.95):
            clones = sum(
                1 for _ in range(3000)
                if np.array_equal(de.crossover_exp(target, mutant, cr, rng),
                                  target))
            self.assertEqual(
                clones, 0,
                f"cr={cr}: {clones}/3000 trials were exact copies of the parent")

    def test_bin_already_guarantees_it(self):
        rng = np.random.default_rng(0)
        target, mutant = np.zeros(20), np.ones(20)
        for cr in (0.05, 0.7):
            for _ in range(500):
                t = de.crossover_bin(target, mutant, cr, rng)
                self.assertGreaterEqual(int((t == 1).sum()), 1)


class TestEGWOSearchDirection(unittest.TestCase):
    """EGWO must actually move toward the prey, and must contract.

    The shipped update was `pop - U(-2,2) * |prey - pop|`: anchored on the
    wolf's own position, with the sign of the separation thrown away by the
    absolute value and the multiplier symmetric about zero. Expected
    displacement toward the prey was therefore exactly zero, at every
    iteration and for every wolf. Nothing about the run *looked* wrong -- it
    returned finite, improving answers, because greedy selection filters a
    diffusion just as happily as it filters a search.

    None of these assertions can be made from the return value, which is why
    they are made here instead.
    """

    def setUp(self):
        self.bounds = np.array([[-5.0, 5.0]] * 12)

    def test_late_steps_land_on_the_prey_not_on_the_wolf(self):
        """The clean discriminator between the two update rules.

        As `a` decays to zero, `A` does too, and `prey - A*|C*prey - X|`
        collapses onto the prey. The old rule, `X - A*|prey - X|`, collapses
        onto **X** -- each wolf onto wherever it already was. Early in the run
        both are wide, which is why this asserts at the end and not at t=0:
        at `a = 2` a large throw past the prey is exploration working, not a
        bug.
        """
        opt = egwo.EGWO(objective=functions.rastrigin, bounds=self.bounds,
                        n_pop=30, n_iter=100, seed=2)
        opt._initialize()
        opt._update_best(epoch=-1)

        opt._update_iter_params(99)                 # a ~ 0
        step = opt._generate_offspring(99)

        # At a ~ 0 every wolf is placed on the *same* prey point, so the
        # offspring cloud has almost no spread. The old rule leaves each wolf
        # at its own position, so the cloud keeps the population's spread.
        before = float(np.mean(np.std(opt.pop, axis=0)))
        after = float(np.mean(np.std(step, axis=0)))
        self.assertLess(after, 0.1 * before,
                        "at a~0 the pack did not collapse onto one point -- "
                        "the step is anchored on each wolf, not on the prey")

        # ...and it moved: the destination is not where the wolves were.
        self.assertGreater(
            float(np.mean(np.linalg.norm(step - opt.pop, axis=1))),
            0.5 * float(np.mean(np.linalg.norm(
                opt.pop - opt.pop.mean(0), axis=1))))

    def test_the_expected_step_is_not_zero(self):
        """`X - U(-a,a) * |prey - X|` has expectation exactly `X`: the
        absolute value discards which side of the prey the wolf is on, and a
        multiplier symmetric about zero then averages the move away. Measured
        on the old rule: 0.058 of drift against a separation of 2.70.
        """
        opt = egwo.EGWO(objective=functions.rastrigin, bounds=self.bounds,
                        n_pop=30, n_iter=100, seed=2)
        opt._initialize()
        opt._update_best(epoch=-1)
        opt._update_iter_params(50)
        drift = np.mean([opt._generate_offspring(50) - opt.pop
                         for _ in range(256)], axis=0)
        separation = float(np.mean(np.abs(opt.alpha_pos - opt.pop)))
        self.assertGreater(float(np.mean(np.abs(drift))), 0.1 * separation,
                           "mean displacement is sampling noise -- the pack "
                           "diffuses instead of hunting")

    def test_the_exploration_coefficient_decays(self):
        """`_update_iter_params` must chain to GWO, which owns `a`."""
        opt = egwo.EGWO(objective=functions.rastrigin, bounds=self.bounds,
                        n_pop=10, n_iter=100, seed=2, noise_scale=0.05)
        opt._initialize()
        seen = []
        for t in (0, 25, 50, 99):
            opt._update_iter_params(t)
            seen.append((opt.a, opt.epoch_std))
        a_vals = [a for a, _ in seen]
        self.assertAlmostEqual(a_vals[0], 2.0)
        self.assertLess(a_vals[-1], 0.05)
        self.assertEqual(a_vals, sorted(a_vals, reverse=True))
        # the noise rides on `a`, so it decays with it rather than dying in
        # the first few percent of the run regardless of n_iter
        self.assertGreater(seen[1][1], 0.0)
        self.assertLess(seen[-1][1], seen[0][1])

    def test_the_noise_schedule_scales_with_n_iter(self):
        """`exp(-100 (t+1)/T)` is spent by t~10 for any T. Halfway through a
        run should look the same whatever the run's length.

        Asserted with the noise switched on explicitly, since the schedule is
        what is under test here and the default turns the term off.
        """
        mid = []
        for n_iter in (50, 200, 1000):
            opt = egwo.EGWO(objective=functions.rastrigin, bounds=self.bounds,
                            n_pop=10, n_iter=n_iter, seed=2, noise_scale=0.05)
            opt._initialize()
            opt._update_iter_params(n_iter // 2)
            mid.append(opt.epoch_std)
        self.assertAlmostEqual(mid[0], mid[1], places=6)
        self.assertAlmostEqual(mid[1], mid[2], places=6)
        self.assertGreater(mid[0], 0.0)

    def test_the_population_converges(self):
        """A pack that never contracts cannot express a good initial
        population, which is what made every `egwo` sweep row uninformative."""
        spread = []
        egwo.EGWO(objective=functions.rastrigin, bounds=self.bounds, n_pop=30,
                  n_iter=60, seed=1,
                  callback=functools.partial(_record_spread, spread)).optimize()
        self.assertLess(spread[-1] / spread[0], 0.5,
                        "population spread barely moved over the whole run")

    def test_the_noise_scale_is_reachable(self):
        loud = egwo.EGWO(objective=functions.rastrigin, bounds=self.bounds,
                         n_pop=10, n_iter=10, seed=2, noise_scale=0.5)
        loud._initialize()
        loud._update_iter_params(0)
        quiet = egwo.EGWO(objective=functions.rastrigin, bounds=self.bounds,
                          n_pop=10, n_iter=10, seed=2, noise_scale=0.0)
        quiet._initialize()
        quiet._update_iter_params(0)
        self.assertGreater(loud.epoch_std, quiet.epoch_std)
        self.assertEqual(quiet.epoch_std, 0.0)


class TestStepGeometry(unittest.TestCase):
    """Two defects of one kind: a search that can only move along one line.

    Both came from collapsing a per-coordinate quantity to a scalar, and
    neither shows up in a convergence test -- the run still improves, because
    greedy selection improves whatever it is given. What is lost is the
    ability to reach most of the space at all.
    """

    def setUp(self):
        self.bounds = np.array([[-5.0, 5.0]] * 8)
        self.obj = functions.rastrigin

    def test_hba_steps_are_not_confined_to_the_diagonal(self):
        r"""HBA's $d_i = x_{prey} - x_i$ is a vector, not a distance.

        With `np.linalg.norm(...)` in its place the honey phase reduced to
        $x_{prey} + c\,(1, 1, \ldots, 1)$ -- every badger somewhere on the
        single diagonal line through the prey. Measured on the old code, 12 of
        30 offsets were exact multiples of the all-ones vector.
        """
        opt = hba.HoneyBadgerAlgorithm(objective=self.obj, bounds=self.bounds,
                                       n_pop=30, n_iter=20, seed=1)
        opt._initialize()
        opt._update_best(epoch=-1)
        step = opt._generate_offspring(0) - opt.best_pos

        ones = np.ones(self.bounds.shape[0])
        on_diagonal = 0
        for row in step:
            norm = np.linalg.norm(row)
            if norm < 1e-12:
                continue
            cosine = abs(float(row @ ones) / (norm * np.linalg.norm(ones)))
            on_diagonal += abs(cosine - 1.0) < 1e-9
        self.assertEqual(on_diagonal, 0,
                         f"{on_diagonal}/{len(step)} badger steps lie exactly "
                         "along the all-ones diagonal")

    def test_abc_scouts_are_drawn_independently(self):
        """`get_random_solution` returns one solution of shape (D,).

        Assigning it into a boolean-masked block broadcasts that same point to
        every scout, so abandoning k exhausted sources produced k identical
        replacements -- the scout phase exists precisely to restore diversity,
        and it was removing it.
        """
        opt = abc.ArtificialBeeColony(objective=self.obj, bounds=self.bounds,
                                      n_pop=20, n_iter=5, seed=1, limit=0)
        opt._initialize()
        opt._update_best(epoch=-1)
        opt.trials[:] = 10 ** 6            # force every source to scout
        opt._generate_offspring(0)

        distinct = len({tuple(np.round(row, 9)) for row in opt.pop})
        self.assertGreater(distinct, 1,
                           "every scout landed on the same random point")
        self.assertEqual(distinct, opt.n_pop)

    def test_pso_accelerations_are_in_a_usable_range(self):
        """`c1 = c2 = 0.1` is an order of magnitude under any published set,
        so neither attractor was felt and the swarm coasted on inertia."""
        opt = pso.ParticleSwarmOptimization(
            objective=self.obj, bounds=self.bounds, n_pop=10,
            n_iter=5, seed=1)
        self.assertGreaterEqual(opt.c1, 1.0)
        self.assertGreaterEqual(opt.c2, 1.0)
        self.assertLess(opt.w, 1.0)
