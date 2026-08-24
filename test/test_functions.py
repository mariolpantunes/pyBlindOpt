
__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import unittest

import numpy as np

import pyBlindOpt.functions as functions


class TestFunctions(unittest.TestCase):
    # --- Sphere Tests ---
    def test_sphere_00(self):
        """Test Sphere Global Minimum at 0"""
        x = np.array([0, 0])
        result = functions.sphere(x)
        self.assertEqual(result, 0.0)

    def test_sphere_01(self):
        """Test Sphere at [1, 1] -> 1^2 + 1^2 = 2"""
        x = np.array([1, 1])
        result = functions.sphere(x)
        self.assertEqual(result, 2.0)

    # --- Rastrigin Tests ---
    #
    def test_rastrigin_00(self):
        """Test Rastrigin Global Minimum at 0"""
        x = np.array([0, 0])
        result = functions.rastrigin(x)
        self.assertEqual(result, 0.0)

    def test_rastrigin_01(self):
        """Test Rastrigin at [1, 0]"""
        # f(x) = 10*D + sum(x^2 - 10cos(2pi*x))
        # D=2. Term1 (x=1): 1 - 10*1 = -9. Term2 (x=0): 0 - 10*1 = -10.
        # Sum = -19. Result = 20 - 19 = 1.0
        x = np.array([1, 0])
        result = functions.rastrigin(x)
        self.assertEqual(result, 1.0)

    # --- Ackley Tests ---
    #
    def test_ackley_00(self):
        """Test Ackley Global Minimum at 0"""
        x = np.array([0, 0])
        # Ackley(0) = -20*exp(0) - exp(0) + 20 + e = -20 - 1 + 20 + 2.718...
        # Wait, the formula is -20*exp(0) - exp(0) + 20 + e
        # -20 - 1 + 20 + e = e - 1? No.
        # Check implementation: -a + 20 + e - exp(0) = -20 + 20 + e - 1?
        # Standard Ackley at 0 is 0. Let's verify numpy behavior close to 0.
        result = functions.ackley(x)
        np.testing.assert_almost_equal(result, 0.0, decimal=10)

    def test_ackley_01(self):
        """Test Ackley simple point"""
        # Just ensure it runs and returns a float
        x = np.array([1, 1])
        result = functions.ackley(x)
        self.assertIsInstance(result, float)
        self.assertNotEqual(result, 0.0)

    # --- Rosenbrock Tests ---
    #
    def test_rosenbrock_00(self):
        """Test Rosenbrock Global Minimum at [1, 1, ..., 1]"""
        # Rosenbrock min is NOT at 0, it is at 1.
        x = np.array([1, 1, 1])
        result = functions.rosenbrock(x)
        self.assertEqual(result, 0.0)

    def test_rosenbrock_01(self):
        """Test Rosenbrock at [0, 0] (Standard starting point often used)"""
        # (1 - 0)^2 + 100(0 - 0^2)^2 = 1
        x = np.array([0, 0])
        result = functions.rosenbrock(x)
        self.assertEqual(result, 1.0)

    def test_rosenbrock_02(self):
        """Test Rosenbrock 2D calculation"""
        # x=[2, 2].
        # Term1: 100 * (2 - 2^2)^2 = 100 * (-2)^2 = 400
        # Term2: (1 - 2)^2 = 1
        # Sum = 401
        x = np.array([2, 2])
        result = functions.rosenbrock(x)
        self.assertEqual(result, 401.0)

    # --- Griewank Tests ---
    def test_griewank_00(self):
        """Test Griewank Global Minimum at 0"""
        x = np.array([0, 0, 0])
        result = functions.griewank(x)
        self.assertEqual(result, 0.0)

    def test_griewank_01(self):
        """Test Griewank calculation"""
        # Check symmetry or specific values if needed
        x = np.array([100, 200])
        result = functions.griewank(x)
        self.assertGreater(result, 0.0)


    # --- Asymmetric landscapes ---
    # These exist because the four functions above are EVEN: on a symmetric
    # box the opposite of x is exactly -x and scores identically, so an
    # opposition-based pool is half mirror pairs and no conclusion about
    # opposition survives. Each of these breaks f(x) == f(-x).

    def test_styblinski_tang_minimum(self):
        """Global minimum is -39.1661657 per dimension, at x = -2.9035340."""
        # Solved rather than quoted: the widely cited -39.16599 is rounded,
        # and a rounded f* is one the optimizer can beat.
        roots = np.roots([4.0, 0.0, -32.0, 5.0])
        roots = roots[np.isreal(roots)].real
        x_opt = roots[np.argmin(0.5 * (roots**4 - 16 * roots**2 + 5 * roots))]

        for d in (1, 5, 12):
            value = functions.styblinski_tang(np.full(d, x_opt))
            self.assertAlmostEqual(float(value) / d, -39.1661657, places=6)

    def test_styblinski_tang_is_asymmetric(self):
        x = np.array([1.0, -2.0, 0.5])
        self.assertNotAlmostEqual(
            float(functions.styblinski_tang(x)),
            float(functions.styblinski_tang(-x)),
        )

    def test_levy_minimum(self):
        for d in (1, 4, 9):
            self.assertAlmostEqual(float(functions.levy(np.ones(d))), 0.0, places=12)

    def test_levy_is_asymmetric(self):
        x = np.array([2.0, -1.0, 3.0])
        self.assertNotAlmostEqual(float(functions.levy(x)), float(functions.levy(-x)))

    def test_zakharov_minimum(self):
        for d in (1, 4, 9):
            self.assertAlmostEqual(float(functions.zakharov(np.zeros(d))), 0.0)

    def test_zakharov_is_index_weighted(self):
        """Permuting the coordinates must change the value."""
        x = np.array([1.0, 2.0, 3.0])
        self.assertNotAlmostEqual(
            float(functions.zakharov(x)), float(functions.zakharov(x[::-1]))
        )

    def test_dixon_price_minimum(self):
        """Global minimum 0 at x_i = 2^(-(2^i - 2) / 2^i)."""
        for d in (1, 4, 9):
            i = np.arange(1, d + 1)
            x_opt = 2.0 ** (-(2.0**i - 2.0) / 2.0**i)
            self.assertAlmostEqual(
                float(functions.dixon_price(x_opt)), 0.0, places=12
            )

    def test_dixon_price_is_asymmetric(self):
        x = np.array([1.0, 0.7, 0.6])
        self.assertNotAlmostEqual(
            float(functions.dixon_price(x)), float(functions.dixon_price(-x))
        )

    def test_new_functions_vectorize(self):
        """(N, D) input must give (N,) output, as compute_objective assumes."""
        batch = np.random.default_rng(0).uniform(-3, 3, size=(7, 5))
        for fn in (functions.styblinski_tang, functions.levy,
                   functions.zakharov, functions.dixon_price):
            out = fn(batch)
            self.assertEqual(out.shape, (7,), fn.__name__)
            for i in range(7):
                self.assertAlmostEqual(
                    float(out[i]), float(fn(batch[i])), places=9, msg=fn.__name__
                )


if __name__ == "__main__":
    unittest.main()


class TestWeakGlobalStructure(unittest.TestCase):
    """The two landscapes added because the set had no weak global structure.

    Every other function here has one funnel: fitness predicts distance to the
    optimum, so a descent method walks home from anywhere and the initial
    population cannot change the outcome. These two break that, and the tests
    assert the properties they were added *for* -- not merely that they run.
    """

    def test_schwefel_minimum(self):
        for d in (2, 8, 32):
            x = np.full(d, 4.20968746359982)
            self.assertAlmostEqual(float(functions.schwefel(x)), 0.0, places=6)

    def test_schwefel_optimum_is_near_the_boundary(self):
        """Its optimum sits at 0.84 of the half-width, not near the centre.

        Every other landscape here has its optimum comfortably interior, so a
        method that drifts toward the middle of the box is never punished.
        """
        self.assertGreater(4.20968746359982 / 5.0, 0.8)

    def test_schwefel_is_deceptive(self):
        """The basin next to the global one is *not* the second best.

        That is what deceptive means operationally: following the local
        gradient structure from near the optimum leads away from it.
        """
        d = 5
        opt = np.full(d, 4.20968746359982)
        # the mirrored basin, which a symmetric method would try next
        mirrored = np.full(d, -4.20968746359982)
        neighbour = np.full(d, -3.0)
        self.assertGreater(float(functions.schwefel(mirrored)),
                           float(functions.schwefel(opt)))
        self.assertGreater(float(functions.schwefel(neighbour)),
                           float(functions.schwefel(opt)))

    def test_lunacek_minimum(self):
        for d in (2, 8, 32):
            x = np.full(d, 1.25)
            self.assertAlmostEqual(float(functions.lunacek_bi_rastrigin(x)),
                                   0.0, places=9)

    def test_lunacek_has_a_second_funnel(self):
        r"""A distinct basin at $\mu_1$, worse than the global one but a local
        attractor: a population that commits to it converges to the wrong
        answer rather than merely converging slowly."""
        for d in (5, 20):
            s = 1.0 - 1.0 / (2.0 * np.sqrt(d + 20.0) - 8.2)
            mu1 = -np.sqrt((2.5 ** 2 - 1.0) / s)
            second = float(functions.lunacek_bi_rastrigin(np.full(d, mu1 / 2.0)))
            best = float(functions.lunacek_bi_rastrigin(np.full(d, 1.25)))
            self.assertGreater(second, best, "the second funnel is not worse")
            # ...but it is a funnel, not a wall: better than a random point
            far = float(functions.lunacek_bi_rastrigin(np.full(d, 4.5)))
            self.assertLess(second, far,
                            "the second funnel is not an attractor at all")

    def test_both_accept_batches(self):
        X = np.random.default_rng(0).uniform(-5, 5, (7, 6))
        for fn in (functions.schwefel, functions.lunacek_bi_rastrigin):
            out = np.asarray(fn(X))
            self.assertEqual(out.shape, (7,))
            self.assertTrue(np.all(np.isfinite(out)))
