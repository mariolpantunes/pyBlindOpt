# coding: utf-8

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


if __name__ == "__main__":
    unittest.main()
