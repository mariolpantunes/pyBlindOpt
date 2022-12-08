import unittest
import numpy as np

import optimization.de as de


# define objective function
def f1(x):
    return np.power(x, 2)[0]


# define objective function
def f2(x):
    return x[0]**2.0 + x[1]**2.0


class TestDE(unittest.TestCase):
    def test_de_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        solution, objective = de.differential_evolution(f1, bounds, n_iter=100, verbose=False)
        self.assertAlmostEqual(solution[0], 0, 5)
    
    def test_de_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, objective = de.differential_evolution(f2, bounds, n_iter=100, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(result, desired, decimal=3)