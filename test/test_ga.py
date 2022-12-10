# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import numpy as np
import optimization.ga as ga


# define objective function
def f1(x):
    return np.power(x, 2)[0]


# define objective function
def f2(x):
    return x[0]**2.0 + x[1]**2.0


# define global variable and callback
total = 0
def callback(epoch, obj_all):
    global total
    total += 1


class TestGA(unittest.TestCase):
    def test_ga_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        solution, objective = ga.genetic_algorithm(f1, bounds, n_iter=100, verbose=False)
        self.assertAlmostEqual(solution[0], 0, 1)
    
    def test_ga_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, objective = ga.genetic_algorithm(f2, bounds, n_iter=100, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(result, desired, decimal=1)
    
    def test_ga_02(self):
        global total
        total = 0
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, objective = ga.genetic_algorithm(f2, bounds, n_iter=10, callback=callback, verbose=False)
        desired = 10
        self.assertEqual(total, desired)