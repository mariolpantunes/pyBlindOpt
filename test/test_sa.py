# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import unittest
import numpy as np
import optimization.sa as sa


# define objective function
def f1(x):
    return np.power(x, 2)[0]


# define objective function
def f2(x):
    return x[0]**2.0 + x[1]**2.0


# define global variable and callback
total = 0
def callback(epoch, best, current):
    global total
    total += 1


class TestSA(unittest.TestCase):
    def test_sa_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        result, _ = sa.simulated_annealing(f1, bounds, n_iter=1500, verbose=False)
        desired = np.array([0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_sa_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, _ = sa.simulated_annealing(f2, bounds, n_iter=1500, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=2)
    
    def test_sa_02(self):
        global total
        total = 0
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        sa.simulated_annealing(f2, bounds, n_iter=10, callback=callback, verbose=False)
        desired = 10
        self.assertEqual(total, desired)