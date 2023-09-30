# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import numpy as np
import pyBlindOpt.init as init


# define objective function
def f1(x):
    return np.power(x, 2)[0]


# define objective function
def f2(x):
    return x[0]**2.0 + x[1]**2.0


class TestInit(unittest.TestCase):
    def test_opposition_00(self):
        bounds = np.asarray([(-3.0, 5.0)])
        population = [np.array([-2]), np.array([4.7])]
        result = init.opposition_based(f1, bounds, population=population)
        desired = [np.array([-2]), np.array([-2.7])]
        self.assertEqual(result, desired)

    def test_opposition_00(self):
        bounds = np.asarray([(-3.0, 5.0), [-5.0, 3.0]])
        population = [np.array([-2, 2.5]), np.array([4.7, -2.5])]
        result = init.opposition_based(f1, bounds, population=population)
        desired = [np.array([-2, 2.5]), np.array([-2.7, 0.5])]
        np.testing.assert_array_almost_equal(result, desired, decimal=1)