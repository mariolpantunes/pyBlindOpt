# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import numpy as np
import optimization.de as de


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

class TestDE(unittest.TestCase):
    def test_de_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        result, objective = de.differential_evolution(f1, bounds, n_iter=100, verbose=False)
        desired = np.array([0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_de_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, objective = de.differential_evolution(f2, bounds, n_iter=100, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_de_02(self):
        global total
        total = 0
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, objective = de.differential_evolution(f2, bounds, n_iter=10, callback=callback, verbose=False)
        desired = 10
        self.assertEqual(total, desired)
    
    def test_de_03(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = [np.array([1,1]), np.array([-1,1]), np.array([2,-2]), np.array([.5,-.5]), np.array([-.5,.5])]
        result, _ = de.differential_evolution(f2, bounds, population=population, n_iter=100, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1)