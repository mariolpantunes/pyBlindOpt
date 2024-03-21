# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import numpy as np
import pyBlindOpt.gwo as gwo


# define objective function
def f1(x):
    return np.power(x, 2)[0]


# define objective function
def f2(x):
    return x[0]**2.0 + x[1]**2.0


# define global variable and callback
total = 0
def callback(epoch, obj, pop):
    global total
    total += 1


class TestPSO(unittest.TestCase):
    def test_gwo_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        result, _ = gwo.grey_wolf_optimization(f1, bounds, n_iter=100, verbose=False)
        desired = np.array([0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_gwo_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, _ = gwo.grey_wolf_optimization(f2, bounds, n_iter=100, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_gwo_02(self):
        global total
        total = 0
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        gwo.grey_wolf_optimization(f2, bounds, n_iter=10, callback=callback, verbose=False)
        desired = 10
        self.assertEqual(total, desired)
    
    def test_gwo_03(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = [np.array([1,1]), np.array([-1,1]), np.array([2,-2]), np.array([.5,-.5]), np.array([-.5,.5])]
        result, _ = gwo.grey_wolf_optimization(f2, bounds, population=population, n_iter=100, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_gwo_04(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = [np.array([1,1]), np.array([-1,1]), np.array([2,-2]), np.array([.5,-.5]), np.array([-.5,.5])]
        result, _ = gwo.grey_wolf_optimization(f2, bounds, population=population, n_iter=100, verbose=False)
        self.assertTrue(isinstance(result,np.ndarray))
    
    def test_gwo_05(self):
        n_iter = 100
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = [np.array([1,1]), np.array([-1,1]), np.array([2,-2]), np.array([.5,-.5]), np.array([-.5,.5])]
        _, _, debug = gwo.grey_wolf_optimization(f2, bounds, population=population, n_iter=n_iter, verbose=False, debug=True)
        
        list_best, list_avg, list_worst = debug
        
        self.assertTrue(isinstance(list_best, list))
        self.assertEqual(len(list_best), n_iter)
        self.assertTrue(isinstance(list_avg, list))
        self.assertEqual(len(list_avg), n_iter)
        self.assertTrue(isinstance(list_worst, list))
        self.assertEqual(len(list_worst), n_iter)