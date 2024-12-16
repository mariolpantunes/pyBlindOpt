# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import numpy as np
import pyBlindOpt.ga as ga
import pyBlindOpt.callback as callback
import pyBlindOpt.functions as functions


class TestGA(unittest.TestCase):
    def test_ga_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        solution, objective = ga.genetic_algorithm(functions.sphere, bounds, n_iter=100, verbose=False)
        self.assertAlmostEqual(solution[0], 0, 1)
    
    def test_ga_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, objective = ga.genetic_algorithm(functions.rastrigin, bounds, n_iter=100, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(result, desired, decimal=1)
    
    def test_ga_02(self):
        c = callback.CountEpochs()
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, objective = ga.genetic_algorithm(functions.rastrigin, bounds, n_iter=10, callback=c.callback, verbose=False)
        desired = 10
        self.assertEqual(c.epoch, desired)
    
    def test_ga_03(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = [np.array([1,1]), np.array([-1,1]), np.array([2,-2]), np.array([.5,-.5]), np.array([-.5,.5])]
        result, _ = ga.genetic_algorithm(functions.rastrigin, bounds, population=population, n_iter=100, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_ga_04(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = [np.array([1,1]), np.array([-1,1]), np.array([2,-2]), np.array([.5,-.5]), np.array([-.5,.5])]
        result, _ = ga.genetic_algorithm(functions.rastrigin, bounds, population=population, n_iter=100, verbose=False)
        self.assertTrue(isinstance(result,np.ndarray))
    
    def test_ga_05(self):
        n_iter = 100
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = [np.array([1,1]), np.array([-1,1]), np.array([2,-2]), np.array([.5,-.5]), np.array([-.5,.5])]
        _, _, debug = ga.genetic_algorithm(functions.rastrigin, bounds, population=population, n_iter=n_iter, verbose=False, debug=True)
        
        list_best, list_avg, list_worst = debug
        
        self.assertTrue(isinstance(list_best, list))
        self.assertEqual(len(list_best), n_iter)
        self.assertTrue(isinstance(list_avg, list))
        self.assertEqual(len(list_avg), n_iter)
        self.assertTrue(isinstance(list_worst, list))
        self.assertEqual(len(list_worst), n_iter)
    
    def test_ga_06(self):
        threshold = 0.1
        n_iter=100
        c = callback.EarlyStopping(threshold)
        bounds = np.asarray([(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])
        result, objective = ga.genetic_algorithm(functions.sphere, bounds, n_iter=n_iter, callback=c.callback, verbose=False)
        #print(f'Epoch {c.epoch} -> {result}|{objective}')
        self.assertTrue(c.epoch < (n_iter-1))
        self.assertTrue(objective < threshold)


if __name__ == '__main__':
    unittest.main()