# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import numpy as np
import pyBlindOpt.sa as sa
import pyBlindOpt.callback as callback
import pyBlindOpt.functions as functions


class TestSA(unittest.TestCase):
    def test_sa_00(self):
        bounds = np.asarray([(-1.0, 1.0)])
        result, _ = sa.simulated_annealing(functions.sphere, bounds, n_iter=1500, verbose=False)
        desired = np.array([0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_sa_01(self):
        bounds = np.asarray([(-1.0, 1.0), (-1.0, 1.0)])
        result, _ = sa.simulated_annealing(functions.rastrigin, bounds, n_iter=1500, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_sa_02(self):
        c = callback.CountEpochs()
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        sa.simulated_annealing(functions.rastrigin, bounds, n_iter=10, callback=c.callback, verbose=False)
        desired = 10
        self.assertEqual(c.epoch, desired)
    
    def test_sa_03(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, _ = sa.simulated_annealing(functions.rastrigin, bounds, n_iter=100, verbose=False)
        self.assertTrue(isinstance(result,np.ndarray))
    
    def test_sa_04(self):
        n_iter = 100
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        _, _, debug = sa.simulated_annealing(functions.rastrigin, bounds, n_iter=n_iter, verbose=False, debug=True)
        
        self.assertTrue(isinstance(debug, list))
        self.assertEqual(len(debug), n_iter)
    
    def test_sa_05(self):
        threshold = 0.1
        n_iter=1000
        c = callback.EarlyStopping(threshold)
        bounds = np.asarray([(-1.0, 1.0), (-1.0, 1.0)])
        result, objective = sa.simulated_annealing(functions.sphere, bounds, n_iter=n_iter, callback=c.callback, verbose=False)
        #print(f'Epoch {c.epoch} -> {result}|{objective}')
        self.assertTrue(c.epoch < (n_iter-1))
        self.assertTrue(objective < threshold)


if __name__ == '__main__':
    unittest.main()