# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import numpy as np
import pyBlindOpt.hc as hc
import pyBlindOpt.callback as callback
import pyBlindOpt.functions as functions


class TestHC(unittest.TestCase):
    def test_hc_00(self):
        bounds = np.asarray([(-1.0, 1.0)])
        result, _ = hc.hillclimbing(functions.sphere, bounds, n_iter=1500, verbose=False)
        desired = np.array([0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_hc_01(self):
        bounds = np.asarray([(-1.0, 1.0), (-1.0, 1.0)])
        result, _ = hc.hillclimbing(functions.rastrigin, bounds, n_iter=1500, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1)
    
    def test_hc_02(self):
        c = callback.CountEpochs()
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        hc.hillclimbing(functions.rastrigin, bounds, n_iter=10, callback=c.callback, verbose=False)
        desired = 10
        self.assertEqual(c.epoch, desired)
    
    def test_hc_03(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, _ = hc.hillclimbing(functions.rastrigin, bounds, n_iter=100, verbose=False)
        self.assertTrue(isinstance(result,np.ndarray))
    
    def test_hc_04(self):
        n_iter = 100
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        _, _, debug = hc.hillclimbing(functions.rastrigin, bounds, n_iter=n_iter, verbose=False, debug=True)
        
        self.assertTrue(isinstance(debug, list))
        self.assertEqual(len(debug), n_iter)
    
    def test_hc_05(self):
        threshold = 0.1
        n_iter=1000
        c = callback.EarlyStopping(threshold)
        bounds = np.asarray([(-1.0, 1.0), (-1.0, 1.0)])
        result, objective = hc.hillclimbing(functions.sphere, bounds, n_iter=n_iter, callback=c.callback, verbose=False)
        #print(f'Epoch {c.epoch} -> {result}|{objective}')
        self.assertTrue(c.epoch < (n_iter-1))
        self.assertTrue(objective < threshold)


if __name__ == '__main__':
    unittest.main()