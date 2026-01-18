# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.1"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import unittest

import numpy as np

import pyBlindOpt.callback as callback
import pyBlindOpt.functions as functions
import pyBlindOpt.gwo as gwo


class TestGWO(unittest.TestCase):
    def test_gwo_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        result, _ = gwo.grey_wolf_optimization(
            functions.sphere, bounds, n_iter=100, verbose=False
        )
        desired = np.array([0])
        np.testing.assert_allclose(result, desired, atol=1)

    def test_gwo_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, _ = gwo.grey_wolf_optimization(
            functions.rastrigin, bounds, n_iter=100, verbose=False
        )
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1)

    def test_gwo_02(self):
        c = callback.CountEpochs()
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        gwo.grey_wolf_optimization(
            functions.rastrigin, bounds, n_iter=10, callback=c.callback, verbose=False
        )
        desired = 10
        self.assertEqual(c.epoch, desired)

    def test_gwo_03(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = np.array(
            [
                np.array([1, 1]),
                np.array([-1, 1]),
                np.array([2, -2]),
                np.array([0.5, -0.5]),
                np.array([-0.5, 0.5]),
            ]
        )
        result, _ = gwo.grey_wolf_optimization(
            functions.rastrigin,
            bounds,
            population=population,
            n_iter=100,
            verbose=False,
        )
        desired = np.array([0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1)

    def test_gwo_04(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = np.array(
            [
                np.array([1, 1]),
                np.array([-1, 1]),
                np.array([2, -2]),
                np.array([0.5, -0.5]),
                np.array([-0.5, 0.5]),
            ]
        )
        result, _ = gwo.grey_wolf_optimization(
            functions.rastrigin,
            bounds,
            population=population,
            n_iter=100,
            verbose=False,
        )
        self.assertTrue(isinstance(result, np.ndarray))

    def test_gwo_05(self):
        n_iter = 100
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        population = np.array(
            [
                np.array([1, 1]),
                np.array([-1, 1]),
                np.array([2, -2]),
                np.array([0.5, -0.5]),
                np.array([-0.5, 0.5]),
            ]
        )
        _, _, debug = gwo.grey_wolf_optimization(
            functions.rastrigin,
            bounds,
            population=population,
            n_iter=n_iter,
            verbose=False,
            debug=True,
        )

        history_best, history_avg, history_worst = debug

        self.assertTrue(isinstance(history_best, np.ndarray))
        self.assertEqual(len(history_best), n_iter)
        self.assertTrue(isinstance(history_avg, np.ndarray))
        self.assertEqual(len(history_avg), n_iter)
        self.assertTrue(isinstance(history_worst, np.ndarray))
        self.assertEqual(len(history_worst), n_iter)

    def test_gwo_06(self):
        threshold = 0.1
        n_iter = 100
        c = callback.EarlyStopping(threshold)
        bounds = np.asarray(
            [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        )
        result, objective = gwo.grey_wolf_optimization(
            functions.sphere, bounds, n_iter=n_iter, callback=c.callback, verbose=False
        )
        # print(f'Epoch {c.epoch} -> {result}|{objective}')
        self.assertTrue(c.epoch < (n_iter - 1))
        self.assertTrue(objective < threshold)


if __name__ == "__main__":
    unittest.main()
