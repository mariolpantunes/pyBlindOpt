# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.1"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import unittest

import numpy as np

import pyBlindOpt.init as init
import pyBlindOpt.pso as pso


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


class LRPSO:
    def _f(self, X, w):
        wt = w[np.newaxis].T
        y_hat = wt[0] + np.dot(X, wt[1:])
        return y_hat

    def _cost(self, X, y, w):
        y_hat = self._f(X, w)
        return np.mean(np.power((y - y_hat.flatten()), 2))

    def fit(self, X, y, n_pop=30, iter=100, verbose=False):
        bounds = np.asarray([(-50.0, 50.0), (-50.0, 50.0)])
        population = init.oblesa(lambda w: self._cost(X, y, w), bounds, n_pop=n_pop)
        solution = pso.particle_swarm_optimization(
            lambda w: self._cost(X, y, w),
            bounds,
            population=population,
            n_iter=iter,
            verbose=verbose,
        )
        self.w = solution[0]

    def predict(self, X):
        wt = self.w[np.newaxis].T
        y_hat = wt[0] + np.dot(X, wt[1:])
        return y_hat.flatten()

    def params(self):
        return self.w


class TestLR(unittest.TestCase):
    def test_lr_00(self):
        x = np.array(
            [
                [0.64768854],
                [0.49671415],
                [-0.23413696],
                [-1.72491783],
                [-0.90802408],
                [-1.4123037],
                [-0.46341769],
                [-1.01283112],
                [-0.23415337],
                [0.24196227],
                [-0.46947439],
                [1.57921282],
                [0.76743473],
                [-0.56228753],
                [1.52302986],
                [-1.91328024],
                [0.54256004],
                [-0.1382643],
                [-0.46572975],
                [0.31424733],
            ]
        )

        y = np.array(
            [
                47.45554635,
                18.24731687,
                -14.28633346,
                -80.27447804,
                -21.96406129,
                -56.05545461,
                -28.28595535,
                -55.0480211,
                -9.82349843,
                7.29963131,
                -19.37407957,
                67.82137943,
                41.63247323,
                -22.91880407,
                72.22045141,
                -87.29766413,
                20.29561687,
                -2.1217765,
                -22.37282384,
                21.97582345,
            ]
        )

        lr = LRPSO()
        lr.fit(x, y)
        self.assertLess(rmse(y, lr.predict(x)), 8.0)


if __name__ == "__main__":
    unittest.main()
