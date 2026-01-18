# coding: utf-8


"""
Test functions for optimization.
"""

__version__ = "0.1"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import numpy as np


def rastrigin(x, a=10.0):
    dim = x.shape[-1]
    return a * dim + np.sum(np.power(x, 2) - a * np.cos(2.0 * np.pi * x), axis=-1)


def sphere(x):
    return np.sum(np.power(x, 2), axis=-1)
