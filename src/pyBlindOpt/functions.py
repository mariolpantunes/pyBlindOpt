# coding: utf-8


'''
Test functions for optimization.
'''


__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import numpy as np


def rastrigin(x, a=10.0):
    return a*len(x)+np.sum(np.power(x, 2)-a*np.cos(2.0*math.pi*x))


def sphere(x):
    return np.sum(np.power(x, 2))