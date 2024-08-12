# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import numpy as np
import pyBlindOpt.utils as utils


class TestUtils(unittest.TestCase):
    
    def test_utils_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        solution = np.asarray([(10)])
        result = utils.check_bounds(solution, bounds)
        desired = np.asarray([(5.0)])
        np.testing.assert_array_almost_equal_nulp(result, desired)
    
    def test_utils_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-1.0, 1.0), (-10.0, 10.0)])
        solution = np.asarray([(10.0, -2.0, 7.0)])
        result = utils.check_bounds(solution, bounds)
        desired = np.asarray([(5.0, -1, 7.0)])
        np.testing.assert_array_almost_equal_nulp(result, desired)
    
    def test_utils_02(self):
        bounds = np.asarray([(-5.0, 5.0), (-1.0, 1.0), (-10.0, 10.0)])
        result = utils.get_random_solution(bounds)
        desired = utils.check_bounds(result, bounds)
        np.testing.assert_array_almost_equal_nulp(result, desired)


if __name__ == '__main__':
    unittest.main()