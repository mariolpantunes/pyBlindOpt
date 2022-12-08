import unittest
import numpy as np

import optimization.utils as utils


class TestUtils(unittest.TestCase):
    
    def test_utils_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        solution = np.asarray([(10)])
        result = utils.check_bounds(solution, bounds)
        desired = np.asarray([(5.0)])
        np.testing.assert_array_equal(result, desired)
    
    def test_utils_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-1.0, 1.0), (-10.0, 10.0)])
        solution = np.asarray([(10.0, -2.0, 7.0)])
        result = utils.check_bounds(solution, bounds)
        desired = np.asarray([(5.0, -1, 7.0)])
        np.testing.assert_array_equal(result, desired)