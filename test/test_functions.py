# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import unittest
import numpy as np
import pyBlindOpt.functions as functions


class TestFunctions(unittest.TestCase):
    def test_functions_00(self):
        x = np.array([0,0])
        result = functions.sphere(x)
        desired = 0.0
        self.assertEqual(result, desired)

    def test_functions_01(self):
        x = np.array([0,0,0,0])
        result = functions.sphere(x)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_functions_02(self):
        x = np.array([0,0])
        result = functions.rastrigin(x)
        desired = 0.0
        self.assertEqual(result, desired)

    def test_functions_03(self):
        x = np.array([0,0,0,0])
        result = functions.rastrigin(x)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_functions_04(self):
        x = np.array([1,0])
        result = functions.sphere(x)
        desired = 1.0
        self.assertEqual(result, desired)
    
    def test_functions_05(self):
        x = np.array([1,0])
        result = functions.rastrigin(x)
        desired = 1.0
        self.assertEqual(result, desired)
    
    def test_functions_06(self):
        x = np.array([1,1])
        result = functions.sphere(x)
        desired = 2.0
        self.assertEqual(result, desired)
    
    def test_functions_07(self):
        x = np.array([1,1])
        result = functions.rastrigin(x)
        desired = 2.0
        self.assertEqual(result, desired)


if __name__ == '__main__':
    unittest.main()