import unittest
import numpy as np

import optimization.ga as ga


# crossover two parents to create two children
def crossover1(p1, p2, r_cross):
    if np.random.rand() < r_cross:
        c1 = np.array([(p1[0]+p2[0])/2])
        c2 = np.array([p1[0]+p2[0]])
        return [c1, c2]
    else:
        return [p1, p2]


# mutation operator
def mutation1(candidate, r_mut, bounds):
    if np.random.rand() < r_mut:
        solution = ga.get_random_solution(bounds)
        candidate[0] = solution[0]


# crossover two parents to create two children
def crossover2(p1, p2, r_cross):
    if np.random.rand() < r_cross:
        c1 = np.array([p1[0], p2[1]])
        c2 = np.array([p2[0], p1[1]])
        return [c1, c2]
    else:
        return [p1, p2]


# mutation operator
def mutation2(candidate, r_mut, bounds):
    if np.random.rand() < r_mut:
        solution = ga.get_random_solution(bounds)
        candidate[0] = solution[0]
        candidate[1] = solution[1]


# define objective function
def f1(x):
    return np.power(x, 2)[0]


# define objective function
def f2(x):
    return x[0]**2.0 + x[1]**2.0


class TestGA(unittest.TestCase):
    def test_ga_00(self):
        bounds = np.asarray([(-5.0, 5.0)])
        solution, objective = ga.genetic_algorithm(f1, bounds, crossover1, mutation1, n_iter=100, verbose=False)
        self.assertAlmostEqual(solution[0], 0, 1)
    
    def test_ga_01(self):
        bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
        result, objective = ga.genetic_algorithm(f2, bounds, crossover2, mutation2, n_iter=100, verbose=False)
        desired = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(result, desired, decimal=1)