import logging
import numpy as np
import optimization.de as de
import optimization.ga as ga
import optimization.sa as sa
import optimization.hc as hc


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    c1 = np.array([p1[0], p2[1]])
    c2 = np.array([p2[0], p1[1]])
    return [c1, c2]


# mutation operator
def mutation(candidate, r_mut, bounds):
    if np.random.rand() < r_mut:
        solution = ga.get_random_solution(bounds)
        candidate[0] = solution[0]
        candidate[1] = solution[1]

# define objective function
def obj(x):
    return x[0]**2.0 + x[1]**2.0

bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])

solutions = []
methods = ['DE', 'GA', 'SA', 'HC']

solutions.append(de.differential_evolution(obj, bounds, debug=True)) 
solutions.append(ga.genetic_algorithm(obj, bounds, crossover, mutation, debug=True))
solutions.append(sa.simulated_annealing(obj, bounds, debug=True))
solutions.append(hc.hillclimbing(obj, bounds, debug=True))

for solution, method in zip(solutions, methods):
    logger.info(f'Solution {method}: f([{np.around(solution[0], decimals=5)}]) = {solution[1]:.5f}')