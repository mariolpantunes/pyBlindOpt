__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import typing
import joblib
import logging
import tempfile
import numpy as np


from tqdm import tqdm


logger = logging.getLogger(__name__)


# Cache from joblib
location = tempfile.gettempdir()
memory = joblib.Memory(location, verbose=0)


def simulated_annealing(objective:typing.Callable, bounds:np.ndarray, n_iter:int=200, step_size:float=0.01, temp:float=20.0, cached=False, debug=False) -> list:
    """
    Simulated annealing algorithm.

    Args:
        objective (typing.Callable): objective fucntion
        bounds (np.ndarray): the bounds of valid solutions
        n_iter (int): the number of iterations (default 200)
        step_size (float): the step size (default 0.01)
        temp (float): initial temperature (default 20.0)

    Returns:
        list: [solution, solution_cost]
    """

    # cache the initial objective function
    if cached:
        # Cache from joblib
        location = tempfile.gettempdir()
        memory = joblib.Memory(location, verbose=0)
        objective_cache = memory.cache(objective)
    else:
        objective_cache = objective

	# min and max for each bound
    bounds_max = bounds.max(axis = 1)
    bounds_min = bounds.min(axis = 1)
    # generate an initial point
    best = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate the initial point
    best_cost = objective_cache(best)
	# current working solution
    curr, curr_cost = best, best_cost
    cost_iter = []
	# run the algorithm
    for i in tqdm(range(n_iter), disable=not debug):
		# take a step
        candidate = curr + np.random.randn(len(bounds)) * step_size
        # Fix out of bounds value
        candidate = np.minimum(candidate, bounds_max)
        candidate = np.maximum(candidate, bounds_min)
		# evaluate candidate point
        candidate_cost = objective_cache(candidate)
		# check for new best solution
        #print(f'{i}/{n_iterations} -> {candidate} -> {candidate_cost}')
        if candidate_cost < best_cost:
			# store new best point
            best, best_eval = candidate, candidate_cost
            cost_iter.append(best_eval)
			# report progress
            #logger.info('>%d f(%s) = %.5f' % (i, best, best_cost))
		# difference between candidate and current point evaluation
        diff = candidate_cost - curr_cost
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = np.exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or np.random.rand() < metropolis:
            # store the new current point
            curr, curr_cost = candidate, candidate_cost
    
    if cached:
        memory.clear(warn=False)

    if debug:
        return (best, best_eval, np.array(cost_iter))
    else:
        return (best, best_eval)
