__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np


# define boundary check operation
def check_bounds(solution, bounds):
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return np.clip(solution, lower, upper)