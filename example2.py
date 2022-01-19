import logging
import numpy as np
import optimization.de as de

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# define objective function
def f(x):
    return np.power(x, 2)[0]

bounds = np.asarray([(-5.0, 5.0)])
solution = de.differential_evolution(f, bounds, n_iter=1, debug=True)
logger.info(f'F({solution[0]})={f(solution[0])}')
