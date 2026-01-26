# coding: utf-8

"""
Optimization Utilities.

Provides mathematical helpers, sampling strategies, and evaluation logic
required by various optimization algorithms.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import abc
import functools
import inspect
import math
from collections.abc import Callable

import joblib
import numpy as np


def inherit_docs(from_obj):
    """
    Unified decorator to inherit docstrings from either a class or a function.

    Args:
        from_obj: The source class or function to pull documentation from.
    """

    def decorator(func):
        # 1. Determine the source of the docstring
        if inspect.isclass(from_obj):
            # Prefer __init__ docstring, fallback to class docstring
            source_doc = from_obj.__init__.__doc__ or from_obj.__doc__
            header = f"\nBase Parameters (from {from_obj.__name__}):\n"
        else:
            source_doc = from_obj.__doc__
            header = f"\nInherited Parameters (from {from_obj.__name__}):\n"

        # 2. Append the documentation if it exists
        if source_doc:
            current_doc = func.__doc__ or ""
            # Use inspect.cleandoc to fix indentation issues from multiline strings
            func.__doc__ = f"{current_doc}\n{header}{inspect.cleandoc(source_doc)}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def scale(
    arr: np.ndarray,
    min_val: float | np.ndarray | None = None,
    max_val: float | np.ndarray | None = None,
) -> tuple[np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Scales an array to the [0, 1] range using Min-Max scaling.

    Scales an array to the range $[0, 1]$.
    $$ x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}} $$

    Args:
        arr (np.ndarray): The input array to scale.
        min_val (float | np.ndarray | None, optional): Minimum value for scaling. If None, computed from arr.
        max_val (float | np.ndarray | None, optional): Maximum value for scaling. If None, computed from arr.

    Returns:
        tuple[np.ndarray, float | np.ndarray, float | np.ndarray]:
            - The scaled array.
            - The minimum value used.
            - The maximum value used.
    """
    # Use strict temporary variables to ensure type safety (guaranteed not None)
    actual_min = np.min(arr) if min_val is None else min_val
    actual_max = np.max(arr) if max_val is None else max_val

    # Avoid division by zero if max == min
    denominator = actual_max - actual_min

    if np.any(denominator == 0):
        scl_arr = np.zeros_like(arr)
    else:
        scl_arr = (arr - actual_min) / denominator

    return scl_arr, actual_min, actual_max


def inv_scale(
    scl_arr: np.ndarray, min_val: float | np.ndarray, max_val: float | np.ndarray
) -> np.ndarray:
    """
    Inverse scales an array from [0, 1] back to the original range.

    Restores values from $[0, 1]$ to $[x_{min}, x_{max}]$.

    Args:
        scl_arr (np.ndarray): The scaled array.
        min_val (float | np.ndarray): The minimum value used in the original scaling.
        max_val (float | np.ndarray): The maximum value used in the original scaling.

    Returns:
        np.ndarray: The array rescaled to the original range.
    """
    return scl_arr * (max_val - min_val) + min_val


class Sampler(abc.ABC):
    """
    Abstract Base Class for Sampling Strategies.

    Defines the interface for generating random or quasi-random numbers
    within the search space.
    """

    def __init__(self, rng: np.random.Generator):
        """
        Args:
            rng (np.random.Generator): The centralized random number generator.
        """
        self.rng = rng

    @abc.abstractmethod
    def sample(self, n_pop: int, bounds: np.ndarray) -> np.ndarray:
        """
        Generates $N$ samples within the given bounds.

        Args:
            n_pop (int): Number of individuals.
            bounds (np.ndarray): Search space bounds (shape: D x 2).

        Returns:
            np.ndarray: Population matrix of shape (n_pop, D).
        """
        pass

    def _scale_to_bounds(
        self, unit_samples: np.ndarray, bounds: np.ndarray
    ) -> np.ndarray:
        """
        Helper to scale [0, 1] samples to [min, max] bounds.
        """
        min_b = bounds[:, 0]
        max_b = bounds[:, 1]
        return inv_scale(unit_samples, min_b, max_b)


class RandomSampler(Sampler):
    """
    Uniform Random Sampling.

    Uses standard pseudo-random generation.
    $$ x \\sim U(lower, upper) $$
    """

    def sample(self, n_pop: int, bounds: np.ndarray) -> np.ndarray:
        return self.rng.uniform(
            low=bounds[:, 0], high=bounds[:, 1], size=(n_pop, bounds.shape[0])
        )


class HLCSampler(Sampler):
    """
        Hyper-Latin Cube Sampling (LHS).

    Stratified sampling that ensures coverage across all dimensions.
    Divides each dimension into $N$ intervals and places exactly one sample per interval,
    minimizing clustering.
    """

    def sample(self, n_pop: int, bounds: np.ndarray) -> np.ndarray:
        dim = bounds.shape[0]
        # 1. Generate stratified samples in [0, 1]
        samples = np.zeros((dim, n_pop))

        # Divide [0,1] into n_pop intervals
        step = 1.0 / n_pop

        for d in range(dim):
            # Create points: [0, 1/N, 2/N, ...]
            points = np.arange(n_pop) * step

            # Add random jitter within each interval
            jitter = self.rng.uniform(0, step, size=n_pop)
            points += jitter

            # Shuffle this dimension independently so dimensions are uncorrelated
            self.rng.shuffle(points)
            samples[d] = points

        # Transpose to (N, D) and scale
        return self._scale_to_bounds(samples.T, bounds)


class SobolSampler(Sampler):
    """
    Sobol Sequence Sampler.

    A low-discrepancy quasi-random sequence. It fills the space more evenly than
    random sampling, reducing gaps and clusters.

    **Implementation:**
    Pure NumPy implementation using Gray Code and pre-computed direction numbers.
    Supports up to 40 dimensions (Joe & Kuo, 2003).
    """

    # Format: [d, s, a, m_i...]
    # d: dimension index
    # s: degree of primitive polynomial
    # a: polynomial coefficient (integer representing the polynomial)
    # m: initial direction numbers
    # Source: Joe & Kuo (2003)
    _DIRECTION_NUMBERS = [
        # Dim 1 (skipped, handled as special case)
        # Dim 2-10
        [2, 1, 0, [1]],
        [3, 2, 1, [1, 3]],
        [4, 3, 1, [1, 3, 1]],
        [5, 3, 2, [1, 1, 1]],
        [6, 4, 1, [1, 1, 3, 3]],
        [7, 4, 4, [1, 3, 5, 13]],
        [8, 5, 2, [1, 1, 5, 5, 17]],
        [9, 5, 4, [1, 1, 5, 5, 5]],
        [10, 5, 7, [1, 1, 7, 11, 19]],
        # Dim 11-20
        [11, 5, 11, [1, 1, 7, 13, 25]],
        [12, 5, 13, [1, 1, 5, 11, 25]],
        [13, 5, 14, [1, 1, 3, 13, 27]],
        [14, 6, 1, [1, 1, 1, 3, 11, 25]],
        [15, 6, 13, [1, 3, 1, 13, 27, 43]],
        [16, 6, 16, [1, 1, 5, 5, 29, 39]],
        [17, 6, 19, [1, 1, 7, 7, 21, 37]],
        [18, 6, 22, [1, 1, 1, 9, 23, 37]],
        [19, 6, 25, [1, 1, 3, 13, 31, 11]],
        [20, 6, 1, [1, 3, 3, 9, 9, 57]],
        # Dim 21-30
        [21, 6, 4, [1, 3, 7, 13, 29, 19]],
        [22, 7, 1, [1, 1, 1, 1, 3, 15, 29]],
        [23, 7, 2, [1, 1, 5, 11, 27, 27, 57]],
        [24, 7, 1, [1, 3, 5, 15, 5, 29, 43]],
        [25, 7, 13, [1, 3, 1, 1, 23, 37, 65]],
        [26, 7, 16, [1, 1, 3, 3, 13, 5, 87]],
        [27, 7, 19, [1, 1, 5, 13, 7, 43, 9]],
        [28, 7, 22, [1, 1, 7, 9, 15, 11, 21]],
        [29, 7, 1, [1, 3, 1, 5, 1, 25, 71]],
        [30, 7, 1, [1, 1, 3, 15, 11, 55, 35]],
        # Dim 31-40
        [31, 7, 4, [1, 1, 1, 11, 21, 17, 105]],
        [32, 7, 4, [1, 3, 5, 3, 7, 25, 61]],
        [33, 7, 7, [1, 3, 1, 1, 29, 17, 111]],
        [34, 7, 7, [1, 1, 5, 9, 19, 53, 59]],
        [35, 7, 7, [1, 1, 3, 3, 11, 63, 13]],
        [36, 7, 19, [1, 1, 7, 5, 23, 49, 101]],
        [37, 7, 19, [1, 1, 1, 7, 5, 17, 77]],
        [38, 7, 21, [1, 1, 5, 15, 27, 5, 89]],
        [39, 7, 21, [1, 3, 3, 9, 21, 15, 31]],
        [40, 7, 21, [1, 3, 5, 13, 7, 39, 27]],
    ]

    def _compute_v(self, dim_idx):
        """Computes direction numbers V for a specific dimension."""
        BITS = 32

        # 1. Handle Dimension 1 (Special Case: Van der Corput)
        if dim_idx == 0:
            V = np.zeros(BITS + 1, dtype=np.uint32)
            for i in range(1, BITS + 1):
                V[i] = 1 << (BITS - i)
            return V

        # 2. Handle Dimensions 2+
        if dim_idx > len(self._DIRECTION_NUMBERS):
            raise ValueError(
                f"Max dimension supported is {len(self._DIRECTION_NUMBERS) + 1}"
            )

        params = self._DIRECTION_NUMBERS[dim_idx - 1]
        s = params[1]
        a = params[2]
        m = [0] + params[3]

        V = np.zeros(BITS + 1, dtype=np.uint32)

        # Initialize first 's' numbers
        for i in range(1, s + 1):
            V[i] = m[i] << (BITS - i)

        # Recurrence for remaining bits
        for i in range(s + 1, BITS + 1):
            v_new = V[i - s] ^ (V[i - s] >> s)
            for k in range(1, s):
                if (a >> (s - 1 - k)) & 1:
                    v_new ^= V[i - k]
            V[i] = v_new
        return V

    def sample(self, n_pop: int, bounds: np.ndarray) -> np.ndarray:
        dim = bounds.shape[0]

        if dim > len(self._DIRECTION_NUMBERS) + 1:
            raise ValueError(
                f"Requested {dim} dimensions, max is {len(self._DIRECTION_NUMBERS) + 1}"
            )

        BITS = 32
        SCALE = 2**BITS

        # 1. Compute V table
        V = np.zeros((dim, BITS + 1), dtype=np.uint32)
        for d in range(dim):
            V[d] = self._compute_v(d)

        # 2. Generate Points (Gray Code)
        samples_int = np.zeros((n_pop, dim), dtype=np.uint32)
        X = np.zeros(dim, dtype=np.uint32)

        # Simplified scrambling
        scramble = self.rng.integers(0, SCALE, size=dim, dtype=np.uint32)

        for i in range(n_pop):
            # Find index of rightmost zero bit (equivalent to rightmost set bit of ~i)
            # This 'c' tells us which Direction Number to XOR
            # i=0 (..00) -> c=1
            # i=1 (..01) -> c=2
            c = 1
            value = i
            while value & 1:
                value >>= 1
                c += 1

            if c < BITS:
                X ^= V[:, c]

            samples_int[i] = X ^ scramble

        return self._scale_to_bounds(samples_int / float(SCALE), bounds)


class ChaoticSampler(Sampler):
    """
    Chaotic Map Sampling (Logistic Map).

    Uses a deterministic chaotic system to generate samples.
    $$ x_{k+1} = r \\cdot x_k (1 - x_k) $$
    where $r=4.0$.
    Chaos is ergodic and can provide better exploration dynamics.
    """

    def sample(self, n_pop: int, bounds: np.ndarray) -> np.ndarray:
        dim = bounds.shape[0]
        # Standard Logistic Map parameter for chaos
        r = 4.0

        # Initialize with random start (avoid 0.0, 0.5, 1.0 fixed points)
        # Shape (D,) one sequence per dimension
        x = self.rng.uniform(0.1, 0.9, size=dim)

        # Burn-in: Run map for 100 iterations to decouple from seed
        for _ in range(100):
            x = r * x * (1.0 - x)

        # Generate samples
        samples = np.zeros((n_pop, dim))
        for i in range(n_pop):
            x = r * x * (1.0 - x)
            samples[i] = x

        # Chaotic maps are naturally in [0, 1]
        return self._scale_to_bounds(samples, bounds)


def assert_bounds(solution: np.ndarray, bounds: np.ndarray) -> bool:
    """
    Verifies if the solution is contained within the defined bounds.

    Args:
        solution (np.ndarray): The solution vector(s) to check.
        bounds (np.ndarray): The bounds of valid solutions (shape: N x 2).

    Returns:
        bool: True if the solution is within bounds, False otherwise.
    """
    x = solution.T
    min_bounds = bounds[:, 0]
    max_bounds = bounds[:, 1]
    rv = ((x > min_bounds[:, np.newaxis]) & (x < max_bounds[:, np.newaxis])).any(1)
    return bool(np.all(rv))


def check_bounds(population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Clips values to stay within bounds.

    Check if a solution is within the given bounds.
    If not, values are clipped to the nearest bound.

    Args:
        solution (np.ndarray): The solution vector to be validated.
        bounds (np.ndarray): The bounds of valid solutions (shape: N x 2).

    Returns:
        np.ndarray: A clipped version of the solution vector.
    """
    return np.clip(population, bounds[:, 0], bounds[:, 1])


def get_random_solution(bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a random solution that is within the bounds.

    Args:
        bounds (np.ndarray): The bounds of valid solutions (shape: N x 2).
                             Column 0 is min, Column 1 is max.
        rng (np.random.Generator | None, optional): A numpy random generator instance.
                                                    If None, a new one is created.

    Returns:
        np.ndarray: A random solution within the bounds.
    """
    return rng.uniform(low=bounds[:, 0], high=bounds[:, 1])


def global_distances(samples: np.ndarray) -> np.ndarray:
    """
    Computes global Euclidean distance sum.

    Calculates $\\sum_j ||x_i - x_j||$ for every sample $i$.
    Used to measure isolation/centrality.

    Args:
        samples (np.ndarray): Shape (N, D).

    Returns:
        np.ndarray: Shape (N,). The sum of distances for each sample.
    """
    # 1. Compute Pairwise Differences
    diff = samples[:, np.newaxis, :] - samples[np.newaxis, :, :]

    # 2. Euclidean Distance Matrix
    sq_dist = np.sum(diff**2, axis=-1)
    dist_matrix = np.sqrt(sq_dist)

    # 3. Sum rows to get total distance to all others
    return np.sum(dist_matrix, axis=1)


def compute_crowding_distance(samples: np.ndarray) -> np.ndarray:
    """
    Crowding Distance Calculation (NSGA-II).

    Estimates the density of solutions surrounding a particular point in the objective space.
    Higher distance = More isolated (Better for diversity).

    Args:
        samples (np.ndarray): Shape (N, D)

    Returns:
        np.ndarray: Shape (N,). Higher value = More isolated (Better).
    """
    N, D = samples.shape
    if N == 0:
        return np.array([])

    distances = np.zeros(N)

    # We compute distance dimension by dimension
    for d in range(D):
        # 1. Sort by the current dimension
        # argsort gives us the indices that would sort the array
        sorted_indices = np.argsort(samples[:, d])
        sorted_samples = samples[sorted_indices, d]

        # 2. Handle Boundaries
        # The min and max points in each dim are always "infinite" distance (most isolated)
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        # 3. Compute Distance for Internal Points
        # Formula: (Next_Val - Prev_Val) / (Max_Val - Min_Val)
        scale = sorted_samples[-1] - sorted_samples[0]

        if scale == 0:
            continue  # All points are identical in this dimension

        # Vectorized difference: P[i+1] - P[i-1]
        # We slice sorted_samples from [2:] and [:-2] to get next/prev neighbors
        dim_dist = (sorted_samples[2:] - sorted_samples[:-2]) / scale

        # Add to the cumulative score of the corresponding original indices
        # Indices [1:-1] correspond to the internal points we just computed
        distances[sorted_indices[1:-1]] += dim_dist

    # Replace infinite values (boundaries) with the max finite value found
    # so probabilities don't break
    finite_mask = np.isfinite(distances)
    if np.any(finite_mask):
        max_dist = np.max(distances[finite_mask])
        distances[~finite_mask] = max_dist * 2.0
    else:
        # If all are infinite (e.g. N=2), fallback to ones
        distances[:] = 1.0

    return distances


def score_2_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax conversion of scores to probabilities.

    Converts minimization costs into selection probabilities.
    $$ P(i) = \\frac{\\exp(-score_i / T)}{\\sum \\exp(-score_j / T)} $$

    Args:
        scores: Cost values (lower is better).
        temperature:
             < 1.0: Sharper distribution (Greedy).
             = 1.0: Standard Boltzmann.
             > 1.0: Flatter distribution (Random).
    """
    # 1. Check for flat scores (avoid division by zero later)
    if np.all(scores == scores[0]):
        return np.ones_like(scores) / len(scores)

    # 2. Softmax Logic
    # We negate scores because Softmax maximizes, but we want to minimize cost.
    neg_scores = -scores

    # 3. Numerical Stability (Log-Sum-Exp trick equivalent)
    # Subtract max(neg_scores) so the largest exponent is 0.
    # (This corresponds to subtracting the *minimum* original score).
    shift = np.max(neg_scores)

    # Apply Temperature and Shift
    # Avoid T=0 crash
    temp = max(temperature, 1e-9)
    exps = np.exp((neg_scores - shift) / temp)

    # 4. Normalize
    probs = exps / np.sum(exps)

    # 5. Safety clamp (fix floating point slight errors < 0 or > 1)
    probs = np.clip(probs, 0.0, 1.0)
    return probs / np.sum(probs)


def compute_objective(
    population: np.ndarray, function: Callable[[object], float], n_jobs: int = 1
) -> np.ndarray:
    """
    Computes the objective function for a population of solutions.

    Strategy:
    1. Optimistic Vectorization: Tries passing the entire population matrix to the function.
    2. Serial (n_jobs=1): Uses np.apply_along_axis for row-wise evaluation.
    3. Parallel (n_jobs!=1): Uses Joblib for multiprocessing.

    Args:
        population (np.ndarray): The population of solutions to evaluate.
        function (Callable[[object], float]): The objective function to apply.
        n_jobs (int, optional): Number of parallel jobs. 1 forces serial. Defaults to 1.

    Returns:
        np.ndarray: A NumPy array of objective values.
    """
    # Ensure input is a standard numpy array for consistent handling
    if isinstance(population, list):
        population = np.array(population)

    # 1. Optimistic Approach: Vectorized Execution
    # If the user's function supports (N, D) -> (N,) input, this is instant.
    try:
        result = function(population)
        # Verify result is a valid array of the correct shape (N,) or (N, 1)
        if isinstance(result, np.ndarray) and result.size == population.shape[0]:
            return result.flatten()
    except Exception:
        # Function does not support matrix input, proceed to row-by-row methods
        pass

    # 2. Serial Execution (User requested np.apply_along_axis)
    if n_jobs == 1:
        # Apply function along axis 1 (rows).
        # Note: apply_along_axis iterates in Python but handles array wrapping cleanly.
        return np.apply_along_axis(function, 1, population)

    # 3. Parallel Execution (Joblib)
    else:
        try:
            # Backend 'loky' is robust for generic Python objects.
            obj_list = joblib.Parallel(backend="loky", n_jobs=n_jobs)(
                joblib.delayed(function)(c) for c in population
            )
        except Exception:
            # Fallback to threading if serialization (pickling) fails
            obj_list = joblib.Parallel(backend="threading", n_jobs=n_jobs)(
                joblib.delayed(function)(c) for c in population
            )

        return np.array(obj_list)


def levy_flight(
    n_rows: int, n_cols: int, beta: float = 1.5, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Lévy Flight Step Generation.

    Generates steps from a heavy-tailed distribution (Lévy distribution),
    simulating the flight patterns of foraging animals (short steps + rare long jumps).

    **Mantegna's Algorithm:**
    $$ \\text{Step} = \\frac{u}{|v|^{1/\\beta}} $$
    where $u \\sim \\mathcal{N}(0, \\sigma_u^2)$ and $v \\sim \\mathcal{N}(0, 1)$.
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma_u = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = rng.normal(0, sigma_u, size=(n_rows, n_cols))
    v = rng.normal(0, 1, size=(n_rows, n_cols))

    # Avoid division by zero
    v[v == 0] = 1e-10

    step = u / (np.abs(v) ** (1 / beta))
    return step
