# coding: utf-8

"""
Utilities for optimization methods.
"""

__version__ = "0.1"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"

import abc
from collections.abc import Callable

import joblib
import numpy as np


class Sampler(abc.ABC):
    """
    Abstract Base Class for Sampling Strategies.
    Stateless regarding the problem dimensions (passed at runtime).
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
        Generates a population matrix.

        Args:
            n_pop (int): Number of individuals.
            bounds (np.ndarray): Search space bounds (shape: D x 2).

        Returns:
            np.ndarray: Population matrix of shape (n_pop, D).
        """
        pass
    
    def _scale_to_bounds(self, unit_samples: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Helper to scale [0, 1] samples to [min, max] bounds."""
        min_b = bounds[:, 0]
        max_b = bounds[:, 1]
        return inv_scale(unit_samples, min_b, max_b)


class RandomSampler(Sampler):
    """
    Standard Uniform Random Sampling.
    """
    def sample(self, n_pop: int, bounds: np.ndarray) -> np.ndarray:
        return self.rng.uniform(
            low=bounds[:, 0], 
            high=bounds[:, 1], 
            size=(n_pop, bounds.shape[0])
        )


class HLCSampler(Sampler):
    """
    Hyper-Latin Cube Sampling (LHS).
    Ensures stratified sampling across all dimensions.
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
    Pure NumPy implementation of the Sobol Sequence.
    Uses pre-computed direction numbers to avoid scipy dependencies.
    Supports up to 40 dimensions.
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
        [14, 6, 1,  [1, 1, 1, 3, 11, 25]],
        [15, 6, 13, [1, 3, 1, 13, 27, 43]],
        [16, 6, 16, [1, 1, 5, 5, 29, 39]],
        [17, 6, 19, [1, 1, 7, 7, 21, 37]],
        [18, 6, 22, [1, 1, 1, 9, 23, 37]],
        [19, 6, 25, [1, 1, 3, 13, 31, 11]],
        [20, 6, 1,  [1, 3, 3, 9, 9, 57]],
        # Dim 21-30
        [21, 6, 4,  [1, 3, 7, 13, 29, 19]],
        [22, 7, 1,  [1, 1, 1, 1, 3, 15, 29]],
        [23, 7, 2,  [1, 1, 5, 11, 27, 27, 57]],
        [24, 7, 1,  [1, 3, 5, 15, 5, 29, 43]],
        [25, 7, 13, [1, 3, 1, 1, 23, 37, 65]],
        [26, 7, 16, [1, 1, 3, 3, 13, 5, 87]],
        [27, 7, 19, [1, 1, 5, 13, 7, 43, 9]],
        [28, 7, 22, [1, 1, 7, 9, 15, 11, 21]],
        [29, 7, 1,  [1, 3, 1, 5, 1, 25, 71]],
        [30, 7, 1,  [1, 1, 3, 15, 11, 55, 35]],
        # Dim 31-40
        [31, 7, 4,  [1, 1, 1, 11, 21, 17, 105]],
        [32, 7, 4,  [1, 3, 5, 3, 7, 25, 61]],
        [33, 7, 7,  [1, 3, 1, 1, 29, 17, 111]],
        [34, 7, 7,  [1, 1, 5, 9, 19, 53, 59]],
        [35, 7, 7,  [1, 1, 3, 3, 11, 63, 13]],
        [36, 7, 19, [1, 1, 7, 5, 23, 49, 101]],
        [37, 7, 19, [1, 1, 1, 7, 5, 17, 77]],
        [38, 7, 21, [1, 1, 5, 15, 27, 5, 89]],
        [39, 7, 21, [1, 3, 3, 9, 21, 15, 31]],
        [40, 7, 21, [1, 3, 5, 13, 7, 39, 27]]
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
            raise ValueError(f"Max dimension supported is {len(self._DIRECTION_NUMBERS)+1}")

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
            raise ValueError(f"Requested {dim} dimensions, max is {len(self._DIRECTION_NUMBERS)+1}")

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


def scale(
    arr: np.ndarray,
    min_val: float | np.ndarray | None = None,
    max_val: float | np.ndarray | None = None,
) -> tuple[np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Scales an array to the [0, 1] range using Min-Max scaling.

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

    Args:
        scl_arr (np.ndarray): The scaled array.
        min_val (float | np.ndarray): The minimum value used in the original scaling.
        max_val (float | np.ndarray): The maximum value used in the original scaling.

    Returns:
        np.ndarray: The array rescaled to the original range.
    """
    return scl_arr * (max_val - min_val) + min_val


def global_distances(samples: np.ndarray) -> np.ndarray:
    """
    Computes the sum of Euclidean distances from each sample to every other sample.
    Vectorized implementation O(N^2).

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
    Computes the Crowding Distance (NSGA-II style).
    Estimates the density of points around each sample.
    
    Complexity: O(D * N log N)
    
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
    max_finite = np.max(distances[np.isfinite(distances)])
    distances[np.isinf(distances)] = max_finite * 2.0  # Give them a boost

    return distances


def score_2_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Converts minimization scores into a probability distribution.
    Uses Min-Max normalization to handle negative values robustly.

    Args:
        scores (np.ndarray): Objective values (lower is better).
        temperature (float): Controls selection pressure (optional). 
                             >1.0 makes distribution flatter (more random).
                             <1.0 makes distribution sharper (greedy).

    Returns:
        np.ndarray: Probabilities summing to 1.0.
    """
    # 1. Handle Constant Scores (Variance is 0)
    if np.all(scores == scores[0]):
        return np.ones_like(scores) / len(scores)

    # 2. Min-Max Normalization -> Maps to [0, 1]
    # S_norm = (S - min) / (max - min)
    min_s = np.min(scores)
    max_s = np.max(scores)
    norm_scores, _, _ = scale(scores, min_s, max_s)

    # 3. Invert (Minimization: Low score -> High prob)
    # 0.0 (Best) becomes 1.0
    # 1.0 (Worst) becomes 0.0
    # We add a small epsilon to avoid 0.0 probability for the worst candidate
    weights = (1.0 - norm_scores) + 1e-6

    # 4. Optional: Temperature Scaling (Weights ^ (1/T))
    if temperature != 1.0:
        weights = weights ** (1.0 / temperature)

    # 5. Normalize to Probabilities
    return weights / np.sum(weights)


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
