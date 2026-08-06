
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
import logging
import math
from collections.abc import Callable

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def inherit_docs(from_obj):
    """
    Unified decorator to inherit docstrings from either a class or a function.

    Args:
        from_obj: The source class or function to pull documentation from.
    """

    def decorator(func):
        # 1. Determine the source of the docstring
        if inspect.isclass(from_obj):
            # Check if the class uses the default object.__init__
            # If so, strictly use the class docstring.
            if from_obj.__init__ is object.__init__:
                source_doc = from_obj.__doc__
            else:
                # Otherwise, prefer the custom __init__ docstring, fallback to class docstring
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
    actual_min = np.nanmin(arr) if min_val is None else min_val
    actual_max = np.nanmax(arr) if max_val is None else max_val

    denominator = actual_max - actual_min

    scl_arr = np.divide(
        (arr - actual_min), denominator, out=np.zeros_like(arr), where=denominator != 0
    )

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
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        return self.rng.uniform(low=lower, high=upper, size=(n_pop, bounds.shape[0]))


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


def _factorize(n: int) -> set:
    """Prime factors of `n` by trial division. `n` here is at most 2^24."""
    factors, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _gf2_mulmod(a: int, b: int, poly: int, s: int) -> int:
    """Multiply two GF(2) polynomials modulo `poly` (degree `s`)."""
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if (a >> s) & 1:
            a ^= poly
    return result


def _gf2_powmod(a: int, e: int, poly: int, s: int) -> int:
    """Exponentiate in GF(2)[x] / poly by square-and-multiply."""
    result = 1
    while e:
        if e & 1:
            result = _gf2_mulmod(result, a, poly, s)
        a = _gf2_mulmod(a, a, poly, s)
        e >>= 1
    return result


def _is_primitive(poly: int, s: int) -> bool:
    """
    Is `poly` primitive over GF(2)?

    Primitive means `x` generates the whole multiplicative group of
    GF(2^s), i.e. `x` has order exactly `2^s - 1`. Checked the standard way:
    `x^(2^s - 1) == 1`, and `x^((2^s - 1) / q) != 1` for every prime `q`
    dividing `2^s - 1`.
    """
    order = (1 << s) - 1
    if _gf2_powmod(2, order, poly, s) != 1:
        return False
    return all(_gf2_powmod(2, order // q, poly, s) != 1 for q in _factorize(order))


def _primitive_polynomials(count: int) -> list:
    """
    The first `count` primitive polynomials over GF(2), by increasing degree
    and then by increasing coefficient value.

    Returned as `(s, a)` pairs in the encoding Sobol construction uses: `s` is
    the degree and `a` the integer formed by the *interior* coefficients (the
    leading and constant terms are always 1, so they carry no information).
    This reproduces the tabulated `(s, a)` column exactly -- see
    `test_utils.py` -- which is unsurprising, since the table is this
    enumeration written down.
    """
    out, s = [], 1
    while len(out) < count:
        # Degree-s polynomials with leading and constant coefficient set.
        for a in range(1 << max(s - 1, 0)):
            if _is_primitive((1 << s) | (a << 1) | 1, s):
                out.append((s, a))
                if len(out) == count:
                    return out
            if s == 1:
                break  # x + 1 is the only degree-1 candidate
        s += 1
    return out


_SOBOL_BITS = 32
# A candidate's projections are scored at *several* point counts, not one.
# Scoring only at 1024 points yields direction numbers whose worst coordinate
# pair correlates at 0.003 there and 0.75 at 256 points -- and population
# sizes in this library are small, so the small-n behaviour is the one that
# matters. Taking the worst case over the range keeps both ends honest.
_SOBOL_SCORE_POINTS = (1 << 6, 1 << 8, 1 << 10)
# Good `m` vectors are rare -- for a degree-7 polynomial only about 1.5% of
# valid draws decorrelate from the dimensions already fixed -- so the search
# draws generously and stops as soon as one is good enough rather than
# always paying for the full budget.
#
# Note what the search cannot fix. With `n` points a coordinate takes at most
# `n` distinct values, so once the dimension count approaches `n` some pair
# must correlate no matter how the design is built: at `d=100, n=64` that is a
# property of having fewer points than dimensions, not of these direction
# numbers. The budget is therefore spent where it buys something -- the
# regime with more points than dimensions -- rather than chasing a limit.
_SOBOL_CANDIDATES = 192
_SOBOL_GOOD_ENOUGH = 0.15

# Gray-code positions, i.e. the 1-based index of the rightmost zero bit of i.
_SOBOL_GRAY = np.zeros(max(_SOBOL_SCORE_POINTS), dtype=np.int64)
for _i in range(max(_SOBOL_SCORE_POINTS)):
    _c, _v = 1, _i
    while _v & 1:
        _v >>= 1
        _c += 1
    _SOBOL_GRAY[_i] = _c

# Direction numbers, one row per dimension, grown on demand. The greedy
# construction below only ever looks at dimensions *before* the one it is
# choosing, so the table is prefix-stable: the rows built for 100 dimensions
# are the rows built for 16 dimensions plus 84 more. That is what makes a
# single growing cache deterministic and safe to slice.
_sobol_directions: list = []
_sobol_columns: list = []  # the matching coordinates, kept for the greedy score


def _sobol_v_row(s: int, a: int, m: list) -> np.ndarray:
    """
    Direction numbers `V` for one dimension, from its primitive polynomial.

    Applies the Sobol recurrence in `V` space, where `V_i = m_i << (BITS - i)`:
    $$ V_i = V_{i-s} \\oplus (V_{i-s} \\gg s)
             \\oplus \\bigoplus_{k=1}^{s-1} a_k V_{i-k} $$
    with `a_k` read off the bits of `a` (the interior polynomial coefficients).
    """
    V = np.zeros(_SOBOL_BITS + 1, dtype=np.uint32)
    for i in range(1, s + 1):
        V[i] = m[i - 1] << (_SOBOL_BITS - i)
    for i in range(s + 1, _SOBOL_BITS + 1):
        v = V[i - s] ^ (V[i - s] >> s)
        for k in range(1, s):
            if (a >> (s - 1 - k)) & 1:
                v ^= V[i - k]
        V[i] = v
    return V


def _sobol_column(V: np.ndarray, n: int) -> np.ndarray:
    """
    One coordinate of the unscrambled sequence, as floats in [0, 1).

    `x_i` is the XOR of `V[c(0)] .. V[c(i-1)]`, so the whole column is one
    cumulative XOR rather than a Python loop -- which matters because the
    direction-number search in :func:`_sobol_extend` builds hundreds of these
    per dimension. Safe to skip the `c < BITS` guard here: `c` only reaches 32
    at index `2**31`, far above any `n` used for scoring.
    """
    out = np.zeros(n, dtype=np.uint32)
    if n > 1:
        out[1:] = np.bitwise_xor.accumulate(V[_SOBOL_GRAY[: n - 1]])
    return out.astype(np.float64) / float(1 << _SOBOL_BITS)


def _sobol_extend(dim: int) -> np.ndarray:
    """
    Direction numbers for `dim` dimensions, extending the cache as needed.

    The initial direction numbers `m_i` are the one part of a Sobol
    construction that cannot be derived: Sobol's conditions only require each
    `m_i` to be odd and below `2^i`, which leaves many valid choices with very
    different two-dimensional projections. Joe & Kuo (2003) resolved this by
    numerically optimizing those projections and publishing the result as a
    table.

    This does the same thing at a smaller scale rather than carrying data it
    cannot check: for each new dimension it draws `_SOBOL_CANDIDATES` valid
    `m` vectors from a generator seeded by the dimension index -- so the
    choice is deterministic and reproducible -- and keeps whichever candidate
    minimizes the largest absolute correlation against every dimension
    already fixed. Measured at 1024 points, that takes the worst pair from
    0.75 down to about 0.003 at 40 dimensions and 0.012 at 100, against a
    ceiling of 1.0 for unoptimized `m_i = 1`.
    """
    if len(_sobol_directions) >= dim:
        return np.array(_sobol_directions[:dim])

    n_max = max(_SOBOL_SCORE_POINTS)
    if not _sobol_directions:
        # Dimension 1 is the Van der Corput sequence: V_i = 2^(BITS - i).
        V = np.zeros(_SOBOL_BITS + 1, dtype=np.uint32)
        for i in range(1, _SOBOL_BITS + 1):
            V[i] = 1 << (_SOBOL_BITS - i)
        _sobol_directions.append(V)
        _sobol_columns.append(_sobol_column(V, n_max))

    polys = _primitive_polynomials(dim)
    while len(_sobol_directions) < dim:
        d = len(_sobol_directions)  # 0-based index of the dimension being added
        s, a = polys[d - 1]
        rng = np.random.default_rng(0x9E3779B9 ^ d)
        prev = np.array(_sobol_columns)
        best_score, best_V, best_col = np.inf, None, None
        for _ in range(_SOBOL_CANDIDATES):
            m = [int(2 * rng.integers(0, 1 << i) + 1) for i in range(s)]
            V = _sobol_v_row(s, a, m)
            col = _sobol_column(V, n_max)
            score = max(
                float(np.abs(np.corrcoef(
                    np.vstack([prev[:, :n], col[:n]])) [-1, :-1]).max())
                for n in _SOBOL_SCORE_POINTS
            )
            if score < best_score:
                best_score, best_V, best_col = score, V, col
                if score < _SOBOL_GOOD_ENOUGH:
                    break
        _sobol_directions.append(best_V)
        _sobol_columns.append(best_col)

    return np.array(_sobol_directions[:dim])


class SobolSampler(Sampler):
    """
    Sobol Sequence Sampler.

    A low-discrepancy quasi-random sequence. It fills the space more evenly than
    random sampling, reducing gaps and clusters.

    **Implementation:**
    Pure NumPy, Gray-code driven, with **no dimension ceiling** and no
    tabulated constants: the primitive polynomial for each dimension is
    generated by :func:`_primitive_polynomials` and the initial direction
    numbers are chosen by the greedy projection search in
    :func:`_sobol_extend`. Everything is derived and therefore checkable,
    which the previous 40-row table was not -- five of its rows carried
    polynomials that are not primitive over GF(2) and six polynomials were
    reused across up to four dimensions, so above dimension 19 it did not
    describe a Sobol sequence at all.
    """

    def sample(self, n_pop: int, bounds: np.ndarray) -> np.ndarray:
        dim = bounds.shape[0]
        SCALE = 1 << _SOBOL_BITS

        V = _sobol_extend(dim)

        # Gray-code recursion: x_0 = 0 and x_i = x_{i-1} XOR V[c(i-1)], where
        # c(k) is the 1-based index of the rightmost zero bit of k. Deriving
        # c from `i` rather than `i - 1` shifts the whole sequence by one
        # point, which silently drops the origin and costs the design its
        # net property -- every dimension then fails to stratify.
        samples_int = np.zeros((n_pop, dim), dtype=np.uint32)
        x = np.zeros(dim, dtype=np.uint32)

        # Random digital shift (Cranley-Patterson in base 2): preserves the
        # net property while decorrelating runs, and moves the origin off 0.
        scramble = self.rng.integers(0, SCALE, size=dim, dtype=np.uint32)
        samples_int[0] = scramble

        for i in range(1, n_pop):
            c = 1
            value = i - 1
            while value & 1:
                value >>= 1
                c += 1
            if c < _SOBOL_BITS:
                x ^= V[:, c]
            samples_int[i] = x ^ scramble

        return self._scale_to_bounds(samples_int / float(SCALE), bounds)


class ChaoticSampler(Sampler):
    """
    Chaotic Map Sampling (Tent Map).

    Uses a deterministic chaotic system to
    generate samples, based on the Tent Map,
    which provides a uniform distribution:
    $$ x_{n+1} = \\begin{cases} 2x_n & \\text{if } x_n < 0.5 \\\\ 2(1-x_n) & \\text{if } x_n \\ge 0.5 \\end{cases} $$

    Chaos is ergodic and can provide better exploration dynamics.
    """

    def sample(self, n_pop: int, bounds: np.ndarray) -> np.ndarray:
        dim = bounds.shape[0]

        # Initialize x with random start (avoid 0.0, 0.5, 1.0 fixed points)
        # We use a slightly tighter range to avoid immediate boundary issues
        x = self.rng.uniform(0.1, 0.9, size=dim)

        # Burn-in: Run map for 100 iterations
        # This decouples the sequence from the random seed
        for _ in range(100):
            # Vectorized Tent Map logic
            x = np.where(x < 0.5, 2.0 * x, 2.0 * (1.0 - x))

        samples = np.zeros((n_pop, dim))

        # Pre-allocate limit constants for speed
        epsilon = 1e-10

        for i in range(n_pop):
            # 1. Update Map
            # np.where is faster and cleaner than boolean indexing here
            x = np.where(x < 0.5, 2.0 * x, 2.0 * (1.0 - x))

            # 2. Robustness Check (Crucial for Tent Map)
            # Tent map is sensitive to floating point bit-shifts.
            # Eventually, values might collapse to exactly 0.0.
            # If that happens, we inject a tiny jitter to restart chaos.
            if np.any(x < epsilon):
                zero_mask = x < epsilon
                # Inject noise only where needed
                jitter = self.rng.uniform(
                    epsilon, 0.01, size=np.count_nonzero(zero_mask)
                )
                x[zero_mask] = jitter

            samples[i] = x

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
    # Handle 1D (single solution) and 2D (population) inputs uniformly
    if solution.ndim == 1:
        solution = solution[np.newaxis, :]
    # Check lower and upper bounds
    check_min = np.all(solution >= bounds[:, 0])
    check_max = np.all(solution <= bounds[:, 1])
    return bool(check_min and check_max)


def check_bounds(population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Clips values to stay within bounds.

    Check if a solution is within the given bounds.
    If not, values are clipped to the nearest bound.

    Args:
        solution (np.ndarray): The population vector to be validated.
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
    It uses the Euclidean distance expansion trick to avoid high memory usage.
    Formula:
    $$||A - B||^2 = ||A||^2 + ||B||^2 - 2<A, B>$$

    Args:
        samples (np.ndarray): Shape (N, D).

    Returns:
        np.ndarray: Shape (N,). The sum of distances for each sample.
    """
    # Compute squared norms of each sample (N,)
    sq_norms = np.sum(samples**2, axis=1)

    # Compute the dot product matrix (N, N)
    dot_products = np.dot(samples, samples.T)

    # Apply expansion formula using broadcasting
    # dist_sq[i, j] = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
    sq_dist_matrix = (
        sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * dot_products
    )

    # Numerical stability, clip negative zeros
    sq_dist_matrix = np.maximum(sq_dist_matrix, 0.0)

    # Sqrt and Sum
    dist_matrix = np.sqrt(sq_dist_matrix)
    return np.sum(dist_matrix, axis=1)


def compute_crowding_distance(samples: np.ndarray) -> np.ndarray:
    """
    Crowding Distance Calculation (NSGA-II).

    Estimates the density of solutions surrounding a particular point in the objective space.
    Higher distance = More isolated (Better for diversity).

    Extremes score twice their single gap rather than infinity. NSGA-II uses
    infinity to protect the endpoints of a 2-3 objective front; over a
    `D`-dimensional decision space the same rule marks up to `2 * D` points --
    88% of a 120-candidate pool at `D=100` -- and ties them all at the top.

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

        # 2. Normalise by this axis's spread, so axes contribute comparably.
        scale = sorted_samples[-1] - sorted_samples[0]

        if scale == 0:
            continue  # All points are identical in this dimension

        # 3. Interior points: the span their two neighbours enclose.
        distances[sorted_indices[1:-1]] += (
            sorted_samples[2:] - sorted_samples[:-2]) / scale

        # 4. Extremes: twice their single gap, matching step 3's units.
        distances[sorted_indices[0]] += (
            2.0 * (sorted_samples[1] - sorted_samples[0]) / scale)
        distances[sorted_indices[-1]] += (
            2.0 * (sorted_samples[-1] - sorted_samples[-2]) / scale)

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
    population: np.ndarray,
    function: Callable[[np.ndarray], float | np.ndarray],
    n_jobs: int = 1,
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

    # Serial Execution (User requested np.apply_along_axis)
    if n_jobs == 1:
        # Optimistic Approach: Vectorized Execution
        try:
            # If the user's function supports (N, D) -> (N,) input
            result = function(population)
            # Verify result is a valid array of the correct shape (N,) or (N, 1)
            if isinstance(result, np.ndarray) and result.size == population.shape[0]:
                return result.flatten()
        except (ValueError, TypeError) as e:
            logger.debug(f"""Function does not support matrix input.
                proceed to row-by-row methods:
                {e}""")
        # Apply function along axis 1 (rows).
        return np.apply_along_axis(function, 1, population)

    # 3. Parallel Execution (Joblib)
    else:
        try:
            # Backend 'loky' is robust for generic Python objects.
            obj_list = joblib.Parallel(backend="loky", n_jobs=n_jobs)(
                joblib.delayed(function)(c) for c in population
            )
        except Exception as e:
            # Fallback to threading if serialization (pickling) fails
            logger.debug(
                f"Fallback to threading if serialization (pickling) fails: {e}"
            )
            obj_list = joblib.Parallel(backend="threading", n_jobs=n_jobs)(
                joblib.delayed(function)(c) for c in population
            )

        return np.array(obj_list)


def levy_flight(
    n_pop: int, dim: int, beta: float = 1.5, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Lévy Flight Step Generation.

    Generates steps from a heavy-tailed distribution (Lévy distribution),
    simulating the flight patterns of foraging animals (short steps + rare long jumps).

    **Mantegna's Algorithm:**
    $$ \\text{Step} = \\frac{u}{|v|^{1/\\beta}} $$
    where $u \\sim \\mathcal{N}(0, \\sigma_u^2)$ and $v \\sim \\mathcal{N}(0, 1)$.

    Args:
        n_pop (int): Number of individuals (rows).
        dim (int): Number of dimensions (columns).
        beta (float, optional): Power law exponent (1 < beta <= 2). Defaults to 1.5.
        rng (np.random.Generator | None, optional): Random number generator.

    Returns:
        np.ndarray: Matrix of steps with shape (n_pop, dim).
    """

    if rng is None:
        rng = np.random.default_rng()

    sigma_u = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = rng.normal(0, sigma_u, size=(n_pop, dim))
    v = rng.normal(0, 1, size=(n_pop, dim))

    # Avoid division by zero
    v[v == 0] = 1e-10

    step = u / (np.abs(v) ** (1 / beta))
    return step


def maximin_fill(
    candidates: np.ndarray,
    k: int,
    chosen: np.ndarray | None = None,
    first: int | None = None,
) -> np.ndarray:
    """
    Sequential maximin: grow a selection by repeatedly taking the candidate
    furthest from everything already selected.

    Distance is to the union of ``chosen`` (points already fixed, not
    selectable) and the candidates picked so far. That union is the reason
    this is not a one-shot diversity score: spread is a property of the
    selected *set*, so it has to be recomputed after every pick.

    Args:
        candidates (np.ndarray): Selectable points, shape (M, D).
        k (int): How many to pick. Clipped to ``M``.
        chosen (np.ndarray | None): Points already fixed, shape (C, D). New
            picks are pushed away from these but none of them are returned.
        first (int | None): Index of a forced first pick (e.g. the fittest
            candidate). If None the first pick is the one furthest from
            ``chosen``, or index 0 when ``chosen`` is empty.

    Returns:
        np.ndarray: Indices into ``candidates``, length ``min(k, M)``.
    """
    m = candidates.shape[0]
    k = int(min(k, m))
    if k <= 0:
        return np.empty(0, dtype=int)

    if chosen is not None and len(chosen) > 0:
        # (M, C) squared distances, reduced to the nearest fixed point.
        d2 = np.min(
            np.sum((candidates[:, None, :] - np.asarray(chosen)[None, :, :]) ** 2, -1),
            axis=1,
        )
    else:
        d2 = np.full(m, np.inf)

    picked = [int(first)] if first is not None else [int(np.argmax(d2))]
    d2 = np.minimum(d2, np.sum((candidates - candidates[picked[0]]) ** 2, axis=1))

    while len(picked) < k:
        d2[picked] = -1.0
        nxt = int(np.argmax(d2))
        picked.append(nxt)
        d2 = np.minimum(d2, np.sum((candidates - candidates[nxt]) ** 2, axis=1))

    return np.array(picked)


def greedy_maximin(
    population: np.ndarray,
    scores: np.ndarray,
    n_pop: int,
    keep: float = 2.0,
) -> np.ndarray:
    """
    Diverse subset selection by sequential maximin over the fittest candidates.

    Truncates the pool to the fittest ``keep * n_pop``, then grows the
    selection one point at a time, each time taking the candidate furthest
    from everything already chosen. Fitness acts as a hard filter rather than
    a term that can be traded away, so a point cannot enter on spread alone.

    **Why this rather than a fitness/diversity blend.** A blend scores
    diversity once against the whole pool, so points that are each isolated
    relative to the pool can still be clustered relative to each other --
    diversity is a property of the selected set, and a one-shot score never
    looks there. Measured against the blend on ``examples/bench_selection.py``
    this dominates it on *both* axes at every dimension from 2 to 40: better
    median fitness and a larger minimum nearest-neighbour gap. It also has no
    weight to tune and needs no diversity metric.

    **What dimension does to it.** The spread available to any selector
    collapses as dimension grows -- against pure fitness selection the gain in
    minimum nearest-neighbour distance is 8.8x at d=2, 3.0x at d=5, 2.0x at
    d=10, 1.5x at d=20 and 1.3x at d=40. In high dimension there is little
    diversity to be bought at any price, which is the measured reason the
    diversity-weighted arms lose ground as dimension rises.

    Args:
        population (np.ndarray): Candidate pool, shape (M, D).
        scores (np.ndarray): Objective values, lower is better.
        n_pop (int): Number of individuals to select.
        keep (float): Fitness truncation, as a multiple of ``n_pop``. 1.0 is
            pure fitness selection; larger values give spread more room at a
            cost in fitness.

    Returns:
        np.ndarray: Indices into ``population``, length ``n_pop``.
    """
    m = int(np.clip(keep * n_pop, n_pop, population.shape[0]))
    cand = np.argpartition(scores, m - 1)[:m] if m < len(scores) \
        else np.arange(len(scores))
    pts = population[cand]

    # Seeded with the fittest candidate, then grown by spread alone.
    picked = maximin_fill(pts, n_pop, first=int(np.argmin(scores[cand])))
    return cand[picked]


def _ranks(values: np.ndarray) -> np.ndarray:
    """
    Positions of `values` in ascending order: 0 for the smallest.

    Ties are broken by index rather than shared, which keeps the result a
    permutation and so keeps a blend of two rank vectors on a fixed scale.
    """
    order = np.argsort(values, kind="stable")
    out = np.empty(values.shape[0], dtype=float)
    out[order] = np.arange(values.shape[0], dtype=float)
    return out


#: Historic spellings of the selection rules. `'random'` in particular never
#: meant uniform sampling -- it was always roulette over the blended score --
#: so it is mapped rather than kept, to stop the name implying otherwise.
_SELECTION_ALIASES = {"probabilistic": "prob", "random": "prob", "greedy": "best"}


def select_indices(
    population: np.ndarray,
    scores: np.ndarray,
    n_pop: int,
    selection: str = "best",
    diversity_weight: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Index-returning form of :func:`select_population`.

    Returned in ascending-score order, which matters more than it looks.
    ``np.argpartition`` orders its output differently depending on how large
    the pool was, so two selectors that pick the *same set* out of pools of
    different size hand the optimizer different row orders. Optimizers index
    their population by position -- DE draws `r1, r2, ...` by index -- so a
    permutation is a different search trajectory. Sorting canonically makes
    "same set" imply "same run", which is what a paired comparison between
    initializers needs in order to be measuring the initializer.

    Args:
        See :func:`select_population`.

    Returns:
        np.ndarray: Indices into ``population``, length ``n_pop``, ordered by
        ascending score.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Normalised up front so an unrecognised name raises instead of silently
    # falling through to roulette, which is what an `else` branch would do to
    # a typo -- and a selector that quietly changes is the hardest kind of
    # benchmark arm to debug.
    selection = _SELECTION_ALIASES.get(selection, selection)
    if selection not in ("best", "prob", "maximin"):
        raise ValueError(
            f"selection must be 'best', 'prob' or 'maximin', got {selection!r}")

    if n_pop >= population.shape[0]:
        idx = np.arange(population.shape[0])
    elif selection == "maximin":
        idx = greedy_maximin(
            population, scores, n_pop, keep=1.0 + 3.0 * diversity_weight)
    elif selection == "best" and diversity_weight == 0.0:
        # SHORTCUT: pure greedy selection, bypass the probability math.
        idx = np.argpartition(scores, n_pop)[:n_pop]
    else:
        # Blended on RANKS, not on softmax probabilities, and that is a
        # correctness matter rather than a preference. `score_2_probs` is a
        # Boltzmann softmax over the raw objective, so its spread depends on
        # the objective's units: on a sphere over [-5, 5]^4 the scores span
        # ~0.01 to ~80, the softmax saturates, and the best candidate alone
        # carries 96% of the mass. Two consequences, both measured:
        # `'prob'` degenerated into greedy selection, and blending in crowding
        # -- whose own softmax is nearly flat, 0.020 to 0.038 -- moved nothing
        # until `diversity_weight` reached exactly 1.0, at which point the
        # selector flipped over completely. A knob inert across three quarters
        # of its range is not a knob.
        #
        # Ranks are scale-free by construction, so the blend means the same
        # thing on every objective, and it is monotone in `diversity_weight`
        # with the endpoints exactly pure fitness and pure diversity.
        rank_fitness = _ranks(scores)
        if diversity_weight > 0:
            # Higher crowding distance is better (more diverse), so negate it.
            rank_dist = _ranks(-compute_crowding_distance(population))
        else:
            rank_dist = np.zeros_like(rank_fitness)

        blended = (
            (1.0 - diversity_weight) * rank_fitness + diversity_weight * rank_dist
        )

        if selection == "best":
            idx = np.argpartition(blended, n_pop)[:n_pop]
        else:
            # 'prob' -- linear-ranking roulette over the blended rank. Note
            # this is *not* uniform sampling: a candidate's chance falls
            # linearly with its rank, so the pressure is still towards good
            # points, just not absolutely. Linear ranking rather than softmax
            # for the reason above -- it fixes the selection pressure at M:1
            # between the best and worst candidate whatever the objective's
            # scale, instead of letting the objective decide it.
            m = blended.shape[0]
            weights = float(m) - _ranks(blended)
            final_probs = weights / weights.sum()
            try:
                idx = rng.choice(
                    population.shape[0], size=n_pop, replace=False, p=final_probs
                )
            except ValueError as e:
                logger.warning(
                    f"Random selection failed ({e}), fallback to best selection"
                )
                idx = np.argpartition(blended, n_pop)[:n_pop]

    return idx[np.argsort(scores[idx], kind="stable")]


def select_population(
    population: np.ndarray,
    scores: np.ndarray,
    n_pop: int,
    selection: str = "best",
    diversity_weight: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Selects the best individuals from a population based on fitness and diversity.

    The selected rows are returned in ascending-score order; see
    :func:`select_indices` for why the order is pinned rather than left to
    ``argpartition``.

    Args:
        population (np.ndarray): The evaluated population.
        scores (np.ndarray): The objective scores for the population (lower is better).
        n_pop (int): The number of individuals to select.
        selection (str): Selection strategy. Three rules, on two different
            axes -- 'maximin' does not supersede 'probabilistic', they trade
            different things:

            * ``'best'`` -- greedy. Take the top ``n_pop`` by score
              (fitness, or fitness blended with crowding when
              ``diversity_weight > 0``). Deterministic, maximum pressure.
            * ``'probabilistic'`` -- roulette wheel over that same blended
              score. Deterministic pressure is traded for a chance that a
              mid-ranked candidate survives. ``'random'`` is accepted as a
              legacy alias and means exactly this; it does *not* mean uniform
              sampling.
            * ``'maximin'`` -- sequential maximin over the fittest
              ``keep * n_pop``. Deterministic like 'best', but spread is a
              property of the selected *set* rather than a per-candidate
              score, which a one-shot blend cannot express. See
              :func:`greedy_maximin`.
        diversity_weight (float): Trade-off between fitness (0.0) and
            diversity (1.0). Under 'maximin' it is reinterpreted as the
            fitness truncation ``keep = 1 + 3 * diversity_weight``, so 0.0
            stays pure fitness and 1.0 spreads over the whole pool.
        rng (np.random.Generator | None): Random number generator.

    Returns:
        np.ndarray: The selected population of shape (n_pop, D), ordered by
        ascending score.
    """
    return population[
        select_indices(population, scores, n_pop, selection, diversity_weight, rng)
    ]
