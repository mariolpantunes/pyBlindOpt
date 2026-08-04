# coding: utf-8

"""
Benchmark functions for evaluating optimization algorithms.

Includes a variety of landscape types to test algorithm performance:
* **Separable vs. Non-Separable:** Can variables be optimized independently?
* **Unimodal vs. Multimodal:** Is there one valley or many?
* **Convex vs. Non-Convex:** Is the gradient always reliable?

Mathematical definitions use vector notation where $x = [x_1, x_2, ..., x_D]$.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import numpy as np


def sphere(x: np.ndarray) -> np.ndarray:
    """
    Sphere Function.

    A simple unimodal, convex, and separable function used to test convergence speed.

    **Equation:**
    $$ f(x) = \\sum_{i=1}^D x_i^2 $$

    **Global Minimum:**
    $f(x) = 0$ at $x = [0, ..., 0]$.

    Args:
        x (np.ndarray): Input vector(s).

    Returns:
        np.ndarray: Computed function values.
    """
    return np.sum(np.power(x, 2), axis=-1)


def rastrigin(x: np.ndarray, a: float = 10.0) -> np.ndarray:
    """
    Rastrigin Function.

    A highly multimodal, non-convex, separable function. It is essentially a sphere
    function modulated by a cosine wave, creating many local minima ("egg carton" shape).

    **Equation:**
    $$ f(x) = A \\cdot D + \\sum_{i=1}^D (x_i^2 - A \\cos(2\\pi x_i)) $$
    where $A=10$.

    **Global Minimum:**
    $f(x) = 0$ at $x = [0, ..., 0]$.

    Args:
        x (np.ndarray): Input vector(s).
        a (float, optional): Modulation amplitude. Defaults to 10.0.

    Returns:
        np.ndarray: Computed function values.
    """
    dim = x.shape[-1]
    return a * dim + np.sum(np.power(x, 2) - a * np.cos(2.0 * np.pi * x), axis=-1)


def ackley(
    x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2 * np.pi
) -> np.ndarray:
    """
    Ackley Function.

    A multimodal, non-separable function. It is characterized by a nearly flat outer region
    and a deep hole at the center. This tests an algorithm's ability to maintain
    exploration (on the flat part) and rapid exploitation (in the hole).

    **Equation:**
    $$ f(x) = -a \\exp\\left(-b \\sqrt{\\frac{1}{D} \\sum x_i^2}\\right) - \\exp\\left(\\frac{1}{D} \\sum \\cos(c x_i)\\right) + a + e $$

    **Global Minimum:**
    $f(x) = 0$ at $x = [0, ..., 0]$.

    Args:
        x (np.ndarray): Input vector(s).
        a, b, c (float): Shape coefficients.

    Returns:
        np.ndarray: Computed function values.
    """
    dim = x.shape[-1]

    sum1 = np.sum(np.power(x, 2), axis=-1)
    sum2 = np.sum(np.cos(c * x), axis=-1)

    term1 = -a * np.exp(-b * np.sqrt(sum1 / dim))
    term2 = -np.exp(sum2 / dim)

    return term1 + term2 + a + np.exp(1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Rosenbrock Function (The Banana Function).

    Unimodal (in low dims), non-convex, and non-separable. The global minimum lies inside
    a long, narrow, parabolic valley. Algorithms often find the valley quickly but struggle
    to converge to the minimum along the valley floor.

    **Equation:**
    $$ f(x) = \\sum_{i=1}^{D-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2] $$

    **Global Minimum:**
    $f(x) = 0$ at $x = [1, ..., 1]$.

    Args:
        x (np.ndarray): Input vector(s).

    Returns:
        np.ndarray: Computed function values.
    """

    # We slice to handle (D,) and (N, D) shapes
    if x.ndim == 1:
        x_next = x[1:]
        x_curr = x[:-1]
    else:
        x_next = x[:, 1:]
        x_curr = x[:, :-1]

    term1 = 100 * np.power(x_next - np.power(x_curr, 2), 2)
    term2 = np.power(1 - x_curr, 2)

    return np.sum(term1 + term2, axis=-1)


def griewank(x: np.ndarray) -> np.ndarray:
    """
    Griewank Function.

    A multimodal, non-separable function. The product term introduces interdependence
    among variables. As dimensions increase, the local minima become smoother.

    **Equation:**
    $$ f(x) = 1 + \\frac{1}{4000}\\sum_{i=1}^D x_i^2 - \\prod_{i=1}^D \\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right) $$

    **Global Minimum:**
    $f(x) = 0$ at $x = [0, ..., 0]$.

    Args:
        x (np.ndarray): Input vector(s).

    Returns:
        np.ndarray: Computed function values.
    """
    # 1-based index for the product term
    indices = np.arange(1, x.shape[-1] + 1)

    term1 = np.sum(np.power(x, 2) / 4000.0, axis=-1)
    term2 = np.prod(np.cos(x / np.sqrt(indices)), axis=-1)

    return term1 - term2 + 1


# ==============================================================================
# Asymmetric landscapes
# ==============================================================================
# Everything above except Rosenbrock is EVEN -- f(x) == f(-x) -- and on the
# symmetric boxes these are normally posed over that makes them a poor test of
# anything opposition-based: the opposite of x is exactly -x and carries
# identical fitness, so an OBL pool is half mirror pairs. Shifting the optimum
# off centre fixes the *location* but not the symmetry, and three of them are
# periodic as well, so a lattice of near-equivalent basins survives the shift.
#
# The three below break that on purpose, each in a different way:
# Styblinski-Tang by an odd term, Lévy by a structure with no symmetry at all,
# Zakharov by index weighting that breaks permutation symmetry too.


def styblinski_tang(x: np.ndarray) -> np.ndarray:
    """
    Styblinski-Tang Function.

    Multimodal and separable, with $2^D$ local minima -- one per orthant --
    of which only one is global. The linear $5x_i$ term is what makes it
    **asymmetric**: $f(x) \\neq f(-x)$, so a reflected point is a genuinely
    different candidate rather than a duplicate with the same score.

    **Equation:**
    $$ f(x) = \\frac{1}{2}\\sum_{i=1}^D (x_i^4 - 16 x_i^2 + 5 x_i) $$

    **Global Minimum:**
    $f(x) = -39.16599 D$ at $x_i = -2.903534$.

    Args:
        x (np.ndarray): Input vector(s).

    Returns:
        np.ndarray: Computed function values.
    """
    return 0.5 * np.sum(np.power(x, 4) - 16.0 * np.power(x, 2) + 5.0 * x, axis=-1)


def levy(x: np.ndarray) -> np.ndarray:
    """
    Lévy Function.

    Highly multimodal and non-separable, built on a change of variables that
    treats the first and last coordinates differently from the rest. That
    construction leaves it with no symmetry to exploit -- neither reflection
    nor permutation -- which is the property the even functions lack.

    **Equation:**
    With $w_i = 1 + (x_i - 1) / 4$,
    $$ f(x) = \\sin^2(\\pi w_1)
       + \\sum_{i=1}^{D-1}(w_i-1)^2\\left[1 + 10\\sin^2(\\pi w_i + 1)\\right]
       + (w_D-1)^2\\left[1 + \\sin^2(2\\pi w_D)\\right] $$

    **Global Minimum:**
    $f(x) = 0$ at $x_i = 1$.

    Args:
        x (np.ndarray): Input vector(s).

    Returns:
        np.ndarray: Computed function values.
    """
    w = 1.0 + (x - 1.0) / 4.0

    first = np.power(np.sin(np.pi * w[..., 0]), 2)
    last = np.power(w[..., -1] - 1.0, 2) * (
        1.0 + np.power(np.sin(2.0 * np.pi * w[..., -1]), 2)
    )

    middle = w[..., :-1]
    body = np.sum(
        np.power(middle - 1.0, 2)
        * (1.0 + 10.0 * np.power(np.sin(np.pi * middle + 1.0), 2)),
        axis=-1,
    )

    return first + body + last


def zakharov(x: np.ndarray) -> np.ndarray:
    """
    Zakharov Function.

    Unimodal, non-separable and increasingly ill-conditioned with dimension.
    The coordinates enter the sums weighted by their **index**, so unlike every
    other function here it is not invariant under permuting them -- a design
    that spreads its points evenly across dimensions is not thereby doing the
    right thing, which makes this a useful counterweight to the separable
    multimodal landscapes.

    Note it is still *even*: every term sees $x$ through an even power of the
    weighted sum, so $f(x) = f(-x)$ and it does not break reflection symmetry.
    It is here for the conditioning and the index weighting, not for that.

    **Equation:**
    $$ f(x) = \\sum_{i=1}^D x_i^2
       + \\left(\\sum_{i=1}^D 0.5\\,i\\,x_i\\right)^2
       + \\left(\\sum_{i=1}^D 0.5\\,i\\,x_i\\right)^4 $$

    **Global Minimum:**
    $f(x) = 0$ at $x = [0, ..., 0]$.

    Args:
        x (np.ndarray): Input vector(s).

    Returns:
        np.ndarray: Computed function values.
    """
    idx = np.arange(1, x.shape[-1] + 1)
    partial = np.sum(0.5 * idx * x, axis=-1)
    return np.sum(np.power(x, 2), axis=-1) + np.power(partial, 2) + np.power(partial, 4)


def dixon_price(x: np.ndarray) -> np.ndarray:
    """
    Dixon-Price Function.

    Unimodal, non-separable, and **asymmetric**: the leading $(x_1 - 1)^2$
    term alone means $f(x) \\neq f(-x)$. Each coordinate is coupled to its
    predecessor, so the valley twists rather than lying along an axis.

    **Equation:**
    $$ f(x) = (x_1 - 1)^2
       + \\sum_{i=2}^{D} i\\left(2x_i^2 - x_{i-1}\\right)^2 $$

    **Global Minimum:**
    $f(x) = 0$ at $x_i = 2^{-(2^i - 2) / 2^i}$.

    Args:
        x (np.ndarray): Input vector(s).

    Returns:
        np.ndarray: Computed function values.
    """
    idx = np.arange(2, x.shape[-1] + 1)
    head = np.power(x[..., 0] - 1.0, 2)
    tail = np.sum(
        idx * np.power(2.0 * np.power(x[..., 1:], 2) - x[..., :-1], 2), axis=-1
    )
    return head + tail
