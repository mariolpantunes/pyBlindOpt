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
