# coding: utf-8

"""
Optimization Callback Utilities.

Provides ready-to-use callbacks that can be injected into the `Optimizer` loop
to modify behavior (e.g., Early Stopping).
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"


import warnings

import numpy as np


class EarlyStopping:
    """
    Target-based Early Stopping.

    Stops the optimization process immediately once a solution with fitness
    below a specific `threshold` is found.

    **Condition:**
    $$ f(x_{best}) < \\text{threshold} $$
    """

    def __init__(self, threshold: float = 0.0) -> None:
        """
        Args:
            threshold (float): The target fitness value.
        """
        self.epoch = 0
        self.threshold = threshold

    def __call__(
        self, epoch: int, fitness: np.ndarray, population: np.ndarray
    ) -> bool | np.ndarray | None:
        """
        Checks the stop condition.

        Returns:
            bool: True if stop condition is met, False otherwise.
        """
        self.epoch = epoch
        # Safely get the minimum, ignoring NaNs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            best_fitness = np.nanmin(fitness)

        # If all values are NaN, or the best fitness hits the threshold, stop.
        if np.isnan(best_fitness) or best_fitness < self.threshold:
            return True

        return False


class PatienceStopping:
    """
    Stagnation-based Early Stopping.

    Stops the optimization if the global best score does not improve by at least
    `min_delta` for `patience` consecutive epochs.

    **Analogy:**
    Giving up after trying for N days without making any meaningful progress.
    """

    def __init__(
        self, patience: int = 10, min_delta: float = 1e-6, percentage: bool = False
    ) -> None:
        """
        Args:
            patience: Number of epochs to wait without improvement.
            min_delta: Minimum change to qualify as an improvement.
            percentage: If True, `min_delta` is treated as a fractional percentage
                        (e.g., 0.01 for 1% improvement) relative to the best score.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.percentage = percentage
        self.wait = 0
        self.best_score = np.inf
        self.epoch = 0

    def __call__(self, epoch: int, fitness: np.ndarray, population: np.ndarray) -> bool:
        """
        Updates internal counter and checks stop condition.

        Returns:
            bool: True if patience is exhausted.
        """
        self.epoch = epoch
        # Safely get the minimum, ignoring NaNs. 
        # Catch warnings if all values are NaN.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            current_best = np.nanmin(fitness)

        # If the entire population failed (all NaNs), treat it as stagnation
        if np.isnan(current_best):
            self.wait += 1
            return self.wait >= self.patience

        # Calculate the required threshold for improvement
        if self.percentage and self.best_score != np.inf:
            # Require an improvement of min_delta * best_score
            # We use abs() in case the objective function yields negative scores
            threshold = self.best_score - (abs(self.best_score) * self.min_delta)
        else:
            threshold = self.best_score - self.min_delta

        # Check for improvement
        if current_best < threshold:
            self.best_score = current_best
            self.wait = 0  # Reset patience
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True

        return False


class CountEpochs:
    """
    Epoch Counter.

    A simple utility to track the actual number of epochs executed (useful when
    early stopping is involved).
    """

    def __init__(self) -> None:
        self.epoch = 0

    def __call__(
        self, epoch: int, fitness: np.ndarray, population: np.ndarray
    ) -> bool | np.ndarray | None:
        """
        Increments internal epoch counter.
        """
        self.epoch = epoch + 1
        return None


class ClampBounds:
    """
    Population Constraint Callback.

    A population modification callback that forcibly clips all particles to
    stay within the defined search bounds at the end of every epoch.

    **Action:**
    $$ x_{i,d} = \\max(\\min(x_{i,d}, upper_d), lower_d) $$
    """

    def __init__(self, bounds: np.ndarray) -> None:
        """
        Args:
            bounds (np.ndarray): The min/max bounds matrix.
        """
        self.bounds = bounds

    def __call__(
        self, epoch: int, fitness: np.ndarray, population: np.ndarray
    ) -> bool | np.ndarray | None:
        """
        Modifies the population in-place (or returns new array).

        Returns:
            np.ndarray: The clipped population.
        """
        return np.clip(population, self.bounds[:, 0], self.bounds[:, 1])
