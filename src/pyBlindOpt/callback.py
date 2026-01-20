# coding: utf-8

"""
Optimization Callback Utilities.

Provides ready-to-use callbacks that can be injected into the `Optimizer` loop
to modify behavior (e.g., Early Stopping) or inspect state (e.g., Logging).
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"


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

    def callback(
        self, epoch: int, fitness: np.ndarray, population: np.ndarray
    ) -> bool | np.ndarray | None:
        """
        Checks the stop condition.

        Returns:
            bool: True if stop condition is met, False otherwise.
        """
        self.epoch = epoch
        best_fitness = np.min(fitness)
        if best_fitness < self.threshold:
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

    def __init__(self, patience: int = 10, min_delta: float = 1e-6) -> None:
        """
        Args:
            patience (int): Number of epochs to wait.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_score = np.inf
        self.epoch = 0

    def callback(
        self, epoch: int, fitness: np.ndarray, population: np.ndarray
    ) -> bool | np.ndarray | None:
        """
        Updates internal counter and checks stop condition.

        Returns:
            bool: True if patience is exhausted.
        """
        self.epoch = epoch
        current_best = np.min(fitness)
        if current_best < (self.best_score - self.min_delta):
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

    def callback(
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

    def callback(
        self, epoch: int, fitness: np.ndarray, population: np.ndarray
    ) -> bool | np.ndarray | None:
        """
        Modifies the population in-place (or returns new array).

        Returns:
            np.ndarray: The clipped population.
        """
        return np.clip(population, self.bounds[:, 0], self.bounds[:, 1])
