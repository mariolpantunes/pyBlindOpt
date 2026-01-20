# coding: utf-8

"""
Base Optimizer Architecture.

Defines the abstract base class `Optimizer` which implements the Template Method
design pattern for population-based meta-heuristics. It handles common infrastructure:
* Random Number Generation (Seeding)
* Caching (Joblib)
* Bound Constraints
* Callback execution
* History logging
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import abc
import collections.abc
import logging
import tempfile

import joblib
import numpy as np
import tqdm

import pyBlindOpt.init as init
import pyBlindOpt.utils as utils

logger = logging.getLogger(__name__)


class Optimizer(abc.ABC):
    """
    Abstract Base Class for optimization algorithms.

    Encapsulates the standard optimization loop:
    Initialize -> Loop [ Update Params -> Generate -> Select -> Update Best -> Callbacks ]
    """

    def __init__(
        self,
        objective: collections.abc.Callable,
        bounds: np.ndarray,
        *,
        population: np.ndarray | None = None,
        callback: "list[collections.abc.Callable] | collections.abc.Callable | None" = None,
        n_iter: int = 100,
        n_pop: int = 10,
        n_jobs: int = 1,
        cached: bool = False,
        debug: bool = False,
        verbose: bool = False,
        seed: int | np.random.Generator | utils.Sampler | None = None,
    ):
        """
        Initializes the optimizer infrastructure.

        Args:
            objective: Target function.
            bounds: Search space constraints.
            n_iter: Max epochs.
            n_pop: Population size.
            cached: Enable disk caching for objective.
        """
        self.objective = objective
        self.bounds = bounds
        self.n_iter = n_iter
        self.n_pop = n_pop
        self.n_jobs = n_jobs
        self.cached = cached
        self.debug = debug
        self.verbose = verbose

        # 1. Setup Random Generator
        self.rng = self._check_random_state(seed)

        # 2. Setup Caching
        self.memory = None
        self.objective_cache = objective
        if self.cached:
            location = tempfile.gettempdir()
            self.memory = joblib.Memory(location, verbose=0)
            self.objective_cache = self.memory.cache(objective)

        # 3. Initialize Population
        self._init_population(population, seed)

        # 4. Initial Evaluation
        self.scores = self.evaluate(self.pop)

        # 5. Global Best Tracking (Subclasses must update these)
        self.best_pos = None
        self.best_score = np.inf

        # 6. Setup History & Callbacks
        self.history = np.zeros((n_iter, 3)) if debug else None

        if callback is None:
            self.callbacks = []
        elif isinstance(callback, collections.abc.Sequence):
            self.callbacks = callback
        else:
            self.callbacks = [callback]

    def _check_random_state(self, seed):
        if isinstance(seed, utils.Sampler):
            return seed.rng
        elif isinstance(seed, np.random.Generator):
            return seed
        else:
            return np.random.default_rng(seed)

    def _init_population(self, population, seed):
        if population is None:
            sampler = (
                seed
                if isinstance(seed, utils.Sampler)
                else utils.RandomSampler(self.rng)
            )
            self.pop = init.get_initial_population(self.n_pop, self.bounds, sampler)
        else:
            self.pop = np.clip(population, self.bounds[:, 0], self.bounds[:, 1])
            self.n_pop = self.pop.shape[0]

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """
        Wrapper for objective function evaluation.

        Handles parallelization and caching logic via `utils.compute_objective`.
        """
        return utils.compute_objective(population, self.objective_cache, self.n_jobs)

    def _check_bounds(self, population: np.ndarray) -> np.ndarray:
        """
        Clamps solution values to the defined search space bounds.
        """
        return np.clip(population, self.bounds[:, 0], self.bounds[:, 1])

    def _process_callbacks(self, epoch: int) -> tuple[bool, bool]:
        """
        Executes all registered callbacks.

        Returns:
            stop_signal (bool): If True, aborts the loop.
            population_changed (bool): If True, indicates a callback modified the population.
        """
        stop_signal = False
        population_changed = False

        for c in self.callbacks:
            pre_callback_pop = self.pop.copy()
            res = c(epoch, self.scores, self.pop)

            if isinstance(res, (bool, np.bool_)) and res:
                stop_signal = True
                break
            elif isinstance(res, np.ndarray):
                if res.shape != self.pop.shape:
                    raise ValueError(
                        f"Callback changed pop shape {self.pop.shape}->{res.shape}"
                    )

                self.pop = res
                changed_mask = np.any(self.pop != pre_callback_pop, axis=1)

                if np.any(changed_mask):
                    population_changed = True
                    self.pop[changed_mask] = self._check_bounds(self.pop[changed_mask])
                    self.scores[changed_mask] = self.evaluate(self.pop[changed_mask])

        return stop_signal, population_changed

    def _update_history(self, epoch: int):
        """
        Logs metrics (Best, Mean, Max) if `debug=True`.
        """
        if self.debug and self.history is not None:
            self.history[epoch, 0] = self.best_score
            self.history[epoch, 1] = np.mean(self.scores)
            self.history[epoch, 2] = np.max(self.scores)

    def cleanup(self):
        """
        Resource cleanup (e.g., clearing Joblib memory cache).
        """
        if self.cached and self.memory is not None:
            self.memory.clear(warn=False)

    def _format_result(self, current_epoch: int):
        """
        Formats the final return value (tuple structure).
        """
        if self.debug and self.history is not None:
            actual_hist = self.history[: current_epoch + 1]
            return (
                self.best_pos,
                self.best_score,
                (actual_hist[:, 0], actual_hist[:, 1], actual_hist[:, 2]),
            )
        else:
            return self.best_pos, self.best_score

    def _initialize(self):
        """
        Hook: Run once before the main loop starts (e.g., initial leader finding).
        """
        pass

    def _update_iter_params(self, epoch: int):
        """
        Hook: Update internal params based on current epoch (e.g., inertia, temperature).
        """
        pass

    @abc.abstractmethod
    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Abstract Hook: Generate new candidate solutions for the next step.
        """
        pass

    @abc.abstractmethod
    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Abstract Hook: Determine which solutions survive to the next generation.
        """
        pass

    @abc.abstractmethod
    def _update_best(self, epoch: int):
        """
        The Main Optimization Loop (Template Method).

        Orchestrates the iterative process:
        1.  Initialize.
        2.  For each epoch:
            * Update parameters (e.g., temperature, inertia).
            * Generate offspring.
            * Evaluate and Select.
            * Update Global Best.
            * Run Callbacks.
            * Log History.
        3.  Cleanup and Return.

        Returns:
            tuple: (best_pos, best_score, [history])
        """
        pass

    def optimize(self) -> tuple:
        self._initialize()

        # Initial Best/Leader update before loop starts
        self._update_best(epoch=-1)

        epoch = 0
        try:
            for epoch in tqdm.tqdm(range(self.n_iter), disable=not self.verbose):
                # 1. Update params (e.g., decay 'a')
                self._update_iter_params(epoch)

                # 2. Algorithm Logic
                offspring = self._generate_offspring(epoch)
                offspring = self._check_bounds(offspring)
                offspring_scores = self.evaluate(offspring)

                # 3. Selection (e.g., Greedy)
                self._selection(offspring, offspring_scores)

                # 4. Update Best/Leaders
                self._update_best(epoch)

                # 5. Callbacks
                stop_signal, population_changed = self._process_callbacks(epoch)

                # If callback mutated population, re-evaluate leaders immediately
                if population_changed:
                    self._update_best(epoch)

                # 6. Logging
                self._update_history(epoch)

                if stop_signal:
                    break
        finally:
            self.cleanup()

        return self._format_result(epoch)
