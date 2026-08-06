
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

    def _update_iter_params(self, epoch: int):
        """
        Hook: Update internal params based on current epoch (e.g., inertia, temperature).
        """

    @abc.abstractmethod
    def _generate_offspring(self, epoch: int) -> np.ndarray:
        """
        Abstract Hook: Generate new candidate solutions for the next step.
        """

    @abc.abstractmethod
    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Abstract Hook: Determine which solutions survive to the next generation.
        """

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

    def _evolve_once(self, epoch: int):
        """One generation: build offspring, evaluate them, select survivors.

        Extracted from `optimize` so a subclass can change *how many* batches
        a generation costs without reimplementing the whole loop. CoDE is the
        reason: it builds three trial vectors per individual and keeps the best
        of each triple, which is three evaluations per generation rather than
        one.

        A subclass that overrides this must keep every `evaluate` call to one
        population's worth of individuals. Batches are not an implementation
        detail here -- pyBlindOpt drives live agents whose engine requires all
        of them connected for a batch, so three batches of `n_pop` is a
        supported shape and one batch of `3 * n_pop` is not.

        Args:
            epoch (int): The current generation index.
        """
        offspring = self._generate_offspring(epoch)
        offspring = self._check_bounds(offspring)
        offspring_scores = self.evaluate(offspring)
        self._selection(offspring, offspring_scores)

    def optimize(self) -> tuple:
        """
        Runs the search and returns the best solution found.

        This is the Template Method: the loop below is fixed, and each
        algorithm supplies its own `_generate_offspring`, `_selection`,
        `_update_best` and `_update_iter_params`. One generation is

        1. update per-epoch parameters (a decaying step, a temperature);
        2. generate offspring, clip to bounds, evaluate;
        3. select survivors;
        4. update the incumbent best;
        5. run callbacks, which may stop the run or edit the population;
        6. append to `history`.

        A callback that edits the population triggers an immediate
        re-evaluation of the incumbent, so an injected solution is visible to
        the next generation rather than one generation later.

        `cleanup` runs in a `finally`, so the joblib cache directory is
        released even if the objective raises.

        Returns:
            tuple: `(best_position, best_score)`, or
            `(best_position, best_score, history)` when the optimizer was
            constructed with `debug=True`.
        """
        self._initialize()

        # Initial Best/Leader update before loop starts
        self._update_best(epoch=-1)

        epoch = 0
        try:
            # 1. Assign the tqdm iterator to a variable
            pbar = tqdm.tqdm(range(self.n_iter), disable=not self.verbose)

            for epoch in pbar:
                # 1. Update params (e.g., decay 'a')
                self._update_iter_params(epoch)

                # 2-3. Generate, evaluate, select.
                self._evolve_once(epoch)

                # 4. Update Best/Leaders
                self._update_best(epoch)

                # 2. Update the progress bar postfix
                if self.verbose:
                    # Formatting to scientific notation (.3e)
                    # keeps the UI clean and prevents the progress bar from jittering.
                    pbar.set_postfix(best_score=f"{self.best_score:.3e}")

                # 5. Callbacks
                stop_signal, population_changed = self._process_callbacks(epoch)

                # If callback mutated population, re-evaluate leaders immediately
                if population_changed:
                    self._update_best(epoch)
                    # 3. Re-update the postfix if a callback improved the score
                    if self.verbose:
                        pbar.set_postfix(best_score=f"{self.best_score:.6e}")

                # 6. Logging
                self._update_history(epoch)

                if stop_signal:
                    break
        finally:
            self.cleanup()

        return self._format_result(epoch)
