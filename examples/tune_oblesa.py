import argparse
import logging
import sys

import numpy as np
import optuna

from pyBlindOpt.callback import EarlyStopping
from pyBlindOpt.de import DifferentialEvolution
from pyBlindOpt.functions import ackley, griewank, rastrigin, rosenbrock, sphere
from pyBlindOpt.init import oblesa

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Tuner")

# ==============================================================================
# 1. Benchmark Configuration
# ==============================================================================
BENCHMARKS = {
    "sphere": {"func": sphere, "bounds": [-100, 100]},
    "rastrigin": {"func": rastrigin, "bounds": [-5.12, 5.12]},
    "ackley": {"func": ackley, "bounds": [-32, 32]},
    "rosenbrock": {"func": rosenbrock, "bounds": [-5, 10]},
    "griewank": {"func": griewank, "bounds": [-600, 600]},
}

DIMENSIONS = [5, 10, 20, 40]
TARGET_EPSILON = 1e-2
MAX_OPTIMIZER_EPOCHS = 1024

# ==============================================================================
# 3. Evaluation Logic
# ==============================================================================


def evaluate_single_run(
    func_name: str,
    dim: int,
    seed: int,
    oblesa_params: dict,
    optimizer_cls=DifferentialEvolution,
) -> int:
    """
    Runs one optimization session and returns the number of epochs used.
    """
    # 1. Setup Environment
    func_info = BENCHMARKS[func_name]
    objective = func_info["func"]

    # Create Bounds vector (Dim, 2)
    bounds = np.zeros((dim, 2))
    bounds[:, 0] = func_info["bounds"][0]
    bounds[:, 1] = func_info["bounds"][1]

    # 2. Initialize Population using OBLESA (The Tuned Component)
    # We must handle the specific init seed to ensure reproducibility
    init_seed = np.random.default_rng(seed)

    try:
        # Note: oblesa assumes 'ess' module is available internally
        population = oblesa(
            objective=objective,
            bounds=bounds,
            n_pop=64,  # Fixed population size for fair comparison
            seed=init_seed,
            **oblesa_params,
        )
    except Exception as e:
        logger.warning(f"OBLESA failed with params {oblesa_params}: {e}")
        return MAX_OPTIMIZER_EPOCHS * 2  # Heavy penalty for crash

    # 3. Setup Optimizer
    # Target is 0.0 for all provided functions. Stop at 0.0 + epsilon.
    target_threshold = 0.0 + TARGET_EPSILON
    stopper = EarlyStopping(threshold=target_threshold)

    # We use a fixed configuration for DE to isolate OBLESA's impact
    optimizer = optimizer_cls(
        objective=objective,
        bounds=bounds,
        population=population,  # Pass the OBLESA population
        n_iter=MAX_OPTIMIZER_EPOCHS,
        variant="best/1/bin",
        callback=stopper.callback,
        seed=seed,
    )

    # 4. Run Optimization
    _ = optimizer.optimize()

    # 5. Extract Metric
    # If the optimizer reached max_iter without stopping, stopper.epoch might be max_iter-1.
    # We check if it actually succeeded.
    if optimizer.best_score < target_threshold:
        return stopper.epoch
    else:
        # Penalize non-convergence
        return MAX_OPTIMIZER_EPOCHS


# ==============================================================================
# 4. Optuna Objective
# ==============================================================================


def objective(trial: optuna.Trial, n_seeds: int):
    """
    Optuna objective function to tune OBLESA parameters.
    """
    # --- A. Sample Hyperparameters ---
    oblesa_params = {
        "epochs": trial.suggest_int("epochs", 128, 2048, step=128),
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "decay": trial.suggest_float("decay", 0.9, 0.99),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
        "search_mode": trial.suggest_categorical("search_mode", ["radius", "knn"]),
        "border_strategy": trial.suggest_categorical(
            "border_strategy", ["repulsive", "clipping"]
        ),
        "selection": trial.suggest_categorical("selection", ["best", "probabilistic"]),
        "diversity_weight": trial.suggest_float("diversity_weight", 0.0, 1.0, step=0.1),
    }

    total_epochs = 0
    step_count = 0

    # --- B. Evaluation Loop ---
    # We loop over functions and dimensions.
    # To speed things up, if the performance is terrible on low dims, we prune.

    for func_name in BENCHMARKS.keys():
        for dim in DIMENSIONS:
            # Aggregate score over multiple seeds
            seed_scores = []

            # Using specific seeds for reproducibility across trials
            seeds = [42 + i for i in range(n_seeds)]

            # We can run seeds in parallel if n_jobs is set in main
            # But here we run sequentially to allow fine-grained pruning
            for seed in seeds:
                epochs_needed = evaluate_single_run(func_name, dim, seed, oblesa_params)
                seed_scores.append(epochs_needed)

            avg_epochs = float(np.mean(seed_scores))
            total_epochs += avg_epochs

            # --- C. Pruning ---
            # Report the cumulative epochs as the intermediate value
            step_count += 1
            trial.report(total_epochs, step_count)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return total_epochs


# ==============================================================================
# 5. Main Execution
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Tune OBLESA hyperparameters using Optuna."
    )
    parser.add_argument(
        "--trials", type=int, default=100, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of random seeds per function config.",
    )
    parser.add_argument(
        "--storage", type=str, default="sqlite:///oblesa_tune.db", help="Database URL."
    )
    parser.add_argument(
        "--study_name", type=str, default="oblesa_optimization", help="Study Name."
    )
    args = parser.parse_args()

    logger.info(f"Starting optimization with {args.trials} trials, {args.seeds} seeds.")
    logger.info("Landscape types being tested:")

    # Visual context for the user (simulated)
    print("")

    # Create Study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",  # We want to minimize Epochs
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    # Optimization Loop
    try:
        study.optimize(
            lambda trial: objective(trial, args.seeds),
            n_trials=args.trials,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user. Saving current progress...")

    # Results
    logger.info("------------------------------------------------")
    logger.info("Optimization Finished.")
    logger.info(f"Best Trial Score (Total Epochs): {study.best_value}")
    logger.info("Best Parameters for OBLESA:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("------------------------------------------------")

    logger.info("------------------------------------------------")


if __name__ == "__main__":
    main()
