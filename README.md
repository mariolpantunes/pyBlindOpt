# ![pyBlindOpt logo](https://raw.githubusercontent.com/mariolpantunes/pyBlindOpt/refs/heads/main/assets/pyblindopt_logo.svg) pyBlindOpt

![PyPI - Version](https://img.shields.io/pypi/v/pyBlindOpt)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyBlindOpt)
![GitHub License](https://img.shields.io/github/license/mariolpantunes/pyBlindOpt)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mariolpantunes/pyBlindOpt/main.yml)
![GitHub last commit](https://img.shields.io/github/last-commit/mariolpantunes/pyBlindOpt)

**pyBlindOpt** is a library that implements several derivative-free optimization algorithms (including genetic and evolutionary methods).

Currently, it implements thirteen different algorithms:

1.  **Random Search (RS)**: A baseline optimization method that iteratively generates candidate solutions from the search space according to a specified probability distribution (usually uniform) and records the best solution found. It serves as a benchmark for comparing the performance of more complex algorithms.
2.  **Hill Climbing (HC)**: A mathematical optimization technique belonging to the family of local search algorithms. It is an iterative method that starts with an arbitrary solution and attempts to find a better one by making incremental changes to the current solution.
3.  **Simulated Annealing (SA)**: A probabilistic technique for approximating the global optimum of a given function. It is a metaheuristic designed to escape local optima by allowing "uphill" moves (worse solutions) with a probability that decreases over time (simulating the cooling process of metallurgy).
4.  **Genetic Algorithm (GA)**: A metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). GA generates high-quality solutions by relying on biologically inspired operators such as mutation, crossover, and selection.
5.  **Differential Evolution (DE)**: A population-based method that optimizes a problem by iteratively improving a candidate solution with regard to a given measure of quality. It makes few to no assumptions about the problem being optimized and is effective for searching very large spaces of candidate solutions. See [Differential Evolution variants](#differential-evolution-variants) for the mutation strategies and adaptation policies available.
6.  **Particle Swarm Optimization (PSO)**: A computational method that optimizes a problem by iteratively improving a candidate solution (particle) with regard to a given measure of quality. Particles move around the search space according to simple mathematical formulas involving their position and velocity. Each particle's movement is guided by its local best-known position and the global best-known position in the search space.
7.  **Grey Wolf Optimization (GWO)**: A population-based metaheuristic algorithm that simulates the leadership hierarchy (Alpha, Beta, Delta, and Omega) and hunting mechanism of grey wolves in nature.
8.  **Enhanced Grey Wolf Optimization (EGWO)**: An advanced variant of the standard GWO that incorporates mechanisms to better balance exploration and exploitation. This modification helps prevent the algorithm from stagnating in local optima, improving convergence speed and solution quality in complex landscapes.
9.  **Artificial Bee Colony (ABC)**: Simulates the foraging behavior of honey bees. The colony consists of employed bees (who exploit food sources), onlooker bees (who select sources based on quality), and scout bees (who find new random sources).
10. **Firefly Algorithm (FA)**: Inspired by the flashing behavior of fireflies. Fireflies are attracted to each other based on brightness (fitness), but the attractiveness decreases with distance, simulating light absorption.
11. **Harris Hawks Optimization (HHO)**: Mimics the cooperative hunting behavior of Harris' hawks, featuring distinct exploration and exploitation phases (like soft and hard besieges) controlled by the prey's escaping energy.
12. **Cuckoo Search (CS)**: Based on the brood parasitism of cuckoos. It uses Lévy flights for global exploration to generate new eggs and simulates nest abandonment to avoid local optima.
13. **Honey Badger Algorithm (HBA)**: Mimics the intelligent foraging behavior of honey badgers, switching between a "digging" phase (using smell intensity) and a "honey" phase (following a guide bird).

All algorithms take advantage of the [joblib](https://joblib.readthedocs.io/en/latest/) library to parallelize objective function evaluations and cache results for improved performance.

> **Note:** The code has been optimized to a certain degree but was primarily created for educational purposes. Please consider libraries like [pymoo](https://pymoo.org/) or [SciPy](https://scipy.org/) if you require a production-grade implementation. Regardless, reported issues will be fixed whenever possible..

## Differential Evolution variants

DE is configured along two independent axes, because the literature's adaptive
methods do not add new mutations -- they choose among the ones already there.

**`variant="base/n/crossover"`** selects the mutation and the recombination:

| base | mutation |
| --- | --- |
| `rand/1` | `v = r1 + F(r2 - r3)` -- standard, good diversity |
| `best/1` | `v = best + F(r1 - r2)` -- greedy, converges fast |
| `rand/2`, `best/2` | two difference vectors; more robust, slower |
| `current-to-best/1` | `v = x + F(best - x) + F(r1 - r2)` -- rotationally invariant |
| `current-to-pbest/1` | as above, but toward a draw from the fittest `p` fraction rather than the single best (JADE's mutation) |
| `current-to-rand/1` | rotationally invariant, exploratory |

with `bin` (independent gene swaps) or `exp` (a contiguous block) as the
crossover.

**`policy=`** selects how `F`, `cr` and the strategy are chosen each
generation:

| policy | behaviour |
| --- | --- |
| `"fixed"` (default) | constant `F` and `cr`, one strategy -- classical DE |
| `"archive"` | JADE's external archive of defeated parents, widening the pool the subtracted difference vector is drawn from |
| `"jade"` | the archive plus `F` and `cr` learned from the values that produce survivors |
| `"ensemble"` | a pool of strategies and parameters, one triple per individual, **kept while it succeeds and resampled when it fails** |

```python
from pyBlindOpt.de import differential_evolution

# JADE's mutation: each individual pulled toward its own draw from the top 10%
best, score = differential_evolution(
    objective, bounds, variant="current-to-pbest/1/bin", p=0.1)

# let the population find the strategy and parameters that suit the landscape
best, score = differential_evolution(objective, bounds, policy="ensemble")
```

`current-to-pbest/1` exists because pulling every individual toward the *same*
point is what makes `current-to-best/1` converge prematurely; pulling each
toward a different good point keeps the population spread. It is bracketed by
two variants already in the table -- `p` below `1/N` is exactly
`current-to-best/1`, and `p = 1` draws from the whole population.

The ensemble matters on multimodal landscapes, where no single strategy is
right for the whole run. Measured on rastrigin at `d=5`, 40 individuals, 250
generations, 8 seeds: the ensemble reached the optimum on **every** seed,
where `best/1/bin` had a median of 1.99 and a worst case of 5.97.

### What to choose

There is no arm that wins everywhere, and the differences are large enough
to matter. Median of 25 seeds, `d=10`, 40 individuals, 300 generations:

| function | `best/1` (default) | p-best | p-best + archive | JADE |
| --- | --- | --- | --- | --- |
| sphere | **9.0e-43** | 2.1e-32 | 6.8e-27 | 3.8e-16 |
| rastrigin | 7.96 | 9.94 | 12.44 | **0.0031** |
| ackley | 1.0e-10 | **4.0e-15** | 6.5e-13 | 2.8e-07 |
| griewank | 0.0984 | **0.0148** | 0.0156 | 0.0342 |
| rosenbrock | 6.77 | 6.93 | 2.51 | **0.95** |
| levy | 0.0895 | **3.8e-30** | 6.4e-26 | 1.7e-14 |
| zakharov | **3.3e-20** | 7.0e-16 | 7.7e-13 | 2.2e-11 |
| dixon_price | **0.667** | 0.667 | 0.667 | 0.667 |
| styblinski_tang | -363.4 | **-391.7** | -391.7 | -391.7 |

The pattern is consistent: **the default is best where the landscape rewards
greed and worst where it punishes it.** `best/1/bin` reaches machine
precision on sphere and zakharov, and is the only arm that fails outright --
7.96 on rastrigin and -363.4 on styblinski_tang, where every other arm finds
-391.7.

**JADE wins exactly where the default fails**, by 2500x on rastrigin and 7x
on rosenbrock, and pays for it with precision on the easy functions: its
heavy-tailed `F` keeps large steps available, which is what escapes local
optima and also what stops it converging to 1e-40 on a bowl.

The archive is worth a note of its own. On its own it is mostly *harmful* --
it made rastrigin worse, 9.94 to 12.44 -- because it buys diversity among
difference vectors and unimodal landscapes want the opposite. Combined with
the adaptation it is part of the arm that wins rastrigin outright. It is a
component of JADE, not a standalone upgrade.


## Installation

The library can be installed directly from GitHub by adding the following line to your `requirements.txt` file:

```bash
git+[https://github.com/mariolpantunes/pyBlindOpt@main#egg=pyBlindOpt](https://github.com/mariolpantunes/pyBlindOpt@main#egg=pyBlindOpt)
```

Alternatively, you can install a specific version from PyPI:

```bash
pyBlindOpt>=0.2.0
```

## Examples

### Simple Example

This example demonstrates how to run a basic optimization using **Simulated Annealing** on the Sphere function.

```python
import numpy as np
import pyBlindOpt

# 1. Define the search space (2 Dimensions, range -5.0 to 5.0)
bounds = np.array([[-5.0, 5.0]] * 2)

# 2. Run the optimization
# Usage: pyBlindOpt.simulated_annealing(objective, bounds, ...)
best_pos, best_score = pyBlindOpt.simulated_annealing(
    objective=pyBlindOpt.functions.sphere,
    bounds=bounds,
    n_iter=100,
    verbose=True
)

print(f"Best Position: {best_pos}")
print(f"Best Score: {best_score}")

```

### Advanced Example

This example demonstrates a complex workflow:

1. Initializing a reproducible random number generator (RNG).
2. Creating a **Hyper-Latin Cube Sampler (HLC)** bound to that RNG.
3. Generating an initial population using **Opposition-Based Learning (OBL)** combined with the HLC sampler.
4. Optimizing using **Grey Wolf Optimization (GWO)**, ensuring the custom population and RNG are passed through.

```python
import numpy as np
import pyBlindOpt

# 1. Setup reproducible RNG
seed = 42
rng = np.random.default_rng(seed)

# 2. Define Problem
bounds = np.array([[-100.0, 100.0]] * 10) # 10 Dimensions
objective = pyBlindOpt.functions.rastrigin
n_pop = 20

# 3. Create Sampler (Hyper-Latin Cube)
sampler = pyBlindOpt.utils.HLCSampler(rng)

# 4. Generate Initial Population using Opposition-Based Learning
# Passing 'sampler' as the population argument tells OBL how to sample the base set
initial_pop = pyBlindOpt.init.opposition_based(
    objective=objective,
    bounds=bounds,
    population=sampler,  # Use HLC for the random part of OBL
    n_pop=n_pop,
    seed=rng
)

# 5. Run GWO with the custom population and shared RNG
best_pos, best_score = pyBlindOpt.grey_wolf_optimization(
    objective=objective,
    bounds=bounds,
    population=initial_pop, # Pass the OBL-optimized population
    n_iter=200,
    n_pop=n_pop,
    verbose=True,
    seed=rng  # Pass the same RNG to ensure reproducibility of GWO internals
)

print(f"Best Position: {best_pos}")
print(f"Best Score: {best_score}")

```

## Documentation

This library is documented using Google-style docstrings.
The full documentation can be accessed [here](https://mariolpantunes.github.io/pyBlindOpt/).

To generate the documentation locally, run the following command:

```bash
pdoc --math -d google -o docs src/pyBlindOpt
```

## Authors

  * **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
