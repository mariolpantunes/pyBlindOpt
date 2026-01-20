"""
pyBlindOpt is a library that implements several derivative-free optimization algorithms (including genetic and evolutionary methods).

Currently, it implements thirteen different algorithms:

1.  **Random Search (RS):** A baseline optimization method that iteratively generates candidate solutions from the search space according to a specified probability distribution (usually uniform) and records the best solution found.
2.  **Hill Climbing (HC):** A local search algorithm that starts with an arbitrary solution and attempts to find a better one by making incremental changes (Greedy approach).
3.  **Simulated Annealing (SA):** A probabilistic technique designed to escape local optima by allowing "uphill" moves (worse solutions) with a probability that decreases over time (simulating the cooling process of metallurgy).
4.  **Genetic Algorithm (GA):** An evolutionary algorithm inspired by natural selection. It generates high-quality solutions using biologically inspired operators such as mutation, crossover, and selection.
5.  **Differential Evolution (DE):** A population-based method that optimizes a problem by iteratively improving a candidate solution using the differences between randomly selected vectors.
6.  **Particle Swarm Optimization (PSO):** A metaheuristic where particles move through the search space guided by their own best-known position and the swarm's global best-known position.
7.  **Grey Wolf Optimization (GWO):** A population-based algorithm that simulates the leadership hierarchy (Alpha, Beta, Delta, Omega) and hunting mechanism of grey wolves.
8.  **Enhanced Grey Wolf Optimization (EGWO):** An advanced variant of GWO that incorporates a weighted prey position and stochastic error to better balance exploration and exploitation.
9.  **Artificial Bee Colony (ABC):** Simulates the foraging behavior of honey bees, utilizing employed, onlooker, and scout bees to exploit food sources and explore new ones.
10. **Firefly Algorithm (FA):** Inspired by the flashing behavior of fireflies, where attraction is proportional to brightness (fitness) and decreases with distance.
11. **Harris Hawks Optimization (HHO):** Mimics the cooperative hunting behavior of Harris' hawks, featuring distinct exploration and exploitation phases controlled by the prey's escaping energy.
12. **Cuckoo Search (CS):** Based on the brood parasitism of cuckoos, utilizing Lévy flights for global exploration and nest abandonment for local optima avoidance.
13. **Honey Badger Algorithm (HBA):** Mimics the intelligent foraging behavior of honey badgers, switching between digging (using smell intensity) and following honeyguides.

All algorithms take advantage of the joblib library to parallelize objective function evaluations and cache results for improved performance.

Note: The code has been optimized to a certain degree but was primarily created for educational purposes. Please consider libraries like pymoo or SciPy if you require a production-grade implementation.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import pyBlindOpt.callback as callback
import pyBlindOpt.functions as functions
import pyBlindOpt.init as init
import pyBlindOpt.utils as utils
from pyBlindOpt.abc_opt import ArtificialBeeColony, artificial_bee_colony
from pyBlindOpt.cs import CuckooSearch, cuckoo_search
from pyBlindOpt.de import DifferentialEvolution, differential_evolution
from pyBlindOpt.egwo import EGWO, enhanced_grey_wolf_optimization
from pyBlindOpt.fa import FireflyAlgorithm, firefly_algorithm
from pyBlindOpt.ga import GeneticAlgorithm, genetic_algorithm
from pyBlindOpt.gwo import GWO, grey_wolf_optimization
from pyBlindOpt.hba import HoneyBadgerAlgorithm, honey_badger_algorithm
from pyBlindOpt.hc import HillClimbing, hill_climbing
from pyBlindOpt.hho import HarrisHawksOptimization, harris_hawks_optimization
from pyBlindOpt.pso import ParticleSwarmOptimization, particle_swarm_optimization
from pyBlindOpt.rs import RandomSearch, random_search
from pyBlindOpt.sa import SimulatedAnnealing, simulated_annealing

__all__ = [
    # Modules
    "functions",
    "utils",
    "init",
    "callback",
    # Algorithms (Classes)
    "RandomSearch",
    "HillClimbing",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "DifferentialEvolution",
    "ParticleSwarmOptimization",
    "GWO",
    "EGWO",
    "ArtificialBeeColony",
    "FireflyAlgorithm",
    "HarrisHawksOptimization",
    "CuckooSearch",
    "HoneyBadgerAlgorithm",
    # Algorithms (Functions)
    "random_search",
    "hill_climbing",
    "simulated_annealing",
    "genetic_algorithm",
    "differential_evolution",
    "particle_swarm_optimization",
    "grey_wolf_optimization",
    "enhanced_grey_wolf_optimization",
    "artificial_bee_colony",
    "firefly_algorithm",
    "harris_hawks_optimization",
    "cuckoo_search",
    "honey_badger_algorithm",
]
