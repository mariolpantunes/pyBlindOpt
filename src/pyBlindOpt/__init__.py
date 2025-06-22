'''
A library that implements several derivation-free optimization algorithms (such as genetic optimization).
Currently, it implements 5 different algorithms:
1. Hill climbing is a mathematical optimization technique that belongs to the family of local search. It is an iterative algorithm that starts with an arbitrary solution to a problem and then attempts to find a better solution by making an incremental change to the solution.
2. Simulated annealing is a probabilistic technique for approximating the global optimum of a given function. Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization problem.
3. Genetic algorithm is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on biologically inspired operators such as mutation, crossover, and selection.
4. Differential evolution is a method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. Such methods are commonly known as metaheuristics as they make few or no assumptions about the problem being optimized and can search very large spaces of candidate solutions.
5. Particle swarm optimization is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions, here dubbed particles, and moving these particles around in the search space according to simple mathematical formula over the particle's position and velocity. Each particle's movement is influenced by its local best-known position but is also guided toward the best-known positions in the search space, which are updated as better positions are found by other particles.

All the algorithms take advantage of the [joblib](https://joblib.readthedocs.io/en/latest/) library to speed up the objective function and cache the results.
The code was optimized to a certain degree but was made for teaching purposes.
Please consider other libraries if you are looking for a stable implementation, such as [pymoo](https://pymoo.org/).
Regardless, any reported issues will be fixed as possible.
'''
import pyBlindOpt.callback as callback
import pyBlindOpt.de as de
import pyBlindOpt.egwo as egwo
import pyBlindOpt.functions as functions
import pyBlindOpt.ga as ga
import pyBlindOpt.gwo as gwo
import pyBlindOpt.hc as hc
import pyBlindOpt.init as init
import pyBlindOpt.pso as pso
import pyBlindOpt.sa as sa
import pyBlindOpt.utils as utils
