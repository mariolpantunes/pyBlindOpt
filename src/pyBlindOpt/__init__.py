'''
pyBlindOpt is a library that implements several derivative-free optimization algorithms (including genetic and evolutionary methods).
Currently, it implements eight different algorithms:
1.  Random Search (RS): A baseline optimization method that iteratively generates candidate solutions from the search space according to a specified probability distribution (usually uniform) and records the best solution found. It serves as a benchmark for comparing the performance of more complex algorithms.
2.  Hill Climbing (HC): A mathematical optimization technique belonging to the family of local search algorithms. It is an iterative method that starts with an arbitrary solution and attempts to find a better one by making incremental changes to the current solution.
3.  Simulated Annealing (SA): A probabilistic technique for approximating the global optimum of a given function. It is a metaheuristic designed to escape local optima by allowing "uphill" moves (worse solutions) with a probability that decreases over time (simulating the cooling process of metallurgy).
4.  Genetic Algorithm (GA): A metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). GA generates high-quality solutions by relying on biologically inspired operators such as mutation, crossover, and selection.
5.  Differential Evolution (DE): A population-based method that optimizes a problem by iteratively improving a candidate solution with regard to a given measure of quality. It makes few to no assumptions about the problem being optimized and is effective for searching very large spaces of candidate solutions.
6.  Particle Swarm Optimization (PSO): A computational method that optimizes a problem by iteratively improving a candidate solution (particle) with regard to a given measure of quality. Particles move around the search space according to simple mathematical formulas involving their position and velocity. Each particle's movement is guided by its local best-known position and the global best-known position in the search space.
7.  Grey Wolf Optimization (GWO): A population-based metaheuristic algorithm that simulates the leadership hierarchy (Alpha, Beta, Delta, and Omega) and hunting mechanism of grey wolves in nature.
8.  Enhanced Grey Wolf Optimization (EGWO): An advanced variant of the standard GWO that incorporates mechanisms to better balance exploration and exploitation. This modification helps prevent the algorithm from stagnating in local optima, improving convergence speed and solution quality in complex landscapes.

All algorithms take advantage of the joblib library to parallelize objective function evaluations and cache results for improved performance.
Note: The code has been optimized to a certain degree but was primarily created for educational purposes. Please consider libraries like pymoo or SciPy if you require a production-grade implementation. Regardless, reported issues will be fixed whenever possible.
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
import pyBlindOpt.rs as rs
import pyBlindOpt.sa as sa
import pyBlindOpt.utils as utils
