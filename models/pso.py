import time
from pyswarm import pso

def run_pso(evaluate_func, bounds, n_dimensions, n_generations):
    start_time = time.perf_counter()
    lb = [bounds[0]] * n_dimensions
    ub = [bounds[1]] * n_dimensions
    xopt, fopt = pso(lambda x: evaluate_func(x)[0], lb, ub, maxiter=n_generations*2, swarmsize=50) # PSO often needs more evaluations
    end_time = time.perf_counter()
    return fopt, None, end_time - start_time
