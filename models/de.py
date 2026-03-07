import time
from scipy.optimize import differential_evolution

def run_de(evaluate_func, bounds, n_dimensions, n_generations):
    start_time = time.perf_counter()
    bounds_list = [(bounds[0], bounds[1])] * n_dimensions
    result = differential_evolution(
        lambda x: evaluate_func(x)[0], bounds_list, maxiter=n_generations, popsize=15, polish=False
    )
    end_time = time.perf_counter()
    return result.fun, None, end_time - start_time
