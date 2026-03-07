import time
import numpy as np
from deap import tools, algorithms

def run_standard_ga(toolbox, evaluate_func, n_population, n_generations):
    start_time = time.perf_counter()
    toolbox.register("evaluate", evaluate_func)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    population = toolbox.population(n=n_population)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    _, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations,
        stats=stats, halloffame=hof, verbose=False
    )
    end_time = time.perf_counter()
    return hof[0].fitness.values[0], logbook, end_time - start_time
