import time
import numpy as np
from deap import tools, algorithms
from models.base_ga import setup_ga
from utils.penalty_funcs import adaptive_penalty

class PPSGA:
    def __init__(self, bounds, n_dimensions, population_size, generations):
        self.bounds = bounds
        self.n_dimensions = n_dimensions
        self.population_size = population_size
        self.generations = generations
        self.push_phase = generations // 2
        self.toolbox = setup_ga(bounds, n_dimensions)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.current_population = []

    def evaluate(self, individual, evaluate_func, gen):
        base_fitness = evaluate_func(individual)[0]
        if gen < self.push_phase:
            return base_fitness,
        penalty = adaptive_penalty(individual, self.current_population, self.bounds)
        return base_fitness + penalty,

    def run(self, evaluate_func):
        start_time = time.perf_counter()
        population = self.toolbox.population(n=self.population_size)
        self.current_population = population

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        logbook = tools.Logbook()
        logbook.header = ["gen", "min", "avg"]

        # Initial evaluation without penalty
        for ind in population:
            ind.fitness.values = (evaluate_func(ind)[0],)
        record = stats.compile(population)
        logbook.record(gen=0, **record)
        hof.update(population)

        for gen in range(1, self.generations + 1):
            self.toolbox.register("evaluate", self.evaluate, evaluate_func=evaluate_func, gen=gen)
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.7, mutpb=0.2)
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population = self.toolbox.select(population + offspring, k=self.population_size)
            self.current_population = population
            
            record = stats.compile(population)
            logbook.record(gen=gen, **record)
            hof.update(population)

        end_time = time.perf_counter()
        final_fitnesses = [self.evaluate(ind, evaluate_func, self.generations)[0] for ind in hof]
        return min(final_fitnesses), logbook, end_time - start_time
