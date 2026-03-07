import time
import random
import numpy as np
from deap import tools
from collections import defaultdict
from utils.penalty_funcs import adaptive_penalty, repair_individual

class RLCGWO:
    def __init__(self, bounds, n_dimensions, population_size, generations):
        self.bounds = bounds
        self.n_dimensions = n_dimensions
        self.population_size = population_size
        self.generations = generations
        self.a = 2.0
        self.actions = ["exploration", "exploitation"]
        self.epsilon, self.alpha, self.gamma = 0.2, 0.1, 0.9
        self.current_action = "balanced"
        self.q_table = defaultdict(lambda: 0)
        self.current_population = []

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.q_table[(state, action)] for action in self.actions]
        return self.actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        max_next_q = max(self.q_table.get((next_state, a), 0) for a in self.actions)
        self.q_table[(state, action)] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[(state, action)])

    def evaluate(self, individual, evaluate_func):
        base_fitness = evaluate_func(individual)[0]
        penalty = adaptive_penalty(individual, self.current_population, self.bounds)
        return base_fitness + penalty

    def run(self, evaluate_func):
        start_time = time.perf_counter()
        population = [repair_individual([random.uniform(self.bounds[0], self.bounds[1]) for _ in range(self.n_dimensions)], self.bounds) for _ in range(self.population_size)]
        self.current_population = population

        stats = tools.Statistics(key=lambda ind: self.evaluate(ind, evaluate_func))
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("diversity", np.std)
        logbook = tools.Logbook()
        logbook.header = ["gen", "min", "avg", "diversity"]
        
        record = stats.compile(population)
        logbook.record(gen=0, **record)
        state = f"{record['min']:.2f}_{record['diversity']:.2f}"

        for gen in range(1, self.generations + 1):
            self.a = 2 * (1 - gen / self.generations)
            self.current_action = self.choose_action(state)
            a_factor = 1.5 if self.current_action == "exploration" else 0.5
            
            sorted_pop = sorted(population, key=lambda ind: self.evaluate(ind, evaluate_func))
            alpha, beta, delta = sorted_pop[0], sorted_pop[1], sorted_pop[2]
            
            new_population = []
            for i in range(self.population_size):
                A1 = a_factor * (2 * np.random.random(self.n_dimensions) - 1)
                A2 = a_factor * (2 * np.random.random(self.n_dimensions) - 1)
                A3 = a_factor * (2 * np.random.random(self.n_dimensions) - 1)
                C1, C2, C3 = 2 * np.random.random(self.n_dimensions), 2 * np.random.random(self.n_dimensions), 2 * np.random.random(self.n_dimensions)

                D_alpha = np.abs(C1 * np.array(alpha) - np.array(population[i]))
                D_beta = np.abs(C2 * np.array(beta) - np.array(population[i]))
                D_delta = np.abs(C3 * np.array(delta) - np.array(population[i]))
                
                X1 = np.array(alpha) - A1 * D_alpha
                X2 = np.array(beta) - A2 * D_beta
                X3 = np.array(delta) - A3 * D_delta
                
                new_individual = (X1 + X2 + X3) / 3.0
                new_population.append(repair_individual(new_individual.tolist(), self.bounds))
            
            population = new_population
            self.current_population = population
            
            record = stats.compile(population)
            logbook.record(gen=gen, **record)
            reward = -record["min"] + 0.1 * record["diversity"]
            next_state = f"{record['min']:.2f}_{record['diversity']:.2f}"
            self.update_q_table(state, self.current_action, reward, next_state)
            state = next_state

        end_time = time.perf_counter()
        final_fitnesses = [self.evaluate(ind, evaluate_func) for ind in population]
        return min(final_fitnesses), logbook, end_time - start_time
