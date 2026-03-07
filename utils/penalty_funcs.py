import numpy as np

def adaptive_penalty(individual, population, bounds):
    if not population:
        return 0
    total_violations = sum(sum(1 for x in ind if not (bounds[0] <= x <= bounds[1])) for ind in population)
    avg_violation = total_violations / len(population)
    penalty_strength = 1e6 * (1 + avg_violation)
    return sum(penalty_strength for x in individual if not (bounds[0] <= x <= bounds[1]))

def repair_individual(individual, bounds):
    return [max(bounds[0], min(bounds[1], x)) for x in individual]

def opposition_based_learning(population, bounds):
    opposite_population = []
    for individual in population:
        opposite_individual = []
        for i, dim in enumerate(individual):
            opposite_dim = (bounds[0] + bounds[1]) - dim
            opposite_individual.append(opposite_dim)
        opposite_population.append(repair_individual(opposite_individual, bounds))
    return opposite_population
