import numpy as np

# --- Benchmark Functions ---
def sphere(individual):
    return sum(x**2 for x in individual),

def rastrigin(individual):
    return 10 * len(individual) + sum(x**2 - 10 * np.cos(2 * np.pi * x) for x in individual),

def rosenbrock(individual):
    return sum(100 * (individual[i+1] - individual[i]**2)**2 + (individual[i] - 1)**2 for i in range(len(individual) - 1)),

def ackley(individual):
    n = len(individual)
    sum_sq_term = -0.2 * np.sqrt(sum(x**2 for x in individual) / n)
    cos_term = sum(np.cos(2 * np.pi * x) for x in individual) / n
    return -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.e,

def griewank(individual):
    sum_part = sum(x**2 / 4000 for x in individual)
    prod_part = np.prod([np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(individual)])
    return sum_part - prod_part + 1,

def michalewicz(individual):
    m = 10
    total = 0
    for i, x in enumerate(individual):
        total += np.sin(x) * (np.sin((i + 1) * x**2 / np.pi))**(2 * m)
    return -total,

def schwefel(individual):
    n = len(individual)
    return 418.9829 * n - sum(x * np.sin(np.sqrt(np.abs(x))) for x in individual),

def zakharov(individual):
    sum1 = sum(x**2 for x in individual)
    sum2 = sum(0.5 * (i + 1) * x for i, x in enumerate(individual))
    return sum1 + sum2**2 + sum2**4,
