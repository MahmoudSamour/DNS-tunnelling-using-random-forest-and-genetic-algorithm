import random
from deap import base, creator, tools

def setup_ga(bounds, n_dimensions):
    # Use existing classes if they are already created to avoid errors on re-runs
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, *bounds)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_dimensions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox
