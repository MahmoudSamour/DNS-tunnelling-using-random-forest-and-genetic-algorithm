import numpy as np
from utils.penalty_funcs import adaptive_penalty, repair_individual, opposition_based_learning

def test_repair_individual_within_bounds():
    individual = [-5.0, 15.0, 3.5]
    bounds = (-10.0, 10.0)
    repaired = repair_individual(individual, bounds)
    assert np.allclose(repaired, [-5.0, 10.0, 3.5]), f"Repair individual failed. Got {repaired}"

def test_adaptive_penalty_no_violation():
    individual = [1.0, 2.0, 3.0]
    population = [[0.0, 1.0, -1.0], [2.0, 3.0, 4.0]]
    bounds = (-5.0, 5.0)
    penalty = adaptive_penalty(individual, population, bounds)
    assert penalty == 0, f"Expected penalty 0, got {penalty}"

def test_adaptive_penalty_with_violation():
    individual = [6.0, 2.0, 3.0] # 6.0 violates the 5.0 bound
    population = [[0.0, 1.0, -1.0], [2.0, 3.0, 4.0]]
    bounds = (-5.0, 5.0)
    penalty = adaptive_penalty(individual, population, bounds)
    assert penalty > 0, f"Expected positive penalty, got {penalty}"

def test_opposition_based_learning():
    population = [[-1.0, 2.0, 5.0], [0.0, -4.0, 2.0]]
    bounds = (-5.0, 5.0)
    opposite = opposition_based_learning(population, bounds)
    
    # Expected: (max+min)-val => (5-5)-val => -val
    # pop 1: [1.0, -2.0, -5.0]
    # pop 2: [0.0, 4.0, -2.0]
    assert np.allclose(opposite[0], [1.0, -2.0, -5.0]), f"OBL failed on pop1. Got {opposite[0]}"
    assert np.allclose(opposite[1], [0.0, 4.0, -2.0]), f"OBL failed on pop2. Got {opposite[1]}"
