import numpy as np
from utils.benchmark_funcs import sphere, rastrigin, rosenbrock, ackley, griewank, michalewicz, schwefel, zakharov

def test_sphere_global_optimum():
    optimum = [0.0, 0.0, 0.0, 0.0]
    result = sphere(optimum)[0]
    assert np.isclose(result, 0.0), f"Sphere failed: Expected 0.0, got {result}"

def test_rastrigin_global_optimum():
    optimum = [0.0, 0.0, 0.0]
    result = rastrigin(optimum)[0]
    assert np.isclose(result, 0.0), f"Rastrigin failed: Expected 0.0, got {result}"

def test_rosenbrock_global_optimum():
    optimum = [1.0, 1.0, 1.0, 1.0]
    result = rosenbrock(optimum)[0]
    assert np.isclose(result, 0.0), f"Rosenbrock failed: Expected 0.0, got {result}"

def test_ackley_global_optimum():
    optimum = [0.0, 0.0, 0.0]
    result = ackley(optimum)[0]
    # Ackley global min is very close to 0 due to float precision
    assert result < 1e-15, f"Ackley failed: Expected ~0.0, got {result}"

def test_griewank_global_optimum():
    optimum = [0.0, 0.0, 0.0, 0.0, 0.0]
    result = griewank(optimum)[0]
    assert np.isclose(result, 0.0), f"Griewank failed: Expected 0.0, got {result}"

def test_zakharov_global_optimum():
    optimum = [0.0, 0.0, 0.0]
    result = zakharov(optimum)[0]
    assert np.isclose(result, 0.0), f"Zakharov failed: Expected 0.0, got {result}"
