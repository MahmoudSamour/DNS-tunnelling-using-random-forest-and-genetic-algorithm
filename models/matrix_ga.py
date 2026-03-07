import numpy as np

class MatrixGA:
    def __init__(self, n_dim, size=5): # 5x5 grid = 25 population
        self.size = size
        self.n_dim = n_dim
        self.population = np.random.uniform(-5.12, 5.12, (size, size, n_dim))

    def run(self, fitness_func, n_gen=20):
        history = []
        for gen in range(n_gen):
            # Evaluate grid
            fitness_grid = np.array([[fitness_func(self.population[i,j])[0] 
                                    for j in range(self.size)] for i in range(self.size)])
            history.append(np.min(fitness_grid))
            
            # 2D Grid Crossover (Mating with neighbors)
            new_pop = self.population.copy()
            for i in range(self.size):
                for j in range(self.size):
                    # Randomly pick a neighbor to mate with
                    ni, nj = (i + np.random.choice([-1, 0, 1])) % self.size, (j + np.random.choice([-1, 0, 1])) % self.size
                    mask = np.random.rand(self.n_dim) > 0.5
                    new_pop[i,j][mask] = self.population[ni,nj][mask]
            
            # Mutation
            if np.random.rand() < 0.1:
                new_pop += np.random.normal(0, 0.1, new_pop.shape)
            self.population = new_pop
        return history
