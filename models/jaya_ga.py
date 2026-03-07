import numpy as np

class HybridJayaGA:
    def __init__(self, n_dim, n_pop=20):
        self.n_dim = n_dim
        self.n_pop = n_pop
        self.pop = np.random.uniform(-5.12, 5.12, (n_pop, n_dim))

    def run(self, fitness_func, n_gen=20):
        history = []
        for gen in range(n_gen):
            fits = np.array([fitness_func(ind)[0] for ind in self.pop])
            best_idx, worst_idx = np.argmin(fits), np.argmax(fits)
            best_ind, worst_ind = self.pop[best_idx], self.pop[worst_idx]
            history.append(fits[best_idx])

            # JAYA Update Rule: Move toward best, away from worst
            r1, r2 = np.random.rand(self.n_pop, self.n_dim), np.random.rand(self.n_pop, self.n_dim)
            new_pop = self.pop + r1 * (best_ind - np.abs(self.pop)) - r2 * (worst_ind - np.abs(self.pop))
            
            # GA Crossover Step
            mask = np.random.rand(self.n_pop, self.n_dim) > 0.5
            self.pop[mask] = new_pop[mask]
        return history
