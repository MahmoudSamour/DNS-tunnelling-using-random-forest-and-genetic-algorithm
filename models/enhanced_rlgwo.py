import time
import random
import numpy as np
from deap import tools
import torch
import torch.nn as nn
import torch.optim as optim
from utils.penalty_funcs import adaptive_penalty, repair_individual, opposition_based_learning

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class EnhancedRLGWO:
    def __init__(self, bounds, n_dimensions, population_size, generations):
        self.bounds = bounds
        self.n_dimensions = n_dimensions
        self.population_size = population_size
        self.generations = generations
        
        self.actions = [
            ("set_a", 0.5),   # Strong exploitation
            ("set_a", 1.0),   # Balanced
            ("set_a", 1.5),   # Strong exploration
            ("activate_obl", None) # Strategic jump
        ]
        
        self.epsilon_start, self.epsilon_end, self.epsilon_decay = 1.0, 0.01, 0.995
        self.epsilon = self.epsilon_start
        self.gamma = 0.99
        self.initial_optimizer_lr = 0.001
        self.lr_decay_rate = 0.01
        self.replay_buffer_size = 2000
        self.batch_size = 64
        self.beta = 0.4
        self.priority_epsilon = 1e-6
        self.tau = 0.005
        self.replay_buffer = []
        self.current_population = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_size = 3
        action_size = len(self.actions)
        
        self.q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.initial_optimizer_lr)

    def add_to_buffer(self, experience):
        max_priority = max([p for p, _ in self.replay_buffer]) if self.replay_buffer else 1.0
        self.replay_buffer.append((max_priority, experience))
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        priorities = np.array([p for p, _ in self.replay_buffer])
        probs = priorities ** self.beta
        probs /= probs.sum()

        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)
        batch = [self.replay_buffer[i][1] for i in indices]

        total = len(self.replay_buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        action_indices = torch.tensor(actions, dtype=torch.int64).to(self.device)

        q_values = self.q_network(states)
        predicted_q = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0]
        
        target_q = rewards + self.gamma * next_q_values
        td_errors = torch.abs(target_q - predicted_q).detach()
        loss = (weights * nn.MSELoss(reduction='none')(predicted_q, target_q)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update_target_network()

        for i, idx in enumerate(indices):
            priority = td_errors[i].item() + self.priority_epsilon
            self.replay_buffer[idx] = (priority, self.replay_buffer[idx][1])

    def soft_update_target_network(self):
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def choose_action_index(self, state):
        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def evaluate(self, individual, evaluate_func):
        base_fitness = evaluate_func(individual)[0]
        penalty = adaptive_penalty(individual, self.current_population, self.bounds)
        return base_fitness + penalty

    def run(self, evaluate_func):
        start_time = time.perf_counter()
        population = [repair_individual([random.uniform(self.bounds[0], self.bounds[1]) for _ in range(self.n_dimensions)], self.bounds) for _ in range(self.population_size)]
        self.current_population = population

        logbook = tools.Logbook()
        logbook.header = ["gen", "min", "avg", "diversity"]
        
        fitnesses = [self.evaluate(ind, evaluate_func) for ind in population]
        min_fit, avg_fit, std_fit = np.min(fitnesses), np.mean(fitnesses), np.std(fitnesses)
        logbook.record(gen=0, min=min_fit, avg=avg_fit, diversity=std_fit)
        state = [min_fit, avg_fit, std_fit]

        for gen in range(1, self.generations + 1):
            new_optimizer_lr = self.initial_optimizer_lr * np.exp(-self.lr_decay_rate * gen)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_optimizer_lr
            self.beta = min(1.0, self.beta + 0.001)

            action_index = self.choose_action_index(state)
            action_type, action_value = self.actions[action_index]

            if action_type == "activate_obl":
                opposite_pop = opposition_based_learning(population, self.bounds)
                combined_pop = population + opposite_pop
                
                combined_fitnesses = [self.evaluate(ind, evaluate_func) for ind in combined_pop]
                sorted_combined = sorted(zip(combined_pop, combined_fitnesses), key=lambda x: x[1])
                
                population = [ind for ind, fit in sorted_combined[:self.population_size]]
                fitnesses = [fit for ind, fit in sorted_combined[:self.population_size]]
            else: # action_type == "set_a"
                a_factor = action_value
                pop_with_fitness = list(zip(population, fitnesses))
                sorted_pop_with_fitness = sorted(pop_with_fitness, key=lambda x: x[1])
                sorted_pop = [ind for ind, fit in sorted_pop_with_fitness]
                alpha_wolf, beta_wolf, delta_wolf = sorted_pop[0], sorted_pop[1], sorted_pop[2]
                
                new_population = []
                for i in range(self.population_size):
                    A1,A2,A3 = a_factor*(2*np.random.random(self.n_dimensions)-1), a_factor*(2*np.random.random(self.n_dimensions)-1), a_factor*(2*np.random.random(self.n_dimensions)-1)
                    C1,C2,C3 = 2*np.random.random(self.n_dimensions), 2*np.random.random(self.n_dimensions), 2*np.random.random(self.n_dimensions)

                    D_alpha = np.abs(C1*np.array(alpha_wolf) - np.array(population[i]))
                    D_beta = np.abs(C2*np.array(beta_wolf) - np.array(population[i]))
                    D_delta = np.abs(C3*np.array(delta_wolf) - np.array(population[i]))
                    
                    X1,X2,X3 = np.array(alpha_wolf)-A1*D_alpha, np.array(beta_wolf)-A2*D_beta, np.array(delta_wolf)-A3*D_delta
                    
                    new_individual = (X1 + X2 + X3) / 3.0
                    new_population.append(repair_individual(new_individual.tolist(), self.bounds))
                population = new_population

            self.current_population = population
            fitnesses = [self.evaluate(ind, evaluate_func) for ind in population]
            min_fit, avg_fit, std_fit = np.min(fitnesses), np.mean(fitnesses), np.std(fitnesses)
            logbook.record(gen=gen, min=min_fit, avg=avg_fit, diversity=std_fit)
            
            reward = -min_fit + 0.1 * std_fit
            next_state = [min_fit, avg_fit, std_fit]

            self.add_to_buffer((state, action_index, reward, next_state))
            self.replay()
            state = next_state
            
            self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

        end_time = time.perf_counter()
        return min(fitnesses), logbook, end_time - start_time
