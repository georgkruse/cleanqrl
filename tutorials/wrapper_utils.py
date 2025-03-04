import jax
import sympy as sp
import numpy as np 
import itertools
import jumanji
import jumanji.wrappers
import gymnasium as gym
from collections import defaultdict
from gymnasium.spaces import Box, Dict, Discrete

from jumanji.environments import TSP, Knapsack, Maze
from jumanji.environments.routing.tsp.generator import UniformGenerator
from jumanji.environments.packing.knapsack.generator import RandomGenerator as RandomGeneratorKnapsack
from jumanji.environments.routing.maze.generator import RandomGenerator as RandomGeneratorMaze


def tsp_create_ising_hamiltonian(self, nodes):
    # Calculate number of cities
    num_cities = nodes.shape[0]
    
    # Calculate distance matrix
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                # Euclidean distance between cities
                distances[i, j] = np.sqrt((nodes[i, 0] - nodes[j, 0])**2 + 
                                        (nodes[i, 1] - nodes[j, 1])**2)
    
    # Create binary variables for QUBO formulation
    # x_i,p = 1 if city i is visited at position p
    x = {}
    for i in range(num_cities):
        for p in range(num_cities):
            x[(i, p)] = sp.symbols(f'x_{i}_{p}')
    
    # QUBO Hamiltonian
    H_qubo = 0
    
    # Constraint 1: Each city must be visited exactly once
    A = 1.0  # Penalty coefficient
    for i in range(num_cities):
        constraint = (sum(x[(i, p)] for p in range(num_cities)) - 1)**2
        H_qubo += A * constraint
    
    # Constraint 2: Each position must have exactly one city
    for p in range(num_cities):
        constraint = (sum(x[(i, p)] for i in range(num_cities)) - 1)**2
        H_qubo += A * constraint
    
    # Objective function: Minimize total distance
    B = 1.0  # Weight for the objective term
    for p in range(num_cities):
        p_next = (p + 1) % num_cities
        for i in range(num_cities):
            for j in range(num_cities):
                H_qubo += B * distances[i, j] * x[(i, p)] * x[(j, p_next)]
    
    # Expand QUBO
    H_qubo = sp.expand(H_qubo)
    
    # Create spin variables for Ising formulation
    s = {}
    for i in range(num_cities):
        for p in range(num_cities):
            s[(i, p)] = sp.symbols(f's_{i}_{p}')
    
    # Convert QUBO to Ising using the transformation x_i = (s_i + 1)/2
    H_ising = H_qubo
    for key in x:
        H_ising = H_ising.subs(x[key], (s[key] + 1)/2)
    
    # Expand the Ising Hamiltonian
    H_ising = sp.expand(H_ising)
    
    return H_ising, s




def convert_jumanji_state_to_ising_model(env_id, config):
    if env_id == 'TSP-v1':
        num_cities = config.get('num_cities', 5)
        generator_tsp = UniformGenerator(num_cities=num_cities)
        env = TSP(generator_tsp)
    elif env_id == 'Knapsack-v1':
        num_items = config.get('num_items', 5)
        total_budget = config.get('total_budget', 2)
        generator_knapsack = RandomGeneratorKnapsack(num_items=num_items, total_budget=total_budget)
        env = Knapsack(generator=generator_knapsack)
        
    elif env_id == 'Maze-v0':
        num_rows = config.get('num_rows', 4)
        num_cols = config.get('num_cols', 4)
        constant_maze = config.get('constant_maze', False)
        generator_maze = RandomGeneratorMaze(num_cols=num_cols, num_rows=num_rows)
        env = Maze(generator=generator_maze)

    env = jumanji.wrappers.JumanjiToGymWrapper(env)
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if env_id == 'TSP-v1':
        env = JumanjiWrapperTSP(env)
    elif env_id == 'Knapsack-v1':
        env = JumanjiWrapperKnapsack(env)
    elif env_id == 'Maze-v0':
        env = JumanjiWrapperMaze(env)
        if constant_maze:
            env.set_seed_constant(0)
    return env