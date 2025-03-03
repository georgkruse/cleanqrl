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



class JumanjiWrapperTSP(gym.Wrapper):
    def step(self, action):
        state, reward, terminate, truncate, info = self.env.step(action)
        num_cities = self.env.unwrapped.num_cities

        nodes = np.reshape(state[num_cities:num_cities*3], (-1, 2))

        H = self.tsp_create_ising_hamiltonian(nodes)
        if truncate:
            num_cities = self.env.unwrapped.num_cities
            nodes = np.reshape(self.previous_state[num_cities:num_cities*3], (-1, 2))
            optimal_tour_length = self.tsp_compute_optimal_tour(nodes)
            info['optimal_tour_length'] = optimal_tour_length
            info['approximation_ratio'] = info['episode']['r']/optimal_tour_length
            if info['episode']['l'] < num_cities:
                info['approximation_ratio'] -= 10
            # else: 
            #     if info['approximation_ratio'] > -1.1:
            #         print(info['approximation_ratio'])
        else:
            info = dict()
        self.previous_state = state
        return state, reward, False, truncate, info


    def tsp_compute_optimal_tour(self, nodes):

        optimal_tour_length = 1000
        for tour_permutation  in itertools.permutations(range(1,nodes.shape[0])):
            tour = [0] + list(tour_permutation)
            tour_length = self.tsp_compute_tour_length(nodes, tour)
            if tour_length < optimal_tour_length:
                optimal_tour_length = tour_length
        return optimal_tour_length


    def tsp_compute_tour_length(self, nodes, tour):
        """
        Compute length of a tour, including return to start node.
        (If start node is already added as last node in tour, 0 will be added to tour length.)
        :param nodes: all nodes in the graph in form of (x, y) coordinates
        :param tour: list of node indices denoting a (potentially partial) tour
        :return: tour length
        """
        tour_length = 0
        for i in range(len(tour)):
            if i < len(tour)-1:
                tour_length += np.linalg.norm(np.asarray(nodes[tour[i]]) - np.asarray(nodes[tour[i+1]]))
            else:
                tour_length += np.linalg.norm(np.asarray(nodes[tour[-1]]) - np.asarray(nodes[tour[0]]))

        return tour_length


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

class JumanjiWrapperKnapsack(gym.Wrapper):

    
    # def __init__(self, env):
    #     super().__init__(env)
    #     spaces = {}
    #     num_items = 4
    #     num_quadratic_terms = sum(range(4))
    #     spaces['linear'] = Box(-np.inf, np.inf, shape = (num_items,2), dtype='float64')
    #     spaces['quadratic'] = Box(-np.inf, np.inf, shape = (num_quadratic_terms,3), dtype='float64')
    #     spaces['annotations'] = Box(-np.inf, np.inf, shape = (num_quadratic_terms,2), dtype='float64')

    #     self._observation_space = Dict(spaces = spaces)
        # self._observation_space = Dict

    # def reset(self, **kwargs):

    #     output = self.env.reset()
    #     spaces = {}
    #     spaces['linear'] = h 
    #     spaces['quadratic'] = J 
    #     spaces['annotations']

    #     return output
    

    def step(self, action):
        state, reward, terminate, truncate, info = self.env.step(action)
        if truncate:
            num_items = self.env.unwrapped.num_items
            total_budget = self.env.unwrapped.total_budget
            values = self.previous_state[num_items*2:num_items*3]
            weights = self.previous_state[-num_items:]
            optimal_value = self.knapsack_optimal_value(weights, values, total_budget)
            info['optimal_value'] = optimal_value
            info['approximation_ratio'] = info['episode']['r']/optimal_value
            # if info['approximation_ratio'] > 0.9:
            #     print(info['approximation_ratio'])
        else:
            info = dict()
            num_items = self.env.unwrapped.num_items
            total_budget = self.env.unwrapped.total_budget
            values = state[num_items*2:num_items*3]
            weights = state[-num_items:]
        self.previous_state = state
        # self.encoding = 'hamiltonian'
        # if self.encoding == 'hamiltonian':
        #     self.weights = weights
        #     self.values = values
        #     self.max_weight = total_budget
        #     offset, QUBO = self.formulate_qubo_unbalanced()
        #     offset, h, J = self.from_Q_to_Ising(offset, QUBO)
        #     state = h, J
        return state, reward, False, truncate, info

    def convert_state_to_cost_hamiltonian(self, state):
        num_items = self.env.unwrapped.num_items
        total_budget = self.env.unwrapped.total_budget
        values = state[num_items*2:num_items*3]
        weights = state[-num_items:]
        self.weights = weights
        self.values = values
        self.max_weight = total_budget
        offset, QUBO = self.formulate_qubo_unbalanced()
        offset, h, J = self.from_Q_to_Ising(offset, QUBO)
        
        return offset, h, J        

    def knapsack_optimal_value(self, weights, values, total_budget, precision=1000):
        """
        Solves the knapsack problem with float weights and values between 0 and 1.
        
        Args:
            weights: List or array of item weights (floats between 0 and 1)
            values: List or array of item values (floats between 0 and 1)
            capacity: Maximum weight capacity of the knapsack (float)
            precision: Number of discretization steps for weights (default: 1000)
            
        Returns:
            The maximum value that can be achieved
        """
        # Convert to numpy arrays
        weights = np.array(weights)
        values = np.array(values)
        
        # Validate input
        if not np.all((0 <= weights) & (weights <= 1)) or not np.all((0 <= values) & (values <= 1)):
            raise ValueError("All weights and values must be between 0 and 1")
        
        if total_budget < 0:
            raise ValueError("Capacity must be non-negative")
        
        n = len(weights)
        if n == 0:
            return 0.0
        
        # Scale weights to integers for dynamic programming
        scaled_weights = np.round(weights * precision).astype(int)
        scaled_capacity = int(total_budget * precision)
        
        # Initialize DP table
        dp = np.zeros(scaled_capacity + 1)
        
        # Fill the DP table
        for i in range(n):
            # We need to go backward to avoid counting an item multiple times
            for w in range(scaled_capacity, scaled_weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - scaled_weights[i]] + values[i])
        
        return float(dp[scaled_capacity])


    def formulate_qubo_unbalanced(self, lambdas = None):
        """
        Formulates the QUBO with the unbalanced penalization method.
        This means the QUBO does not use additional slack variables.
        Params:
            lambdas: Correspond to the penalty factors in the unbalanced formulation.
        """
        if lambdas is None:
            lambdas = [0.96, 0.0371]
        num_items = len(self.values)
        x = [sp.symbols(f"{i}") for i in range(num_items)]
        cost = 0
        constraint = 0

        for i in range(num_items):
            cost -= x[i] * self.values[i]
            constraint += x[i] * self.weights[i]

        H_constraint = self.max_weight - constraint
        H_constraint_taylor = 1 - lambdas[0] * H_constraint + 0.5 * lambdas[1] * H_constraint**2
        H_total = cost + H_constraint_taylor
        H_total = H_total.expand()
        H_total = sp.simplify(H_total)

        for i in range(len(x)):
            H_total = H_total.subs(x[i] ** 2, x[i])

        H_total = H_total.expand()
        H_total = sp.simplify(H_total)

        "Transform into QUBO matrix"
        coefficients = H_total.as_coefficients_dict()

        # Remove the offset
        try:
            offset = coefficients.pop(1)
        except IndexError:
            print("Warning: No offset found in coefficients. Using default of 0.")
            offset = 0
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            offset = 0

        # Get the QUBO
        QUBO = np.zeros((num_items, num_items))
        for key, value in coefficients.items():
            key = str(key)
            parts = key.split("*")
            if len(parts) == 1:
                QUBO[int(parts[0]), int(parts[0])] = value
            elif len(parts) == 2:
                QUBO[int(parts[0]), int(parts[1])] = value / 2
                QUBO[int(parts[1]), int(parts[0])] = value / 2
        return offset, QUBO


    # def from_Q_to_Ising(self, offset, Q):
    #     """Convert the matrix Q of Eq.3 into Eq.13 elements J and h"""
    #     n_qubits = len(Q)  # Get the number of qubits (variables) in the QUBO matrix
    #     # Create default dictionaries to store h and pairwise interactions J
    #     h = defaultdict(int)
    #     J = defaultdict(int)

    #     # Loop over each qubit (variable) in the QUBO matrix
    #     for i in range(n_qubits):
    #         # Update the magnetic field for qubit i based on its diagonal element in Q
    #         h[(i,)] -= Q[i, i] / 2
    #         # Update the offset based on the diagonal element in Q
    #         offset += Q[i, i] / 2
    #         # Loop over other qubits (variables) to calculate pairwise interactions
    #         for j in range(i + 1, n_qubits):
    #             # Update the pairwise interaction strength (J) between qubits i and j
    #             J[(i, j)] += Q[i, j] / 4
    #             # Update the magnetic fields for qubits i and j based on their interactions in Q
    #             h[(i,)] -= Q[i, j] / 4
    #             h[(j,)] -= Q[i, j] / 4
    #             # Update the offset based on the interaction strength between qubits i and j
    #             offset += Q[i, j] / 4
    #     # Return the magnetic fields, pairwise interactions, and the updated offset
    #     return offset, h, J


class JumanjiWrapperMaze(gym.Wrapper):

    def reset(self, **kwargs):
        if hasattr(self, 'constant_seed'): 
            output = self.env.reset(seed=self.seed)
        else:
            output = self.env.reset()
        print(output)
        return output
    
    def step(self, action):
        state, reward, terminate, truncate, info = self.env.step(action)
        if not truncate:
            info = dict()
        self.previous_state = state
        return state, reward, False, truncate, info

    def set_seed_constant(self, seed):
        self.constant_seed = True
        self.seed = seed


class MinMaxNormalizationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MinMaxNormalizationWrapper, self).__init__(env)
        self.low = env.observation_space.low
        self.high = env.observation_space.high

    def observation(self, observation):
        normalized_obs = -np.pi + 2 * np.pi * (observation - self.low) / (self.high - self.low)
        return normalized_obs


class ArctanNormalizationWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)



def convert_QUBO_to_ising(offset, Q):
    """Convert the matrix Q of Eq.3 into Eq.13 elements J and h"""
    n_qubits = len(Q)  # Get the number of qubits (variables) in the QUBO matrix
    # Create default dictionaries to store h and pairwise interactions J
    h = np.zeros(Q.shape[0])
    J = []

    # Loop over each qubit (variable) in the QUBO matrix
    for i in range(n_qubits):
        # Update the magnetic field for qubit i based on its diagonal element in Q
        h[i] -= Q[i, i] / 2
        # Update the offset based on the diagonal element in Q
        offset += Q[i, i] / 2
        # Loop over other qubits (variables) to calculate pairwise interactions
        for j in range(i + 1, n_qubits):
            # Update the pairwise interaction strength (J) between qubits i and j
            J.append([i, j, Q[i, j] / 4])
            # Update the magnetic fields for     qubits i and j based on their interactions in Q
            h[i] -= Q[i, j] / 4
            h[j] -= Q[i, j] / 4
            # Update the offset based on the interaction strength between qubits i and j
            offset += Q[i, j] / 4
    J = np.stack(J)
    J = np.expand_dims(J, axis=0)        # Return the magnetic fields, pairwise interactions, and the updated offset
    h = np.expand_dims(h, axis=0)        # Return the magnetic fields, pairwise interactions, and the updated offset
    return offset, h, J

def create_jumanji_env(env_id, config):
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