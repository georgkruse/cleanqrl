# from jumanji.environments.routing.tsp.generator import Generator, UniformGenerator
# # import jax
# generator_tsp = UniformGenerator(num_cities=5)
# # generator_knapsack = RandomGenerator(total_budget=2, num_items=2)
# import gymnasium as gym
# import jumanji.wrappers
# from jumanji.environments import Knapsack, TSP
# from jumanji.environments.packing.knapsack.generator import RandomGenerator
# import numpy as np
# import itertools


# def compute_optimal_tour(nodes):

#     optimal_tour_length = 1000
#     for tour_permutation  in itertools.permutations(range(1,nodes.shape[0])):
#         tour = [0] + tour_permutation
#         tour_length = compute_tour_length(nodes, tour)
#         if tour_length < optimal_tour_length:
#             optimal_tour_length = tour_length
#     return optimal_tour_length

# def compute_tour_length(nodes, tour):
#     """
#     Compute length of a tour, including return to start node.
#     (If start node is already added as last node in tour, 0 will be added to tour length.)
#     :param nodes: all nodes in the graph in form of (x, y) coordinates
#     :param tour: list of node indices denoting a (potentially partial) tour
#     :return: tour length
#     """
#     tour_length = 0
#     for i in range(len(tour)):
#         if i < len(tour)-1:
#             tour_length += np.linalg.norm(np.asarray(nodes[tour[i]]) - np.asarray(nodes[tour[i+1]]))
#         else:
#             tour_length += np.linalg.norm(np.asarray(nodes[tour[-1]]) - np.asarray(nodes[tour[0]]))

#     return tour_length

# class JumanjiWrapper(gym.Wrapper):
#     def step(self, action):
#         state, reward, terminate, truncate, info = self.env.step(action)
#         num_cities = self.env.action_space.n
#         nodes = np.reshape(state[num_cities:num_cities*3], (-1, 2))
#         optimal_tour_length = compute_optimal_tour(nodes)
#         info['optimal_tour_length'] = optimal_tour_length
#         return state, reward, False, truncate, info

# generator_knapsack = RandomGenerator(total_budget=2, num_items=5)
# # env = Knapsack(generator=generator_knapsack)
# env = TSP(generator=generator_tsp)

# env = jumanji.wrappers.AutoResetWrapper(env)
# env = jumanji.wrappers.JumanjiToGymWrapper(env)
# env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
# env = gym.wrappers.RecordEpisodeStatistics(env)
# env = JumanjiWrapper(env)
# state, _ = env.reset()

# for i in range(10):
#     state, reward, terminate, truncate, info = env.step(i)
#     print(state, reward, terminate, truncate)

# env = jumanji.make('TSP-v1')


# def knapsack(values, weights, total_budget, max_items):
#     """
#     Calculate the optimal value of the knapsack.

#     Args:
#         values (list): List of item values.
#         weights (list): List of item weights.
#         total_budget (int): Maximum weight capacity of the knapsack.
#         max_items (int): Maximum number of items to include.

#     Returns:
#         int: Optimal value of the knapsack.
#     """

#     # Initialize a 3D table to store the maximum value at each subproblem
#     dp = [[[0 for _ in range(total_budget + 1)] for _ in range(max_items + 1)] for _ in range(len(values) + 1)]

#     # Iterate over each item
#     for i in range(1, len(values) + 1):
#         # Iterate over each possible number of items
#         for j in range(1, min(i, max_items) + 1):
#             # Iterate over each possible weight
#             for w in range(1, total_budget + 1):
#                 # If the current item's weight exceeds the current weight limit, skip it
#                 if weights[i - 1] > w:
#                     dp[i][j][w] = dp[i - 1][j][w]
#                 # Otherwise, choose the maximum value between including and excluding the current item
#                 else:
#                     dp[i][j][w] = max(dp[i - 1][j][w], dp[i - 1][j - 1][w - weights[i - 1]] + values[i - 1])

#     # Return the maximum value in the table
#     return dp[-1][-1][-1]

# # Example usage:
# values = [60, 100, 120, 240]
# weights = [10, 20, 30, 30]
# total_budget = 50
# max_items = 2

# optimal_value = knapsack(values, weights, total_budget, max_items)
# print(f"Optimal knapsack value: {optimal_value}")


import jax
import jumanji.wrappers
from jumanji.environments import Knapsack
from jumanji.environments.packing.knapsack.generator import RandomGenerator

key = jax.random.PRNGKey(0)
generator_knapsack = RandomGenerator(total_budget=10, num_items=5)
env = Knapsack(generator=generator_knapsack)
env = jumanji.wrappers.JumanjiToGymWrapper(env)
state, _ = env.reset(seed=0)

for i in range(5):
    state, _ = env.reset(seed=1)
    print(state)
    # state, reward, terminate, truncate, info = env.step(i)
    # print(reward, terminate, truncate, info)

print("done")
