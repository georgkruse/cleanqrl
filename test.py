
from jumanji.environments.routing.tsp.generator import Generator, UniformGenerator
# import jax
generator_tsp = UniformGenerator(num_cities=5)
# generator_knapsack = RandomGenerator(total_budget=2, num_items=2)
import gymnasium as gym 
import jumanji.wrappers
from jumanji.environments import Knapsack, TSP
from jumanji.environments.packing.knapsack.generator import RandomGenerator
import numpy as np 
import itertools
def compute_optimal_tour(nodes):

    optimal_tour_length = 1000
    for tour_permutation  in itertools.permutations(range(1,nodes.shape[0])):
        tour = [0] + tour_permutation 
        tour_length = compute_tour_length(nodes, tour)
        if tour_length < optimal_tour_length:
            optimal_tour_length = tour_length
    return optimal_tour_length

def compute_tour_length(nodes, tour):
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

class JumanjiWrapper(gym.Wrapper):
    def step(self, action):
        state, reward, terminate, truncate, info = self.env.step(action)
        num_cities = self.env.action_space.n
        nodes = np.reshape(state[num_cities:num_cities*3], (-1, 2))
        optimal_tour_length = compute_optimal_tour(nodes)
        info['optimal_tour_length'] = optimal_tour_length
        return state, reward, False, truncate, info

generator_knapsack = RandomGenerator(total_budget=2, num_items=5)
# env = Knapsack(generator=generator_knapsack)
env = TSP(generator=generator_tsp)

env = jumanji.wrappers.AutoResetWrapper(env)
env = jumanji.wrappers.JumanjiToGymWrapper(env)
env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
env = gym.wrappers.RecordEpisodeStatistics(env)
env = JumanjiWrapper(env)
state, _ = env.reset()

for i in range(10):
    state, reward, terminate, truncate, info = env.step(i)
    print(state, reward, terminate, truncate)

env = jumanji.make('TSP-v1')

print('done')
