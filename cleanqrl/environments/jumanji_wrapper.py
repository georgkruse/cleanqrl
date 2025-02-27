import numpy as np 
import itertools
import jumanji
import jumanji.wrappers
import gymnasium as gym
        
from jumanji.environments import TSP, Knapsack 
from jumanji.environments.routing.tsp.generator import UniformGenerator
from jumanji.environments.packing.knapsack.generator import RandomGenerator


def compute_optimal_tour(nodes):

    optimal_tour_length = 1000
    for tour_permutation  in itertools.permutations(range(1,nodes.shape[0])):
        tour = [0] + list(tour_permutation)
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

class JumanjiWrapperTSP(gym.Wrapper):
    def step(self, action):
        self.env.unwrapped
        state, reward, terminate, truncate, info = self.env.step(action)
        if truncate:
            num_cities = self.env.action_space.n
            nodes = np.reshape(self.previous_state[num_cities:num_cities*3], (-1, 2))
            optimal_tour_length = compute_optimal_tour(nodes)
            info['optimal_tour_length'] = optimal_tour_length
            info['approximation_ratio'] = info['episode']['r']/optimal_tour_length
            if info['episode']['l'] < num_cities:
                info['approximation_ratio'] -= 10
            # else: 
            #     if info['approximation_ratio'] > -1.01:
            #         print(info['approximation_ratio'], 'dfajfdjsafdj')
        else:
            info = dict()
        self.previous_state = state
        return state, reward, False, truncate, info

def create_jumanji_env(env_id, config):
        if env_id == 'TSP-v1':
            num_cities = config.get('num_cities', 5)
            generator_tsp = UniformGenerator(num_cities=num_cities)
            env = TSP(generator_tsp)
        elif env_id == 'Knapsack-v1':
            num_items = config.get('num_items', 5)
            total_budget = config.get('total_budget', 3)
            generator_knapsack = RandomGenerator(num_items=num_items, total_budget=total_budget)
            env = Knapsack(generator=generator_knapsack)

        env = jumanji.wrappers.AutoResetWrapper(env)
        env = jumanji.wrappers.JumanjiToGymWrapper(env)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = JumanjiWrapperTSP(env)

        # # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=config['gamma'])
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env