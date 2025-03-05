import itertools
from copy import deepcopy

import gymnasium as gym
import jumanji
import jumanji.wrappers
import numpy as np
from jumanji.environments import TSP, Knapsack, Maze
from jumanji.environments.packing.knapsack.generator import (
    RandomGenerator as RandomGeneratorKnapsack,
)
from jumanji.environments.routing.maze.generator import (
    RandomGenerator as RandomGeneratorMaze,
)
from jumanji.environments.routing.tsp.generator import UniformGenerator


class JumanjiWrapperTSP(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episodes = 0

    def step(self, action):
        state, reward, terminate, truncate, info = self.env.step(action)
        num_cities = self.env.unwrapped.num_cities
        nodes = np.reshape(state[num_cities : num_cities * 3], (-1, 2))

        if truncate:
            if self.episodes % 100 == 0:
                num_cities = self.env.unwrapped.num_cities
                nodes = np.reshape(
                    self.previous_state[num_cities : num_cities * 3], (-1, 2)
                )
                optimal_tour_length = self.tsp_compute_optimal_tour(nodes)
                info["optimal_tour_length"] = optimal_tour_length
                info["approximation_ratio"] = info["episode"]["r"] / optimal_tour_length
                if info["episode"]["l"] < num_cities:
                    info["approximation_ratio"] -= 10
                # else:
                #     if info['approximation_ratio'] > -1.1:
                #         print(info['approximation_ratio'])
            self.episodes += 1
        else:
            info = dict()
        self.previous_state = state
        return state, reward, False, truncate, info

    def tsp_compute_optimal_tour(self, nodes):

        optimal_tour_length = 1000
        for tour_permutation in itertools.permutations(range(1, nodes.shape[0])):
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
            if i < len(tour) - 1:
                tour_length += np.linalg.norm(
                    np.asarray(nodes[tour[i]]) - np.asarray(nodes[tour[i + 1]])
                )
            else:
                tour_length += np.linalg.norm(
                    np.asarray(nodes[tour[-1]]) - np.asarray(nodes[tour[0]])
                )

        return tour_length


class JumanjiWrapperKnapsack(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episodes = 0

    def step(self, action):
        state, reward, terminate, truncate, info = self.env.step(action)
        if truncate:
            if self.episodes % 100 == 0:
                num_items = self.env.unwrapped.num_items
                total_budget = self.env.unwrapped.total_budget
                values = self.previous_state[num_items * 2 : num_items * 3]
                weights = self.previous_state[-num_items:]
                optimal_value = self.knapsack_optimal_value(weights, values, total_budget)
                info["optimal_value"] = optimal_value
                info["approximation_ratio"] = info["episode"]["r"] / optimal_value
            # if info['approximation_ratio'] > 0.9:
            #     print(info['approximation_ratio'])
            self.episodes += 1
        else:
            info = dict()
        self.previous_state = state

        return state, reward, False, truncate, info

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
        if not np.all((0 <= weights) & (weights <= 1)) or not np.all(
            (0 <= values) & (values <= 1)
        ):
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


class JumanjiWrapperMaze(gym.Wrapper):
    def reset(self, **kwargs):
        if self.constant_maze:
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


class MinMaxNormalizationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MinMaxNormalizationWrapper, self).__init__(env)
        self.low = env.observation_space.low
        self.high = env.observation_space.high

    def observation(self, observation):
        normalized_obs = -np.pi + 2 * np.pi * (observation - self.low) / (
            self.high - self.low
        )
        return normalized_obs


class ArctanNormalizationWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)


class ReplayBufferWrapper(gym.Wrapper):
    def step(self, action):
        state, reward, terminate, truncate, info = self.env.step(action)
        if truncate or terminate:
            info["final_observation"] = deepcopy(self.previous_state)
        self.previous_state = state

        return state, reward, terminate, truncate, info


def create_jumanji_env(env_id, config):
    if env_id == "TSP-v1":
        num_cities = config.get("num_cities", 5)
        generator_tsp = UniformGenerator(num_cities=num_cities)
        env = TSP(generator_tsp)
    elif env_id == "Knapsack-v1":
        num_items = config.get("num_items", 5)
        total_budget = config.get("total_budget", 2)
        generator_knapsack = RandomGeneratorKnapsack(
            num_items=num_items, total_budget=total_budget
        )
        env = Knapsack(generator=generator_knapsack)

    elif env_id == "Maze-v0":
        num_rows = config.get("num_rows", 4)
        num_cols = config.get("num_cols", 4)
        constant_maze = config.get("constant_maze", False)
        generator_maze = RandomGeneratorMaze(num_cols=num_cols, num_rows=num_rows)
        env = Maze(generator=generator_maze)

    env = jumanji.wrappers.JumanjiToGymWrapper(env)
    env = gym.wrappers.FlattenObservation(
        env
    )  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if env_id == "TSP-v1":
        env = JumanjiWrapperTSP(env)
    elif env_id == "Knapsack-v1":
        env = JumanjiWrapperKnapsack(env)
    elif env_id == "Maze-v0":
        env = JumanjiWrapperMaze(env)
        env.constant_maze = constant_maze
        env.seed = config.get("seed", 42)
    return env
