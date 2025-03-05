# from openqaoa.problems import FromDocplex2IsingModel
# from docplex.mp.model import Model
import math
from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from pyqubo import Binary


class KnapsackSequentialDynamic(gym.Env):
    def __init__(self, env_config):
        self.config = env_config
        self.instances_size = self.config["instances_size"]
        self.dataset_size = 100

        self.values = self.config["values"]
        self.weights = self.config["weights"]
        self.maximum_weight = self.config["maximum_weight"]

        self.lambdas = self.config["lambdas"]

        self.timestep = np.random.randint(low=0, high=self.dataset_size)

        self.a = {}
        for item in range(self.instances_size):
            self.a[str(item)] = np.pi

        self.scale_qvalues = {}
        for item in range(self.instances_size):
            self.scale_qvalues[str(item)] = 1.0

        self.mdl = [
            self.KP(self.values[i], self.weights[i], self.maximum_weight[i])
            for i in range(self.dataset_size)
        ]
        self.solution_str = [self.solve_knapsack(mdl) for mdl in self.mdl]

        self.ising_hamiltonian = [
            FromDocplex2IsingModel(
                self.mdl[i],
                unbalanced_const=True,
                strength_ineq=[self.lambdas[0], self.lambdas[1]],
            ).ising_model
            for i in range(self.dataset_size)
        ]

        self.state = OrderedDict()
        # self.mdl = [self.KP(self.values[i],self.weights[i],self.maximum_weight[i]) for i in range(self.dataset_size)]
        # self.solution_str = [self.solve_knapsack(mdl) for mdl in self.mdl]
        self.current_value = 0
        self.current_weight = 0

        num_quadratic_terms = int(
            math.factorial(self.instances_size)
            / (math.factorial(self.instances_size - 2) * math.factorial(2))
        )

        spaces = {}
        spaces["linear_0"] = Box(
            -np.inf, np.inf, shape=(self.instances_size, 2), dtype="float64"
        )
        spaces["quadratic_0"] = Box(
            -np.inf, np.inf, shape=(num_quadratic_terms, 3), dtype="float64"
        )
        spaces["annotations"] = Box(
            -np.inf, np.inf, shape=(self.instances_size, 2), dtype="float64"
        )
        spaces["scale_qvalues"] = Box(
            -np.inf, np.inf, shape=(self.instances_size, 2), dtype="float64"
        )

        self.observation_space = Dict(spaces=spaces)
        self.action_space = Discrete(self.instances_size)

    def KP(self, values: list, weights: list, maximum_weight: int):
        """
        Crete a Docplex model of the Knapsack problem:

        args:
            values - list containing the values of the items
            weights - list containing the weights of the items
            maximum_weight - maximum weight allowed in the "backpack"
        """

        mdl = Model("Knapsack")
        num_items = len(values)
        x = mdl.binary_var_list(range(num_items), name="x")
        cost = -mdl.sum(x[i] * values[i] for i in range(num_items))
        mdl.minimize(cost)
        mdl.add_constraint(
            mdl.sum(x[i] * weights[i] for i in range(num_items)) <= maximum_weight
        )
        return mdl

    def calculate_reward(self, action, timestep):
        reward = sum(
            action[i] * self.values[timestep][i] for i in range(self.instances_size)
        )

        # Check if constraint was broken
        weight = sum(
            action[i] * self.weights[timestep][i] for i in range(self.instances_size)
        )
        if round(weight, 2) > self.maximum_weight[timestep]:
            reward = 0
        return reward

    def solve_knapsack(self, mdl):
        docplex_sol = mdl.solve()
        solution = ""
        for ii in mdl.iter_binary_vars():
            solution += str(int(np.round(docplex_sol.get_value(ii), 1)))
        return solution

    def is_constraint_broken(self, action):
        constraint_is_broken = False
        if self.current_weight > self.maximum_weight:
            constraint_is_broken = True
        return constraint_is_broken

    def reset(self, seed=None, options=None, use_specific_timestep=False, timestep=0):
        if use_specific_timestep:
            self.timestep = timestep
        else:
            self.timestep = np.random.randint(low=0, high=self.dataset_size)

        self.current_weight = 0
        self.current_value = 0

        for item in range(self.instances_size):
            self.a[str(item)] = np.pi
            self.scale_qvalues[str(item)] = self.values[self.timestep][item]

        list_terms_coeffs = list(
            zip(
                self.ising_hamiltonian[self.timestep].terms,
                self.ising_hamiltonian[self.timestep].weights,
            )
        )
        linear_terms = [[i[0], w] for i, w in list_terms_coeffs if len(i) == 1]
        quadratic_terms = [[i, w] for i, w in list_terms_coeffs if len(i) == 2]

        linear = {str(item[0]): item[1] for item in linear_terms}
        quadratic = {
            tuple(str(item[0][i]) for i in range(len(item[0]))): item[1]
            for item in quadratic_terms
        }

        self.state["linear_0"] = np.stack(
            [[int(key[0]), value] for key, value in zip(linear.keys(), linear.values())]
        )
        self.state["quadratic_0"] = np.stack(
            [
                [int(key[0]), int(key[1]), value]
                for key, value in zip(quadratic.keys(), quadratic.values())
            ]
        )
        self.state["annotations"] = np.stack(
            [[int(key), value] for key, value in zip(self.a.keys(), self.a.values())]
        )
        self.state["scale_qvalues"] = np.stack(
            [
                [int(key), value]
                for key, value in zip(
                    self.scale_qvalues.keys(), self.scale_qvalues.values()
                )
            ]
        )

        return self.state, {}

    def step(self, action):
        self.a[str(action)] = 0
        self.state["annotations"] = np.stack(
            [[int(key), value] for key, value in zip(self.a.keys(), self.a.values())]
        )
        self.current_weight += self.weights[self.timestep][action]

        done = False

        if self.current_weight <= self.maximum_weight[self.timestep]:
            reward = 0
            self.current_value += self.values[self.timestep][action]
        else:
            reward = self.current_value
            done = True

        return self.state, reward, done, False, {}


# if __name__ == "__main__":
#    config = {
#        "instances_size":5,
#        "lambdas": [0.96,0.03]
#    }
#
#    env = KnapsackSequentialDynamic(config)
#
#    for i in range(10):
#        state,_ = env.reset()
#        done = False
#        while not done:
#            action = [0,1,1]
#            state, reward, done, _, _ = env.step(action)
#            print(reward)
#            if done == True:
#                break
