import random
import jumanji
import jax

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import pickle
from itertools import combinations
import copy

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

class TSPEnv(gym.Env):
    def __init__(self, config):
        
        self.config = config

        if 'quantum' in config['agent']:
            self.num_cities = config["num_qubits"]
            print('If a quantum agent is used, the number of cities is determined by the number of qubits')
        else:
            if 'num_cities' in config.keys():
                self.num_cities = config['num_cities']
            else:
                self.num_cities = 10
                print('No number of cities specified in config, defaulting to "10".')

        if 'action_space_type' in config.keys():
            self.action_space_type = config['action_space_type']
        else:
            self.action_space_type = 'nodes'
            print('No action space specified in config, defaulting to "nodes".')

        if self.action_space_type == 'nodes':
            self.action_space = Discrete(self.num_cities)
        elif self.action_space_type == 'edges':
            self.action_space = Discrete(self.num_cities-1)
       
        self.fully_connected_qubits = list(combinations(list(range(self.num_cities)), 2))
        self.state = OrderedDict()

        self.annotations = {}
        for city in range(self.num_cities):
            self.annotations[str(city)] = np.pi

        self.env = jumanji.make('TSP-v1')
        state_space = {}
        state_space['linear_terms'] = Box(-np.inf, np.inf, shape = (self.num_cities,2), dtype='float64')
        state_space['quadratic_terms'] = Box(-np.inf, np.inf, shape = (len(self.fully_connected_qubits),3), dtype='float64')       
        state_space['annotations'] = Box(-np.inf, np.inf, shape = (self.num_cities,2), dtype='float64')
        state_space['current_city'] = Box(-np.inf, np.inf, shape = (1,), dtype='float64')

        self.observation_space = Dict(spaces = state_space)

            
    @staticmethod
    def cost(nodes, tour):
        return -compute_tour_length(nodes, tour)

    def compute_reward(self, nodes, old_state, state):
        return self.cost(nodes, state) - self.cost(nodes, old_state)
    
    def reset(self, seed = None, options = None,use_specific_timestep = False, timestep = 0):
        
        key = jax.random.PRNGKey(0)
        state, timestep = self.env.reset(key)
        self.tsp_graph_nodes = state.coordinates 
        self.optimal_tour_length = 1 #compute_tour_length(self.tsp_graph_nodes, [int(x - 1) for x in self.data_y[instance_number][:-1]])


        self.fully_connected_edges = []
        self.edge_weights = {}
        for edge in self.fully_connected_qubits:
            self.fully_connected_edges.append((self.tsp_graph_nodes[edge[0]], self.tsp_graph_nodes[edge[1]]))
            edge_distance = np.linalg.norm(
                np.asarray(self.tsp_graph_nodes[edge[0]]) - np.asarray(self.tsp_graph_nodes[edge[1]]))
            self.edge_weights[edge] = edge_distance

        self.tour = [0]  # w.l.o.g. we always start at city 0
        self.tour_edges = []
        self.step_rewards = []
        self.available_nodes = list(range(1, self.num_cities))
        self.annotations = np.zeros(self.num_cities)

        self.prev_tour = copy.deepcopy(self.tour)

        state = OrderedDict()

        linear = {}

        for node in range(self.num_cities):
            linear[str(node)] = 0

        self.partial_tour = np.stack([[int(self.fully_connected_qubits[idx][0]),
                                        int(self.fully_connected_qubits[idx][1]),
                                        0] for idx in range(len(self.fully_connected_qubits))])
        
        state['linear_terms'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        state['quadratic_terms'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in self.edge_weights.items()])


        self.current_node = np.array([0])

        state['current_node'] = self.current_node

        state['annotations'] = self.annotations

        return deepcopy(state), {}
    
    def sample_valid_action(self):
        if self.action_space_type == 'nodes':
            return np.random.choice(self.available_nodes)
        elif self.action_space_type == 'edges':
            return np.random.choice(self.available_edges)

    def step(self, action):        
        
        if self.action_space_type == 'edges':
            potential_edges = []
            for (node1, node2) in self.fully_connected_qubits:
                if self.current_node == node1:
                    potential_edges.append([node1, node2])
                elif self.current_node == node2:
                    potential_edges.append([node1, node2])
            (node1, node2) = potential_edges[action]
            if self.current_node == node1:
                next_node = node2
            elif self.current_node == node2:
                next_node = node1

        elif self.action_space_type == 'nodes':
            next_node = action

        self.tour.append(next_node)
        self.tour_edges.append((deepcopy(self.tour[-1]), next_node))
        remove_node_ix = self.available_nodes.index(next_node)
        self.annotations[next_node] = np.pi
        self.current_node = np.array([self.tour[-1]])

        del self.available_nodes[remove_node_ix]

        if len(self.tour) > 1:
            reward = self.compute_reward(self.tsp_graph_nodes, self.prev_tour, self.tour)
            self.step_rewards.append(reward)
            terminations = False if len(self.available_nodes) > 1 else True
            

        if len(self.available_nodes) == 1:
            self.tour.append(self.available_nodes[0])
            self.tour.append(self.tour[0])
            reward = self.compute_reward(self.tsp_graph_nodes, self.prev_tour, self.tour)
            self.step_rewards.append(reward)
        
        if terminations:
            tour_length = compute_tour_length(self.tsp_graph_nodes, self.tour)
            ratio = tour_length/self.optimal_tour_length
            self.ratio = ratio
            info = {} #{'episode': {'r': tour_length, 'ratio': ratio}}
            truncations = True
        else: 
            info = {}
            truncations = False

        self.prev_tour = copy.deepcopy(self.tour)
        
        next_state = OrderedDict()

        linear = {}

        for node in range(self.num_cities):
            linear[str(node)] = 0
        
        next_state['linear_terms'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        next_state['quadratic_terms'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in self.edge_weights.items()])

        next_state['current_node'] = self.current_node
        next_state['annotations'] = self.annotations

        return deepcopy(next_state), reward, terminations, truncations, info