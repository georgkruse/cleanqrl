import numpy as np
import matplotlib.pyplot as plt
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import os, datetime
from copy import deepcopy
import gymnasium as gym
from gymnasium.core import ObservationWrapper
from games.maze.mazes import *


maze_types = {'crawford':       create_crawford,
              'mueller_3x3':    create_mueller_3x3,
              'neumann_a':      create_neumann_a,
              'neumann_b':      create_neumann_b,
              'neumann_c':      create_neumann_c,
            #   'neumann_d':      create_neumann_d,
             }

optimal_policy_types = {'crawford':       optimal_policy_crawford,
                        'mueller_3x3':    optimal_policy_mueller_3x3,
                        'neumann_a':      optimal_policy_neumann_a,
                        'neumann_b':      optimal_policy_neumann_b,
                        'neumann_c':      optimal_policy_neumann_c,
                        # 'neumann_d':      create_neumann_d,
                        }
class MazeGame(Env):
    '''
    Base class for EggholderGame
    '''

    def __init__(self, config):
        super(MazeGame, self).__init__()
        self.config = config
        self.game_mode = config['game_mode']                    # basic, frozen_lake
        self.state_encoding = config['state_encoding']          # binary, onehot
        self.legal_actions_type = config['legal_actions_type']  # restricted, all
        self.total_goal_states = 0
        if self.game_mode == 'basic':
            self.action_array = [0, 1, 2, 3]
            self.action_space = Discrete(4)
            self.env_type = config['env_type']
            self.P = config['P']
            self.A = config['A']
            self.W = config['W']
            self.R = config['R']
            self.n = config['n']
            self.default_reward = config['default_reward']
            obs = maze_types[self.env_type](self.P, self.A, self.W, self.R, n=self.n)
            self.optimal_policy_tuple = optimal_policy_types[self.env_type](n=self.n)
            self.maze_size_x = obs.shape[1]
            self.maze_size_y = obs.shape[0]
            self.obs_size = int(self.maze_size_y*self.maze_size_x)

            if self.state_encoding == 'binary':
                num_bits = (self.maze_size_x*self.maze_size_y-1).bit_length() 
                self.observation_space = Box(-np.inf, np.inf, shape = (num_bits,))
            elif self.state_encoding == 'onehot':
                self.observation_space = Box(-np.inf, np.inf, shape = (self.maze_size_x*self.maze_size_y,))            
            elif self.state_encoding == 'integer':
                self.observation_space = Box(-np.inf, np.inf, shape = (self.maze_size_x*self.maze_size_y,))
            
            self.render_interval = 2
            self.render_bool = True
            self.start_mid = True
            self.path = 'logs/ray'
            self.episode = 0

        elif self.game_mode == 'frozen_lake':

            self.map_name = self.config['map_name']
            self.is_slippery = self.config['is_slippery']
            env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=False)

            self.action_space = env.action_space
            self.maze_size_x = int(self.map_name[0])
            self.maze_size_y = int(self.map_name[-1])
            self.obs_size = int(self.maze_size_y*self.maze_size_x)

            if self.state_encoding == 'binary':
                num_bits = int((env.observation_space.n-1)).bit_length()  # Number of bits needed to represent the maximum index
                self.observation_space = Box(-np.inf, np.inf, shape = (num_bits,))
            elif self.state_encoding == 'onehot':
                self.observation_space = Box(-np.inf, np.inf, shape = (self.maze_size_x*self.maze_size_y,)) 
            elif self.state_encoding == 'integer':
                self.observation_space = Box(-np.inf, np.inf, shape = (self.maze_size_x*self.maze_size_y,))
            self.A = 1

            maze_4x4 = [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG"
                ]

            maze_8x8 = [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG",
            ]
            if self.map_name == '4x4':
                self.maze = maze_4x4
            elif self.map_name == '8x8':
                self.maze = maze_8x8
    
        elif self.game_mode == 'cliffwalking':

            self.map_name = self.config['map_name']
            self.is_slippery = self.config['is_slippery']
            env = gym.make('CliffWalking-v0')

            self.action_space = env.action_space
            self.maze_size_x = 12 
            self.maze_size_y = 4
            self.obs_size = int(self.maze_size_y*self.maze_size_x)

            if self.state_encoding == 'binary':
                num_bits = int((env.observation_space.n)).bit_length()  # Number of bits needed to represent the maximum index
                self.observation_space = Box(-np.inf, np.inf, shape = (num_bits,))
            elif self.state_encoding == 'onehot':
                self.observation_space = Box(-np.inf, np.inf, shape = (self.maze_size_x*self.maze_size_y,)) 
            elif self.state_encoding == 'integer':
                self.observation_space = Box(-np.inf, np.inf, shape = (self.maze_size_x*self.maze_size_y,))
            self.A = 1

            self.maze = [
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "SHHHHHHHHHHG",
            ]

    def legal_actions(self, obs):
        '''
        Returns an array of playable actions. In the case of the Eggholder game, 
        these are always all actions.

        :return: list of size len(action_space)
        :rtype: list
        '''
        if self.game_mode == 'basic':
            if self.legal_actions_type == 'restricted': 
                if len(obs.shape) == 1 and self.state_encoding == 'binary':
                    integer_value = int(''.join(map(str, obs.astype(np.int16))), 2)
                    obs =  np.zeros(self.obs_size)
                    obs[integer_value] = self.A
                    obs = np.reshape(obs, (self.maze_size_y,self.maze_size_x))
                elif len(obs.shape) == 1 and self.state_encoding == 'integer':
                    integer_value = int(obs)
                    obs =  np.zeros(self.obs_size)
                    obs[integer_value] = self.A
                    obs = np.reshape(obs, (self.maze_size_y,self.maze_size_x))
                obs = np.reshape(obs, (self.maze_size_y, self.maze_size_x))

                agent_index_y, agent_index_x = np.where(obs==self.A)
                y, x = agent_index_y[0], agent_index_x[0] #np.unravel_index(np.argmax(obs), obs.shape)
                actions = []
                if (y < self.maze_size_y-1) and (obs[y+1,x] !=self.W):
                    actions.append(0)
                if(y > 0) and (obs[y-1,x] !=self.W):
                    actions.append(1)
                if (x < self.maze_size_x -1) and (obs[y,x+1] !=self.W):
                    actions.append(2)
                if (x > 0) and (obs[y,x-1] !=self.W):
                    actions.append(3)

            else:
                actions = self.action_array
        else:
            actions = [0, 1, 2, 3]
        return np.stack(actions)

    

    def reset(self, seed=42, options=None, render=True):
        '''
        Resets the game environment and randomly samples, if specified, new parameters 
        for the new game episode. Also calculates the meshgrid for plotting. Plots at the 
        beginning of each episode the last episode played, if specified. 
        For AlphaZero additional render variable is added, so MCTS is not included in plotting.

        :param render: Plotting argument for AlphaZero MCTS. default==True
        :type render: boolean

        :return: numpy array of state with timestep 0 of shape [4]
        :rtype: numpy array
        '''
        if self.game_mode == 'basic':
            obs = maze_types[self.env_type](self.P, self.A, self.W, self.R, n=self.n)
            self.initial_obs = deepcopy(obs)
            agent_index_y, agent_index_x = np.where(self.initial_obs==self.A)
            y, x = agent_index_y[0], agent_index_x[0] #np.unravel_index(np.argmax(obs), obs.shape)
            self.current_obs = np.zeros(self.initial_obs.shape)
            self.current_obs[y, x] = self.A
        elif self.game_mode == 'frozen_lake':
            self.current_obs = np.zeros((self.maze_size_y, self.maze_size_x))
            self.current_obs[0, 0] = self.A

        elif self.game_mode == 'cliffwalking':
            self.current_obs = np.zeros((self.maze_size_y, self.maze_size_x))
            self.current_obs[3, 0] = self.A

        if self.state_encoding == 'binary':
            index_of_one = np.argmax(np.reshape(self.current_obs, (-1)))  # Returns the index of the first occurrence of 1
            # Determine the number of bits needed to represent the index based on the array size
            num_bits = (len(np.reshape(self.current_obs, (-1)))-1).bit_length()  # Number of bits needed to represent the maximum index
            # Convert the index to binary string representation with leading zeros to fit the bit length
            binary_rep = format(index_of_one, f'0{num_bits}b')
            state = np.array([int(char) for char in binary_rep], dtype=int)
            return  np.reshape(deepcopy(state), -1), {}

        elif self.state_encoding == 'onehot':
            return  np.reshape(deepcopy(self.current_obs), -1), {}
        
        elif self.state_encoding == 'integer':
            return  deepcopy(np.argmax(np.reshape(self.current_obs, (-1)))), {}


    def step(self, action, obs=np.array([None])):
        '''
        Does one step in the game environment. Note that the discrete actions are hard
        coded and must be changed according to the game config file.

        :param action: action taken by the agent. Either one value or array of values.
        :type action: int oder np.array
        :return: deepcopy of new observation, deepcopy of reward, done and empty dict.
        :rtype: np.array, float, boolean, dict
        '''
        
        # obs = deepcopy(obs)
        done = False
        reward = 0
        if not sum(obs == None):
            if self.state_encoding == 'binary':
                integer_value = int(''.join(map(str, obs.astype(np.int16))), 2)
                self.current_obs = np.zeros(self.obs_size)
                self.current_obs[integer_value] = self.A
                self.current_obs = np.reshape(self.current_obs, (self.maze_size_y,self.maze_size_x))
            elif self.state_encoding == 'integer':
                integer_value = int(obs)
                obs =  np.zeros(self.obs_size)
                obs[integer_value] = self.A
                obs = np.reshape(obs, (self.maze_size_y,self.maze_size_x))
            else:
                # print(obs)
                self.current_obs = deepcopy(np.reshape(obs, (self.maze_size_y,self.maze_size_x)))

        agent_index_y, agent_index_x = np.where(self.current_obs==self.A)
        y, x = agent_index_y[0], agent_index_x[0] #np.unravel_index(np.argmax(obs), obs.shape)
        self.current_obs[y, x] = 0
        if self.game_mode == 'basic':

            if action == 0:
                if (y < self.maze_size_y-1) and (self.current_obs[y+1,x] !=self.W):
                    y += 1
            elif action == 1:
                if(y > 0) and (self.current_obs[y-1,x] !=self.W):
                    y -= 1
            elif action == 2:
                if (x < self.maze_size_x -1) and (self.current_obs[y,x+1] !=self.W):
                    x += 1
            elif action == 3:
                if (x > 0) and (self.current_obs[y,x-1] !=self.W):
                    x -= 1
            if self.initial_obs[y, x] == self.P:
                reward = self.P 
            elif self.initial_obs[y, x] == self.R:
                reward = self.R
                done = True
            else:
                reward = self.default_reward
        
        elif self.game_mode == 'frozen_lake' or self.game_mode == 'cliffwalking':
            if action == 0:
                if (y < self.maze_size_y-1):
                    y += 1
            elif action == 1:
                if(y > 0):
                    y -= 1
            elif action == 2:
                if (x < self.maze_size_x -1):
                    x += 1
            elif action == 3:
                if (x > 0):
                    x -= 1
            if self.game_mode == 'frozen_lake':
                if self.maze[y][x] == 'G':
                    reward = 1 
                    done = True
                elif self.maze[y][x] == 'H':
                    reward = 0
                    done = True
                else:
                    reward = 0
            elif self.game_mode == 'cliffwalking':
                if self.maze[y][x] == 'G':
                    reward = 100 
                    self.total_goal_states += 1 
                    done = True
                    print('#################################################################')
                elif self.maze[y][x] == 'H':
                    reward = -100
                    done = True
                else:
                    reward = -1

        self.current_obs[y, x] = self.A

        # print('########################')
        # print(np.reshape(obs, (3,3)))
        # print(action)
        # print(np.reshape(self.current_obs, (3,3)))
        # print(self.current_obs)

        if self.state_encoding == 'binary':
            index_of_one = np.argmax(np.reshape(self.current_obs, (-1)))  # Returns the index of the first occurrence of 1
            # Determine the number of bits needed to represent the index based on the array size
            num_bits = (len(np.reshape(self.current_obs, (-1)))-1).bit_length()  # Number of bits needed to represent the maximum index
            # Convert the index to binary string representation with leading zeros to fit the bit length
            binary_rep = format(index_of_one, f'0{num_bits}b')
            state = np.array([int(char) for char in binary_rep], dtype=int)
            return np.reshape(deepcopy(state), -1), deepcopy(reward), done, False, {}

        elif self.state_encoding == 'onehot':
            return deepcopy(np.reshape(self.current_obs, -1)), deepcopy(reward), done, False, {}
       
        elif self.state_encoding == 'integer':
            return  deepcopy(np.argmax(np.reshape(self.current_obs, (-1)))), deepcopy(reward), done, False, {}