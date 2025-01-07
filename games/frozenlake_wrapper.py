import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box


class FrozenLake_Wrapper(gym.Env):
    def __init__(self, env_config):
        
        self.env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
        self.action_space = self.env.action_space
       
        self.num_bits = (self.env.observation_space.n).bit_length()  # Number of bits needed to represent the maximum index
        self.observation_space = Box(-np.inf, np.inf, shape = (self.num_bits,))
        # self.norm = np.array([np.pi/2, np.pi/2, np.pi/16])

    def reset(self, seed=None, options=None):
        self.counter = 0
        obs, info = self.env.reset()
        binary_rep = format(obs, f'0{self.num_bits}b')
        state = np.array([int(char) for char in binary_rep], dtype=int)
        # obs *= self.norm
        return state, info
    
    def step(self, action):
        self.counter += 1
        done = False
        next_obs, reward, done1, done2, info = self.env.step(action)
        binary_rep = format(next_obs, f'0{self.num_bits}b')
        next_state = np.array([int(char) for char in binary_rep], dtype=int)
        if done1 or done2:
            done = True
        # next_state *= self.norm
        # <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>
        return next_state, reward, done, False, info
    
    def legal_actions(self, obs):
        '''
        Returns an array of playable actions. In the case of the Eggholder game, 
        these are always all actions.

        :return: list of size len(action_space)
        :rtype: list
        '''
        if len(obs.shape) == 1 and not self.state_encoding == 'standard':
            integer_value = int(''.join(map(str, obs.astype(np.int16))), 2)
            obs =  np.zeros(self.obs_size)
            obs[integer_value] = 1
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

        actions.append(4)

        return np.stack(actions)
    def render(self):
        return self.env.render()
    
