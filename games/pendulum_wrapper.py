import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

class Pendulum_Wrapper_no_norm(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make('Pendulum-v1')
        self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        self.observation_space = self.env.observation_space # Box(-1.6, 1.6, shape = (3,), dtype='float64')
    def reset(self, seed=None, options=None):
        self.counter = 0
        obs, info = self.env.reset()
        return obs, info
    def step(self, action):
        self.counter += 1
        done = False
        next_state, reward, done1, done2, info = self.env.step(action)
        if done1 or done2:
            done = True
        # <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>
        return next_state, reward, done, False, info

    def render(self):
        return self.env.render()
    

class Pendulum_Wrapper(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make('Pendulum-v1')
        self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        self.observation_space = Box(-1.6, 1.6, shape = (3,), dtype='float64')
        self.norm = np.array([np.pi/2, np.pi/2, np.pi/16])
    def reset(self, seed=None, options=None):
        self.counter = 0
        obs, info = self.env.reset()
        obs *= self.norm
        return obs, info
    def step(self, action):
        self.counter += 1
        done = False
        next_state, reward, done1, done2, info = self.env.step(action)
        if done1 or done2:
            done = True
        next_state *= self.norm
        # <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>
        return next_state, reward, done, False, info

    def render(self):
        return self.env.render()
    

class Pendulum_Wrapper_discrete(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make('Pendulum-v1')
        self.action_space = Discrete(8)
        self.observation_space = Box(-1.6, 1.6, shape = (3,), dtype='float64')
        self.norm = np.array([np.pi/2, np.pi/2, np.pi/16])
    
    def reset(self, seed=None, options=None):
        self.counter = 0
        obs, info = self.env.reset()
        obs *= self.norm
        return obs, info
    
    def step(self, action):
        # action_space_car = np.linspace(-2.,  2., 8)
        # action = action_space_car[int(action)]

        self.counter += 1
        done = False
        next_state, reward, done1, done2, info = self.env.step(np.array([action]))
        if done1 or done2:
            done = True
        next_state *= self.norm
        return next_state, reward, done, False, info

    def render(self):
        return self.env.render()