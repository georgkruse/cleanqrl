import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

class LunarLander_Wrapper(gym.Env):
    def __init__(self, env_config):
        if 'render_mode' in env_config.keys():
            self.env = gym.make('LunarLander-v2', continuous=True, render_mode = env_config['render_mode'])
        else:
            self.env = gym.make('LunarLander-v2', continuous=True)
        self.action_space = self.env.action_space
        self.observation_space =  Box(-np.inf, np.inf, shape = (8,), dtype='float64')
        self.env_config = env_config
        # self.observation_space =  Box(-np.inf, np.inf, shape = (8,), dtype='float32')
        # obs high:  [1.5 1.5 5. 5. 3.14 5. 1. 1. ]
        # obs low:  [-1.5 -1.5 -5. -5. -3.14 -5. -0. -0. ]
        # [-90. -90. -5. -5. -3.1415927 -5. -0. -0. ], [90. 90. 5. 5. 3.1415927 5. 1. 1. ]
        # self.norm = np.array([(1/1.5)*(np.pi/2), (1/1.5)*(np.pi/2), (1/5)*(np.pi/2),
        #                       (1/5)*(np.pi/2), (1/3.14)*(np.pi/2), (1/5)*(np.pi/2),
        #                       np.pi, np.pi])
        self.norm = np.array([(1/90)*(np.pi/2), (1/90)*(np.pi/2), (1/5)*(np.pi/2),
                              (1/5)*(np.pi/2), (1/3.14)*(np.pi/2), (1/5)*(np.pi/2),
                              np.pi, np.pi])
    def reset(self, seed=None, options=None):
        self.counter = 0

        obs = self.env.reset()[0]
        if self.env_config['norm']:
            obs *= self.norm
        return obs, {}
    
    def step(self, action):
        self.counter += 1
        done = False
        next_state, reward, done1, done2, info = self.env.step(action)
        # if next_state[0] >= 1.5 or next_state[1] >= 1.5:
        #     print(next_state, 'sadkjfahsfkjhaskfjhaskfjshfkjashfkajslfhsdkjl')
        if done1 or done2:
            done = True
        if self.env_config['norm']:
            next_state *= self.norm
        # print(next_state)

        # if self.env_config['mode'] == 'high_reward':
        #     reward = (reward / 100) + 1
        # if reward >= 99:
        #     reward = 0
        # else:
        #     reward = (reward - 10)*0.1
        # <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>

        return next_state, reward, done, False, info

    def render(self):
        return self.env.render()
    
class LunarLander_Wrapper_discrete(gym.Env):
    def __init__(self, env_config):
        self.env_config = env_config
        self.env = gym.make('LunarLander-v2', continuous=True)
        self.action_space = MultiDiscrete([16, 16])
        
        self.observation_space =  Box(-6.16, 6.16, shape = (8,), dtype='float64')
        
        # self.observation_space =  Box(-np.inf, np.inf, shape = (8,), dtype='float32')
        # obs high:  [1.5 1.5 5. 5. 3.14 5. 1. 1. ]
        # obs low:  [-1.5 -1.5 -5. -5. -3.14 -5. -0. -0. ]
        
        self.norm = np.array([(1/1.5)*(np.pi/2), (1/1.5)*(np.pi/2), (1/5)*(np.pi/2),
                              (1/5)*(np.pi/2), (1/3.14)*(np.pi/2), (1/5)*(np.pi/2),
                              np.pi, np.pi])
    def reset(self):
        self.counter = 0
        obs = self.env.reset()[0]
        obs *= self.norm

        return obs
    def step(self, action):

        action_space_car = np.linspace(-1.,  1., 16)
        a1 = action_space_car[int(action[0])]
        a2 = action_space_car[int(action[1])]

        self.counter += 1
        done = False
        next_state, reward, done1, done2, info = self.env.step(np.array([a1, a2]))
        if done1 or done2:
            done = True
        next_state *= self.norm
        if self.env_config['mode'] == 'high_reward':
            reward = (reward / 100) + 1
        # if reward >= 99:
        #     reward = 0
        # else:
        #     reward = (reward - 10)*0.1
        return next_state, reward, done, info

    def render(self):
        return self.env.render()

