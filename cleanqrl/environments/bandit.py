import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MultiArmedBanditEnv(gym.Env):
    def __init__(self, config):
        super(MultiArmedBanditEnv, self).__init__()

        if hasattr(config, 'n_arms'):
            self.n_arms = config['n_arms']
        else:
            self.n_arms = 10
        if hasattr(config, 'reward_probabilities'):
            reward_probabilities = config['reward_probabilities']
        else:
            reward_probabilities = None
        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Discrete(1)  # No observation, only action selection
        
        # Define reward probabilities for each arm
        if reward_probabilities is None:
            self.reward_probabilities = np.random.rand(self.n_arms)
        else:
            assert len(reward_probabilities) == self.n_arms
            self.reward_probabilities = reward_probabilities
        
    def reset(self):
        # Return a dummy observation (no real state in bandit problem)
        return 0, {}
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        
        # Reward is 1 with probability equal to the arm's probability, else 0
        reward = 1 if np.random.rand() < self.reward_probabilities[action] else 0
        
        # Bandit problem is stateless, so we always return the same dummy observation
        done = True  # Each step is a complete episode
        info = {}
        
        return 0, reward, done, False, info