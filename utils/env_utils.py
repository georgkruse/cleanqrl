import numpy as np
import gymnasium as gym
from envs.bandit import MultiArmedBanditEnv
from envs.maze import MazeEnv
from envs.tsp import TSPEnv


class MinMaxNormalizationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MinMaxNormalizationWrapper, self).__init__(env)
        self.low = env.observation_space.low
        self.high = env.observation_space.high

    def observation(self, observation):
        normalized_obs = -np.pi + 2 * np.pi * (observation - self.low) / (self.high - self.low)
        return normalized_obs


class ArctanNormalizationWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)


def make_env(env_id, config=None):
    
    def thunk():
        custom_envs = {
            "bandit": MultiArmedBanditEnv,  # Add your custom environments here
            "maze": MazeEnv,
            'TSP-v1': TSPEnv,
        }

        if env_id in custom_envs:
            env = custom_envs[env_id](config)
        else:
            try:
                env = gym.make(env_id)
                # Add here some if condition for a wrapper, specified in the config?
                if 'env_wrapper' in config.keys():
                    if config['env_wrapper'] == 'min_max':
                        env = MinMaxNormalizationWrapper(env)
                    elif config['env_wrapper'] == 'arctan':
                        env = ArctanNormalizationWrapper(env)

            except gym.error.Error:
                raise ValueError(f"Environment ID {env_id} is not valid or not supported"
                                 "by Gym or custom environments.")
            
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
