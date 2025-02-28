import numpy as np
import gymnasium as gym
from environments.bandit import MultiArmedBanditEnv
from environments.maze import MazeEnv
from environments.tsp import TSPEnv





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
