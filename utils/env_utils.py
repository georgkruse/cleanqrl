import numpy as np
import gymnasium as gym
from envs.bandit import MultiArmedBanditEnv

class ContinuousEncoding(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)


def make_env(env_id, config=None):
    def thunk():
        custom_envs = {
            "bandit": MultiArmedBanditEnv,  # Add your custom environments here
            # "maze": MazeEnv,
        }

        if env_id in custom_envs:
            env = custom_envs[env_id](config)
        else:
            try:
                env = gym.make(env_id)
                # Add here some if condition for a wrapper, specified in the config?
            except gym.error.Error:
                raise ValueError(f"Environment ID {env_id} is not valid or not supported"
                                 "by Gym or custom environments.")
            
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
