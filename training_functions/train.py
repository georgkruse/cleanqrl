from training_functions.ppo import PPO
from training_functions.dqn import DQN
from training_functions.reinforce import reinforce


def train(config):
    if config["algo"] == "DPPO":   #Discrete PPO
        PPO(config)
    elif config["algo"] == "DQN":
        DQN(config)
    elif config["algo"] == "REINFORCE":
        reinforce(config)