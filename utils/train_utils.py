from agents.ppo_classical import ppo_classical
from agents.ppo_quantum import ppo_quantum
from agents.dqn_classical import dqn_classical
from agents.dqn_quantum import dqn_quantum
from agents.reinforce_classical import reinforce_classical
from agents.reinforce_quantum import reinforce_quantum


agent_switch = {
    "PPO_classical": ppo_classical, 
    "PPO_quantum": ppo_quantum,
    "DQN_classical": dqn_classical,
    "DQN_quantum": dqn_quantum,
    "REINFORCE_classical": reinforce_classical,
    "REINFORCE_quantum": reinforce_quantum
}

def train_agent(config):
    agent_switch[config["agent"]](config)
