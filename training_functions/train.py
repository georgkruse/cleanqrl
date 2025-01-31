from training_functions.ppo_classical import ppo_classical
from training_functions.ppo_quantum import ppo_quantum
from training_functions.dqn_classical import dqn_classical
from training_functions.dqn_quantum import dqn_quantum
from training_functions.reinforce_classical import reinforce_classical
from training_functions.reinforce_quantum import reinforce_quantum

agent_switch = {
    "DPPO": {"classical": ppo_classical, "quantum": ppo_quantum}, # discrete PPO
    "DQN": {"classical": dqn_classical, "quantum": dqn_quantum},
    "REINFORCE": {"classical": reinforce_classical, "quantum": reinforce_quantum}
}

def train_agent(config):
    agent_switch[config["agent"]][config["agent_type"]](config)
