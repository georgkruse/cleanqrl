from cleanqrl.ppo_classical import ppo_classical
from cleanqrl.ppo_classical_continuous import ppo_classical_continuous
from cleanqrl.ppo_quantum import ppo_quantum
from cleanqrl.dqn_classical import dqn_classical
from cleanqrl.dqn_quantum import dqn_quantum
from cleanqrl.reinforce_classical import reinforce_classical
from cleanqrl.reinforce_classical_continuous import reinforce_classical_continuous
from cleanqrl.reinforce_quantum import reinforce_quantum

from cleanqrl.dqn_quantum_jumanji import dqn_quantum_hamiltonian

agent_switch = {
    "PPO_classical": ppo_classical, 
    "PPO_classical_continuous": ppo_classical_continuous,
    "PPO_quantum": ppo_quantum,
    "DQN_classical": dqn_classical,
    "DQN_quantum": dqn_quantum,
    "REINFORCE_classical": reinforce_classical,
    "REINFORCE_classical_continuous": reinforce_classical_continuous,
    "REINFORCE_quantum": reinforce_quantum,
    'DQN_quantum_hamiltonian': dqn_quantum_hamiltonian
}

def train_agent(config):
    agent_switch[config["agent"]](config)
