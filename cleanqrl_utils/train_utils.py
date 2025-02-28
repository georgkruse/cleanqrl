import os
import sys 

# This is important for the import of the cleanqrl package. Do not delete this line
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, 'cleanqrl'))

from cleanqrl.ppo_classical import ppo_classical
from cleanqrl.ppo_classical_continuous import ppo_classical_continuous
from cleanqrl.ppo_classical_jumanji import ppo_classical_jumanji
from cleanqrl.ppo_quantum import ppo_quantum
from cleanqrl.ppo_quantum_continuous import ppo_quantum_continuous
from cleanqrl.ppo_quantum_jumanji import ppo_quantum_jumanji
from cleanqrl.dqn_classical import dqn_classical
# from cleanqrl.dqn_classical_jumanji import dqn_classical_jumanji
from cleanqrl.dqn_quantum import dqn_quantum
# from cleanqrl.dqn_quantum_jumanji import dqn_quantum_jumanji
from cleanqrl.reinforce_classical import reinforce_classical
from cleanqrl.reinforce_classical_continuous import reinforce_classical_continuous
# from cleanqrl.reinforce_classical_jumanji import reinforce_classical_jumanji
from cleanqrl.reinforce_quantum import reinforce_quantum
from cleanqrl.reinforce_quantum_continuous import reinforce_quantum_continuous
# from cleanqrl.reinforce_quantum_jumanji import reinforce_quantum_jumanji


agent_switch = {
    "ppo_classical": ppo_classical, 
    "ppo_classical_continuous": ppo_classical_continuous,
    "ppo_classical_jumanji": ppo_classical_jumanji,
    "ppo_quantum": ppo_quantum,
    "ppo_quantum_continuous": ppo_quantum_continuous,
    "ppo_quantum_jumanji": ppo_quantum_jumanji,
    "dqn_classical": dqn_classical,
    # "dqn_classical_jumanji": dqn_classical_jumanji,    
    "dqn_quantum": dqn_quantum,
    # 'dqn_quantum_hamiltonian': dqn_quantum_jumanji,
    "reinforce_classical": reinforce_classical,
    "reinforce_classical_continuous": reinforce_classical_continuous,
    # "reinforce_classical_jumanji": reinforce_classical_jumanji,
    "reinforce_quantum": reinforce_quantum,
    "reinforce_quantum_continuous": reinforce_quantum_continuous,
    # "reinforce_quantum_jumanji": reinforce_quantum_jumanji,
}

def train_agent(config):
    agent_switch[config["agent"].lower()](config)
