from agents.qpg_agent import QPG
from agents.qdqn_agent import QDQN
from agents.fe_agent import QBM
from agents.aa_agent import Grover_agent

qrl_switch = {
              'QPG':    QPG, 
              'QDQN+':  QDQN, 
              'QBM':    QBM, 
              'Grover': Grover_agent
              }

def create_algorithm(config):
           
    algorithm = qrl_switch[config.alg] 
        
    return algorithm