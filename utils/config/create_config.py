"""
@author: Georg Kruse
georg.kruse@iisb.fraunhofer.de
Fraunhofer Institute for Integrated Systems and Device Technology IISB
"""

from utils.config.common import add_hyperparameters
from utils.config.create_env import create_env

def create_config(config, env_config):
   
    create_env(config)

    tune_config = {}
    alg_config = add_hyperparameters(config.algorithm_config)
    
    tune_config['alg_config'] = alg_config
    tune_config['env_config'] = env_config
    tune_config['env_config']['env'] = config.env

    return tune_config

