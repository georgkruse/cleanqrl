"""
@author: Georg Kruse
georg.kruse@iisb.fraunhofer.de
Fraunhofer Institute for Integrated Systems and Device Technology IISB
"""

from utils.config.common import add_hyperparameters
from utils.config.create_env import create_env

def create_config(config, env_config):

    # keys to lower-case for case insensitivity and improved fault tolerance
    config_switch = {k.lower(): v for k, v in config_switch.items()}

    if config.alg.lower() not in config_switch:
        err_msg = "There does not exist any default configuration for provided ray algorithm %s" \
                " check whether it is a custom algorithm".format(config.alg)
        raise ValueError(err_msg)
    
    create_env(config)

    tune_config = {}
    alg_config = add_hyperparameters(config.algorithm_config)
    
    tune_config['alg_config'] = alg_config
    tune_config['env_config'] = env_config
    tune_config['env_config']['env'] = config.env

    return tune_config

