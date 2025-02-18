import os
import yaml

from cleanqrl_utils.train_utils import train_agent


def test_all_configs():


    # Specify the path to the config file
    # Get all config files in the configs folder
    config_files = [f for f in os.listdir('configs') if f.endswith('.yaml')]

    for config_file in config_files:
        config_path = os.path.join('configs', config_file)
    config_path = 'configs/dqn_quantum.yaml'

    # Load the config file 
    with open(config_path) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    train_agent(parameter_config)