import os
import sys
import yaml

# This is important for the import of the cleanqrl package. Do not delete this line
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleanqrl'))

from cleanqrl_utils.train_utils import train_agent
from cleanqrl_utils.config_utils import generate_config
    

if __name__ == "__main__":
    # Specify the path to the config file
    # Get all config files in the configs folder
    config_files = [f for f in os.listdir('configs') if f.endswith('.yaml')]

    for config_file in config_files:
        config_path = os.path.join('configs', config_file)
        # Load the config file 
        with open(config_path) as f:
            config = yaml.load(f, Loader = yaml.FullLoader)
        config = generate_config(config['algorithm_config'])
        config['total_timesteps'] = 100
        config['path'] = os.path.join(os.getcwd(), 'tests')    
        os.makedirs(os.path.dirname(config['path'] + '/'), exist_ok=True)
        train_agent(config)