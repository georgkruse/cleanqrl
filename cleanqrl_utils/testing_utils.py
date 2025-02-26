import os
import sys
import yaml
import datetime 

# This is important for the import of the cleanqrl package. Do not delete this line
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, 'cleanqrl'))

# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleanqrl'))

from cleanqrl_utils.train_utils import train_agent
from cleanqrl_utils.config_utils import generate_config
    

if __name__ == "__main__":
    # Specify the path to the config file
    # Get all config files in the configs folder
    # config_files = [f for f in os.listdir('configs') if f.endswith('.yaml')]
    config_files = ['dqn_classical.yaml', 
                    'reinforce_classical.yaml', 
                    'reinforce_classical_continuous.yaml',
                    'ppo_classical.yaml', 
                    'ppo_classical_continuous.yaml']

    for config_file in config_files:
        config_path = os.path.join(repo_path, 'configs', config_file)
        # Load the config file 
        with open(config_path) as f:
            config = yaml.load(f, Loader = yaml.FullLoader)
        parameter_config = generate_config(config['algorithm_config'])
        parameter_config['total_timesteps'] = 10000
        parameter_config['trial_name'] = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config['tune_config']["trial_name"]
        parameter_config['path'] = os.path.join(os.getcwd(), 'tests', parameter_config['trial_name'])

        os.makedirs(os.path.dirname(parameter_config['path'] + '/'), exist_ok=True)
        print('################################')
        print('Start test config: ', config_file)
        train_agent(parameter_config)
    
    print('Tested all config files successfully!')