import os
import sys
import yaml
import datetime
import shutil

# This is important for the import of the cleanqrl package. Do not delete this line
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleanqrl'))

from cleanqrl_utils.config_utils import generate_config
from cleanqrl_utils.train_utils import train_agent
from cleanqrl_utils.plotting_utils import plot_single_run


if __name__ == "__main__":
    # Specify the path to the config file
    config_path = 'configs/reinforce_classical_continuous.yaml'

    # Load the config file 
    with open(config_path) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    # Generate the parameter space for the experiment from the config file
    parameter_config = generate_config(config['algorithm_config'])
    tune_config = config['tune_config']
    
    # Based on the current time, create a unique name for the experiment
    parameter_config['trial_name'] = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + tune_config["trial_name"]
    parameter_config['path'] = os.path.join(os.path.dirname(os.getcwd()), tune_config["trial_path"], parameter_config['trial_name'])

    # Create the directory and save a copy of the config file so 
    # that the experiment can be replicated
    os.makedirs(os.path.dirname(parameter_config['path'] + '/'), exist_ok=True)
    shutil.copy(config_path, os.path.join(parameter_config['path'], 'config.yaml'))

    # Start the agent training 
    train_agent(parameter_config)
    # Plot the results of the training
    # plot_single_run(config['path'])





