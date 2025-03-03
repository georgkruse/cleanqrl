import os
import yaml
from datetime import datetime
import shutil
from cleanqrl_utils.train_utils import train_agent
from cleanqrl_utils.plotting_utils import plot_single_run


if __name__ == "__main__":
    # Specify the path to the config file
    config_path = 'configs/ppo_quantum_jumanji.yaml'

    # Load the config file 
    with open(config_path) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
   
    # Based on the current time, create a unique name for the experiment
    config['trial_name'] = datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config['trial_name']
    config['path'] = os.path.join(os.path.dirname(os.getcwd()), config['trial_path'], config['trial_name'])

    # Create the directory and save a copy of the config file so 
    # that the experiment can be replicated
    os.makedirs(os.path.dirname(config['path'] + '/'), exist_ok=True)
    shutil.copy(config_path, os.path.join(config['path'], 'config.yaml'))

    # Start the agent training 
    train_agent(config)
    # Plot the results of the training
    # plot_single_run(config['path'])





