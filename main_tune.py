import os
import sys
import ray
from ray import tune
import yaml
import datetime
import shutil
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

# This is important for the import of the cleanqrl package. Do not delete this line

from cleanqrl_utils.config_utils import generate_config
from cleanqrl_utils.train_utils import train_agent
from cleanqrl_utils.plotting_utils import plot_tune_run

if __name__ == "__main__":
    # Specify the path to the config file
    config_path = 'configs/ppo_classical_continuous.yaml'

    # Load the config file 
    with open(config_path) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    # Generate the parameter space for the experiment from the config file
    parameter_config = generate_config(config['algorithm_config'])
    tune_config = config['tune_config']
    
    # Based on the current time, create a unique name for the experiment
    trial_name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + tune_config['trial_name']
    path = os.path.join(os.getcwd(), tune_config['trial_path'], trial_name)
    parameter_config['trial_name'] = trial_name
    parameter_config['path'] = path

    # Create the directory and save a copy of the config file so 
    # that the experiment can be replicated
    os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
    shutil.copy(config_path, path + '/config.yml')

    # Instead of running a single agent as before, we will use ray.tune to run multiple agents
    # in parallel. We will use the same train_agent function as before.
    ray.init(local_mode = tune_config['ray_local_mode'],
             num_cpus = tune_config['num_cpus'],
             num_gpus= tune_config['num_gpus'],
             _temp_dir=os.path.join(os.path.dirname(os.getcwd()), 't'),
             include_dashboard = False)
    
    # We need an addtional function to create subfolders for each hyperparameter configuration
    def trial_name_creator(trial):
        return trial.__str__() + '_' + trial.experiment_tag + ','
    
    # We will use the tune.Tuner class to run multiple agents in parallel
    tuner = tune.Tuner(
            train_agent,
            param_space=parameter_config,
            run_config=tune.RunConfig(storage_path=path), 
            tune_config=tune.TuneConfig(num_samples=tune_config['ray_num_trial_samples'],
                                        trial_dirname_creator=trial_name_creator))

            
    # The fit function will start the hyperparameter search
    tiral = tuner.fit()

    # After the experiment is done, we will plot the results.
    ray.shutdown()
    plot_tune_run(path)




