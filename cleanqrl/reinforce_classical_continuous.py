import os
import ray
import json
import wandb
import time
import yaml
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from ray.train._internal.session import get_session

def make_env(env_id, config=None):
    def thunk():

        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env, gamma=config['gamma'])
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


class ReinforceAgentClassical(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, np.prod(envs.single_action_space.shape))
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_action_and_logprob(self, x):
        action_mean = self.actor_mean(x) 
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        action = probs.sample()
        return action, probs.log_prob(action)


def log_metrics(config, metrics, report_path=None):
    if config['wandb']:
        wandb.log(metrics)
    if ray.is_initialized():
        ray.train.report(metrics=metrics)
    else:
        with open(os.path.join(report_path, 'result.json'), "a") as f:
            json.dump(metrics, f)
            f.write("\n")


def reinforce_classical_continuous(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    lr = config["lr"]
    gamma = config["gamma"]

    if not ray.is_initialized():
        report_path = config["path"]
        name = config['trial_name']
        with open(os.path.join(report_path, "result.json"), "w") as f:
            f.write("")
    else:
        session = get_session()
        report_path = session.storage.trial_fs_path 
        name = session.trial_id

    if config['wandb']:
        wandb.init(
            project='cleanqrl',
            sync_tensorboard=True,
            config=config,
            name=name,
            monitor_gym=True,
            save_code=True,
            dir=report_path
        )

    device = torch.device("cuda" if (torch.cuda.is_available() and config["cuda"]) else "cpu")
    assert env_id in gym.envs.registry.keys(), f"{env_id} is not a valid gymnasium environment"

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, config) for _ in range(num_envs)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Here, the classical agent is initialized with a Neural Network
    agent = ReinforceAgentClassical(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    global_step = 0
    start_time = time.time()
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(device)
    global_episodes = 0
    episode_returns = []
    losses = []

    while global_step < total_timesteps:
        log_probs = []
        rewards = []
        done = False
        metrics = {}

        # Episode loop
        while not done:
            action, log_prob = agent.get_action_and_logprob(obs)
            log_probs.append(log_prob)
            obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards.append(reward)
            obs = torch.Tensor(obs).to(device)
            done = np.any(terminations) or np.any(truncations)
        
        global_episodes +=1

        # Not sure about this?
        global_step += len(rewards) * num_envs

        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate policy gradient loss
        loss = torch.cat([-log_prob * Gt for log_prob, Gt in zip(log_probs, discounted_rewards)]).sum()

        # Update the policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # If the episode is finished, report the metrics
        # Here addtional logging can be added
        if "episode" in infos:
            for idx, finished in enumerate(infos["_episode"]):
                if finished:
                    metrics = {}
                    global_episodes +=1
                    episode_returns.append(infos['episode']['r'].tolist()[idx])
                    metrics['episode_reward'] = infos['episode']['r'].tolist()[idx]
                    metrics['episode_length'] = infos['episode']['l'].tolist()[idx]
                    metrics['global_step'] = global_step
                    metrics["policy_loss"] = loss.item()
                    metrics["SPS"] = int(global_step / (time.time() - start_time))
                    log_metrics(config, metrics, report_path)
                    
            if global_episodes % 10 == 0 and not ray.is_initialized():
                print('Global step: ', global_step, ' Mean return: ', np.mean(episode_returns[-1:]))
                       
    if config['save_model']:
        model_path = f"{os.path.join(report_path, name)}.cleanqrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
    envs.close()
    if config['wandb']:
        wandb.finish()


if __name__ == '__main__':

    @dataclass
    class Config:
        # General parameters
        trial_name: str = 'reinforce_classical_continuous'  # Name of the trial
        trial_path: str = 'logs'  # Path to save logs relative to the parent directory
        wandb: bool = True # Use wandb to log experiment data 

        # Algorithm parameters
        env_id: str = "Pendulum-v1"  # Environment ID
        num_envs: int = 2  # Number of environments
        total_timesteps: int = 200000  # Total number of timesteps
        gamma: float = 0.9  # discount factor
        lr: float = 0.001  # Learning rate for network weights
        cuda: bool = False  # Whether to use CUDA
        save_model: bool = True # Save the model after the run
        
    config = vars(Config())
    
    # Based on the current time, create a unique name for the experiment
    config['trial_name'] = datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config['trial_name']
    config['path'] = os.path.join(os.path.dirname(os.getcwd()), config['trial_path'], config['trial_name'])

    # Create the directory and save a copy of the config file so that the experiment can be replicated
    os.makedirs(os.path.dirname(config['path'] + '/'), exist_ok=True)
    config_path = os.path.join(config['path'], 'config.yml')
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    # Start the agent training 
    reinforce_classical_continuous(config)    