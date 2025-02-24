# This file is an adaptation from https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import ray
import json
import yaml
import random
import time
import gymnasium as gym
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from replay_buffer import ReplayBuffer

from environments.bandit import MultiArmedBanditEnv
from environments.maze import MazeEnv
from environments.tsp import TSPEnv
from environments.wrapper import *


def make_env(env_id, config=None):
    def thunk():
        custom_envs = {
            "bandit": MultiArmedBanditEnv,  # Add your custom environments here
            "maze": MazeEnv,
            'TSP-v1': TSPEnv,
        }

        env = custom_envs[env_id](config)
        env = MinMaxNormalizationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def dqn_classical_jumanji(config: dict):
    cuda = config["cuda"]
    env_id = config["env_id"]
    num_envs = config["num_envs"]
    lr = config["lr"]
    buffer_size = config["buffer_size"]
    total_timesteps = config["total_timesteps"]
    start_e = config["start_e"]
    end_e = config["end_e"]
    exploration_fraction = config["exploration_fraction"]
    learning_starts = config["learning_starts"]
    train_frequency = config["train_frequency"]
    batch_size = config["batch_size"]
    gamma = config["gamma"]
    target_network_frequency = config["target_network_frequency"]
    tau = config["tau"]

    if not ray.is_initialized():
        report_path = os.path.join(config["path"], "result.json")
        with open(report_path, "w") as f:
            f.write("")

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id,) for i in range(num_envs)]
    )

    assert num_envs == 1, "environment vectorization not possible in DQN"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    target_network = QNetwork(envs).to(device)    

    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    global_episodes = 0
    episode_returns = []
    losses = []

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # Dont know about this as well?
        # for idx, trunc in enumerate(truncations):
            # if trunc:
            #     real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
          
        metrics = {}
        # ALGO LOGIC: training.
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                data = rb.sample(batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                steps_per_second = int(global_step / (time.time() - start_time))

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(float(loss.item()))
                metrics["loss"] = float(loss.item())

            # update target network
            if global_step % target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                    )


        if "episode" in infos:
            for idx, finished in enumerate(infos["_episode"]):
                if finished:
                    global_episodes +=1
                    episode_returns.append(float(infos["episode"]["r"][idx]))
                    metrics["episodic_return"] = float(infos["episode"]["r"][idx])
                    metrics["global_step"] = global_step
                    metrics["episode"] = global_episodes             

                    if ray.is_initialized():
                        ray.train.report(metrics=metrics)
                    else:
                        with open(report_path, "a") as f:
                            json.dump(metrics, f)
                            f.write("\n")
        
        if global_episodes >= 1:
            if global_step % 100 == 0:
                print(f"Global step: {global_step}, " 
                      f"Mean Return: {np.round(np.mean(episode_returns[-10:]), 2)}, "
                      f"Mean Loss: {np.round(np.mean(losses[-10:]), 5)}, "
                      f"SPS: {steps_per_second}")    
    envs.close()


if __name__ == '__main__':

    @dataclass
    class Config:
        trial_name: str = 'dqn_classical_jumanji'  # Name of the trial
        trial_path: str = 'logs'  # Path to save logs relative to the parent directory
        env_id: str = "TSP-v1"  # Environment ID
        num_envs: int = 1  # Number of environments
        buffer_size: int = 10000  # Size of the replay buffer
        total_timesteps: int = 100000  # Total number of timesteps
        start_e: float = 1.0  # Starting value of epsilon for exploration
        end_e: float = 0.01  # Ending value of epsilon for exploration
        exploration_fraction: float = 0.1  # Fraction of total timesteps for exploration
        learning_starts: int = 1000  # Timesteps before learning starts
        train_frequency: int = 1  # Frequency of training
        batch_size: int = 32  # Batch size for training
        gamma: float = 0.99  # Discount factor
        target_network_frequency: int = 100  # Frequency of target network updates
        tau: float = 0.01  # Soft update coefficient
        lr: float = 0.01  # Learning rate for network weights
        cuda: bool = False  # Whether to use CUDA


    config = vars(Config())
    
    # Based on the current time, create a unique name for the experiment
    name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config["trial_name"]
    path = os.path.join(os.path.dirname(os.getcwd()), config["trial_path"], name)
    config['path'] = path

    # Create the directory and save a copy of the config file so 
    # that the experiment can be replicated
    os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
    config_path = os.path.join(path, 'config.yml')
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    # Start the agent training 
    dqn_classical_jumanji(config)    