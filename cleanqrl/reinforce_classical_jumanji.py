import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from ray.train._internal.session import get_session
from torch.distributions.categorical import Categorical
from cleanqrl.wrapper_jumanji import create_jumanji_env


def make_env(env_id, config):
    def thunk():
        env = create_jumanji_env(env_id, config)

        return env

    return thunk


class ReinforceAgentClassical(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, envs.single_action_space.n),
        )

    def get_action_and_logprob(self, x):
        logits = self.network(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action)


def log_metrics(config, metrics, report_path=None):
    if config["wandb"]:
        wandb.log(metrics)
    if ray.is_initialized():
        ray.train.report(metrics=metrics)
    else:
        with open(os.path.join(report_path, "result.json"), "a") as f:
            json.dump(metrics, f)
            f.write("\n")


def reinforce_classical_jumanji(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    lr = config["lr"]
    gamma = config["gamma"]

    if config["seed"] == "None":
        config["seed"] = None

    if not ray.is_initialized():
        report_path = config["path"]
        name = config["trial_name"]
        with open(os.path.join(report_path, "result.json"), "w") as f:
            f.write("")
    else:
        session = get_session()
        report_path = session.storage.trial_fs_path
        name = session.storage.trial_fs_path.split("/")[-1]

    if config["wandb"]:
        wandb.init(
            project=config["project_name"],
            sync_tensorboard=True,
            config=config,
            name=name,
            monitor_gym=True,
            save_code=True,
            dir=report_path,
        )

    if config["seed"] is None:
        seed = np.random.randint(0, 1e9)
    else:
        seed = config["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config["cuda"]) else "cpu"
    )

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, config) for _ in range(num_envs)],
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Here, the classical agent is initialized with a Neural Network
    agent = ReinforceAgentClassical(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    # global parameters to log
    global_step = 0
    global_episodes = 0
    print_interval = 50
    episode_returns = deque(maxlen=print_interval)
    episode_approximation_ratio = deque(maxlen=print_interval)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(device)

    while global_step < total_timesteps:
        log_probs = []
        rewards = []
        done = False

        # Episode loop
        while not done:
            action, log_prob = agent.get_action_and_logprob(obs)
            log_probs.append(log_prob)
            obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            rewards.append(reward)
            obs = torch.Tensor(obs).to(device)
            done = np.any(terminations) or np.any(truncations)

        global_episodes += 1

        # Not sure about this?
        global_step += len(rewards) * num_envs

        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # Normalize rewards
        discounted_rewards = torch.tensor(np.array(discounted_rewards)).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-9
        )

        # Calculate policy gradient loss
        loss = torch.cat(
            [-log_prob * Gt for log_prob, Gt in zip(log_probs, discounted_rewards)]
        ).sum()

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
                    global_episodes += 1
                    episode_returns.append(infos["episode"]["r"].tolist()[idx])
                    metrics["episode_reward"] = infos["episode"]["r"].tolist()[idx]
                    metrics["episode_length"] = infos["episode"]["l"].tolist()[idx]
                    metrics["global_step"] = global_step
                    metrics["policy_loss"] = loss.item()
                    metrics["SPS"] = int(global_step / (time.time() - start_time))
                    if "approximation_ratio" in infos.keys():
                        metrics["approximation_ratio"] = infos["approximation_ratio"][
                            idx
                        ]
                        episode_approximation_ratio.append(
                            metrics["approximation_ratio"]
                        )
                    log_metrics(config, metrics, report_path)

            if global_episodes % print_interval == 0 and not ray.is_initialized():
                logging_info = f"Global step: {global_step}  Mean return: {np.mean(episode_returns)}"
                if len(episode_approximation_ratio) > 0:
                    logging_info += f"  Mean approximation ratio: {np.mean(episode_approximation_ratio)}"
                print(logging_info)

    if config["save_model"]:
        model_path = f"{os.path.join(report_path, name)}.cleanqrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if config["wandb"]:
        wandb.finish()


if __name__ == "__main__":

    @dataclass
    class Config:
        # General parameters
        trial_name: str = "reinforce_classical_jumanji"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = True  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

        # Environment parameters
        env_id: str = "TSP-v1"  # Environment ID
        num_cities: int = 4

        # Algorithm parameters
        num_envs: int = 2  # Number of environments
        seed: int = None  # Seed for reproducibility
        total_timesteps: int = 100000  # Total number of timesteps
        gamma: float = 0.99  # discount factor
        lr: float = 0.01  # Learning rate for network weights
        cuda: bool = False  # Whether to use CUDA
        save_model: bool = True  # Save the model after the run

    config = vars(Config())

    # Based on the current time, create a unique name for the experiment
    config["trial_name"] = (
        datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + "_" + config["trial_name"]
    )
    config["path"] = os.path.join(
        Path(__file__).parent.parent, config["trial_path"], config["trial_name"]
    )

    # Create the directory and save a copy of the config file so that the experiment can be replicated
    os.makedirs(os.path.dirname(config["path"] + "/"), exist_ok=True)
    config_path = os.path.join(config["path"], "config.yml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    # Start the agent training
    reinforce_classical_jumanji(config)
