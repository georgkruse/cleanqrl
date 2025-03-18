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
import pennylane as qml
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from ray.train._internal.session import get_session
from torch.distributions.categorical import Categorical
from wrapper_jumanji import create_jumanji_env


# ENV LOGIC: create your env (with config) here:
def make_env(env_id, config):
    def thunk():
        env = create_jumanji_env(env_id, config)

        return env

    return thunk


# QUANTUM CIRCUIT: define your ansatz here:
def parametrized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions
):

    # This block needs to be adapted depending on the environment.
    # The input vector is of shape [4*num_actions] for the Knapsack:
    # [action mask, selected items, values, weights]

    annotations = x[:, num_qubits : num_qubits * 2]
    values_kp = x[:, num_qubits * 2 : num_qubits * 3]
    weights_kp = x[:, 3*num_qubits:]

    for layer in range(num_layers):
        for block, features in enumerate([annotations, values_kp, weights_kp]):
            for i in range(num_qubits):
                qml.RX(input_scaling[layer, block, i] * features[:, i], wires=[i])

            for i in range(num_qubits):
                qml.RY(weights[layer, block, i], wires=[i])

            for i in range(num_qubits):
                qml.RZ(weights[layer, block, i+num_qubits], wires=[i])
        
            if num_qubits == 2:
                qml.CZ(wires=[0, 1])
            else:
                for i in range(num_qubits):
                    qml.CZ(wires=[i, (i + 1) % num_qubits])

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]


# ALGO LOGIC: initialize your agent here:
class ReinforceAgentQuantum(nn.Module):
    def __init__(self, num_actions, config):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.block_size = 3  # number of subblocks depends on environment

        # input and output scaling are always initialized as ones
        self.input_scaling = nn.Parameter(
            torch.ones(self.num_layers, self.block_size, self.num_qubits), requires_grad=True
        )
        self.output_scaling = nn.Parameter(
            torch.ones(self.num_actions), requires_grad=True
        )
        # trainable weights are initialized randomly between -pi and pi
        self.weights = nn.Parameter(
            torch.rand(self.num_layers, self.block_size, self.num_qubits * 2) * 2 * torch.pi - torch.pi,
            requires_grad=True,
        )

        device = qml.device(config["device"], wires=range(self.num_qubits))
        self.quantum_circuit = qml.QNode(
            parametrized_quantum_circuit,
            device,
            diff_method=config["diff_method"],
            interface="torch",
        )

    def get_action_and_logprob(self, x):
        logits = self.quantum_circuit(
            x,
            self.input_scaling,
            self.weights,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
        )
        logits = torch.stack(logits, dim=1)
        logits = logits * self.output_scaling
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


# MAIN TRAINING FUNCTION
def reinforce_quantum_jumanji(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    gamma = config["gamma"]
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]
    num_qubits = config["num_qubits"]

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

    num_actions = envs.single_action_space.n
    
    assert (
        num_qubits >= num_actions
    ), "Number of qubits must be greater than or equal to the number of actions"

    # Here, the quantum agent is initialized with a parameterized quantum circuit
    agent = ReinforceAgentQuantum(num_actions, config).to(device)
    optimizer = optim.Adam(
        [
            {"params": agent.input_scaling, "lr": lr_input_scaling},
            {"params": agent.output_scaling, "lr": lr_output_scaling},
            {"params": agent.weights, "lr": lr_weights},
        ]
    )

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
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
        #     discounted_rewards.std() + 1e-9
        # )

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
        trial_name: str = "reinforce_quantum_jumanji"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = False  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

        # Environment parameters
        env_id: str = "Knapsack-v1"  # Environment ID
        num_items: int = 4
        total_budget: float = 2

        # Algorithm parameters
        num_envs: int = 1  # Number of environments
        seed: int = None  # Seed for reproducibility
        total_timesteps: int = 100000  # Total number of timesteps
        gamma: float = 0.99  # discount factor
        lr_input_scaling: float = 0.01  # Learning rate for input scaling
        lr_weights: float = 0.01  # Learning rate for variational parameters
        lr_output_scaling: float = 0.01  # Learning rate for output scaling
        cuda: bool = False  # Whether to use CUDA
        num_qubits: int = 4  # Number of qubits
        num_layers: int = 2  # Number of layers in the quantum circuit
        device: str = "lightning.qubit"  # Quantum device
        diff_method: str = "adjoint"  # Differentiation method
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
    reinforce_quantum_jumanji(config)
