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

# import wandb
import yaml
from maze_game import MazeGame
from ray.train._internal.session import get_session
from torch.distributions.categorical import Categorical


def make_env(env_id, config):
    def thunk():
        config["game_mode"] = "frozen_lake"
        config["state_encoding"] = "binary"
        config["legal_actions_type"] = "restricted"
        config["map_name"] = "4x4"
        config["is_slippery"] = False
        env = MazeGame(config)
        # env = gym.make(env_id, is_slippery=False)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def parameterized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions, observation_size
):

    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    for layer in range(num_layers):
        for i in range(observation_size):
            qml.RX(input_scaling[layer, i] * x[:, i], wires=[i])
        z = 0
        for i in range(num_qubits):
            qml.RZ(weights[layer, z], wires=[i])
            qml.RY(weights[layer, z + 1], wires=[i])
            z += 2

        if num_qubits == 2:
            qml.CZ(wires=[0, 1])
        else:
            for i in range(num_qubits):
                qml.CZ(wires=[i, (i + 1) % num_qubits])

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]


class ReinforceAgentQuantum(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_encoding = config["state_encoding"]
        self.observation_size = 4
        # if self.state_encoding == "onehot":
        #     self.observation_size = envs.single_observation_space.n
        # elif self.state_encoding == "binary":
        #     self.observation_size = len(bin(envs.single_observation_space.n - 1)[2:])

        self.num_actions = 4
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]

        assert (
            self.num_qubits >= self.observation_size
        ), "Number of qubits must be greater than or equal to the observation size"
        assert (
            self.num_qubits >= self.num_actions
        ), "Number of qubits must be greater than or equal to the number of actions"

        # input and output scaling are always initialized as ones
        self.input_scaling = nn.Parameter(
            torch.ones(self.num_layers, self.num_qubits), requires_grad=True
        )
        self.output_scaling = nn.Parameter(
            torch.ones(self.num_actions), requires_grad=True
        )
        # trainable weights are initialized randomly between -pi and pi
        self.weights = nn.Parameter(
            torch.FloatTensor(self.num_layers, self.num_qubits * 2).uniform_(0, 0.1),
            requires_grad=True,
        )

        device = qml.device(config["device"], wires=range(self.num_qubits))
        self.quantum_circuit = qml.QNode(
            parameterized_quantum_circuit,
            device,
            diff_method=config["diff_method"],
            interface="torch",
        )

    def forward(self, state):

        state = state.unsqueeze(0)
        # state = torch.from_numpy(state).float().unsqueeze(0)

        logits = self.quantum_circuit(
            state,
            self.input_scaling,
            self.weights,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size,
        )
        logits = torch.stack(logits, dim=1)
        probs = logits * self.output_scaling
        softmax = nn.Softmax(dim=-1)
        probs = softmax(probs)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    # def forward(self, x):
    #     # x_encoded = self.encode_input(x)
    #     logits = self.quantum_circuit(
    #         x,
    #         self.input_scaling,
    #         self.weights,
    #         self.num_qubits,
    #         self.num_layers,
    #         self.num_actions,
    #         self.observation_size
    #     )
    #     logits = torch.stack(logits, dim=1)
    #     logits = logits * self.output_scaling
    #     # softmax = nn.Softmax(dim=-1)
    #     # logits = softmax(logits)
    #     # probs = Categorical(logits=logits)
    #     # action = probs.sample()
    #     # print(action)
    #     return logits #action, probs.log_prob(action)

    def encode_input(self, x):
        if self.state_encoding == "onehot":
            x_onehot = torch.zeros((x.shape[0], self.observation_size))
            for i, val in enumerate(x):
                x_onehot[i, int(val.item())] = np.pi
            return x_onehot
        elif self.state_encoding == "binary":
            x_binary = torch.zeros((x.shape[0], self.observation_size))
            for i, val in enumerate(x):
                binary = bin(int(val.item()))[2:]
                padded = binary.zfill(self.observation_size)
                x_binary[i] = torch.tensor([int(bit) for bit in padded])
            return x_binary


def log_metrics(config, metrics, report_path=None):
    if config["wandb"]:
        wandb.log(metrics)
    if ray.is_initialized():
        ray.train.report(metrics=metrics)
    else:
        with open(os.path.join(report_path, "result.json"), "a") as f:
            json.dump(metrics, f)
            f.write("\n")


def normalize_rewards(rewards):
    rewards = torch.cat(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
    return rewards


def discounted_rewards(rewards):
    discounted_rewards = []
    R = 0
    gamma = 0.9
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return torch.tensor(discounted_rewards, dtype=torch.float32)


def reinforce_quantum_discrete_state(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    gamma = config["gamma"]
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]

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
    assert (
        env_id in gym.envs.registry.keys()
    ), f"{env_id} is not a valid gymnasium environment"
    config["game_mode"] = "frozen_lake"
    config["state_encoding"] = "binary"
    config["legal_actions_type"] = "restricted"
    config["map_name"] = "4x4"
    config["is_slippery"] = False
    envs = MazeGame(config)
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(env_id, config) for _ in range(num_envs)],
    # )

    # assert isinstance(
    #     envs.single_action_space, gym.spaces.Discrete
    # ), "only discrete action space is supported"
    # assert isinstance(
    #     envs.single_observation_space, gym.spaces.Discrete
    # ), "only discrete state space is supported"

    # Here, the classical agent is initialized with a Neural Network
    agent = ReinforceAgentQuantum(config).to(device)
    optimizer = optim.Adam(
        [
            {"params": agent.input_scaling, "lr": lr_input_scaling},
            {"params": agent.output_scaling, "lr": lr_output_scaling},
            {"params": agent.weights, "lr": lr_weights},
        ],
        amsgrad=True,
        weight_decay=0,
    )

    # global parameters to log
    global_step = 0
    global_episodes = 0
    print_interval = 10
    episode_returns = deque(maxlen=print_interval)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, _ = envs.reset()
    x = 0
    while global_step < total_timesteps:

        rewards_epoch = []

        rewards_epoch_train = []
        log_probs_epoch_train = []

        steps_per_epoch = 0
        step = 0
        for _ in range(200):
            log_probs = []
            rewards = []

            done = False
            # Episode loop
            while not done:
                obs = torch.tensor(obs, dtype=torch.float32)

                action, log_prob = agent.forward(obs)

                obs, reward, terminations, truncations, infos = envs.step(
                    np.array([action])
                )
                rewards.append(reward)
                log_probs.append(log_prob)
                done = np.any(terminations) or np.any(truncations)
                step += 1
                x += 1

            if done or (step == steps_per_epoch - 1):
                if done or len(rewards_epoch) == 0:
                    rewards_epoch.append(np.sum(rewards))
                if step == steps_per_epoch - 1:
                    rewards_epoch_train.append(
                        torch.tensor(rewards, dtype=torch.float32)
                    )
                else:
                    # Calculate discounted rewards
                    discounted_rewards = []
                    cumulative_reward = 0
                    for reward in reversed(rewards):
                        cumulative_reward = reward + gamma * cumulative_reward
                        discounted_rewards.insert(0, cumulative_reward)

                    rewards_epoch_train.append(
                        torch.tensor(discounted_rewards, dtype=torch.float32)
                    )
                log_probs_epoch_train.append(torch.cat(log_probs))

                obs, _ = envs.reset()

            if step >= 199:
                break

        print(x, np.mean(rewards_epoch))

        new_rewards = normalize_rewards(rewards_epoch_train)
        new_log_probs = torch.cat(log_probs_epoch_train)
        policy_loss = -(new_log_probs * new_rewards).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

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
        trial_name: str = "reinforce_quantum_discrete_state"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = False  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

        # Environment parameters
        env_id: str = "FrozenLake-v1"  # Environment ID
        is_slippery: bool = False

        # Algorithm parameters
        num_envs: int = 1  # Number of environments
        seed: int = None  # Seed for reproducibility
        total_timesteps: int = 20000  # Total number of timesteps
        gamma: float = 0.9  # discount factor
        lr_input_scaling: float = 0.025  # Learning rate for input scaling
        lr_weights: float = 0.025  # Learning rate for variational parameters
        lr_output_scaling: float = 0.1  # Learning rate for output scaling
        cuda: bool = False  # Whether to use CUDA
        num_qubits: int = 4  # Number of qubits
        num_layers: int = 5  # Number of layers in the quantum circuit
        device: str = "lightning.qubit"  # Quantum device
        diff_method: str = "adjoint"  # Differentiation method
        save_model: bool = True  # Save the model after the run
        state_encoding: str = (
            "binary"  # Type of state encoding, either "binary" or "onehot"
        )

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
    reinforce_quantum_discrete_state(config)
