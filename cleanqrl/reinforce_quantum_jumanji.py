import os
import ray
import json
import time
import yaml
import datetime
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from dataclasses import dataclass
import pennylane as qml
from environments.bandit import MultiArmedBanditEnv
from environments.maze import MazeEnv
from environments.tsp import TSPEnv
from environments.wrapper import *


class ArctanNormalizationWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)


def make_env(env_id, config=None):
    def thunk():
        custom_envs = {
            "bandit": MultiArmedBanditEnv,  # Add your custom environments here
            "maze": MazeEnv,
            'TSP-v1': TSPEnv,
        }

        env = custom_envs[env_id](config)
        # env = MinMaxNormalizationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk



def hardware_efficient_ansatz(x, input_scaling_weights, variational_weights, wires, layers, num_actions):
    for layer in range(layers):
        for i, wire in enumerate(wires):
            qml.RX(input_scaling_weights[layer, i] * x[:,i], wires = [wire])
    
            for i, wire in enumerate(wires):
                qml.RY(variational_weights[layer, i], wires = [wire])

            for i, wire in enumerate(wires):
                qml.RZ(variational_weights[layer, i+len(wires)], wires = [wire])

            if len(wires) == 2:
                qml.CZ(wires = wires)
            else:
                for i in range(len(wires)):
                    qml.CZ(wires = [wires[i],wires[(i+1)%len(wires)]])
        # TODO: make observation dependent on num_actions
        return [qml.expval(qml.PauliZ(0)@qml.PauliZ(1)), qml.expval(qml.PauliZ(2)@qml.PauliZ(3))]


def calculate_a(a,S,L):
    left_side = np.sin(2 * a * np.pi) / (2 * a * np.pi)
    right_side = (S * (2 * L - 1) - 2) / (S * (2 * L + 1))
    return left_side - right_side

class ReinforceAgentQuantum(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        self.config = config
        self.num_features = np.array(envs.single_observation_space.shape).prod()
        self.num_actions = envs.single_action_space.n
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.wires = range(self.num_qubits)

        # input and output scaling are always initialized as ones      
        self.register_parameter(name="input_scaling", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=True))
        self.register_parameter(name="output_scaling", param = nn.Parameter(torch.ones(self.num_actions), requires_grad=True))
        # trainable weights are initialized randomly between -pi and pi
        self.register_parameter(name="weights", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi - torch.pi, requires_grad=True))
        
        device = qml.device(config["device"], wires = self.wires)
        self.quantum_circuit = qml.QNode(hardware_efficient_ansatz, device, diff_method = config["diff_method"], interface = "torch")
        
    def get_action_and_logprob(self, x):
        x = x.repeat(1, len(self.wires) // len(x[0]) + 1)[:, :len(self.wires)]
        logits = self.quantum_circuit(x, 
                         self._parameters["input_scaling"], 
                         self._parameters["weights"], 
                         self.wires, 
                         self.num_layers, 
                         self.num_actions)
        
        if type(logits) == list:
            logits = torch.stack(logits, dim = 1)
        logits_scaled = logits * self._parameters["output_scaling"]
        probs = Categorical(logits=logits_scaled)
        action = probs.sample()
        return action, probs.log_prob(action)


def reinforce_quantum_jumanji(config):
    cuda = config["cuda"]
    env_id = config["env_id"]
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    gamma = config["gamma"]
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]

    if not ray.is_initialized():
        report_path = os.path.join(config["path"], "result.json")
        with open(report_path, "w") as f:
            f.write("")

    device = torch.device("cuda" if (torch.cuda.is_available() and config["cuda"]) else "cpu")
    assert env_id in gym.envs.registry.keys(), f"{env_id} is not a valid gymnasium environment"

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id) for _ in range(num_envs)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = ReinforceAgentQuantum(envs, config).to(device)
    optimizer = optim.Adam([
        {"params": agent._parameters["input_scaling"], "lr": lr_input_scaling},
        {"params": agent._parameters["output_scaling"], "lr": lr_output_scaling},
        {"params": agent._parameters["weights"], "lr": lr_weights}
    ])

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

        global_step += len(rewards) * num_envs
        steps_per_second = int(global_step / (time.time() - start_time))

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
            losses.append(float(loss.item()))
            metrics["loss"] = float(float(loss.item()))

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
        
        print(f"Global step: {global_step}, " 
              f"Mean Return: {np.round(np.mean(episode_returns[-10:]), 2)}, "
              f"Mean Loss: {np.round(np.mean(losses[-10:]), 5)}, "
              f"SPS: {steps_per_second}")
        
    envs.close()


if __name__ == '__main__':

    @dataclass
    class Config:
        trial_name: str = 'reinforce_quantum_jumanji'  # Name of the trial
        trial_path: str = 'logs'  # Path to save logs relative to the parent directory
        env_id: str = "TSP-v1"  # Environment ID
        num_envs: int = 1  # Number of environments
        total_timesteps: int = 100000  # Total number of timesteps
        gamma: float = 0.99  # discount factor
        lr_input_scaling: float = 0.01  # Learning rate for input scaling
        lr_weights: float = 0.01  # Learning rate for variational parameters
        lr_output_scaling: float = 0.01  # Learning rate for output scaling
        cuda: bool = False  # Whether to use CUDA
        num_qubits: int = 4  # Number of qubits
        num_layers: int = 2  # Number of layers in the quantum circuit
        device: str = "default.qubit"  # Quantum device
        diff_method: str = "backprop"  # Differentiation method

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
    reinforce_quantum_jumanji(config)    