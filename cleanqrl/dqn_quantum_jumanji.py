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
import pennylane as qml
from dataclasses import dataclass
from replay_buffer import ReplayBuffer
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


def layer_hamiltonian_encoding_ansatz(linear_terms, quadratic_terms, annotation, rotational_params, wires):
    """
    x: input (batch_size,num_features)
    input_scaling_params: vector of parameters (num_features)
    rotational_params:  vector of parameters (num_features*2)
    """

    parameter_idx = 0
    for idx in range(linear_terms.shape[1]):
        qml.RZ(linear_terms[:,idx,0]*rotational_params[parameter_idx], wires = idx)

    parameter_idx += 1
    for idx in range(quadratic_terms.shape[1]):
        qml.CRZ(quadratic_terms[:,idx,2]*rotational_params[parameter_idx], wires = [int(quadratic_terms[0,idx,0].detach().numpy()), int(quadratic_terms[0,idx,1].detach().numpy())])

    parameter_idx += 1
    for idx in range(annotation.shape[0]):
        qml.RZ(annotation[:,idx]*rotational_params[parameter_idx], wires = idx)


def hamiltonian_encoding_ansatz(linear_terms, quadratic_terms, annotation, variational_weights, wires, layers, type_, observables = "None"):
    for layer in range(layers):
        layer_hamiltonian_encoding_ansatz(linear_terms, quadratic_terms, annotation, variational_weights[layer], wires)
    if type_ == "critic":
        return qml.expval(qml.PauliZ(0))
    elif type_ == "actor":
        return [qml.expval(qml.PauliZ(i)) for i in wires]
    

class DQNAgentQuantumJumanji(nn.Module):
    def __init__(self,envs,config):
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
        
    def forward(self, x):
        x = x.repeat(1, len(self.wires) // len(x[0]) + 1)[:, :len(self.wires)]
        logits = self.quantum_circuit(x, self._parameters["input_scaling"], self._parameters["weights"], self.wires, self.num_layers, self.num_actions)
        if type(logits) == list:
            logits = torch.stack(logits, dim = 1)
        logits_scaled = logits * self._parameters["output_scaling"]
        return logits_scaled

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def dqn_quantum_jumanji(config):
    cuda = config["cuda"]
    env_id = config["env_id"]
    num_envs = config["num_envs"]
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
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]

    if not ray.is_initialized():
        report_path = os.path.join(config["path"], "result.json")
        with open(report_path, "w") as f:
            f.write("")

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    assert env_id in gym.envs.registry.keys(), f"{env_id} is not a valid gymnasium environment"

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id,config) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert num_envs == 1, "environment vectorization not possible in DQN"


    q_network = DQNAgentQuantumJumanji(envs, config).to(device)
    optimizer = optim.Adam([
        {"params": q_network._parameters["input_scaling"], "lr": lr_input_scaling},
        {"params": q_network._parameters["output_scaling"], "lr": lr_output_scaling},
        {"params": q_network._parameters["weights"], "lr": lr_weights}
    ])

    target_network = DQNAgentQuantumJumanji(envs, config).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # global parameters to log
    global_step = 0
    global_episodes = 0
    global_circuit_executions = 0
    steps_per_eposide = 0
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
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
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
                      f"Mean Loss: {np.round(np.mean(losses[-10:]), 5)}")
    envs.close()



if __name__ == '__main__':

    @dataclass
    class Config:
        trial_name: str = 'dqn_quantum_jumanji'  # Name of the trial
        trial_path: str = 'logs'  # Path to save logs relative to the parent directory
        env_id: str = "TSP-v1"  # Environment ID
        num_envs: int = 1  # Number of environments
        buffer_size: int = 10000  # Size of the replay buffer
        total_timesteps: int = 10000  # Total number of timesteps
        start_e: float = 1.0  # Starting value of epsilon for exploration
        end_e: float = 0.01  # Ending value of epsilon for exploration
        exploration_fraction: float = 0.1  # Fraction of total timesteps for exploration
        learning_starts: int = 1000  # Timesteps before learning starts
        train_frequency: int = 1  # Frequency of training
        batch_size: int = 32  # Batch size for training
        gamma: float = 0.99  # Discount factor
        target_network_frequency: int = 100  # Frequency of target network updates
        tau: float = 0.01  # Soft update coefficient
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
    dqn_quantum_jumanji(config)    

