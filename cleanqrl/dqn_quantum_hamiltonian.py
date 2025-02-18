# This file is an adaptation from https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import ray
import json
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pennylane as qml
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
        # if observables == "local":
        #     half = len(wires) // 2

        #     # Create the tensor product for the first half
        #     left_observable = qml.PauliZ(0)
        #     for i in range(1, half):
        #         left_observable = left_observable @ qml.PauliZ(i)

        #     # Create the tensor product for the second half
        #     right_observable = qml.PauliZ(half)
        #     for i in range(half + 1, len(wires)):
        #         right_observable = right_observable @ qml.PauliZ(i)
        
        # elif observables == "global":

        #     left_observable = qml.PauliZ(0)
        #     right_observable = qml.PauliX(0)

        #     for i in range(1,len(wires)):
        #         left_observable = left_observable @ qml.PauliZ(i)
        #         right_observable = right_observable @ qml.PauliX(i)
            
        #     left_observable = qml.Hamiltonian([1.0], [left_observable])
        #     right_observable = qml.Hamiltonian([1.0], [right_observable])
        
        return [qml.expval(qml.PauliZ(i)) for i in wires]
    

def calculate_a(a,S,L):
    left_side = np.sin(2 * a * np.pi) / (2 * a * np.pi)
    right_side = (S * (2 * L - 1) - 2) / (S * (2 * L + 1))
    return left_side - right_side

class QRLAgentDQNHamiltonian(nn.Module):
    def __init__(self,envs,config):
        super().__init__()
        self.config = config
        self.observation_space = envs.observation_space
        self.num_actions = envs.single_action_space.n
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.ansatz = config["ansatz"]
        self.init_method = config["init_method"]
        self.observables = config["observables"]
        if self.observables == "global":
            self.S = self.num_qubits
        elif self.observables == "local":
            self.S = self.num_qubits // 2
        self.wires = range(self.num_qubits)


        # Input and Output Scaling weights are always initialized as 1s        
        self.register_parameter(name="output_scaling_actor", param = nn.Parameter(torch.ones(self.num_actions), requires_grad=True))

        # The variational weights are initialized differently according to the config file
        if self.init_method == "uniform":
            self.register_parameter(name="variational_actor", param = nn.Parameter(torch.rand(self.num_layers,3) * 2 * torch.pi - torch.pi, requires_grad=True))
        else:
            raise ValueError("Invalid initialization method")
        
        dev = qml.device(config["device"], wires = self.wires)

        self.qc = qml.QNode(hamiltonian_encoding_ansatz, dev, diff_method = config["diff_method"], interface = "torch")
        
    def forward(self, linear_terms, quadratic_terms, annotations):

        # x = x.repeat(1, len(self.wires) // len(x[0]) + 1)[:, :len(self.wires)]
        linear_terms = linear_terms.unsqueeze(0) 
        quadratic_terms = quadratic_terms.unsqueeze(0) 
        annotations = annotations.unsqueeze(0) 
        logits = self.qc(linear_terms, quadratic_terms, annotations, self._parameters["variational_actor"], self.wires, self.num_layers, "actor", self.observables)
        if type(logits) == list:
            logits = torch.stack(logits, dim = 1)
        logits_scaled = logits * self._parameters["output_scaling_actor"]
        return logits_scaled

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def dqn_quantum_hamiltonian(config):
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
    lr_variational = config["lr_variational"]
    lr_output_scaling = config["lr_output_scaling"]

    if not ray.is_initialized():
        report_path = os.path.join(config["path"], "result.json")
        with open(report_path, "w") as f:
            f.write("")

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id,config) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert num_envs == 1, "environment vectorization not possible in DQN"


    q_network = QRLAgentDQNHamiltonian(envs, config).to(device)
    optimizer = optim.Adam([
        {"params": q_network._parameters["output_scaling_actor"], "lr": lr_output_scaling},
        {"params": q_network._parameters["variational_actor"], "lr": lr_variational}
    ])
    target_network = QRLAgentDQNHamiltonian(envs, config).to(device)

    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size,
        envs.observation_space,
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
    global_step_returns = []

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
        if random.random() < epsilon:
            actions = envs.env.sample_valid_action() # for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs['linear_terms']).to(device), torch.Tensor(obs['quadratic_terms']).to(device), torch.Tensor(obs['annotations']).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if len(infos)>0:
            global_episodes += 1
            episode_returns.append(float(infos["episode"]["r"]))
            global_step_returns.append(global_step)
            metrics = {
                "episodic_return": float(infos["episode"]["r"]),
                "global_step": global_step,
                "episode": global_episodes
            }
            # obs, _ = envs.reset()

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                data = rb.sample(batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 20 == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                    )
        
        if (global_step > learning_starts and global_step % train_frequency == 0) or "episode" in infos:
            if ray.is_initialized():
                ray.train.report(metrics=metrics)
            else:
                with open(report_path, "a") as f:
                    json.dump(metrics, f)
                    f.write("\n") 

    envs.close()