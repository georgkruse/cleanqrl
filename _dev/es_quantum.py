import os
import ray
import json
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import pennylane as qml

from utils.env_utils import make_env

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
        self.ansatz = config["ansatz"]
        self.init_method = config["init_method"]
        self.observables = config["observables"]
        if self.observables == "global":
            self.S = self.num_qubits
        elif self.observables == "local":
            self.S = self.num_qubits // 2
        self.wires = range(self.num_qubits)

        if self.ansatz == "hea":
            # Input and Output Scaling weights are always initialized as 1s        
            self.register_parameter(name="input_scaling_actor", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=True))
            self.register_parameter(name="output_scaling_actor", param = nn.Parameter(torch.ones(self.num_actions), requires_grad=True))

            # The variational weights are initialized differently according to the config file
            if self.init_method == "uniform":
                self.register_parameter(name="variational_actor", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi - torch.pi, requires_grad=True))
            else:
                raise ValueError("Invalid initialization method")    
        
        dev = qml.device(config["device"], wires = self.wires)
        if self.ansatz == "hea":
            self.qc = qml.QNode(hea, dev, diff_method = config["diff_method"], interface = "torch")
        
    def get_action_and_logprob(self, x):
        x = x.repeat(1, len(self.wires) // len(x[0]) + 1)[:, :len(self.wires)]
        logits = self.qc(x, 
                         self._parameters["input_scaling_actor"], 
                         self._parameters["variational_actor"], 
                         self.wires, 
                         self.num_layers, 
                         "actor", 
                         self.observables)
        
        if type(logits) == list:
            logits = torch.stack(logits, dim = 1)
        logits_scaled = logits * self._parameters["output_scaling_actor"]
        probs = Categorical(logits=logits_scaled)
        action = probs.sample()
        return action, probs.log_prob(action)


def es_quantum(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    learning_rate = config["learning_rate"]
    gamma = config["gamma"]

    if not ray.is_initialized():
        report_path = os.path.join(config["path"], "result.json")
        with open(report_path, "w") as f:
            f.write("")

    device = torch.device("cuda" if (torch.cuda.is_available() and config["cuda"]) else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id) for _ in range(num_envs)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Here, the classical agent is initialized with a Neural Network
    agent = ReinforceAgentQuantum(envs, config).to(device)

    parameter_vector_size = np.prod(agent._parameters["input_scaling_actor"].shape) + np.prod(agent._parameters["output_scaling_actor"].shape) + np.prod(agent._parameters["variational_actor"].shape)
    size_input_scaling = agent._parameters["input_scaling_actor"].shape
    size_output_scaling = agent._parameters["output_scaling_actor"].shape
    size_variational_actor = agent._parameters["variational_actor"].shape

    sigma = 0.001
    from copy import deepcopy
    apv_input_scaling = deepcopy(agent._parameters["input_scaling_actor"]).detach().numpy()
    apv_output_scaling = deepcopy(agent._parameters["output_scaling_actor"]).detach().numpy()
    apv_variational_actor = deepcopy(agent._parameters["variational_actor"]).detach().numpy()

    # apv_input_scaling = np.random.normal(0, sigma**2, size_input_scaling)
    # apv_output_scaling = np.random.normal(0, sigma**2, size_output_scaling)
    # apv_variational_actor = np.random.normal(0, sigma**2, size_variational_actor)

    num_mutations = 10
    global_step = 0
    start_time = time.time()
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(device)
    global_episodes = 0
    episode_returns = []
    global_step_returns = []

    while global_step < total_timesteps:

        mpv_input_scaling = np.random.normal(0, 1, (num_mutations,*size_input_scaling))
        mpv_output_scaling = np.random.normal(0, 1, (num_mutations,*size_output_scaling))
        mpv_variational_actor = np.random.normal(0, 1, (num_mutations,*size_variational_actor))
        fitness_of_generation = np.zeros(num_mutations)

        for idx in range(num_mutations):

            log_probs = []
            rewards = []
            done = False
            # set the parameters of the agent
            
            agent._parameters["input_scaling_actor"] = apv_input_scaling + sigma*mpv_input_scaling[idx]
            agent._parameters["output_scaling_actor"] = apv_output_scaling + sigma*mpv_output_scaling[idx]
            agent._parameters["variational_actor"] = apv_variational_actor + sigma*mpv_variational_actor[idx]

            # Episode loop
            while not done:
                action, log_prob = agent.get_action_and_logprob(obs)
                log_probs.append(log_prob)
                obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                rewards.append(reward)
                obs = torch.Tensor(obs).to(device)
                done = np.any(terminations) or np.any(truncations)
            fitness_of_generation[idx] = np.sum(rewards)
            
        print(np.mean(fitness_of_generation))
        fitness_of_generation = (fitness_of_generation - np.mean(fitness_of_generation)) / np.std(fitness_of_generation)

        update_vector_input_scaling = mpv_input_scaling * fitness_of_generation[:, np.newaxis, np.newaxis]
        update_vector_output_scaling = mpv_output_scaling * fitness_of_generation[:, np.newaxis]
        update_vector_variational_actor = mpv_variational_actor * fitness_of_generation[:, np.newaxis, np.newaxis]

        update_vector_input_scaling = np.sum(update_vector_input_scaling, axis=0)
        update_vector_output_scaling = np.sum(update_vector_output_scaling, axis=0)
        update_vector_variational_actor = np.sum(update_vector_variational_actor, axis=0)

        apv_input_scaling += learning_rate * (1/(num_mutations*sigma)) * update_vector_input_scaling
        apv_output_scaling += learning_rate * (1/(num_mutations*sigma)) * update_vector_output_scaling
        apv_variational_actor += learning_rate * (1/(num_mutations*sigma)) * update_vector_variational_actor
        global_episodes +=1

        # Eval:

        log_probs = []
        rewards = []
        done = False
        # set the parameters of the agent
        
        agent._parameters["input_scaling_actor"] = apv_input_scaling
        agent._parameters["output_scaling_actor"] = apv_output_scaling
        agent._parameters["variational_actor"] = apv_variational_actor

        # Episode loop
        while not done:
            action, log_prob = agent.get_action_and_logprob(obs)
            log_probs.append(log_prob)
            obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards.append(reward)
            obs = torch.Tensor(obs).to(device)
            done = np.any(terminations) or np.any(truncations)
        print(np.sum(rewards))
        # Not sure about this?
        global_step += len(rewards) * num_envs

        print("SPS:", int(global_step / (time.time() - start_time)))

        metrics = {
            "episodic_return": int(infos["final_info"][0]["episode"]["r"][0]),
            "global_step": global_step,
            "episode": global_episodes,
            "mean_fitness": np.mean(fitness_of_generation)
        }
        if ray.is_initialized():
            ray.train.report(metrics=metrics)
        else:
            with open(report_path, "a") as f:
                json.dump(metrics, f)
                f.write("\n")

        print(f"Global step: {global_step}, Return: {sum(rewards)}, mean fitness: {np.mean(fitness_of_generation)}")

    envs.close()