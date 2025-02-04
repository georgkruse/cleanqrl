import ray
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import pennylane as qml

from ansatz.hea import hea

def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ContinuousEncoding(env)
        return env
    return thunk

class ContinuousEncoding(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)

def calculate_a(a,S,L):
    left_side = np.sin(2 * a * np.pi) / (2 * a * np.pi)
    right_side = (S * (2 * L - 1) - 2) / (S * (2 * L + 1))
    return left_side - right_side

class ReinforceAgentQuantum(nn.Module):
    def __init__(self,envs,config):
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
            elif self.init_method == "small_random":
                self.register_parameter(name="variational_actor", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 0.2 - 0.1, requires_grad=True))
            elif self.init_method == "reduced_domain":
                initial_guess = 0.1
                alpha = fsolve(lambda a: calculate_a(a,self.S,self.num_layers), initial_guess)
                self.register_parameter(name="variational_actor", param = nn.Parameter((torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi - torch.pi) * alpha, requires_grad=True))    
        
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
        
        if self.config["diff_method"] == "backprop" or x.shape[0] == 1:
            logits = logits.reshape(x.shape[0], self.num_actions)
        logits_scaled = logits * self._parameters["output_scaling_actor"]
        probs = Categorical(logits=logits_scaled)
        action = probs.sample()
        return action, probs.log_prob(action)


def reinforce_quantum(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    gamma = config["gamma"]
    lr_input_scaling = config["lr_input_scaling"]
    lr_variational = config["lr_variational"]
    lr_output_scaling = config["lr_output_scaling"]

    device = torch.device("cuda" if (torch.cuda.is_available() and config["cuda"]) else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id) for _ in range(num_envs)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


    agent = ReinforceAgentQuantum(envs, config).to(device)
    optimizer = optim.Adam([
        {"params": agent._parameters["input_scaling_actor"], "lr": lr_input_scaling},
        {"params": agent._parameters["output_scaling_actor"], "lr": lr_output_scaling},
        {"params": agent._parameters["variational_actor"], "lr": lr_variational}
    ])


    global_step = 0
    start_time = time.time()
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(device)
    global_episodes = 0
    episode_returns = []
    global_step_returns = []

    while global_step < total_timesteps:
        log_probs = []
        rewards = []
        done = False

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

        print("SPS:", int(global_step / (time.time() - start_time)))

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

        metrics = {
            "episodic_return": infos["episode"]["r"][0],
            "global_step": global_step,
            "episode": global_episodes,
            "loss": loss.item()
        }
        ray.train.report(metrics = metrics)

        print(f"Global step: {global_step}, Return: {sum(rewards)}, Loss: {loss.item()}")

    envs.close()