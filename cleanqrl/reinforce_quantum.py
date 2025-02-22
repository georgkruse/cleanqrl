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

class ArctanNormalizationWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)


def make_env(env_id, config=None):
    def thunk():
        env = gym.make(env_id)
        env = ArctanNormalizationWrapper(env)       
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


def reinforce_quantum(config):
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

        if len(infos)>0:
            for info in infos:
                if info and "episode" in info:
                    global_episodes +=1
                    episode_returns.append(float(info["episode"]["r"]))
                    global_step_returns.append(global_step)
                    metrics = {
                        "episodic_return": float(info["episode"]["r"]),
                        "global_step": global_step,
                        "episode": global_episodes
                    }

            # This needs to be placed at the end to include loss loggings
            if ray.is_initialized():
                ray.train.report(metrics=metrics)
            else:
                with open(report_path, "a") as f:
                    json.dump(metrics, f)
                    f.write("\n")

        print(f"Global step: {global_step}, Return: {sum(rewards)}, Loss: {loss.item()}")

    envs.close()