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

def make_env(env_id, config=None):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ReinforceAgentClassical(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_action_and_logprob(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action)

def reinforce_classical(config):
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
    assert env_id in gym.envs.registry.keys(), f"{env_id} is not a valid gymnasium environment"

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id) for _ in range(num_envs)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Here, the classical agent is initialized with a Neural Network
    agent = ReinforceAgentClassical(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

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

        # Not sure about this?
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

        if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        global_episodes +=1
                        episode_returns.append(float(info["episode"]["r"][0]))
                        global_step_returns.append(global_step)
                        metrics = {
                            "episodic_return": float(info["episode"]["r"][0]),
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