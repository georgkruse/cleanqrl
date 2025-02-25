# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import ray
import json
import random
import time
import yaml
from ray import tune
import datetime
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from ray.train._internal.session import get_session

import wandb
def make_env(env_id, gamma):
    def thunk():

        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def ppo_classical_continuous(config):
    num_envs = config["num_envs"]
    num_steps = config["num_steps"]
    num_minibatches = config["num_minibatches"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    num_envs = config["num_envs"]
    learning_rate = config["learning_rate"]
    anneal_lr = config["anneal_lr"]
    gamma = config["gamma"]
    gae_lambda = config["gae_lambda"]
    update_epochs = config["update_epochs"]
    clip_coef = config["clip_coef"]
    norm_adv = config["norm_adv"]
    clip_vloss = config["clip_vloss"]
    ent_coef = config["ent_coef"]
    vf_coef = config["vf_coef"]
    target_kl = config["target_kl"]
    max_grad_norm = config["max_grad_norm"]
    seed = config["seed"]
    cuda = config["cuda"]

    if target_kl == "None":
        target_kl = None

    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size
    
    if not ray.is_initialized():
        report_path = os.path.join(config["path"], "result.json")
        name = config['trial_name']
        with open(report_path, "w") as f:
            f.write("")
    else:
        session = get_session()
        report_path = session.storage.trial_fs_path 
        name = session.trial_id
    print('shfaskdflksahfslkjdf', name)
    wandb.init(
        project='cleanqrl',
        sync_tensorboard=True,
        config=config,
        name=name,
        monitor_gym=True,
        save_code=True,
        dir=report_path
    )
    
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size


    # TRY NOT TO MODIFY: seeding
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, gamma) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "episode" in infos:
                print(infos['episode']['r'])
                print(global_step)
                metrics = {}
                metrics['episodic_return'] = infos['episode']['r']
                metrics['episodic_length'] = infos['episode']['l']
                metrics['global_step'] = global_step
                wandb.log(metrics)
                if ray.is_initialized():
                    ray.train.report(metrics=metrics)
                else:
                    with open(report_path, "a") as f:
                        json.dump(metrics, f)
                        f.write("\n")
                          
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        metrics = {}
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        metrics["value_loss"] = v_loss.item()
        metrics["policy_loss"] = pg_loss.item()
        metrics["entropy"] = entropy_loss.item()
        metrics["old_approx_kl"] =  old_approx_kl.item()
        metrics["approx_kl"] = approx_kl.item()
        metrics["clipfrac"] = np.mean(clipfracs)
        metrics["explained_variance"] = np.mean(explained_var)
        metrics["SPS"] = int(global_step / (time.time() - start_time))
        
        wandb.log(metrics)
        if ray.is_initialized():
            ray.train.report(metrics=metrics)
        else:
            with open(report_path, "a") as f:
                json.dump(metrics, f)
                f.write("\n")
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)



    envs.close()
    # writer.close()

if __name__ == "__main__":
    
    @dataclass
    class Config:
        trial_name: str = 'ppo_classical'  # Name of the trial
        trial_path: str = 'logs'  # Path to save logs relative to the parent directory
        exp_name: str = os.path.basename(__file__)[: -len(".py")]
        """the name of this experiment"""
        seed: int = 1
        """seed of the experiment"""
        torch_deterministic: bool = True
        """if toggled, `torch.backends.cudnn.deterministic=False`"""
        cuda: bool = True
        """if toggled, cuda will be enabled by default"""
        track: bool = False
        """if toggled, this experiment will be tracked with Weights and Biases"""
        wandb_project_name: str = "cleanRL"
        """the wandb's project name"""
        wandb_entity: str = None
        """the entity (team) of wandb's project"""
        capture_video: bool = False
        """whether to capture videos of the agent performances (check out `videos` folder)"""
        save_model: bool = False
        """whether to save model into the `runs/{run_name}` folder"""
        upload_model: bool = False
        """whether to upload the saved model to huggingface"""
        hf_entity: str = ""
        """the user or org name of the model repository from the Hugging Face Hub"""

        # Algorithm specific arguments
        env_id: str = "Pendulum-v1"
        """the id of the environment"""
        total_timesteps: int = 1000000
        """total timesteps of the experiments"""
        learning_rate: float = 3e-4
        """the learning rate of the optimizer"""
        num_envs: int = 1
        """the number of parallel game environments"""
        num_steps: int = 2048
        """the number of steps to run in each environment per policy rollout"""
        anneal_lr: bool = True
        """Toggle learning rate annealing for policy and value networks"""
        gamma: float = 0.9
        """the discount factor gamma"""
        gae_lambda: float = 0.95
        """the lambda for the general advantage estimation"""
        num_minibatches: int = 32
        """the number of mini-batches"""
        update_epochs: int = 10
        """the K epochs to update the policy"""
        norm_adv: bool = True
        """Toggles advantages normalization"""
        clip_coef: float = 0.2
        """the surrogate clipping coefficient"""
        clip_vloss: bool = True
        """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
        ent_coef: float = 0.0
        """coefficient of the entropy"""
        vf_coef: float = 0.5
        """coefficient of the value function"""
        max_grad_norm: float = 0.5
        """the maximum norm for the gradient clipping"""
        target_kl: float = None
        """the target KL divergence threshold"""

        # to be filled in runtime
        batch_size: int = 0
        """the batch size (computed in runtime)"""
        minibatch_size: int = 0
        """the mini-batch size (computed in runtime)"""
        num_iterations: int = 0
        """the number of iterations (computed in runtime)"""
    
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
    ppo_classical_continuous(config)