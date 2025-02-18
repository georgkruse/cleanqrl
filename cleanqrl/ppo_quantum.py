# This file is an adaptation from https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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


def hardware_efficient_ansatz(x, input_scaling_weights, variational_weights, wires, layers):
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

        return [qml.expval(qml.PauliZ(0)@qml.PauliZ(1)), qml.expval(qml.PauliZ(2)@qml.PauliZ(3))]


class PPOAgentQuantum(nn.Module):
    def __init__(self,envs,config):
        super().__init__()
        self.num_features = np.array(envs.single_observation_space.shape).prod()
        self.num_actions = envs.single_action_space.n
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.wires = range(self.num_qubits)

        # input and output scaling are always initialized as ones      
        self.register_parameter(name="input_scaling_critic", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=True))
        self.register_parameter(name="input_scaling_actor", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=True))
        self.register_parameter(name="output_scaling_critic", param = nn.Parameter(torch.ones(1), requires_grad=True))
        self.register_parameter(name="output_scaling_actor", param = nn.Parameter(torch.ones(self.num_actions), requires_grad=True))
        # trainable weights are initialized randomly between -pi and pi
        self.register_parameter(name="weights_critic", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi, requires_grad=True))
        self.register_parameter(name="weights_actor", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi, requires_grad=True))

        device = qml.device(config["device"], wires = self.wires)
        self.quantum_circuit = qml.QNode(hardware_efficient_ansatz, device, diff_method = config["diff_method"], interface = "torch")
        
    def get_value(self,x):
        return self._parameters["output_scaling_critic"] * self.quantum_circuit(x, 
                         self._parameters["input_scaling_actor"], 
                         self._parameters["weights_actor"], 
                         self.wires, 
                         self.num_layers, 
                         "critic")
        

    def get_action_and_value(self, x, action=None):
        logits = self.quantum_circuit(x, 
                         self._parameters["input_scaling_actor"], 
                         self._parameters["weights_critic"], 
                         self.wires, 
                         self.num_layers, 
                         "actor")
        
        logits = torch.stack((logits[0], logits[1]), dim=1)
        logits_scaled = logits * self._parameters["output_scaling_actor"]
        probs = Categorical(logits=logits_scaled)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self._parameters["output_scaling_critic"] * self.quantum_circuit(x, 
                         self._parameters["input_scaling_actor"], 
                         self._parameters["weights_critic"], 
                         self.wires, 
                         self.num_layers, 
                         "critic")


def ppo_quantum(config):
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
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]

    if target_kl == "None":
        target_kl = None

    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size

    if not ray.is_initialized():
        report_path = os.path.join(config["path"], "result.json")
        with open(report_path, "w") as f:
            f.write("")

    device = torch.device("cuda" if (torch.cuda.is_available() and config["cuda"]) else "cpu")
    assert env_id in gym.envs.registry.keys(), f"{env_id} is not a valid gymnasium environment"

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id) for i in range(num_envs)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = PPOAgentQuantum(envs, config).to(device)  # This is what I need to change to fit quantum into the picture
    #optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    optimizer = optim.Adam([
        {"params": agent._parameters["input_scaling_critic"], "lr": lr_input_scaling},
        {"params": agent._parameters["input_scaling_actor"], "lr": lr_input_scaling},
        {"params": agent._parameters["output_scaling_critic"], "lr": lr_output_scaling},
        {"params": agent._parameters["output_scaling_actor"], "lr": lr_output_scaling},
        {"params": agent._parameters["weights_critic"], "lr": lr_weights},
        {"params": agent._parameters["weights_actor"], "lr": lr_weights}
    ])

    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    # global parameters to log
    global_step = 0
    global_episodes = 0
    global_circuit_executions = 0
    steps_per_eposide = 0
    episode_returns = []
    global_step_returns = []

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

            if global_episodes >= 100:
                if global_step % 1000 == 0:
                    print(np.mean(episode_returns[-10:]))


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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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

        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()