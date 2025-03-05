# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy

# this file has been mainly adapted from https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy

import os
import random
import time
from dataclasses import dataclass
from ray import tune 
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, List, Union
from copy import deepcopy


from games.maze.maze_game import MazeGame
from games.frozenlake_wrapper import FrozenLake_Wrapper
from torch.optim.lr_scheduler import LinearLR
from agents.qpg_model import QuantumPGModel

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def stack_dict_values(list_of_dicts):
    # Get all unique keys from all dictionaries
    all_keys = set().union(*list_of_dicts)
    
    # Create a new dictionary with lists of values for each key
    return {key: torch.cat([d[key] for d in list_of_dicts if key in d], dim=0) 
            for key in all_keys if any(key in d for d in list_of_dicts)}

class QPG(tune.Trainable):

    def setup(self, config: dict):
      
        from utils.config.create_env import wrapper_switch
        self.config = config['alg_config']
        self.env = wrapper_switch[config['env_config']['env']](config['env_config'])
        # self.env = FrozenLake_Wrapper(config['env_config'])

        self.model = QuantumPGModel(self.env.observation_space, self.env.action_space, self.env.action_space.n, config['alg_config'], 'model ')
        # self.target_network = QuantumPGModel(self.env.observation_space, self.env.action_space, self.env.action_space.n, config['model'], 'name')
        # self.target_network.load_state_dict(self.model.state_dict())

        self.optimizer = self.create_optimizer()[0]
        self.schedulder = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-3, total_iters=5)

        self.episodes_trained_total = 0
        self.steps_trained_total = 0
        self.steps_per_epoch = self.config['steps_per_epoch']
        self.circuit_executions = 0


    def create_optimizer(self): 
        '''
        In this function, the ray optimizer is overwritten. Use custom learning rate for variational parameters, 
        input scaling parameters and output scaling parameters
        '''
        if hasattr(self, "config"):
            
            params = []    

            if self.config['mode'] == 'quantum':
                
                # Use custom learning rate for variational parameters, input scaling parameters and output scaling parameters
                params.append({'params': [self.model._parameters[f'weights_actor']], 'lr': self.config['lr']})
                params.append({'params': [self.model._parameters[f'input_scaling_actor']], 'lr': self.config['lr']})
                
                if 'lr_output_scaling' in self.config.keys():
                    print('Using lr_output_scaling:', self.config['lr_output_scaling'])
                    custom_lr = self.config['lr_output_scaling']
                else:                            
                    print('NOT using lr_output_scaling:', self.config['lr'])
                    custom_lr = self.config['lr']

                if 'output_scaling_actor' in self.model._parameters.keys():
                    params.append({'params': self.model._parameters['output_scaling_actor'], 'lr': custom_lr})

                if 'weight_decay' in self.config.keys():
                    weight_decay = self.config['weight_decay']
                else:
                    weight_decay = 0

                if self.config['custom_optimizer'] == 'Adam':
                    optimizers = [
                        torch.optim.Adam(params, amsgrad=True, weight_decay=weight_decay)
                    ]
                elif self.config['custom_optimizer'] == 'SGD':
                    optimizers = [
                        torch.optim.SGD(params)
                    ]
                elif self.config['custom_optimizer'] == 'RMSprop':
                    optimizers = [
                        torch.optim.RMSprop(params)
                    ]
                elif self.config['custom_optimizer'] == 'LBFGS':
                    optimizers = [
                    torch.optim.LBFGS(params)
                    ]

                print('Using optimizer:', self.config['custom_optimizer'])

            elif self.config['mode'] == 'classical':
                optimizers = [torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])]
            else:
                print('Incomplete config file.')
                exit()
        else:
            optimizers = [torch.optim.Adam(self.model.parameters())]
        
        if getattr(self, "exploration", None):
            optimizers = self.exploration.get_exploration_optimizer(optimizers)
        return optimizers

    def normalize_rewards(self, rewards):
        rewards = torch.cat(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        return rewards
    
    def discounted_rewards(self, rewards):
        discounted_rewards = []
        R = 0
        gamma = 0.9
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        return torch.tensor(discounted_rewards, dtype=torch.float32)
    
    def update_policy(self, rewards, log_probs):  
        
        # policy_loss = []
        # for log_prob, R in zip(log_probs, rewards):
        #     policy_loss.append(-log_prob * R)re
        policy_loss = -(log_probs * rewards).sum()
        # print(rewards.shape)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.circuit_executions += log_probs.shape[0]*self.model.total_model_parameters
        print(log_probs.shape[0])
        return policy_loss.detach().numpy()
    
    def select_action(self, state):
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, _ = self.model.forward({'obs':state})
        self.circuit_executions += 1
        softmax = nn.Softmax(dim=-1)
        probs = softmax(probs)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def step(self):  
        
        done = True
        
        self.steps_trained_epoch = 0
        self.episodes_trained_epoch = 0
        self.rewards_epoch = []
        self.policy_loss_epoch = []
        self.env_steps_per_epoch = []
        
        rewards_epoch_train = []
        log_probs_epoch_train = []
        steps_per_episode = 0
        rewards_episode = []
        log_probs_episode = []
        state, _ = self.env.reset()

        for step in range(self.steps_per_epoch):

            action, log_prob = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)            
            state = next_state

            rewards_episode.append(reward)
            log_probs_episode.append(log_prob)

            steps_per_episode += 1
            self.steps_trained_epoch += 1
            self.steps_trained_total += 1

            if done or (step == self.steps_per_epoch-1):
                if done or len(self.rewards_epoch) == 0:        
                    self.rewards_epoch.append(np.sum(rewards_episode))
                self.env_steps_per_epoch.append(steps_per_episode)
                if step == self.steps_per_epoch-1:
                    rewards_epoch_train.append(torch.tensor(rewards_episode, dtype=torch.float32))
                else:
                    rewards_epoch_train.append(self.discounted_rewards(rewards_episode))
                log_probs_epoch_train.append(torch.cat(log_probs_episode))
                self.episodes_trained_total += 1            
                self.episodes_trained_epoch += 1
                steps_per_episode = 0
                rewards_episode = []
                log_probs_episode = []
                state, _ = self.env.reset()

        policy_loss_episode = self.update_policy(self.normalize_rewards(rewards_epoch_train), torch.cat(log_probs_epoch_train))
        
        return deepcopy({
                         "steps_trained_total": self.steps_trained_total,
                         "circuit_executions": self.circuit_executions,
                         "episode_reward_mean": np.mean(self.rewards_epoch),
                         "episode_reward_max": np.max(self.rewards_epoch),
                         "episode_reward_min": np.min(self.rewards_epoch),
                         "env_steps_per_epoch": self.env_steps_per_epoch,
                         "episode_length_mean": np.mean(self.env_steps_per_epoch),
                         "mean_env_steps_per_episode": np.mean(self.env_steps_per_epoch),
                         "policy_loss_epoch": policy_loss_episode,
                         "steps_trained_epoch": self.steps_trained_epoch,
                         "episodes_trained_total": self.episodes_trained_total,
                         "episodes_trained_epoch": self.episodes_trained_epoch,
                         "num_env_steps_sampled": self.steps_trained_total,
                         "total_goal_states": self.env.total_goal_states
                         })

    def save_checkpoint(self, checkpoint_dir):
        pass


# logits, _ = self.model.forward(obs)
            # prob = torch.softmax(logits).cpu().numpy()[0]
            # action = np.random.choice(self.env.action_space, p=prob)

            # next_obs, rewards, done, truncations, infos = self.env.step(action)

            # # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            # obs = next_obs

            # episode_reward += rewards
            # if done:
            #     episodes_trained +=1
            #     # approximation_ratio = float(episode_reward)/self.env.optimal_cost_
            #     episode_reward_list.append(episode_reward)
            #     # episode_approximation_ratio_list.append(approximation_ratio)
            #     # print(global_step, approximation_ratio, episode_actions)
            # # ALGO LOGIC: training.

            #     sample_obs, sample_actions, sample_rewards, sample_next_obs, sample_dones = rb.sample(args.batch_size)
            #     sample_obs = stack_dict_values(sample_obs)
            #     sample_next_obs = stack_dict_values(sample_next_obs)

            #     # with torch.no_grad():
            #     #     target_values, _ = target_network.forward(sample_next_obs)
            #     #     target_max = target_values.max(dim=1).values.detach().numpy()
            #     #     td_target = (sample_rewards.flatten() + args.gamma * target_max * (1 - sample_dones.flatten())).astype(np.float32)
            #     # val, _ = q_network.forward(sample_obs)
            #     # val_online = val.gather(1,torch.from_numpy(sample_actions).unsqueeze(1)).squeeze()
            #     # loss = F.mse_loss(torch.from_numpy(td_target), val_online)
            #     # Create an action distribution object.
            #     action_dist = dist_class(dist_inputs, model)

            #     # Calculate the vanilla PG loss based on:
            #     # L = -E[ log(pi(a|s)) * A]
            #     log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
            #     # optimize the model
            #     self.optimizer.zero_grad()
            #     self.loss.backward()
            #     self.optimizer.step()


                
        # print(self.schedulder.get_last_lr())
        # self.schedulder.step()
        # print('epoch mean:', epoch, np.mean(episode_approximation_ratio_list[-episodes_trained:]))