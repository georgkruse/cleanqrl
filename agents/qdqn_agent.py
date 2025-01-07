# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy

# this file has been mainly adapted from https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy

import random
from ray import tune 
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from torch.optim.lr_scheduler import LinearLR
from agents.qdqn_model import QuantumDQN_Model
from agents.replay_buffer import ReplayBuffer

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def stack_dict_values(list_of_dicts):
    # Get all unique keys from all dictionaries
    all_keys = set().union(*list_of_dicts)
    
    # Create a new dictionary with lists of values for each key
    return {key: torch.cat([d[key] for d in list_of_dicts if key in d], dim=0) 
            for key in all_keys if any(key in d for d in list_of_dicts)}

class QDQN(tune.Trainable):
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

    def setup(self, config: dict):
      
        from utils.config.create_env import wrapper_switch
        self.config = config['model']['custom_model_config']
        self.env = wrapper_switch[config['env_config']['env']](config['env_config'])
        # self.env = FrozenLake_Wrapper(config['env_config'])

        self.model = QuantumDQN_Model(self.env.observation_space, self.env.action_space, self.env.action_space.n, config['model'], 'name')
        self.target_network = QuantumDQN_Model(self.env.observation_space, self.env.action_space, self.env.action_space.n, config['model'], 'name')
        self.target_network.load_state_dict(self.model.state_dict())

        self.optimizer = self.create_optimizer()[0]
        self.schedulder = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-3, total_iters=5)

        self.episodes_trained_total = 0
        self.steps_trained_total = 0
        self.global_step = 0
        self.circuit_executions = 0
        self.steps_per_epoch = self.config['steps_per_epoch']
        self.gamma =self.config['gamma']

        self.inital_epsilon = self.config['exploration_config']['initial_epsilon']
        self.final_epsilon = self.config['exploration_config']['final_epsilon']
        self.epsilon_timesteps = self.config['exploration_config']['epsilon_timesteps']

        self.learning_starts = self.config['num_steps_sampled_before_learning_starts']
        self.train_frequency = 10
        self.train_batch_size = self.config['train_batch_size']

        self.target_network_update_freq = self.config['target_network_update_freq']
        self.tau = self.config['tau']

        self.replay_buffer = ReplayBuffer(size=self.config['replay_buffer_config']['capacity'])




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
    
    def select_action(self, state):
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        epsilon = linear_schedule(self.inital_epsilon, self.final_epsilon, self.epsilon_timesteps, self.global_step)
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            q_values, _ = self.model.forward({'obs':state})
            self.circuit_executions += 1
            action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
       
        return action

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

            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)           
            self.replay_buffer.add(state, action, reward, next_state, done)
 
            state = next_state

            rewards_episode.append(reward)

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
                
                self.episodes_trained_total += 1            
                self.episodes_trained_epoch += 1
                steps_per_episode = 0
                rewards_episode = []
                log_probs_episode = []
                state, _ = self.env.reset()

            if self.global_step > self.learning_starts:
                if self.global_step % self.train_frequency == 0:
                    sample_obs, sample_actions, sample_rewards, sample_next_obs, sample_dones = self.replay_buffer.sample(self.train_batch_size)

                    with torch.no_grad():
                        target_values, _ = self.target_network.forward({'obs':torch.tensor(sample_next_obs)})
                        self.circuit_executions += self.train_batch_size
                        target_max = target_values.max(dim=1).values.detach().numpy()
                        td_target = (sample_rewards.flatten() + self.gamma * target_max * (1 - sample_dones.flatten())).astype(np.float64)
                    
                    values, _ = self.model.forward({'obs':torch.tensor(sample_obs)})
                    self.circuit_executions += self.train_batch_size*self.model.total_model_parameters
                    values_action = values.gather(1,torch.from_numpy(sample_actions).unsqueeze(1)).squeeze()
                    loss = F.mse_loss(torch.from_numpy(td_target), values_action)

                    # optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if self.global_step % self.target_network_update_freq == 0:
                for target_network_param, q_network_param in zip(self.target_network.parameters(), self.model.parameters()):
                    target_network_param.data.copy_(
                        self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                    )
            self.global_step += 1

        return deepcopy({
                         "steps_trained_total": self.steps_trained_total,
                         "circuit_executions": self.circuit_executions,
                         "episode_reward_mean": np.mean(self.rewards_epoch),
                         "episode_reward_max": np.max(self.rewards_epoch),
                         "episode_reward_min": np.min(self.rewards_epoch),
                         "env_steps_per_epoch": self.env_steps_per_epoch,
                         "episode_length_mean": np.mean(self.env_steps_per_epoch),
                         "mean_env_steps_per_episode": np.mean(self.env_steps_per_epoch),
                        #  "policy_loss_epoch": policy_loss_episode,
                         "steps_trained_epoch": self.steps_trained_epoch,
                         "episodes_trained_total": self.episodes_trained_total,
                         "episodes_trained_epoch": self.episodes_trained_epoch,
                         "num_env_steps_sampled": self.steps_trained_total,
                         "total_goal_states": self.env.total_goal_states
                         })

    def save_checkpoint(self, checkpoint_dir):
        pass
