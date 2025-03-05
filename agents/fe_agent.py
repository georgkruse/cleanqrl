# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from ray import tune 
import numpy as np
import torch
from copy import deepcopy

import math
from copy import deepcopy
from ray import tune
from neal import SimulatedAnnealingSampler
# from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from collections import defaultdict

import yaml 
from utils.config.create_env import wrapper_switch
from agents.replay_buffer import ReplayBufferQBM

from scipy.sparse import csr_matrix, kron, identity, diags
from scipy.linalg import eigh


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (float(end_e) - float(start_e)) / float(duration)
    return max(slope * t + float(start_e), float(end_e))

def stack_dict_values(list_of_dicts):
    # Get all unique keys from all dictionaries
    all_keys = set().union(*list_of_dicts)
    
    # Create a new dictionary with lists of values for each key
    return {key: torch.cat([d[key] for d in list_of_dicts if key in d], dim=0) 
            for key in all_keys if any(key in d for d in list_of_dicts)}



def compute_rho(H, beta, diagonal=False):
    """
    Computes the trace normalized density matrix rho.

    :param H: Hamiltonian matrix.
    :param beta: Inverse temperature beta = 1 / (k_B * T).
    :param diagonal: Flag to indicate whether H is a diagonal matrix or not.

    :return: Density matrix rho.
    """
    # if diagonal then compute directly, else use eigen decomposition
    if diagonal:
        Lambda = H.diagonal()
        exp_beta_Lambda = np.exp(-beta * (Lambda - Lambda.min()))
        return np.diag(exp_beta_Lambda / exp_beta_Lambda.sum())
    else:
        Lambda, S = eigh(H)
        exp_beta_Lambda = np.exp(-beta * (Lambda - Lambda.min()))
        return (S * (exp_beta_Lambda / exp_beta_Lambda.sum())) @ S.T

def compute_H(h, J, A, B, n_qubits, pauli_kron):
    """
    Computes the Hamiltonian of the annealer at relative time s.

    :param h: Linear Ising terms.
    :param J: Quadratic Ising terms.
    :param A: Coefficient of the off-diagonal terms, e.g. A(s).
    :param B: Coefficient of the diagonal terms, e.g. B(s).
    :param n_qubits: Number of qubits.
    :param pauli_kron: Kronecker product Pauli matrices dict.

    :returns: Hamiltonian matrix H.
    """
    # diagonal terms
    H_diag = np.zeros(2 ** n_qubits)
    for i in range(n_qubits):
        # linear terms
        if h[i] != 0:
            H_diag += (B * h[i]) * pauli_kron["z_diag", i]

        # quadratic terms
        for j in range(i + 1, n_qubits):
            if J[i, j] != 0:
                H_diag += (B * J[i, j]) * pauli_kron["zz_diag", i, j]

    # return just the diagonal if H is a diagonal matrix
    if A == 0:
        return np.diag(H_diag)

    # off-diagonal terms
    H = csr_matrix((2 ** n_qubits, 2 ** n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        H -= A * pauli_kron["x", i]

    return (H + diags(H_diag, format="csr")).toarray()



def execute_dwave(h, J, config, sampler, path, sample_index, embedding_path, layers):

	sampler_kwargs = {
		'num_reads': config.num_reads,
		'reduce_intersample_correlation': config.reduce_intersample_correlation,
		'programming_thermalization': config.programming_thermalization,
		'readout_thermalization': config.readout_thermalization,
		'auto_scale': config.auto_scale,
		'answer_mode': config.answer_mode,
		'return_embedding': config.return_embedding
	}


	# apply a random gauge
	if config.use_gauge:
		h = deepcopy(h)
		J = deepcopy(J)
		gauge = np.random.choice([-1, 1], J.shape[0])
		h *= gauge
		J *= np.outer(gauge, gauge)

	chain_strength = config.prefactor_chain_strength
	chain_strength *= max(np.abs(h).max(), np.abs(J).max())
	chain_strength = min(chain_strength, 1)
	sampler_kwargs['chain_strength'] = chain_strength
	start = time.time()
	sample_set = sampler.sample_ising(h=h, J=J, **sampler_kwargs)
	print(time.time() - start)
	
	# undo the gauge
	if config.use_gauge:
		sample_set.record.sample *= gauge

	if config.return_embedding:
		embedding_data = {}
		embedding_data['embedding'] = dict(sample_set.info['embedding_context']['embedding'])
		embedding_data['chain_strength'] = float(sample_set.info['embedding_context']['chain_strength'])
		embedding_data['chain_break_method'] = str(sample_set.info['embedding_context']['chain_break_method'])
		embedding_data['embedding_parameters'] = dict(sample_set.info['embedding_context']['embedding_parameters'])

		np.save(f"{embedding_path}/{config['embedding_id']}.npy", embedding_data)
		del sample_set.info['embedding_context']

	if sample_index != 0:
		np.save(f'{path}/sample_set_{sample_index}.npy', sample_set)

	return sample_set


class QBM(tune.Trainable):

	def setup(self, config: dict):

		self.config = config['alg_config']
		self.env = wrapper_switch[config['env_config']['env']](config['env_config'])

	# def __init__(self, config: dict):

	# 	self.config = config._asdict() #['model']['custom_model_config']
	# 	self.env = wrapper_switch[self.config['env_config']['env']](self.config['env_config'])
	# 	self.config = self.config['algorithm_config']
		self.episodes_trained_total = 0
		self.steps_trained_total = 0
		self.global_step = 0        
		self.circuit_executions = 0
		# self.schedulder = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-3, total_iters=5)
		self.steps_per_epoch = self.config['steps_per_epoch']
		self.gamma = self.config['gamma']

		self.inital_epsilon = self.config['exploration_config']['initial_epsilon']
		self.final_epsilon = self.config['exploration_config']['final_epsilon']
		self.epsilon_timesteps = self.config['exploration_config']['epsilon_timesteps']

		self.learning_starts = self.config['num_steps_sampled_before_learning_starts']
		self.train_frequency = 10
		self.train_batch_size = self.config['train_batch_size']

		self.target_network_update_freq = self.config['target_network_update_freq']
		self.tau = self.config['tau']

		self.replay_buffer = ReplayBufferQBM(size=self.config['replay_buffer_config']['capacity'])
	
		self.layers = self.config['layers']
		self.total_qubits = int(sum(self.layers))
		self.mean = self.config['mean']
		self.variance = self.config['variance']

		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		
		self.hamiltonian_type = self.config['hamiltonian_type']
		self.sampler_type = self.config['sampler_type']
		self.embedding_path = self.config['embedding_path']
		self.big_gamma = self.config['big_gamma']
		self.beta = self.config['beta']
		self.lr = self.config['lr']
		self.small_gamma = self.config['small_gamma']
		self.eps = 1e-5

		if 'H_(d+1)' in self.hamiltonian_type:
			if self.hamiltonian_type[-2] == '_': 
				self.num_replicas = int(self.hamiltonian_type[-1])
			else:
				self.num_replicas = int(self.hamiltonian_type[-2:])
		else:
			self.num_replicas = 1

		# if self.hamiltonian_type == 'H':
		# 	self.total_qubits = self.w_hh.shape[0]
		
		self.init_weights()		

		if self.sampler_type in ['dwave-sim', 'dwave-sim-SA','dwave-sim-SQA', 'dwave-sim-SA-H-eff']:
			self.sampler = SimulatedAnnealingSampler()

		elif self.sampler_type  in ['dwave-qpu', 'dwave-qpu-SA','dwave-qpu-SQA']:
			with open('/home/users/kruse/Hamiltonian-based-QRL/token.yaml') as f:
				token_file = yaml.load(f, Loader=yaml.FullLoader)
			token = token_file['token']

			qpu = DWaveSampler(solver=dict(topology__type='pegasus', name='Advantage_system6.4'), token=token)
			if not config['fixed_embedding']:
				self.sampler = EmbeddingComposite(qpu, embedding_parameters={'max_no_improvement': config['embedding']['max_no_improvement'],
											'random_seed': config['embedding']['random_seed']})
			else:
				embedding = np.load(f"{self.embedding_path}/{self.layers}/{self.hamiltonian_type}/{config['embedding_id']}.npy", allow_pickle=True).item()
				self.sampler = FixedEmbeddingComposite(qpu, embedding['embedding'])

	
	def generate_ising_model(self, visible_nodes):
	
		if 'H_(d+1)' in self.hamiltonian_type:
			w_plus = math.log10(math.cosh(self.big_gamma*self.beta/self.num_replicas)/math.sinh(self.big_gamma*self.beta/self.num_replicas))/(2*self.beta)
			
			J = np.zeros((self.w_hh.shape[0]*self.num_replicas,self.w_hh.shape[1]*self.num_replicas))
			for k in range(self.num_replicas):
				for i in range(self.w_hh.shape[0]):
					for j in range(self.w_hh.shape[1]):
						if j > i:
							J[i+self.w_hh.shape[0]*k,j+self.w_hh.shape[0]*k] = self.w_hh[i,j]/self.num_replicas

			for i in range(self.w_vh.shape[1]):
				for k in range(self.num_replicas-1):
					J[i+(k*self.w_vh.shape[1]), i+((k+1)*self.w_vh.shape[1])] = w_plus
				J[i, i+((self.num_replicas-1)*self.w_vh.shape[1])] = w_plus
			
			h = np.zeros(self.w_vh.shape[1]*self.num_replicas)
			for k in range(self.num_replicas):
				for i in range(self.w_vh.shape[1]):
					for j in range(visible_nodes.shape[0]):
						h[i+self.w_vh.shape[1]*k] += (self.w_vh[j,i]*visible_nodes[j])/self.num_replicas
		
		elif self.hamiltonian_type == 'H':
			J = self.w_hh.copy()
			h = np.zeros(self.total_qubits)
			for i in range(self.w_vh.shape[0]):
				for j in range(self.w_vh.shape[1]):
					h[j] += self.w_vh[i,j]*visible_nodes[i]
		
		# substract small value in order to cope with dwave instabilities
		h = np.clip(h, -4.0+self.eps, 4.0-self.eps)
		J = np.clip(J, -1.0+self.eps, 1.0-self.eps)

		return h, J 
	
	def calculate_entropy(self, samples):
		random.shuffle(samples)
		key_list = np.arange(samples[0].shape[0]) 
		a_sum = 0
		div_factor = samples.shape[0] // self.num_replicas
		prob_dict = defaultdict(int)

		for i_sample in range(0,samples.shape[0],self.num_replicas):
			c_iterable = list()
			for s in samples[i_sample : i_sample + self.num_replicas]:
				for k in key_list:
					c_iterable.append(s[k])
			prob_dict[tuple(c_iterable)] +=1

		for c in prob_dict.values():
			a_sum += c * np.log10( c / div_factor ) / div_factor

		entropy = a_sum / self.beta
		return entropy 
	
	def get_free_energy_and_samples(self, visible_nodes):

		h, J = self.generate_ising_model(visible_nodes)
		samples, energies = self.execute_sampler(h, J)
		entropy = self.calculate_entropy(samples)
		average_hamiltonian_energy = - np.mean(energies)
		F = (average_hamiltonian_energy + entropy) / 1

		return F, average_hamiltonian_energy, entropy, samples

	def init_weights(self):
		
		# Initialize the adjacency matrix with zeros
		self.w_hh = np.zeros((self.total_qubits, self.total_qubits))
		self.w_vh = np.zeros(self.total_qubits)

		# Fully connected if only one layer is specified
		if len(self.layers) == 1:
			upper_indices = np.triu_indices(self.total_qubits, k=1)
			self.w_hh[upper_indices] = np.random.normal(self.mean, self.variance, len(upper_indices[0]))
			self.w_vh = np.random.normal(self.mean, self.variance, (self.action_size+self.state_size, self.total_qubits))

		else:
			for qubits_layer_0 in range(self.layers[0]):
				for qubits_layer_1 in range(self.layers[0], self.layers[0]+self.layers[1]):
					self.w_hh[qubits_layer_0, qubits_layer_1] = np.random.normal(self.mean, self.variance)
			if len(self.layers) == 3:
				for qubits_layer_1 in range(self.layers[0], self.layers[0]+self.layers[1]):
					for qubits_layer_2 in range(sum(self.layers[:2]), sum(self.layers)):
						self.w_hh[qubits_layer_1, qubits_layer_2] =  np.random.normal(self.mean, self.variance)
			
			self.w_vh = np.zeros((self.state_size+self.action_size,self.total_qubits))

			for state_qubit in range(self.state_size):
				for qubits_layer_0 in range(self.layers[0]):
					self.w_vh[state_qubit, qubits_layer_0] = np.random.normal(self.mean, self.variance)
			if len(self.layers) == 2:
				for action_qubit in range(self.state_size, self.state_size+self.action_size):
					for qubits_layer_1 in range(self.layers[0], self.layers[0]+self.layers[1]):
						self.w_vh[action_qubit, qubits_layer_1] = np.random.normal(self.mean, self.variance)
			if len(self.layers) == 3:
				for action_qubit in range(self.state_size, self.state_size+self.action_size):
					for qubits_layer_2 in range(sum(self.layers[:2]), sum(self.layers)):
						self.w_vh[action_qubit, qubits_layer_2] = np.random.normal(self.mean, self.variance)
			


	def update_weights(self, next_state, samples, reward, free_energy, visible_nodes):
		
		# for idx in range(self.train_batch_size):
		# 	next_state = sample_next_obs[idx]
		# 	samples = sample_samples[idx]
		# 	reward = sample_rewards[idx]
		# 	free_energy = sample_free_energy[idx]
		# 	visible_nodes = sample_visible_nodes[idx]
		# free_energy_next_timestep = None
		samples = np.mean(np.reshape(samples, (samples.shape[0], self.num_replicas, -1)), axis=1)

		for action_next_time_step in self.env.legal_actions(next_state):
			next_state, reward_next_time_step, done, _, infos = self.env.step(action_next_time_step, next_state)
			
			agent_index = np.where(next_state==self.env.A)
			states_array = -1*np.ones(next_state.shape[0])
			action_array = -1*np.ones(self.env.action_space.n)
			states_array[agent_index] = 1
			action_array[action_next_time_step] = 1
			visible_nodes_next_time_step = np.concatenate([states_array, action_array])
			free_energy_next_timestep, average_hamiltonian_energy, entropy, samples_next_time_step = self.get_free_energy_and_samples(visible_nodes=visible_nodes_next_time_step)

			# if free_energy_next_timestep is None or free_energy_next_timestep < free_energy_next_timestep_tmp:
			# 	free_energy_next_timestep = free_energy_next_timestep_tmp
		
			n = self.w_hh.shape[0]
			upper_triangle_indices = np.triu_indices(n, k=1)
			probabilities = np.zeros((self.w_hh.shape[0], self.w_hh.shape[0]))

			probabilities[upper_triangle_indices] += np.sum(samples[:,upper_triangle_indices[0]]*samples[:,upper_triangle_indices[1]]/samples.shape[0], axis=0)
			probabilities += np.diag(np.sum(samples, axis=0)/samples.shape[0])
			update = self.lr * (reward + (self.small_gamma * free_energy_next_timestep - free_energy))

			self.w_hh[upper_triangle_indices] -= update * probabilities[upper_triangle_indices]
			for i in range(self.w_vh.shape[1]):
				for j in range(visible_nodes.shape[0]):
					self.w_vh[j,i] -= update * visible_nodes[j] * probabilities[i,i]


	def execute_sampler(self, h, J):

		start = time.time()
		n = J.shape[0]
		upper_triangle_indices = np.triu_indices(n, k=1)
		J_dict = {}
		for i, j in zip(upper_triangle_indices[0], upper_triangle_indices[1]):
			J_dict[(i,j)] = J[i,j]
		h_dict = {}
		for i in range(h.shape[0]):
			h_dict[i] = h[i] 


		if self.sampler_type in ['dwave-qpu', 'dwave-qpu-SA','dwave-qpu-SQA']:
			dwave_path = path + f'/{layers}/{hamiltonian_type}/dwave'
			embedding_path_layers = config.embedding_path + f'/{layers}/{hamiltonian_type}'
			os.makedirs(dwave_path, exist_ok=True)
			os.makedirs(embedding_path_layers, exist_ok=True)
			samples_dwave = []
			dwave_index = 0
			# for dwave_idx in range(config['dwave_samples_per_sample']):
			# 	dwave_index = f'{sample_index}_dwave_iter_{dwave_idx}'
			sample_set = execute_dwave(h, J, config, sampler, dwave_path, dwave_index, embedding_path_layers, layers)
			samples_dwave.append(sample_set.record.sample)
			samples = np.vstack(samples_dwave)


		elif self.sampler_type in ['dwave-sim', 'dwave-sim-SA','dwave-sim-SQA', 'dwave-sim-SA-H-eff']:
			sample_set = self.sampler.sample_ising(h=h_dict, J=J_dict, **self.config['simulator_config'])
			self.circuit_executions += 1
			samples_ = list(sample_set.samples())
			num_samples = len(samples_)
			num_nodes = len(samples_[0].values())
			samples = np.zeros((num_samples, num_nodes), dtype=int)
			for i, sample in enumerate(samples_):
				for node, value in sample.items():
					samples[i, node] = value
			samples = np.stack(samples)
			random.shuffle(samples)
		elif self.sampler_type in ['state-vector']:
			H = compute_H(h, J, A, B, qubits, pauli_kron)
			rho = compute_rho(H, beta, diagonal=(A == 0))
			distributions[s_value, T] = {
				"E": H.diagonal().copy(),
				"p": rho.diagonal().copy(),
				"E_total": sum(H.diagonal()*rho.diagonal())
			}
		energies = sample_set.data_vectors['energy']
		return samples, energies


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
    
    
	def make_action(self, state):
		data_tuple = None
		# current_obs = deepcopy(np.reshape(state, (3,3)))
		# new_obs = np.zeros((3,3))
		# agent_index_y, agent_index_x = np.where(current_obs==1.0)
		# y, x = agent_index_y[0], agent_index_x[0] #np.unravel_index(np.argmax(obs), obs.shape)
		#state = torch.from_numpy(state).float() #.unsqueeze(0)
		epsilon = linear_schedule(self.inital_epsilon, self.final_epsilon, self.epsilon_timesteps, self.global_step)
		if random.random() < epsilon:
			action = np.random.choice(self.env.legal_actions(state))
			next_state, reward, done, _,  infos = self.env.step(action, state)
			# reward = reward*100
			agent_index = np.where(state==self.env.A)
			states_array = -1*np.ones(state.shape[0])
			action_array = -1*np.ones(self.env.action_space.n)
			states_array[agent_index] = 1
			action_array[action] = 1
			visible_nodes = np.concatenate([states_array, action_array])
			free_energy, average_hamiltonian_energy, entropy, samples = self.get_free_energy_and_samples(visible_nodes=visible_nodes)
			data_tuple = (free_energy, action, samples, visible_nodes, reward, next_state, done)
			# if action == 0:
			# 	y += 1
			# elif action == 1:
			# 	y -= 1
			# elif action == 2:
			# 	x += 1
			# elif action == 3:
			# 	x -= 1
			
			# new_obs[y,x] = free_energy
		else:

			for action in self.env.legal_actions(state):
				
				# agent_index_y, agent_index_x = np.where(current_obs==1.0)
				# y, x = agent_index_y[0], agent_index_x[0] #np.unravel_index(np.argmax(obs), obs.shape)
				#current_obs[y, x] = 0

				next_state, reward, done, _, infos = self.env.step(action, state)
				
				# reward = reward*100

				agent_index = np.where(state==self.env.A)
				states_array = -1*np.ones(state.shape[0])
				action_array = -1*np.ones(self.env.action_space.n)
				states_array[agent_index] = 1
				action_array[action] = 1
				visible_nodes = np.concatenate([states_array, action_array])

				free_energy, average_hamiltonian_energy, entropy, samples = self.get_free_energy_and_samples(visible_nodes=visible_nodes)

				if data_tuple is None or data_tuple[0] < free_energy:
					data_tuple = deepcopy((free_energy, action, samples, visible_nodes, reward, next_state, done))
			
				# if action == 0:
				# 	y += 1
				# elif action == 1:
				# 	y -= 1
				# elif action == 2:
				# 	x += 1
				# elif action == 3:
				# 	x -= 1
				
		# 		new_obs[y,x] = free_energy
		# print(new_obs)
		# print(data_tuple[1])
		# if self.global_step % 100 == 0:
		# 	x = 0
		# 	pass
		return deepcopy(data_tuple)


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
		if 'learning_rate_schedule' in self.config.keys():
			self.lr = linear_schedule(*self.config['learning_rate_schedule'], self.global_step)
			print(self.lr)
		# if hasattr(config, 'gamma_schedule'):
		# 	big_gamma = linear_schedule(config.gamma_schedule, idx)
		for step in range(self.steps_per_epoch):
			# print(step)
			free_energy, action, samples, visible_nodes, reward, next_state, done = self.make_action(deepcopy(state))

			self.replay_buffer.add(state, action, reward, next_state, done, free_energy, samples, visible_nodes)
			# print('########################')
			# print(np.reshape(state, (3,3)))
			# print(action)
			# print(np.reshape(next_state, (3,3)))

			state = next_state

			# sample_obs, sample_actions, sample_rewards, sample_next_obs, sample_dones, \
			# sample_free_energy, sample_samples, sample_visible_nodes = self.replay_buffer.sample(1)
			# # next_state, samples, reward, free_energy, visible_nodes
			# self.update_weights(sample_next_obs, sample_samples, sample_rewards, sample_free_energy, sample_visible_nodes)

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
					for batch in range(self.train_batch_size):
						sample_obs, sample_actions, sample_rewards, sample_next_obs, sample_dones, \
						sample_free_energy, sample_samples, sample_visible_nodes = self.replay_buffer.sample(1)
						# next_state, samples, reward, free_energy, visible_nodes
						self.update_weights(sample_next_obs, sample_samples, sample_rewards, sample_free_energy, sample_visible_nodes)

			# if self.global_step % self.target_network_update_freq == 0:
			# 		for target_network_param, q_network_param in zip(self.target_network.parameters(), self.model.parameters()):
			# 			target_network_param.data.copy_(
			# 				self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
			# 			)
			self.global_step += 1

		if hasattr(self.config, 'learning_rate_schedule'):
			self.lr = linear_schedule(self.config['learning_rate_schedule'], self.global_step)
		if hasattr(self.config, 'gamma_schedule'):
			self.big_gamma = linear_schedule(self.config['gamma_schedule'], self.global_step)
		
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

