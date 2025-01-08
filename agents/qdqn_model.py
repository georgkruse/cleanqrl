from abc import ABC
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import pennylane as qml
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Type, Union

from circuits.postprocessing import *
from circuits.quantum_circuits import vqc_generator

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Dict

        
class QuantumDQN_Model(TorchModelV2,nn.Module,ABC):
    def __init__(self,obs_space,action_space,num_actions,config,name):
        TorchModelV2.__init__(
            self, obs_space,action_space,num_actions,config,name
        )
        nn.Module.__init__(self)
        self.counter = -3
        self.reset = True
        self.config = config
        self.mode = self.config['mode']
        self.num_params = self.config['num_variational_params']

        if isinstance(self.action_space, Box):
            self.num_outputs = self.action_space.shape[0]*2
        elif isinstance(self.action_space, Discrete):
            self.num_outputs = self.action_space.n
        elif isinstance(self.action_space, MultiBinary):
            self.num_outputs = self.action_space.n
        elif isinstance(self.action_space, MultiDiscrete): 
            self.num_outputs = np.sum(action.n for action in self.action_space)

        if isinstance(self.obs_space, gym.spaces.Dict):
            self.num_inputs = 5 #sum([box.shape[0] for box in self.obs_space.values()])
        else:
            self.num_inputs = self.obs_space.shape[0]

        self.init_variational_params = self.config['init_variational_params']
        self.measurement_type_actor = self.config['measurement_type_actor']
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_actions = num_actions
        self.config['num_layers'] = int(self.config['num_layers'])
        self.num_layers = int(self.config['num_layers'])
        self.num_qubits = int(self.config['vqc_type'][1])
        self.config['num_qubits'] = self.num_qubits
        self.layerwise_training = self.config['layerwise_training']
        self.gradient_clipping = self.config['gradient_clipping']
        self.use_input_scaling = self.config['use_input_scaling']
        self.use_output_scaling_actor = self.config['use_output_scaling_actor']
        self.init_output_scaling_actor = self.config['init_output_scaling_actor']
        self.num_scaling_params = self.config['num_scaling_params']
        self.action_masking = self.config['action_masking']

        self.use_classical_layer = self.config['use_classical_layer']        
        self.layer_size = self.config['layer_size']

        self.weight_logging_interval = self.config['weight_logging_interval']
        self.weight_plotting = self.config['weight_plotting']
        self._value_out = None

        def init_weights(size):
            if self.config['init_variational_params_mode'] == 'plus-zero-uniform':
                return torch.FloatTensor(*size,).uniform_(0, self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'plus-minus-uniform':
                return torch.FloatTensor(*size,).uniform_(-self.init_variational_params, self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'plus-zero-normal':
                return torch.FloatTensor(*size,).normal_(0., self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'constant':
                return torch.tensor(np.full((*size,), self.init_variational_params))

        if self.mode == 'quantum': 

            if self.config['encoding_type'] == 'graph_encoding':
            
                if self.config['graph_encoding_type'] in ['sge-sgv', 'sge-sgv-linear', 'sge-sgv-quadratic']:
                    size_vqc = 1
                    size_input_scaling = 1
                elif self.config['graph_encoding_type'] == 'mge-mgv':
                    size_vqc = self.num_qubits
                    size_input_scaling = sum(range(self.num_qubits+1))+self.num_qubits
                elif self.config['graph_encoding_type'] == 'mge-mgv-linear':
                    size_vqc = self.num_qubits
                    size_input_scaling = self.num_qubits
                elif self.config['graph_encoding_type'] == 'mge-mgv-quadratic':
                    size_vqc = self.num_qubits
                    size_input_scaling = sum(range(self.num_qubits+1))
                elif self.config['graph_encoding_type'] == 'mge-sgv':
                    size_vqc = 1
                    size_input_scaling = sum(range(self.num_qubits+1))+self.num_qubits
                elif self.config['graph_encoding_type'] == 'mge-sgv-linear':
                    size_vqc = 1
                    size_input_scaling = self.num_qubits + 1
                elif self.config['graph_encoding_type'] == 'mge-sgv-quadratic':
                    size_vqc = 1
                    size_input_scaling = sum(range(self.num_qubits+1)) + 1
                elif self.config['graph_encoding_type'] in ['angular', 'angular-hea']:
                    size_vqc = self.num_qubits*self.num_params
                    size_input_scaling = self.num_qubits*self.config['num_scaling_params']
                elif self.config['graph_encoding_type'] == 'hamiltonian-hea':
                    size_vqc = self.num_qubits*self.num_params
                    size_input_scaling = 0
                if self.config['block_sequence'] in ['enc_var_ent', 'enc_var', 'enc_ent_var']:
                    size_vqc += self.num_qubits*self.num_params
            else:
                size_vqc = self.num_qubits*self.num_params
                size_input_scaling = self.num_qubits*self.config['num_scaling_params']    
            
            self.total_model_parameters = 0

            self.register_parameter(name=f'weights_actor', param=torch.nn.Parameter(init_weights((self.num_layers, size_vqc)), requires_grad=True))
            
            self.total_model_parameters += self.num_layers * size_vqc
            
            if self.use_input_scaling:
                self.register_parameter(name=f'input_scaling_actor', param=torch.nn.Parameter(torch.tensor(np.full((self.num_layers, size_input_scaling), 1.)), requires_grad=True))
            
                self.total_model_parameters += self.num_layers * size_input_scaling

            if self.use_output_scaling_actor:
                if isinstance(self.init_output_scaling_actor, list):
                    self.register_parameter(name='output_scaling_actor', param=torch.nn.Parameter(torch.tensor(np.full(self.num_outputs, self.config['init_output_scaling_actor'][0])), requires_grad=True))
                    self.total_model_parameters += self.num_outputs
                else:
                    self.register_parameter(name='output_scaling_actor', param=torch.nn.Parameter(torch.tensor(self.config['init_output_scaling_actor']), requires_grad=True))
                    self.total_model_parameters += 1

            if self.gradient_clipping:
                self.weights_actor.register_hook(lambda grad: torch.clip(grad, -1., 1.))
                                
            dev_actor = qml.device(self.config['backend_name'], wires=self.num_qubits)

            self.qnode_actor = qml.QNode(vqc_generator, dev_actor, interface=self.config['interface'], diff_method=self.config['diff_method']) #, argnum=0)

            if self.use_classical_layer:
                self.classical_layer_actor = nn.Linear(in_features=self.num_inputs, out_features=self.num_actions, dtype=torch.float32)
           
        elif self.mode == 'classical':
            if self.config['activation_function'] == 'relu':
                activation = nn.ReLU()
            elif self.config['activation_function'] == 'leaky_relu':
                activation = nn.LeakyReLU()
            if self.config['activation_function'] == 'tanh':
                activation = nn.Tanh()
            if len(self.layer_size) == 1:
                self.actor_network = nn.Sequential(nn.Linear(in_features=self.num_inputs, out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.num_outputs))
                
            elif len(self.layer_size) == 2:
                
                self.actor_network = nn.Sequential(nn.Linear(in_features=self.num_inputs, out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.layer_size[1]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[1], out_features=self.num_outputs))
                
            elif len(self.layer_size) == 3:
                self.actor_network = nn.Sequential(nn.Linear(in_features=self.num_inputs, out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.layer_size[1]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[1], out_features=self.layer_size[2]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[2], out_features=self.num_outputs))
                
                
    
    def forward(self, input_dict, state=None, seq_lens=None):
        
        state = input_dict['obs']


        if self.mode == 'quantum':

            # Check the encoding block type in order to adapt obs/state accordingly
            if 'double' in self.config['vqc_type'][0]:
                state = torch.concat([state, state], dim=1)
            elif 'triple' in self.config['vqc_type'][0]:
                state = torch.concat([state, state, state], dim=1)
            elif 'circular' in self.config['vqc_type'][0]:
                reps = round((self.num_qubits*self.num_layers)/state.shape[1] + 0.5)
                state = torch.concat([state for _ in range(reps)], dim=1)
            else:
                if not isinstance(self.obs_space, gym.spaces.Dict): 
                    state = torch.reshape(state, (-1, self.obs_space.shape[0]))

            if self.config['measurement_type_actor'] == 'edge':
                prob = []
                for i in range(state['quadratic_0'].shape[1]):
                    if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                        self.config['edge_measurement'] = state['quadratic_0'][0,i,:2]
                    else:
                        self.config['edge_measurement'] = [torch.tensor(0.), torch.tensor(1.)]

                    prob.append(self.qnode_actor(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None))

                q_values = torch.reshape(torch.stack(prob).T, (-1, state['quadratic_0'].shape[1]))
            else:
                q_values = self.qnode_actor(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None)
                # Higher Pennylane Version
                if state.shape[0] == 1:
                    q_values = torch.hstack(q_values)
                else:
                    q_values = torch.stack(q_values).T
            if self.use_output_scaling_actor:
                logits = torch.reshape(postprocessing(q_values, self.config, self.num_outputs, self._parameters, 'actor'), (-1, self.num_outputs))
            else:
                logits = q_values

        elif self.mode == 'classical':

            if isinstance(self.action_space, MultiBinary):
                logits = []
                action = torch.tensor([0.]).repeat(state.shape[0], 1)
                for i in range(len(self.action_space)):
                    action_one_hot = F.one_hot(torch.tensor([i]), num_classes=len(self.action_space))
                    action_one_hot = action_one_hot[0].repeat(state.shape[0], 1)
                    input_actor = torch.concatenate([state, action_one_hot, action], dim=1)
                    probs = self.actor_network(input_actor)
                    action = torch.reshape(torch.argmax(probs, dim=1), (-1, 1))
                    action_mask = F.one_hot(torch.argmax(probs, dim=1), num_classes=2)
                    logits.append(probs*action_mask)
                logits = torch.hstack(logits)

            elif isinstance(self.action_space, MultiDiscrete):
                linear = state['linear_0'][:,:,1]
                quadratic = state['quadratic_0'][:,:,1]
                input_state = torch.from_numpy(np.concatenate([linear, quadratic], axis=1))
                logits = self.actor_network(input_state)
            
            else:
                logits = self.actor_network(state)
                self._logits = logits
        self.counter += 1

        if self.config['output_scaling_schedule']:
            self.counter += 1
            if self.counter % 2000 == 0:
                if self.counter <= 75000:
                    if logits.shape[0] == 1.:
                        self._parameters['output_scaling_actor'] += 0.5
                    else:
                        self.counter -= 1
                # print('output_scaling_actor:', self._parameters['output_scaling_actor'])

        elif self.config["problem_scaling"]:
            if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                nodes = state["scale_qvalues"].shape[1]
                batch_dim = state["scale_qvalues"].shape[0]
                for batch in range(batch_dim):
                    for node in range(nodes):
                        logits[batch][node] *= state["scale_qvalues"][batch][node][1]


        if isinstance(self.obs_space, Dict):
            if "annotations" in state.keys():
                if 'current_node' in state.keys():
                    if state['current_node'][0].to(torch.int).item() != -1:
                        if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                            new_logits_batch = []
                            
                            for batch_dim in range(state["quadratic_0"].shape[0]):
                                new_logits = []
                                weights_comp = []
                                for idx, (node1, node2, value) in enumerate(state['quadratic_0'][batch_dim]):
                                    node1 = int(node1)
                                    node2 = int(node2)
                                    if node1 == state['current_node'][batch_dim,0]:
                                        if state['annotations'][batch_dim,node2,1] == np.pi:
                                            # new_logits.append(logits[batch_dim,idx]*value)
                                            new_logits.append(logits[batch_dim,idx])
                                            weights_comp.append(value)
                                        elif state['annotations'][batch_dim,node2,1] == 0:
                                            new_logits.append(torch.tensor(-10_000))
                                            weights_comp.append(10_000)
                                    elif node2 == state['current_node'][batch_dim,0]:
                                        if state['annotations'][batch_dim,node1,1] == np.pi:
                                            # logits[idx] = logits[idx]*value
                                            # new_logits.append(logits[batch_dim,idx]*value)
                                            new_logits.append(logits[batch_dim,idx])
                                            weights_comp.append(value)

                                        elif state['annotations'][batch_dim,node1,1] == 0:
                                            # logits[idx] = -10_000
                                            new_logits.append(torch.tensor(-10_000))
                                            weights_comp.append(10_000)
                                    
                                new_logits_batch.append(torch.stack(new_logits))
                            logits = torch.stack(new_logits_batch, dim=0)
                        else:
                            logits = logits[:,:self.action_space.n]

                    elif state['current_node'][0].to(torch.int).item() == -1:
                        nodes = state["annotations"].shape[1]
                        batch_dim = state["annotations"].shape[0]
                        if batch_dim == 1:
                            for batch in range(batch_dim):
                                for node in range(nodes):
                                    if state["annotations"][batch][node][1] == 0:
                                        logits[batch][node] = -np.inf
                                    else:
                                        logits[batch][node] *= -1
                        else:
                            for batch in range(batch_dim):
                                for node in range(nodes):
                                    if state["annotations"][batch][node][1] == 0:
                                        logits[batch][node] = -10000
                                    else:
                                        logits[batch][node] *= -1
            
                else:
                    if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                        # if self.config['measurement_type_actor'] == 'edge':
                        #     nodes = state["annotations"].shape[1]
                        #     batch_dim = state["annotations"].shape[0]
                        #     if batch_dim == 1:
                        #         for batch in range(batch_dim):
                        #             for node, annotation in state["annotations"][batch]:
                        #                 if annotation == 0:
                        #                     for idx, (node0, node1, value) in enumerate(state['edges'][batch]):
                        #                         if node0 == node:
                        #                             logits[batch][idx] = -np.inf
                        #                         elif node1 == node:
                        #                             logits[batch][idx] = -np.inf
                        # else:
                        nodes = state["annotations"].shape[1]
                        batch_dim = state["annotations"].shape[0]
                        if batch_dim == 1:
                            for batch in range(batch_dim):
                                for node in range(nodes):
                                    if state["annotations"][batch][node][1] == 0:
                                        logits[batch][node] = -np.inf
                        else:
                            for batch in range(batch_dim):
                                for node in range(nodes):
                                    if state["annotations"][batch][node][1] == 0:
                                        logits[batch][node] = -10000
        
    
        return logits, []

