from abc import ABC
import pennylane as qml
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Dict

from circuits.postprocessing import *
from circuits.quantum_circuits import vqc_generator


class QuantumPGModel(TorchModelV2, nn.Module, ABC):
    '''
    Quantum Model for Policy Gradient.
    '''

    def __init__(self, obs_space, action_space, num_actions, config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_actions, config, name
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
            self.num_inputs = sum([box.shape[0] for box in self.obs_space.values()])
        elif isinstance(self.obs_space, gym.spaces.Discrete):
            self.num_inputs = self.obs_space.n
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
        self.init_input_scaling_actor = self.config['init_input_scaling_actor']

        self.use_classical_layer = self.config['use_classical_layer']        
        self.layer_size = self.config['layer_size']

        self.weight_logging_interval = self.config['weight_logging_interval']
        self.weight_plotting = self.config['weight_plotting']
        self._value_out = None
       
        def init_weights(size):
            if self.config['init_variational_params_mode'] == 'plus-zero-uniform':
                return torch.FloatTensor(*size).uniform_(0, self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'plus-minus-uniform':
                return torch.FloatTensor(*size).uniform_(-self.init_variational_params, self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'plus-zero-normal':
                return torch.FloatTensor(*size).normal_(0., self.init_variational_params)
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

            if self.config["init_variational_params_mode"] == "ramp":
                delta_beta = self.config["delta_beta"]
                params = torch.zeros(self.num_layers,1)
                for i in range(self.num_layers):
                    params[i][0] = (1 - (i/self.num_layers)) * delta_beta
                self.register_parameter(name=f'weights_actor', param = torch.nn.Parameter(params, requires_grad = True))
            else:
                self.register_parameter(name=f'weights_actor', param=torch.nn.Parameter(init_weights((self.num_layers, size_vqc)), requires_grad=True))
            
            self.total_model_parameters += self.num_layers * size_vqc

            if self.use_input_scaling:
                if isinstance(self.config["init_input_scaling_actor"], list):
                    self.register_parameter(name=f'input_scaling_actor', param=torch.nn.Parameter(torch.tensor(np.full((self.num_layers, size_input_scaling), 1.)), requires_grad=True))
                elif self.config["init_input_scaling_actor"] == "ramp":
                    delta_gamma = self.config["delta_gamma"]
                    params = torch.zeros(self.num_layers,1)
                    for i in range(self.num_layers):
                        params[i][0] = ((1 + i) / self.num_layers) * delta_gamma
                    self.register_parameter(name=f"input_scaling_actor",param = torch.nn.Parameter(params, requires_grad = True))
                
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
       
        # Set gradients in layers to True/False if layerwise training
        if self.layerwise_training:
            self.set_layerwise_training()

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


            # If vqc_type is relu or qcnn, two function calls are required
            if 'relu' in self.config['vqc_type'][0]:    
                activations_actor = self.qnode_activations(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None)
                prob = self.qnode(theta=state, weights=self._parameters, config=self.config, type='actor', activations=activations_actor)
            elif 'qcnn' in self.config['vqc_type'][0]:    
                activations_actor = self.qnode(theta=state, weights=self._parameters, config=self.config, type='activations_actor', activations=None)
                activations_actor = torch.reshape(activations_actor, (-1, 4))
                prob = self.qnode(theta=state, weights=self._parameters, config=self.config, type='actor', activations=activations_actor)
            else:
                if self.config['measurement_type_actor'] == 'probs':
                    prob = []
                    for i in range(state['linear_0'].shape[0]):
                        tmp_state = {}
                        tmp_state['linear_0'] = np.reshape(state['linear_0'][i], (-1, *state['linear_0'].shape[1:]))
                        tmp_state['quadratic_0'] = np.reshape(state['quadratic_0'][i], (-1, *state['quadratic_0'].shape[1:]))

                        prob.append(self.qnode_actor(theta=tmp_state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None))

                    prob = torch.stack(prob)
                elif self.config['measurement_type_actor'] == 'edge':
                    prob = []
                    for i in range(state['quadratic_0'].shape[1]):
                        if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                            self.config['edge_measurement'] = state['quadratic_0'][0,i,:2]
                        else:
                            self.config['edge_measurement'] = [torch.tensor(0.), torch.tensor(1.)]
                        prob.append(self.qnode_actor(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None))
                    prob = torch.stack(prob).T
                    prob = torch.reshape(prob, (-1, state['quadratic_0'].shape[1]))

                else:
                    prob = self.qnode_actor(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None)
                    # Higher Pennylane Version
                    if state.shape[0] == 1:
                        prob = torch.hstack(prob)
                    else:
                        prob = torch.stack(prob).T

            if isinstance(self.action_space, MultiDiscrete):
                
                # Do for now test if action space has 2 elements / is binary
                if self.action_space[0].n == 2:
                    prob = (torch.ones(self.num_qubits) + prob)/2
                    prob = torch.reshape(prob, (-1,self.num_qubits)).repeat_interleave(2, dim=1)*torch.tensor([1., -1.]).repeat(self.num_qubits)
                    prob = torch.tensor([0., 1.]).repeat(self.num_qubits) + prob
                # else:
                #     prob = torch.reshape(prob, (-1, self.action_space.shape[0], self.action_space.shape[1]))

            if self.use_output_scaling_actor:
                logits = torch.reshape(postprocessing(prob, self.config, self.num_outputs, self._parameters, 'actor'), (-1, self.num_outputs))
            else:
                logits =  torch.reshape(prob, (-1, self.num_outputs)) #prob[:,:self.num_outputs]


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
                input_state = []
                for idx in range(int(len(state.keys())/2)):
                    linear = state[f'linear_{idx}'][:,:,1]
                    quadratic = state[f'quadratic_{idx}'][:,:,1]
                    input_state.append(linear)
                    # input_state.append(quadratic)

                input_state = torch.from_numpy(np.hstack(input_state)).type(torch.float32)
                logits = self.actor_network(input_state)
            
            else:
                logits = self.actor_network(state)
                
        if self.config['output_scaling_schedule']:
            self.counter += 1
            if self.counter % self.config["steps_for_output_scaling_update"] == 0:
                if self.counter <= self.config["max_steps_output_scaling_update"]:
                    if logits.shape[0] == 1.:
                        self._parameters['output_scaling_actor'] += self.config["output_scaling_update"]
                    else:
                        self.counter -= 1
                # print('output_scaling_actor:', self._parameters['output_scaling_actor'])
                
        if self.config['use_temperature']:
            
            logits = logits/self.temperature

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
                        nodes = state["annotations"].shape[1]
                        batch_dim = state["annotations"].shape[0]
                        for batch in range(batch_dim):
                            for node in range(nodes):
                                if state["annotations"][batch][node][1] == 0:
                                    logits[batch][node] = -np.inf
        self._logits = logits
        return logits, []
    

