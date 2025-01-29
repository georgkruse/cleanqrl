import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from quantum.circuits import ansatz_hwe
from torch.distributions.categorical import Categorical
from scipy.optimize import fsolve

def calculate_a(a,S,L):
    left_side = np.sin(2 * a * np.pi) / (2 * a * np.pi)
    right_side = (S * (2 * L - 1) - 2) / (S * (2 * L + 1))
    return left_side - right_side

class QRLAgentReinforce(nn.Module):
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

        if self.ansatz == "hwe":
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
        if self.ansatz == "hwe":
            self.qc = qml.QNode(ansatz_hwe, dev, diff_method = config["diff_method"], interface = "torch")
        
    def get_action_and_logprob(self, x):
        x = x.repeat(1, len(self.wires) // len(x[0]) + 1)[:, :len(self.wires)]
        logits = self.qc(x, self._parameters["input_scaling_actor"], self._parameters["variational_actor"], self.wires, self.num_layers, "actor", self.observables)
        if self.config["diff_method"] == "backprop" or x.shape[0] == 1:
            logits = logits.reshape(x.shape[0], self.num_actions)
        logits_scaled = logits * self._parameters["output_scaling_actor"]
        probs = Categorical(logits=logits_scaled)
        action = probs.sample()
        return action, probs.log_prob(action)

class QRLAgentDQN(nn.Module):
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

        if self.ansatz == "hwe":
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
        if self.ansatz == "hwe":
            self.qc = qml.QNode(ansatz_hwe, dev, diff_method = config["diff_method"], interface = "torch")
        
    def forward(self, x):
        x = x.repeat(1, len(self.wires) // len(x[0]) + 1)[:, :len(self.wires)]
        logits = self.qc(x, self._parameters["input_scaling_actor"], self._parameters["variational_actor"], self.wires, self.num_layers, "actor", self.observables)
        if self.config["diff_method"] == "backprop" or x.shape[0] == 1:
            logits = logits.reshape(x.shape[0], self.num_actions)
        logits_scaled = logits * self._parameters["output_scaling_actor"]
        return logits_scaled

class QRLAgent(nn.Module):
    def __init__(self,envs,config):
        super().__init__()
        self.num_features = np.array(envs.single_observation_space.shape).prod()
        self.num_actions = envs.single_action_space.n
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.ansatz = config["ansatz"]
        self.dynamic_meas = config["dynamic_meas"]
        self.wires = range(self.num_qubits)

        if self.ansatz == "hwe":
            self.register_parameter(name="input_scaling_critic", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=True))
            self.register_parameter(name="variational_critic", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi, requires_grad=True))
        
            self.register_parameter(name="input_scaling_actor", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=True))
            self.register_parameter(name="variational_actor", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi, requires_grad=True))

            self.register_parameter(name="output_scaling_critic", param = nn.Parameter(torch.ones(1), requires_grad=True))
            self.register_parameter(name="output_scaling_actor", param = nn.Parameter(torch.ones(self.num_actions), requires_grad=True))
        
        self.measured_qubits = torch.randint(low = 0, high = self.num_qubits, size = (self.num_layers,2), requires_grad = False)

        dev = qml.device("lightning.qubit", wires = self.wires)
        if self.ansatz == "hwe":
            if self.dynamic_meas:
                self.qc = qml.QNode(ansatz_hwe, dev,mcm_method="deferred", diff_method = "backprop", interface = "torch")
            else:
                self.qc = qml.QNode(ansatz_hwe, dev, diff_method = "backprop", interface = "torch")
        
    def get_value(self,x):
        return self._parameters["output_scaling_critic"] * self.qc(x, self._parameters["input_scaling_critic"], self._parameters["variational_critic"], self.wires, self.num_layers, "critic", self.dynamic_meas, self.measured_qubits)

    def get_action_and_value(self, x, action=None):
        logits = self.qc(x, self._parameters["input_scaling_actor"], self._parameters["variational_actor"], self.wires, self.num_layers, "actor", self.dynamic_meas, self.measured_qubits)
        logits = torch.stack((logits[0], logits[1]), dim=1)
        logits_scaled = logits * self._parameters["output_scaling_actor"]
        probs = Categorical(logits=logits_scaled)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self._parameters["output_scaling_critic"] * self.qc(x, self._parameters["input_scaling_critic"], self._parameters["variational_critic"], self.wires, self.num_layers, "critic", self.dynamic_meas, self.measured_qubits)

