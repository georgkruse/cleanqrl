"""
Example of an nn.Module class that uses qml.qnn.TorchLayer

Unfortunately only runs on latest pennylane versions and those do not have lighnting.qubit on our cluster.
"""

class QRLAgentDQN(nn.Module):
    def __init__(self,envs,config):
        super().__init__()
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

        dev = qml.device("default.qubit", wires = self.wires)

        if self.ansatz == "hwe":

            weight_shapes = {"input_scaling_weights": (self.num_layers, self.num_qubits),
                         "variational_weights": (self.num_layers, self.num_qubits*2)}
            
            # Input scaling weights are always initialized as 1s
            init_method = {
            "input_scaling_weights": torch.nn.init.ones_,
            }

            if self.init_method == "uniform":
                init_method["variational_weights"] = lambda tensor: torch.nn.init.uniform_(tensor,a=-torch.pi, b = torch.pi)
            elif self.init_method == "small_random":
                init_method["variational_weights"] = lambda tensor: torch.nn.init.uniform_(tensor,a=-0.1,b=0.1)
            elif self.init_method == "reduced_domain":
                initial_guess = 0.1
                alpha = fsolve(lambda a: calculate_a(a,self.S,self.num_layers), initial_guess)
                init_method["variational_weights"] = lambda tensor: torch.nn.init.uniform_(tensor,a=-torch.pi, b = torch.pi).mul_(alpha)   
        
            self.qc = qml.QNode(self.ansatz_hwe_new, dev, diff_method = "best", interface = "torch")
        
        # Define the torchlayer to allow for batch optimization
        self.qlayer = qml.qnn.TorchLayer(self.qc, weight_shapes, init_method)

        # Output Scaling weights are always initialized as 1s        
        self.register_parameter(name="output_scaling_actor", param = nn.Parameter(torch.ones(self.num_actions), requires_grad=True))

    def layer_hwe_new(self,x, input_scaling_params, rotational_params, wires):
        """
        x: input (batch_size,num_features)
        input_scaling_params: vector of parameters (num_features)
        rotational_params:  vector of parameters (num_features*2)
        """
        for i, wire in enumerate(wires):
            qml.RX(input_scaling_params[i] * x[:,i], wires = [wire])
        
        for i, wire in enumerate(wires):
            qml.RY(rotational_params[i], wires = [wire])

        for i, wire in enumerate(wires):
            qml.RZ(rotational_params[i+len(wires)], wires = [wire])

        if len(wires) == 2:
            qml.broadcast(unitary=qml.CZ, pattern = "chain", wires = wires)
        else:
            qml.broadcast(unitary=qml.CZ, pattern = "ring", wires = wires)
    
    def ansatz_hwe_new(self,inputs, input_scaling_weights,variational_weights):
        layers = input_scaling_weights.shape[0]
        wires = range(input_scaling_weights.shape[1])
        for layer in range(layers):
            self.layer_hwe_new(inputs, input_scaling_weights[layer], variational_weights[layer], wires)

        if self.observables == "global":
        
            left_observable = qml.PauliZ(0)
            right_observable = qml.PauliX(0)

            for i in range(1,len(wires)):
                left_observable = left_observable @ qml.PauliZ(i)
                right_observable = right_observable @ qml.PauliX(i)

            left_observable = qml.Hamiltonian([1.0], [left_observable])
            right_observable = qml.Hamiltonian([1.0], [right_observable])
        
        elif self.observables == "local":
            half = len(wires) // 2

            # Create the tensor product for the first half
            left_observable = qml.PauliZ(0)
            for i in range(1, half):
                left_observable = left_observable @ qml.PauliZ(i)

            # Create the tensor product for the second half
            right_observable = qml.PauliZ(half)
            for i in range(half + 1, len(wires)):
                right_observable = right_observable @ qml.PauliZ(i)

        return [qml.expval(left_observable), qml.expval(right_observable)]
        
    def forward(self, x):
        x = x.repeat(1, len(self.wires) // len(x[0]) + 1)[:, :len(self.wires)]
        logits = self.qc(x, self._parameters["input_scaling_actor"], self._parameters["variational_actor"], self.wires, self.num_layers, "actor", self.dynamic_meas, self.measured_qubits, self.observables)
        if x.shape[0] == 1:
            logits = logits.reshape(x.shape[0], logits.shape[0])
        logits_scaled = logits * self._parameters["output_scaling_actor"]
        return logits_scaled