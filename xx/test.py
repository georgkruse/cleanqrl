import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import time
from quantum.hea import ansatz_hwe

class QAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.num_actions = 2
        self.num_qubits = self.num_features
        self.num_layers = 5
        self.ansatz = "hwe"
        self.init_method = "uniform"
        self.observables = "local"
        self.wires = range(self.num_qubits)
        if self.ansatz == "hwe":
            # Input and Output Scaling weights are always initialized as 1s        
            self.register_parameter(name="input_scaling_actor", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=True))
            # The variational weights are initialized differently according to the config file
            if self.init_method == "uniform":
                self.register_parameter(name="variational_actor", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi - torch.pi, requires_grad=True))

        dev = qml.device("default.qubit", wires = self.wires)
        if self.ansatz == "hwe":
            self.qc = qml.QNode(ansatz_hwe, dev, diff_method = "backprop", interface = "torch")

    def forward(self,x):
        return self.qc(x, self._parameters["input_scaling_actor"], self._parameters["variational_actor"], self.wires, self.num_layers, "actor", self.observables)

class QAgentlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.num_actions = 2
        self.num_qubits = self.num_features
        self.num_layers = 5
        self.ansatz = "hwe"
        self.init_method = "uniform"
        self.observables = "local"
        self.wires = range(self.num_qubits)
        
        dev = qml.device("default.qubit", wires = self.wires)
        if self.ansatz == "hwe":
            self.qc = qml.QNode(self.ansatz_hwe_new, dev, diff_method = "best", interface = "torch")
        
        weight_shapes = {"input_scaling_weights": (self.num_layers, self.num_qubits),
                         "variational_weights": (self.num_layers, self.num_qubits*2)}
        
        init_method = {
            "input_scaling_weights": torch.nn.init.ones_,
            "variational_weights": lambda tensor: torch.nn.init.uniform_(tensor,a=-torch.pi, b = torch.pi)
        }
        
        self.qlayer = qml.qnn.TorchLayer(self.qc, weight_shapes, init_method)

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
        logits = self.qlayer(x)
        return logits

# Sample data preparation (replace with your actual data)
# X: (N, 4) where N is batch size and 4 is the number of features
# y: (N,) where N is batch size

# Example: Random tensors for demonstration
X = torch.randn(100, 4)  # 100 samples, 4 features
y = torch.randint(0, 2, (100,))  # 100 samples, 10 classes
batch_size = 48

# Dataset and DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = QAgent()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        start = time.time()
        outputs = model(inputs)
        print(f"Batch Size of {batch_size} took {(time.time()-start)} seconds")
        outputs = torch.stack(outputs, dim=1)
    
        loss = criterion(outputs, labels)
        start = time.time()
        loss.backward()
        print(f"Loss Backward took {time.time() - start} seconds")
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

print('Training complete.')