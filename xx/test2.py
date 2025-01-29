import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import pennylane as qml
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Create your data
X, y = make_moons(n_samples=200, noise=0.1)
y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_qubits = 2
        self.n_layers = 6
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weight_shapes = {"weights": (self.n_layers, self.n_qubits)}
        self.qnode = qml.QNode(self.circuit, self.dev, diff_method = "best", interface = "torch")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)

    def circuit(self,inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
    
    def forward(self,x):
        return self.qlayer(x)
    
model = Model()

# Choose an optimizer and a loss function
opt = torch.optim.SGD(model.parameters(), lr=0.2)
loss = torch.nn.L1Loss()

# Set your data to something trainable
X = torch.tensor(X, requires_grad=True).float()
y_hot = y_hot.float()

# Get your data ready for training in batches
batch_size = 64
batches = 200 // batch_size

data_loader = torch.utils.data.DataLoader(
    list(zip(X, y_hot)), batch_size=batch_size, shuffle=True, drop_last=True
)

# Choose your epochs and start your training
epochs = 6

for epoch in range(epochs):

    running_loss = 0

    for xs, ys in data_loader:
        opt.zero_grad()

        start = time.time()
        for i in range(100):
            loss_evaluated = loss(model(xs), ys)
        print(f"Batch Size of {batch_size} took {(time.time()-start)/100} seconds on average")
        loss_evaluated.backward()

        opt.step()

        running_loss += loss_evaluated

    avg_loss = running_loss / batches
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

# Calculate your accuracy
y_pred = model(X)
predictions = torch.argmax(y_pred, axis=1).detach().numpy()

correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
print(f"Accuracy: {accuracy * 100}%")