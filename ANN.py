import torch
from torch import nn
from torch import optim

class ANN(nn.Module):
    def __init__(self, input_size, hidden_layers=3, neurons_per_layer=20, output_size=4):
        super(ANN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, neurons_per_layer))
        
        # Hidden layers
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        
        # Output layer
        self.layers.append(nn.Linear(neurons_per_layer, output_size))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return torch.softmax(self.layers[-1](x), dim=1)
    
    def train(self, dataloader, lr=0.001, epochs=10):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
        pass