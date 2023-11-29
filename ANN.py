from PCA import performPCA
features, labels = performPCA()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

print(isinstance(features, np.ndarray))
print(isinstance(labels, np.ndarray))

# Assuming features and labels as NumPy arrays
label_mapping = {'spiral': 0, 'merger': 1, 'Elliptical': 2, 'star': 3}
labels = np.array([label_mapping[label] for label in labels])

# Assuming you have your features and labels as NumPy arrays
# (Code for creating dummy data remains the same)

# Convert NumPy arrays to PyTorch tensors
features_tensor = torch.tensor(features).float()  # Convert to float
labels_tensor = torch.tensor(labels, dtype=torch.long)  # Assuming integer labels

# Create a custom neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1).float()  # Set dtype to float
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2).float()  # Set dtype to float
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size).float()  # Set dtype to float
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Define hyperparameters
input_size = 25
hidden_size1 = 50
hidden_size2 = 30
output_size = 4  # Number of classes

# Create the neural network, loss function, and optimizer
model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a DataLoader for batch training
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 50
losses = []  # To store the losses for plotting
kl_divergences = []  # To store KL divergences for plotting

for epoch in range(num_epochs):
    epoch_loss = 0.0
    kl_divergence = 0.0

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        log_outputs = torch.log(outputs + 1e-10)  # Add a small epsilon to avoid log(0)

        # Construct target probabilities (convert class indices to one-hot vectors)
        target_probs = torch.zeros_like(outputs)
        target_probs.scatter_(1, targets.unsqueeze(1), 1.0)

        # Calculate KL Divergence
        kl_divergence += nn.KLDivLoss()(log_outputs, target_probs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_kl_divergence = kl_divergence / len(dataloader)

    losses.append(avg_epoch_loss)
    kl_divergences.append(avg_kl_divergence)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, KL Divergence: {avg_kl_divergence:.4f}')

# Plot the losses and KL divergence
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Convert the KL divergences to NumPy using detach()
kl_divergences_np = [value.detach().item() for value in kl_divergences]

plt.subplot(1, 2, 2)
plt.plot(kl_divergences_np, label='KL Divergence')
plt.title('KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.legend()

plt.show()

# Testing the model
model.eval()
with torch.no_grad():
    test_input = torch.tensor(features[0], dtype=torch.float32)
    predicted_probs = model(test_input)
    predicted_label = torch.argmax(predicted_probs).item()

# Map predicted label back to original label
reverse_mapping = {v: k for k, v in label_mapping.items()}
predicted_class = reverse_mapping[predicted_label]

print(f"Predicted class: {predicted_class}")
