import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# Specify the path to the 'processedData' folder
data_folder = os.path.join('dataProcessing', 'processedData')

# Import the three text files from the 'processedData' folder
file1_path = os.path.join(data_folder, 'testPCAList.txt')
file2_path = os.path.join(data_folder, 'trainPCAList.txt')
file3_path = os.path.join(data_folder, 'validPCAList.txt')

label_mapping = {'Spiral': 0, 'Merger': 1, 'Elliptical': 2, 'Star': 3}

with open(file1_path, 'r') as test:
    # Process file1 contents
    lines = test.readlines()
    first_elements = []
    other_elements = []
    for line in lines:
        elements = line.split()
        first_element = elements[0]
        other_element = elements[1:]
        first_elements.append(first_element)
        Test_labels = np.array([label_mapping[label] for label in first_elements])
        Test_labels_tensor = torch.tensor(Test_labels, dtype=torch.long)
        other_elements.append(other_element)
        Test_features = other_elements
        Test_features = [[float(value) for value in sublist] for sublist in Test_features]
        Test_features_tensor = torch.tensor(Test_features).float()
        print('test done')

with open(file2_path, 'r') as train:
    # Process file2 contents
    lines = train.readlines()
    first_elements = []
    other_elements = []
    for line in lines:
        elements = line.split()
        first_element = elements[0]
        other_element = elements[1:]
        first_elements.append(first_element)
        Train_labels = np.array([label_mapping[label] for label in first_elements])
        Train_labels_tensor = torch.tensor(Train_labels, dtype=torch.long)
        other_elements.append(other_element)
        Train_features = other_elements
        Train_features = [[float(value) for value in sublist] for sublist in Train_features]
        Train_features_tensor = torch.tensor(Train_features).float()
        print('train done')

with open(file3_path, 'r') as validation:
    # Process file3 contents
    lines = validation.readlines()
    first_elements = []
    other_elements = []
    for line in lines:
        elements = line.split()
        first_element = elements[0]
        other_element = elements[1:]
        first_elements.append(first_element)
        Valid_labels = np.array([label_mapping[label] for label in first_elements])
        Valid_labels_tensor = torch.tensor(Valid_labels, dtype=torch.long)
        other_elements.append(other_element)
        Valid_features = other_elements
        Valid_features = [[float(value) for value in sublist] for sublist in Valid_features]
        Valid_features_tensor = torch.tensor(Valid_features).float()    
        print('valid done')



print(isinstance(Test_labels, np.ndarray))
print(isinstance(Test_features, np.ndarray))

# Assuming you have your features and labels as NumPy arrays
# (Code for creating dummy data remains the same)

# Create a custom neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1).float()  # Set dtype to float
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2).float()  # Set dtype to float
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size).float()  # Set dtype to float
        self.softmax = nn.Softmax(dim=-1)  # Use dim=-1 to infer the correct dimension

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
train_dataset = TensorDataset(Train_features_tensor, Train_labels_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create a DataLoader for validation
valid_dataset = TensorDataset(Valid_features_tensor, Valid_labels_tensor)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 400
train_losses = []  # To store the training losses for plotting
valid_losses = []  # To store the validation losses for plotting
kl_divergences = []  # To store KL divergences for plotting

print('on to training now')

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    kl_divergence = 0.0

    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        log_outputs = torch.log(outputs + 1e-10)  # Add a small epsilon to avoid log(0)

        target_probs = torch.zeros_like(outputs)
        target_probs.scatter_(1, targets.unsqueeze(1), 1.0)

        kl_divergence += nn.KLDivLoss()(log_outputs, target_probs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_dataloader)
    avg_kl_divergence = kl_divergence / len(train_dataloader)

    train_losses.append(avg_train_loss)
    kl_divergences.append(avg_kl_divergence)

    print(f'Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, KL Divergence: {avg_kl_divergence:.4f}')

    # Validation phase
    model.eval()
    epoch_valid_loss = 0.0

    with torch.no_grad():
        for inputs, targets in valid_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_valid_loss += loss.item()

    avg_valid_loss = epoch_valid_loss / len(valid_dataloader)
    valid_losses.append(avg_valid_loss)

    print(f'Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_valid_loss:.4f}')

# Plot the losses and KL divergence for both training and validation
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
kl_divergences_np = [value.detach().item() for value in kl_divergences]
plt.plot(kl_divergences_np, label='KL Divergence')
plt.title('KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.legend()

plt.show()

# Test the model
model.eval()
with torch.no_grad():
    test_input = Test_features_tensor
    predicted_probs = model(test_input)
    predicted_labels = torch.argmax(predicted_probs, dim=1)
    predicted_labels_array = predicted_labels.numpy()

    # Compare predicted labels to test labels
    correct_predictions = (predicted_labels_array == Test_labels)
    accuracy = (correct_predictions.sum() / len(Test_labels)) * 100

    print(f"Percentage of correctness: {accuracy:.2f}%")

# Map predicted labels back to original labels
reverse_mapping = {v: k for k, v in label_mapping.items()}
predicted_classes_name = [reverse_mapping[label.item()] for label in predicted_labels]

print(f"Predicted classes: {predicted_classes_name}")

