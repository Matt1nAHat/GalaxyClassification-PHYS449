import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import argparse

def main():
    # Create parser
    parser = argparse.ArgumentParser(description='ANN for Galaxy Morphology Classification')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--hidden_size_1', type=int, default=12, help='Number of neurons in the first hidden layer')
    parser.add_argument('--hidden_size_2', type=int, default=24, help='Number of neurons in the second hidden layer')
    parser.add_argument('--hidden_size_3', type=int, default=16, help='Number of neurons in the third hidden layer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    # Parse arguments
    args = parser.parse_args()


    # Specify the path to the 'processedData' folder
    data_folder = os.path.join('dataProcessing', 'processedData')

    # Import the three text files from the 'processedData' folder
    file1_path = os.path.join(data_folder, 'PCA_85K_test.txt')
    file2_path = os.path.join(data_folder, 'PCA_85k_train.txt')
    file3_path = os.path.join(data_folder, 'PCA_85K_valid.txt')

    label_mapping = {'Spiral': 0, 'Merger': 1, 'Elliptical': 2, 'Star': 3}

    # Process Test file
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file1_path, delim_whitespace=True, header=None)

    # Process the DataFrame
    Test_labels = df[0].map(label_mapping).values
    Test_labels_tensor = torch.tensor(Test_labels, dtype=torch.long)

    Test_features = df.drop(0, axis=1).values.astype(float)
    Test_features_tensor = torch.tensor(Test_features).float()

    print('test done')

    # Process Training file
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file2_path, delim_whitespace=True, header=None)

    # Process the DataFrame
    Train_labels = df[0].map(label_mapping).values
    Train_labels_tensor = torch.tensor(Train_labels, dtype=torch.long)

    Train_features = df.drop(0, axis=1).values.astype(float)
    Train_features_tensor = torch.tensor(Train_features).float()

    print('train done')

    # Process Validation file
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file3_path, delim_whitespace=True, header=None)

    # Process the DataFrame
    Valid_labels = df[0].map(label_mapping).values
    Valid_labels_tensor = torch.tensor(Valid_labels, dtype=torch.long)

    Valid_features = df.drop(0, axis=1).values.astype(float)
    Valid_features_tensor = torch.tensor(Valid_features).float()

    print('valid done')



    print(isinstance(Test_labels, np.ndarray))
    print(isinstance(Test_features, np.ndarray))

    # Assuming you have your features and labels as NumPy arrays
    # (Code for creating dummy data remains the same)

    # Create a custom neural network class
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1).float()
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size1, hidden_size2).float()
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, hidden_size3).float()
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_size3, output_size).float()
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.fc4(x)
            x = self.softmax(x)
            return x

    # Define hyperparameters
    input_size = 25
    hidden_size1 = args.hidden_size_1
    hidden_size2 = args.hidden_size_2
    hidden_size3 = args.hidden_size_3
    output_size = 4  # Number of classes

    # Create the neural network, loss function, and optimizer
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Create a DataLoader for batch training
    train_dataset = TensorDataset(Train_features_tensor, Train_labels_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create a DataLoader for validation
    valid_dataset = TensorDataset(Valid_features_tensor, Valid_labels_tensor)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Training loop
    num_epochs = args.epochs
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

        print(f'Training - Epoch: [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, KL Divergence: {avg_kl_divergence:.4f}')

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

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(Test_labels, predicted_labels_array)

    # Calculate precision, recall, and F-score for each class
    precision = precision_score(Test_labels, predicted_labels_array, average=None)
    recall = recall_score(Test_labels, predicted_labels_array, average=None)
    f_score = f1_score(Test_labels, predicted_labels_array, average=None)

    # Print the confusion matrix and performance metrics
    print("Confusion Matrix:")
    print(confusion_mat)
    print("")

    print("Performance Metrics:")
    print(f"Elliptical - Precision: {precision[0]:.2f}, Recall: {recall[0]:.2f}, F-score: {f_score[0]:.2f}")
    print(f"Merger - Precision: {precision[1]:.2f}, Recall: {recall[1]:.2f}, F-score: {f_score[1]:.2f}")
    print(f"Spiral - Precision: {precision[2]:.2f}, Recall: {recall[2]:.2f}, F-score: {f_score[2]:.2f}")
    print(f"Star - Precision: {precision[3]:.2f}, Recall: {recall[3]:.2f}, F-score: {f_score[3]:.2f}")

    # print(f"Predicted classes: {predicted_classes_name}")

if __name__ == '__main__':
    main()