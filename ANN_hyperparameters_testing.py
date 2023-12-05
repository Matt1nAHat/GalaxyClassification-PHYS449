#Importing the libraries
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
from sklearn.model_selection import ParameterGrid

def param_testing():
    """
    Perform hyperparameter testing for the ANN model.

    This function reads the input data, defines the neural network model,
    and performs hyperparameter testing using a grid search approach.

    Returns:
        None
    """
    # Create parser
    parser = argparse.ArgumentParser(description='ANN for Galaxy Morphology Classification')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for training')
    parser.add_argument('--hidden_size_1', type=int, default=12, help='Number of neurons in the first hidden layer')
    parser.add_argument('--hidden_size_2', type=int, default=24, help='Number of neurons in the second hidden layer')
    parser.add_argument('--hidden_size_3', type=int, default=12, help='Number of neurons in the third hidden layer')
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0003, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training')

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

    # Process Training file
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file2_path, delim_whitespace=True, header=None)

    # Process the DataFrame
    Train_labels = df[0].map(label_mapping).values
    Train_labels_tensor = torch.tensor(Train_labels, dtype=torch.long)

    Train_features = df.drop(0, axis=1).values.astype(float)
    Train_features_tensor = torch.tensor(Train_features).float()

    # Process Validation file
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file3_path, delim_whitespace=True, header=None)

    # Process the DataFrame
    Valid_labels = df[0].map(label_mapping).values
    Valid_labels_tensor = torch.tensor(Valid_labels, dtype=torch.long)

    Valid_features = df.drop(0, axis=1).values.astype(float)
    Valid_features_tensor = torch.tensor(Valid_features).float()

    # Assuming you have your features and labels as NumPy arrays
    # (Code for creating dummy data remains the same)

    # Create a custom neural network class
    class NeuralNetwork(nn.Module):
        """
        Neural network model for galaxy morphology classification.

        This class defines the architecture of the neural network model
        used for galaxy morphology classification. It consists of multiple
        fully connected layers with ReLU activation and a softmax output layer.

        Args:
            input_size (int): Number of input features
            hidden_size1 (int): Number of neurons in the first hidden layer
            hidden_size2 (int): Number of neurons in the second hidden layer
            hidden_size3 (int): Number of neurons in the third hidden layer
            output_size (int): Number of output classes

        Attributes:
            fc1 (nn.Linear): First fully connected layer
            relu1 (nn.ReLU): ReLU activation function
            fc2 (nn.Linear): Second fully connected layer
            relu2 (nn.ReLU): ReLU activation function
            fc3 (nn.Linear): Third fully connected layer
            relu3 (nn.ReLU): ReLU activation function
            fc4 (nn.Linear): Output fully connected layer
            softmax (nn.Softmax): Softmax activation function
        """
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
            """
                Initialize the neural network model.

                Args:
                    input_size (int): The size of the input layer.
                    hidden_size1 (int): The size of the first hidden layer.
                    hidden_size2 (int): The size of the second hidden layer.
                    hidden_size3 (int): The size of the third hidden layer.
                    output_size (int): The size of the output layer.
                """
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
            """
            Performs the forward pass of the neural network.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after passing through the network.
            """
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
    param_grid = {
        'hidden_size_1': [10],
        'hidden_size_2': [25],
        'hidden_size_3': [25],
        'lr': [0.0001],
        'wd': [0.0001, 0.0002],
        'batch_size': [25, 30]
    }

    grid = ParameterGrid(param_grid)
    best_accuracy = 0
    best_params = None  

    # Loop over the grid
    for i, params in enumerate(grid):
        print(f'Running configuration {i+1} of {len(grid)}: {params}')

        # Update the args with the current parameters
        hidden_size1 = params['hidden_size_1']
        hidden_size2 = params['hidden_size_2']
        hidden_size3 = params['hidden_size_3']
        hidden_size1 = params['hidden_size_1']
        hidden_size2 = params['hidden_size_2']
        hidden_size3 = params['hidden_size_3']
        args.lr = params['lr']
        args.wd = params['wd']
        args.batch_size = params['batch_size']
        input_size = 25
        output_size = 4  # Number of classes

        # Create the neural network, loss function, and optimizer
        model = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
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

        for epoch in range(num_epochs):
            # Training phase
            model.train()

            for inputs, targets in train_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)

                target_probs = torch.zeros_like(outputs)
                target_probs.scatter_(1, targets.unsqueeze(1), 1.0)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            epoch_valid_loss = 0.0

            with torch.no_grad():
                for inputs, targets in valid_dataloader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    epoch_valid_loss += loss.item()


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

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params

    # Print the best accuracy and parameters
    print(f'Best accuracy: {best_accuracy:.2f}%')
    print(f'Best parameters: {best_params}')

# Run the main function
if __name__ == '__main__':
    param_testing()