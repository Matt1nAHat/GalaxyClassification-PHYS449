from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import torch
from PIL import Image
import numpy as np
import os

# Specify the directory where your images are stored
image_dir = 'path_to_your_image_directory'

# Get a list of all image paths in the directory
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.png')]

# Initialize an empty list for your data
data = []

# Process each image
for image_path in image_paths:
    # Load the image
    image = Image.open(image_path)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Initialize an empty list for your features
    features = []

    # Add a dummy feature for each color channel
    for channel in range(3):  # Assuming the image is in RGB format
        features.append(image_array[:, :, channel].mean())

    # Add the feature vector to your data
    data.append(features)

# Convert the data to a numpy array
data = np.array(data)

# Normalize the data
data = (data - data.mean(axis=0)) / data.std(axis=0)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
data_pca = pca.fit_transform(data)

# Convert to PyTorch tensors
data_pca = torch.tensor(data_pca, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Create a DataLoader for the training data
dataset = TensorDataset(data_pca, labels)
loader = DataLoader(dataset, batch_size=32)

# Now you can use `loader` to train your ANN