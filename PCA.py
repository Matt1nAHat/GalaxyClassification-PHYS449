from sklearn.decomposition import PCA
import numpy as np



# Load the data from the text file into a 2D numpy array
features = []
labels = []

# Open the file
with open('galaxyDataset.txt', 'r') as f:
    # Read the entire file as a single string
    data = f.read()

# Split the string into 1D arrays
arrays = data.split('][')

# Process each 1D array
for array in arrays:
    # Remove leading and trailing brackets and split into columns
    columns = array.strip('[]').split()
    #print(len(columns))
    # Convert the necessary columns to floats and add them to the features array
    features.append([float(x.strip("'")) for x in columns[7:]])
    labels.append([x.strip("'") for x in columns[:7]])

# Convert the features and labels list to a numpy array
features = np.array(features)
labels = np.array(labels)
print(features.shape)
print(labels.shape)

# Assuming features is your feature vector
pca = PCA()
pca.fit(features[:,2:])

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(cumulative_variance > 0.95)[0][0] + 1

# Now you can initialize a new PCA object with the determined number of components
pca = PCA(n_components=n_components)
pca.fit(features[:,2:])
transformed_features = pca.transform(features[:,2:])
print(transformed_features.shape)