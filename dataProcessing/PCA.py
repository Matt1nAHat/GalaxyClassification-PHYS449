from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

def performPCA(path, outPath):

    # Load the data from the text file into a 2D numpy array
    features = []
    labels = []

    # Open the file
    with open(path, 'r') as f:
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

    # Finalize the label array from probabilities to a single label
    final_labels = []
    for label in labels:
        if label[1] == 'Star':
            final_labels.append('Star')
        else:
            # Get the index of the maximum probability among the galaxy types
            max_index = np.argmax(label[2:7])
            if max_index == 0:
                final_labels.append('Elliptical')
            elif max_index in [1, 2, 3]:
                final_labels.append('Spiral')
            else:
                final_labels.append('Merger')

    # Convert the final_labels list to a numpy array
    final_labels = np.array(final_labels)

    # Standardize the features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    # Perform PCA
    pca = PCA(n_components=25)
    pca.fit(standardized_features)
    transformed_features = pca.transform(standardized_features)

    # Renormalize the PCA components
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(transformed_features)

    # Concatenate the labels and the PCA values
    data = np.column_stack((final_labels, normalized_features))

    # Write the data to a .txt file
    np.savetxt(outPath, data, fmt='%s')
    