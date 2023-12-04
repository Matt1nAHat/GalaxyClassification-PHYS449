import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# This function assumes that 'train' returns the loss at each epoch
def plot(components, variance):
    """
    This function generates a line plot of the variance in data accounted by number of components. 
    The plot is saved as a png.

    Args:
    - components (int): The total number of components to perform PCA on.
    - variance (list): A list or containing the variance for each component.

    Returns:
    Plot of the variance saved as a png.
    """
    # Generate the plot
    plt.plot(range(components), variance)
    plt.xlabel('No. of Components')
    plt.ylabel('Variance in Data (%)')
    plt.title('Total Variance Accounted by No. of Components')

    # Draw a horizontal line at the paper's variance
    plt.axhline(y=97.4, color='red', linestyle='--')

    # Add a label to the line
    plt.text(0, 97.4, 'Paper variance', color='red', va='bottom')

    # Save the plot to a file
    plt.savefig('dataProcessing/PCAStats/pca_plots.png')
    plt.close()
     
def performPCA(path, outPath):
    """
    This function performs Principal Component Analysis (PCA) on a dataset and writes the lower dimensional data to a text file.
    It also creates a plot of the variance in data accounted by number of components.
    Additionally it creates a list of the top 5 features for each PCA component and saves it to an Excel file.

    Args:
    - path (str): The path to the text file containing the dataset. 
    - outPath (str): The path where the output file will be saved.
    Returns:
    None. The function saves the PCA results to an output file.
    """
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
        # Convert the necessary columns to floats and add them to the features array
        features.append([float(x.strip("'")) for x in columns[7:]])
        # Add the labels including GZ predictions to the labels array
        labels.append([x.strip("'") for x in columns[:7]])

    # Convert the features and labels list to a numpy array
    features = np.array(features)
    labels = np.array(labels)

    # Finalize the label array from probabilities to a single label
    final_labels = []
    for label in labels:
        if label[1] == 'STAR':
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
    pcaVariance = []
    for i in range(1, 26):
        pca = PCA(n_components=i)
        pca.fit(standardized_features)
        transformed_features = pca.transform(standardized_features)

        # Calculate the cumulative sum of the explained variance ratios
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

        # Print the total explained variance for the first 25 components
        pcaVariance.append(cumulative_explained_variance[-1]*100)

    # Plot the variance
    plot(25, pcaVariance)

    # Write the top 5 features for each PCA component to an Excel file
    # Get the feature names from the features.txt file
    with open('dataProcessing/features.txt', 'r') as f:
        next(f)  # Skip headers
        next(f)
        next(f)
        feature_names = f.read().splitlines()

    # Create a DataFrame for all the PCA components
    components_df = pd.DataFrame(pca.components_, columns=feature_names)

    # Replace the weights with the feature names
    components_df = components_df.apply(lambda row: pd.Series(row.nlargest(5).index), axis=1)

    # Create a new Excel writer object
    with pd.ExcelWriter('dataProcessing/PCAStats/pca_components.xlsx') as writer:
        # Write the DataFrame to the Excel file
        components_df.to_excel(writer, header=False)

    # Renormalize the PCA components
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(transformed_features)

    # Concatenate the labels and the PCA values
    data = np.column_stack((final_labels, normalized_features))

    # Write the data to a .txt file
    np.savetxt(outPath, data, fmt='%s')
    