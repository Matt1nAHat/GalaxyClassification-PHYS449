# GalaxyClassification-PHYS449
This project is a collaborative effort aimed at classifying objects from the Sloan Digital Sky Survey (SDSS) into one of four categories: Stars, Spirals, Ellipticals, and Mergers. The classification is based on a paper by Moonzarin Reza and utilizes machine learning techniques to achieve this goal.

### Members:
- Matthew Charbonneau - 20822976
- Jakob Devey - 20854189
- Tom Harding - 20825392
- Aaron Herschberger - 20778142

## Overview

### Models & Outputs
Two distinct machine learning models are developed for this purpose: Extra Trees and Artificial Neural Network (ANN).

1. Extra Trees: This is an ensemble learning method fundamentally based on decision trees. Extra Trees operates by creating a multitude of decision trees at training time, and outputting a vote by the trees in the forest, weighted by their probability estimates. That is, the predicted class is the one with highest mean probability estimate across the trees. Our ET model produces a pickle (file type) of the best model, and plots validation accuracy for every possible combination of depth and number of trees, both of which are saved to the ET_Results folder.  

2. ANN: This is a computational model consisting of interconnected processing nodes, or "neurons", as well as hidden layers, and can learn from and make decisions and predictions based on data. In verbose mode our ANN model produces a plot of the training and validation loss as well as a plot of the KL-divergence and saves it to the ANN_Results folder.

Both models also produce a confusion matrix, a specific table layout that allows visualization of the performance of the algorithm. Each column of the matrix represents the instances in a predicted class, while each row represents the instances in an actual class. They also both print the precision, recall, and f-scores, which are metrics used to measure the accuracy of a classification model.Precision is the ratio of correctly predicted positive observations to the total predicted positives, recall is the ratio of correctly predicted positive observations to all actual positives, and the F-score is the harmonic mean of precision and recall.

Use of AI tools, namely copilot, were used to assist in writing code for this project. 

### Structure
The GalaxyClassification-PHYS449 project is organized into several modules, each with a specific purpose related to the data acquisition, processing, and analysis stages of the project.

1. dataAcquisition: This folder is responsible for obtaining object IDs. It includes code for querying the SDSS to fetch the required data. The object ID lists obtained from these queries are stored as CSV files in the Split_data_IDs folder.

2. dataProcessing: This folder handles all the preprocessing of the data. It fetches the features of the objects using the IDs obtained in the data acquisition stage. The fetched features are stored in the featureVectors folder. After fetching the features, the module performs Principal Component Analysis (PCA) on them to create the input vectors for the machine learning models. The processed data is stored in the processedData folder. Additionally, this module performs some analysis on the PCA-transformed data, the results of which are stored in the PCAAnalysis subfolder within the dataProcessing folder.

In the main GalaxyClassification folder, there are several key files:

1. main.py: This is the main script from which everything is run. It orchestrates the data acquisition, processing, and model training processes using the run command below.

2. ANN_4_main.py: This file contains the code for the ANN model. It defines the structure of the ANN and includes functions for training the model and evaluating its performance.

3. ET.py: This file contains the code for the Extra Trees model. It defines the structure of the Extra Trees model and includes functions for training the model and evaluating its performance.

By organizing the code in this way, each folder and file can focus on a specific part of the data pipeline, making the code easier to understand and maintain. The use of separate folders for different types of data also helps keep the project organized and makes it easier to track the flow of data through the pipeline.


## Run instructions
To train Extra Trees on existing post-PCA files with default hyperparameters:
```sh
python main.py -ET
```

To train the ANN on existing post-PCA files with default hyperparameters:
```sh
python main.py -ANN
```

To preprocess a list of objects to use for the models (fetch the features and perform PCA):
```sh
python main.py -PROCESS --OBJ_LIST "test.csv" 
```

*NOTE*
When specifying file paths be sure to follow the following:
- For training/validation/testing parameters the file name input must match the file in the dataProcessing/processedData folder
- For --OBJ_LIST the file name input must match the file in the dataAcquisition/Split_data_IDs folder
- For --FEATURE_OUT the file name input must match the file in the dataProcessing/featureVectors folder
- For --PCA_OUT the file name input must match the file in the dataProcessing/processedData folder

## Dependancies
### This project utilizes the following libraries:
- torch
- numpy
- SDSS
- csv
- sklearn
- pandas
- io
- matplotlib

# Citations
Paper: Moonzarin Reza, Galaxy morphology classification using automated machine learning, Astronomy and Computing, Volume 37, 2021, 100492, ISSN 2213-1337, https://doi.org/10.1016/j.ascom.2021.100492.

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. (https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
