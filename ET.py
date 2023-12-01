import numpy as np

from sklearn.ensemble import ExtraTreesClassifier

import pickle

# Step 1: Load data
def txtToList(filePath):
    """Takes filepath of a .txt of the form:
    label1 feature1.1 feature1.2 feature1.3 ...
    label2 feature2.1 feature2.2 feature2.3 ...
    
    And returns two lists:
    trainFeatures: listof(listof(float))
    trainLabels: listof(str)"""
    with open(filePath, 'r') as f:
        trainFeatures = []
        trainLabels = []
        for line in f.readlines():
            line = line.split(' ')
            trainLabels.append(line[0])
            trainFeatures.append(list(map(float, line[1:])))
    return trainFeatures, trainLabels
            

trainPath = './dataProcessing/processedData/trainPCAList.txt'
trainFeatures, trainLabels = txtToList(trainPath)
testPath = './dataProcessing/processedData/testPCAList.txt'
testFeatures, testLabels = txtToList(testPath)
validPath = './dataProcessing/processedData/validPCAList.txt'
validFeatures, validLabels = txtToList(validPath)

# Step 2: Train model
# Initialize the ExtraTreesClassifier
clf = ExtraTreesClassifier(
    n_estimators=10, 
    criterion='gini', 
    max_depth=20, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features=1, 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    bootstrap=False,
    oob_score=False,
    n_jobs=None, 
    random_state=None, 
    verbose=0, 
    warm_start=False, 
    class_weight=None)


# Fit the model using the training data
clf.fit(trainFeatures, trainLabels)

inference = clf.predict(trainFeatures)
# print(inference[0:10])
# print(trainLabels[0:10])
trainAccuracy = clf.score(trainFeatures, trainLabels)
print(f"Training   (70% of Data) Accuracy: {trainAccuracy}")

# Step 3: Evaluate model

# Evaluate the model using the testing data
testAccuracy = clf.score(testFeatures, testLabels)
print(f"Testing    (20% of Data) Accuracy: {testAccuracy}")

# Step 4: Validation
# Evaluate the model using the validation data
validAccuracy = clf.score(validFeatures, validLabels)
print(f"Validation (10% of Data) Accuracy: {validAccuracy}")

# Write All accuracies to a file, ET_accuracies.txt
with open('./ET_accuracies.txt', 'w') as f:
    f.write(f"""Training   (70% of Data) Accuracy: {trainAccuracy}
Testing    (20% of Data) Accuracy: {testAccuracy}
Validation (10% of Data) Accuracy: {validAccuracy}""")

# Step 5: Save model
# Save the model for future use
with open('trained_ET_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
