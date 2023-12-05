import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def ET(trainPath = './dataProcessing/processedData/trainPCAList.txt', 
    testPath = './dataProcessing/processedData/testPCAList.txt',
    validPath = './dataProcessing/processedData/validPCAList.txt',
    depthRange = range(5, 15, 2),
    numOfTreesRange = range(10, 150, 10),
    verbose = False):
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
                

    
    trainPath = './dataProcessing/processedData/PCA_85k_train.txt'
    trainPath = './dataProcessing/processedData/PCA_85k_test.txt'
    validPath = './dataProcessing/processedData/PCA_85k_valid.txt'
    
    testFeatures, testLabels = txtToList(testPath)
    trainFeatures, trainLabels = txtToList(trainPath)
    validFeatures, validLabels = txtToList(validPath)

    depthRange = [10] # To have a fixed depth
    numOfTreesRange = [150] # To have a fixed number of trees
    depthRange = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] # To have a range of depths
    numOfTreesRange = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # To have a range of number of trees
    validationResults = []
    maxValidAccuracy = 0
    maxTrainAccuracy = 0


# Step 2: Train model (initialize, fit) for each combination of hyperparameters
    for depth in depthRange:
        validationRow = []
        for num_of_trees in numOfTreesRange:
            # Initialize the ExtraTreesClassifier
            clf = ExtraTreesClassifier(
                n_estimators=num_of_trees, 
                criterion='gini', 
                max_depth=depth, 
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
                class_weight={'Elliptical':0.473, 'Spiral': 0.284, 'Star':0.237, 'Merger':0.006})

            # Fit the model using the training data
            clf.fit(trainFeatures, trainLabels)

            trainAccuracy = clf.score(trainFeatures, trainLabels)
            

            # Step 3: Evaluate model
            # Evaluate the model using the testing data
            testAccuracy = clf.score(testFeatures, testLabels)


            # Step 4: Validation
            # Evaluate the model using the validation data
            validAccuracy = clf.score(validFeatures, validLabels)
            validationRow.append(round(validAccuracy, 5))
            
            # Save the best model
            if validAccuracy > maxValidAccuracy:
                maxValidAccuracy = validAccuracy
                bestValidDepth = depth
                bestValidNumOfTrees = num_of_trees
                bestModel = clf
            if trainAccuracy > maxTrainAccuracy:
                maxTrainAccuracy = trainAccuracy
                bestTrainDepth = depth
                bestTrainNumOfTrees = num_of_trees

            ResultsString = f"""Training   (70% of Data) Accuracy: {trainAccuracy}
Testing    (10% of Data) Accuracy: {testAccuracy}
Validation (20% of Data) Accuracy: {validAccuracy}"""
            
            if verbose:
                print(f"Depth: {depth}, Num of Trees: {num_of_trees}")
                print(ResultsString)
        validationResults.append(validationRow)
    
    inference = bestModel.predict(trainFeatures)
    print(inference[0:30])
    print(trainLabels[0:30])

    # Step 5: Plot validation results

    # Create a figure
    fig = plt.figure()

    # Create a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid for depthRange and numOfTreesRange
    depthRange, numOfTreesRange = np.meshgrid(depthRange, numOfTreesRange)
    # Convert validationResults to a numpy array
    validationResults = np.array(validationResults)
    validationResults = validationResults.transpose()

    # Now you can plot the surface
    ax.plot_surface(numOfTreesRange, depthRange, validationResults)

    # Set labels
    ax.set_ylabel('Depth')
    ax.set_xlabel('Number of Trees')
    ax.set_zlabel('Validation Results')
    ax.set_title(f'Highest validation accuracy: {maxValidAccuracy}, at depth {bestValidDepth} and {bestValidNumOfTrees} trees\nHighest training accuracy: {trainAccuracy}, at depth {bestTrainDepth} and {bestTrainNumOfTrees} trees')
    

    # Show the plot
    plt.show()
    
    # Step 6: Create confusion matrix for best model

    # Get the predicted labels for the validation set
    validPredictions = bestModel.predict(validFeatures)

    # Calculate the confusion matrix
    confMatrix = confusion_matrix(validLabels, validPredictions, labels=['Elliptical', 'Spiral', 'Star', 'Merger'])
    print('Confusion Matrix:\n', confMatrix)

    # Calculate precision and recall
    precision = precision_score(validLabels, validPredictions, average='macro', zero_division=0)
    recall = recall_score(validLabels, validPredictions, average='macro')
    print('Precision:', round(precision, 5))
    print('Recall:', round(recall, 5))


    # Write All accuracies to a file, ET_accuracies.txt
    with open('./ET_Results/ET_accuracies.txt', 'w') as f:
        f.write(ResultsString)

    # Step 7: Save best model as pickel file for future use
    with open('./ET_Results/trained_ET_model.pkl', 'wb') as f:
        pickle.dump(bestModel, f)

ET()