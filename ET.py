import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
                

    
    # trainPath = './dataProcessing/processedData/PCA_50k_train.txt'
    trainFeatures, trainLabels = txtToList(trainPath)
    # trainPath = './dataProcessing/processedData/PCA_50k_test.txt'
    testFeatures, testLabels = txtToList(testPath)
    # validPath = './dataProcessing/processedData/PCA_50k_valid.txt'
    validFeatures, validLabels = txtToList(validPath)

    # depthRange = [10] # To have a fixed depth
    # numOfTreesRange = [50] # To have a fixed number of trees
    validationResults = []
    maxAccuracy = 0

    for depth in depthRange:
        validationRow = []
        for num_of_trees in numOfTreesRange:
            # Step 2: Train model (initialize, fit)
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
                class_weight=None)

            # Fit the model using the training data
            clf.fit(trainFeatures, trainLabels)

            inference = clf.predict(trainFeatures)
            # print(inference[0:10])
            # print(trainLabels[0:10])
            trainAccuracy = clf.score(trainFeatures, trainLabels)
            

            # Step 3: Evaluate model
            # Evaluate the model using the testing data
            testAccuracy = clf.score(testFeatures, testLabels)


            # Step 4: Validation
            # Evaluate the model using the validation data
            validAccuracy = clf.score(validFeatures, validLabels)
            validationRow.append(round(validAccuracy, 5))
            
            # Save the best model
            if validAccuracy > maxAccuracy:
                maxAccuracy = validAccuracy
                bestDepth = depth
                bestNumOfTrees = num_of_trees
                bestModel = clf

            ResultsString = f"""Training   (70% of Data) Accuracy: {trainAccuracy}
            Testing    (10% of Data) Accuracy: {testAccuracy}
            Validation (20% of Data) Accuracy: {validAccuracy}"""
            
            if verbose:
                print(f"Depth: {depth}, Num of Trees: {num_of_trees}")
                print(ResultsString)
        validationResults.append(validationRow)

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
    ax.set_title(f'Highest accuracy: {maxAccuracy}, at depth {bestDepth} and {bestNumOfTrees} trees')

    # Show the plot
    plt.show()


    # Write All accuracies to a file, ET_accuracies.txt
    with open('./ET_Results/ET_accuracies.txt', 'w') as f:
        f.write(ResultsString)

    # Step 6: Save model as pickel file for future use
    with open('./ET_Results/trained_ET_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

