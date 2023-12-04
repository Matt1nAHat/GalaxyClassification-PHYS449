import argparse
from ET import ETModel
#from ANN import ANNModel
from sklearn.ensemble import ExtraTreesClassifier
import pickle
from dataProcessing.featureVector import saveFeatureVectors
from dataProcessing.PCA import performPCA 
import dataProcessing.mySDSS

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Provide hyperparameters & dataset filepaths.")

    # Add the arguments. First 3 are whether to run PROCESS, ET, or ANN. The rest are hyperparameters and/or dataset filepaths.
    parser.add_argument('--PROCESS', default=False, type=bool, help='Run data preprocessing on object list to obtain input data for ML models') # PROCESS
    parser.add_argument('--ET', default=False, type=bool, help='Run the ET model; specify hyperparameters and/or datasets or neither for default values.') # ET
    parser.add_argument('--ANN', default=False, type=bool, help='Run the ANN model; specify hyperparameters and/or datasets or neither for default values.') # ANN
    # ET arguments
    parser.add_argument('--TRAIN_PATH', type=str, help='The path to the training dataset')
    parser.add_argument('--TEST_PATH', type=str, help='The path to the testing dataset')
    parser.add_argument('--VALID_PATH', type=str, help='The path to the validation dataset')
    parser.add_argument('--DEPTH', type=str, help='The max depth; can be an int or a list of ints')
    parser.add_argument('--NUM_OF_TREES', type=int, help='The number of trees in the forest')
    parser.add_argument('--VERBOSE', type=bool, help='The verbosity of the model')
    # ANN arguments
    parser.add_argument('--NUM_EPOCHS', type=int, help='The number of epochs to train the model for')
    # Preprocess arguments
    parser.add_argument("--OBJ_LIST", default="dataAcquisition/Split_data_IDs/test.csv", help="path to the object list you want to process")
    parser.add_argument("--FEATURE_OUT", default="dataProcessing/featureVectors/featuresList.txt", help="path to save the feature vectors")
    parser.add_argument("--PCA_OUT", default="dataProcessing/processedData/PCAList.txt", help="path to save the final PCA data with labels")
    
    
    # Execute the parse_args() method, obtain whether to run ET or ANN
    args = parser.parse_args()
    run_PROCESS = args.PROCESS
    run_ET = args.ET
    run_ANN = args.ANN

    # If Extra Trees is chosen:
    if run_ET:
        TRAIN_PATH = args.TRAIN_PATH
        TEST_PATH = args.TEST_PATH
        VALID_PATH = args.VALID_PATH
        VERBOSE = args.VERBOSE

        # Now parse the DEPTH argument
        try:
            # Try to parse it as an integer
            DEPTH = int(args.DEPTH)
        except ValueError:
            # If that fails, try to parse it as a list of integers
            DEPTH = [int(item) for item in args.DEPTH.split(',')]
        
        # Now parse the NUM_OF_TREES argument
        try:
            # Try to parse it as an integer
            NUM_OF_TREES = int(args.NUM_OF_TREES)
        except ValueError:
            # If that fails, try to parse it as a list of integers
            NUM_OF_TREES = [int(item) for item in args.NUM_OF_TREES.split(',')]
        
        ETModel(TRAIN_PATH, TEST_PATH, VALID_PATH, DEPTH, NUM_OF_TREES, VERBOSE)

    if run_ANN:
        # Code here for ANN
        print("change your code to a function named ANNModel that takes in parameters from here instead")
        #ANNModel(args.NUM_EPOCHS)


    # If PROCESS is chosen:
    if run_PROCESS:
        # Run the preprocessing
        saveFeatureVectors(args.OBJ_LIST, args.FEATURE_OUT)
        performPCA(args.FEATURE_OUT, args.PCA_OUT)