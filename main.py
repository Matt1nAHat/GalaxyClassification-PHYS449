import argparse
from ET import ETModel
#from ANN import ANNModel
from sklearn.ensemble import ExtraTreesClassifier
import pickle
from dataProcessing.featureVector import saveFeatureVectors
from dataProcessing.PCA import performPCA 
import dataProcessing.mySDSS
from ANN_4_main import ANN

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
    parser.add_argument('--EPOCHS', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--HS1', type=int, default=12, help='Number of neurons in the first hidden layer')
    parser.add_argument('--HS2', type=int, default=24, help='Number of neurons in the second hidden layer')
    parser.add_argument('--HS3', type=int, default=16, help='Number of neurons in the third hidden layer')
    parser.add_argument('--LR', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--WD', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch size for training')
    parser.add_argument('-v', type=bool, default=False, help='Print out loss plots')

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
        # Parse the ANN arguments
        EPOCHS = args.EPOCHS
        HS1 = args.HS1
        HS2 = args.HS2
        HS3 = args.HS3
        LR = args.LR
        WD = args.WD
        BATCH_SIZE = args.BATCH_SIZE
        VERBOSE = args.v

        ANN(EPOCHS, HS1, HS2, HS3, LR, WD, BATCH_SIZE, VERBOSE)     


    # If PROCESS is chosen:
    if run_PROCESS:
        # Run the preprocessing
        saveFeatureVectors(args.OBJ_LIST, args.FEATURE_OUT)
        performPCA(args.FEATURE_OUT, args.PCA_OUT)