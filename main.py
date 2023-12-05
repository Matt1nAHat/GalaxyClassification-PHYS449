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
    parser.add_argument('-PROCESS', default=False, action='store_true', help='Run data preprocessing on object list to obtain input data for ML models') # PROCESS
    parser.add_argument('-ET', default=False, action='store_true', help='Run the ET model; specify hyperparameters and/or datasets or neither for default values.') # ET
    parser.add_argument('-ANN', default=False, action='store_true', help='Run the ANN model; specify hyperparameters and/or datasets or neither for default values.') # ANN

    # ET arguments
    parser.add_argument('--DEPTH', type=str, default=10, help='The max depth; can be an int or a list of ints')
    parser.add_argument('--NUM_OF_TREES', type=int, default=50, help='The number of trees in the forest')

    # ANN arguments
    parser.add_argument('--EPOCHS', type=int, default=25, help='Number of epochs for training')
    parser.add_argument('--HS1', type=int, default=12, help='Number of neurons in the first hidden layer')
    parser.add_argument('--HS2', type=int, default=24, help='Number of neurons in the second hidden layer')
    parser.add_argument('--HS3', type=int, default=12, help='Number of neurons in the third hidden layer')
    parser.add_argument('--LR', type=float, default=0.0004, help='Learning rate')
    parser.add_argument('--WD', type=float, default=0.0003, help='Weight decay')
    parser.add_argument('--BATCH_SIZE', type=int, default=30, help='Batch size for training')
    parser.add_argument('-V', action='store_true', default=False, help='Print out loss plots')

    # Input file paths
    parser.add_argument('--TEST_PATH', type=str, default='PCA_85k_test.txt', help='Path to testing dataset txt')
    parser.add_argument('--TRAIN_PATH', type=str, default='PCA_85k_train.txt', help='Path to training dataset txt')
    parser.add_argument('--VALID_PATH', type=str, default='PCA_85k_valid.txt', help='Path to validation dataset txt')

    # Preprocess arguments
    parser.add_argument("--OBJ_LIST", default="test.csv", help="path to the object list you want to process (csv file) - MUST BE IN DATAACQUISITION/SPLIT_DATA_IDS/")
    parser.add_argument("--FEATURE_OUT", default="featuresList.txt", help="path to save the feature vectors (txt file) - MUST BE IN DATAPROCESSING/FEATUREVECTORS/")
    parser.add_argument("--PCA_OUT", default="PCAList.txt", help="path to save the final PCA data with labels (txt file) - MUST BE IN DATAPROCESSING/PROCESSEDDATA/")
    
    
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
        VERBOSE = args.V

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
        
        ETModel(f"dataProcessing/processedData/{TRAIN_PATH}", f"dataProcessing/processedData/{TEST_PATH}", f"dataProcessing/processedData/{VALID_PATH}", DEPTH, NUM_OF_TREES, VERBOSE)

    if run_ANN:
        # Parse the ANN arguments
        EPOCHS = args.EPOCHS
        HS1 = args.HS1
        HS2 = args.HS2
        HS3 = args.HS3
        LR = args.LR
        WD = args.WD
        BATCH_SIZE = args.BATCH_SIZE
        VERBOSE = args.V
        test_path = args.TEST_PATH
        train_path = args.TRAIN_PATH
        valid_path = args.VALID_PATH

        ANN(EPOCHS, HS1, HS2, HS3, LR, WD, BATCH_SIZE, VERBOSE, test_path, train_path, valid_path)     


    # If PROCESS is chosen:
    if run_PROCESS:
        # Run the preprocessing of given files
        saveFeatureVectors(f"dataAcquisition/Split_data_IDs/{args.OBJ_LIST}", f"dataProcessing/featureVectors/{args.FEATURE_OUT}")
        performPCA(f"dataProcessing/featureVectors/{args.FEATURE_OUT}", f"dataProcessing/processedData/{args.PCA_OUT}")