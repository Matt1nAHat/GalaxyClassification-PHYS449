import argparse
from ET import ET
from sklearn.ensemble import ExtraTreesClassifier
import pickle

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Provide hyperparameters & dataset filepaths.")

    # Add the arguments
    parser.add_argument('run_ET', type=bool, help='Run the ET model; specify hyperparameters and/or datasets or neither for default values.') # ET
    parser.add_argument('run_ANN', type=bool, help='Run the ANN model; specify hyperparameters and/or datasets or neither for default values.') # ANN
    # ET arguments
    parser.add_argument('TRAIN_PATH', type=str, help='The path to the training dataset')
    parser.add_argument('TEST_PATH', type=str, help='The path to the testing dataset')
    parser.add_argument('VALID_PATH', type=str, help='The path to the validation dataset')
    parser.add_argument('DEPTH', type=str, help='The max depth; can be an int or a list of ints')
    parser.add_argument('NUM_OF_TREES', type=int, help='The number of trees in the forest')
    parser.add_argument('VERBOSE', type=bool, help='The verbosity of the model')
    # ANN arguments
    parser.add_argument('NUM_EPOCHS', type=int, help='The number of epochs to train the model for')
    
    
    # Execute the parse_args() method, obtain whether to run ET or ANN
    args = parser.parse_args()
    run_ET = args.run_ET
    run_ANN = args.run_ANN
    
    # If Extra Trees is chosen:
    if run_ET:
        TRAIN_PATH = args.TRAIN_PATH
        TEST_PATH = args.TEST_PATH
        VALID_PATH = args.VALID_PATH
        VERBOSE = args.VERBOSE

        # Parse the arguments
        args = parser.parse_args()

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
        
        ET(TRAIN_PATH, TEST_PATH, VALID_PATH, DEPTH, NUM_OF_TREES, VERBOSE)