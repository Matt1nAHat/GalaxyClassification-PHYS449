from featureVector import saveFeatureVectors
from PCA import performPCA
import argparse

#Parser for command-line options, arguments and sub-commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull data from obeject list, create feature vectors and perform PCA on the data")
    parser.add_argument("--objList", default="../dataAcquisition/Split_data_IDs/train.csv", help="path to the object list you want to process")
    parser.add_argument("--fOut", default="featureVectors/featuresList.txt", help="path to save the feature vectors")
    parser.add_argument("--pcaOut", default="processedData/PCAList.txt", help="path to save the final PCA data with labels")

    #Add arguments to the parser
    args = parser.parse_args()


    saveFeatureVectors(args.objList, args.fOut)
    performPCA(args.fOut, args.pcaOut)
