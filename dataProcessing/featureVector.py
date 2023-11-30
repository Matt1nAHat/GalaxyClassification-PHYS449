import csv
import numpy as np
from mySDSS import PhotoObj 

def calcModelColour(magnitudeDict):
    """
    Calculate model colour as the difference of adjacent filter's model magnitude.

    Parameters:
    modelMagnitudeDict (dict): A dictionary where the key is the filter name ('u', 'g', 'r', 'i', 'z') and the value is the corresponding model magnitude.

    Returns:
    dict: A dictionary where the key is the model colour name ('u-g', 'g-r', 'r-i', 'i-z') and the value is the corresponding model colour.
    """
    colorDict = {}
    filters = ['u', 'g', 'r', 'i', 'z']
    for i in range(len(filters) - 1):
        colorDict[filters[i] + '-' + filters[i+1]] = magnitudeDict[filters[i]] - magnitudeDict[filters[i+1]]
    return colorDict

def calcConcentration(petroR90, petroR50):
    """
    Calculate the concentration indicies for each filter by taking the ratio of the petroR90 and petroR50 values.

    Parameters:
    petroR90, petroR50 (dict): The dictionaries.

    Returns:
    dict: A dictionary where the key is the same as the inputs, and the value is the ratio of the input values.
    """
    ratioDict = {}
    for key in petroR90.keys():
        ratioDict[key] = petroR90[key] / petroR50[key]
    return ratioDict

def getFeatures(obj_ID):
    """
    Get the features of the object from the SDSS database and create the feature vector.

    Parameters:
    obj_ID (int): The ID of the object.

    Returns:
    list: The feature vector.
    """
    ph = PhotoObj(obj_ID)
    ph.cutout_image()
    featureList = []
    
    #Append features to featureList
    featureList.append(obj_ID)
    featureList.append(ph.type) # galaxy or star
    featureList.append(ph.p_el) # probability of being an elliptical galaxy
    featureList.append(ph.p_cw) # probability of being a clockwise spiral galaxy
    featureList.append(ph.p_acw) # probability of being an anticlockwise spiral galaxy
    featureList.append(ph.p_edge) # probability of being an edge-on galaxy
    featureList.append(ph.p_mg) # probability of being a merger
    featureList.extend(calcModelColour(ph.fiberColour).values())
    featureList.extend(calcModelColour(ph.modelColour).values())
    featureList.extend(calcModelColour(ph.petroColour).values())
    featureList.extend(calcConcentration(ph.petroR90,ph.petroR50).values())
    featureList.extend(ph.secondMoment.values())
    featureList.extend(ph.fourthMoment.values())
    featureList.extend(ph.axisRatioDEV.values())
    featureList.extend(ph.axisRatioEXP.values())
    featureList.extend(ph.ellipticityE1.values())
    featureList.extend(ph.ellipticityE2.values())
    featureList.extend(ph.modelFitDEV.values())
    featureList.extend(ph.modelFitEXP.values())
    featureList.extend(ph.modelFitSTAR.values())

    # Convert the feature list to a numpy array
    feature_vector = np.array(featureList)

    return feature_vector

def saveFeatureVectors(csvPath='Objectlist.csv', outPath='galaxyDataset_1000ea.txt'):
    """
    This function reads a CSV file containing object IDs, retrieves the feature vectors for each object ID using the getFeatures function, 
    and writes the feature vectors to an output text file.

    Parameters:
    csvPath (str): The path to the CSV file containing the object IDs. Defaults to 'Objectlist.csv'.
    outPath (str): The path to the output text file. Defaults to 'galaxyDataset_1000ea.txt'.

    Returns:
    None
    """
    
    # Open the CSV file
    with open(csvPath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        next(reader)  # Skip the second row

        # Open the output text file for writing
        with open(outPath, 'w') as out_file:

            # Loop through each row in the CSV file
            for row in reader:
                # Get the object ID from the first column
                object_id = int(row[0])
                out_file.write(str(getFeatures(object_id)))