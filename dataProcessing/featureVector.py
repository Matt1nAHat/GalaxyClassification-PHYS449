import csv
import numpy as np
from .mySDSS import PhotoObj 

def calcColourDiff(magnitudeDict):
    """
    Calculate colour as the difference of adjacent filter's magnitude.

    Parameters:
    magnitudeDict (dict): A dictionary where the key is the filter name ('u', 'g', 'r', 'i', 'z') and the value is the corresponding magnitude.

    Returns:
    dict: A dictionary where the key is the colour name ('u-g', 'g-r', 'r-i', 'i-z') and the value is the corresponding colour.
    """
    # Initialize the dictionary
    colorDict = {}
    filters = ['u', 'g', 'r', 'i', 'z']
    # Calculate the colour for each filter
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
    # Initialize the dictionary
    ratioDict = {}
    # Calculate the ratio for each filter
    for key in petroR90.keys():
        ratioDict[key] = petroR90[key] / petroR50[key]
    return ratioDict

def getFeatures(obj_ID, zoo):
    """
    Get the features of the object from the SDSS database and create the feature vector.

    Parameters:
    obj_ID (int): The ID of the object.

    Returns:
    np array: The feature vector.
    """
    # Initialize the feature list and create a photo object from the object ID
    featureList = []
    ph = PhotoObj(obj_ID)
    #For objects that are catelogued in Galaxy Zoo
    if zoo:
        ph.download()
        featureList.append(obj_ID)
        featureList.append(ph.type) # galaxy or star
        featureList.append(ph.p_el) # probability of being an elliptical galaxy
        featureList.append(ph.p_cw) # probability of being a clockwise spiral galaxy
        featureList.append(ph.p_acw) # probability of being an anticlockwise spiral galaxy
        featureList.append(ph.p_edge) # probability of being an edge-on galaxy
        featureList.append(ph.p_mg) # probability of being a merger
    
    #For all stars and galaxies not catelogued in Galaxy Zoo
    else:
        ph.downloadNoZoo()
        featureList.append(obj_ID)
        featureList.append(ph.type) # galaxy or star
        #Additional appends to retain feature vector dimensionality
        featureList.append(0) # probability of being an elliptical galaxy
        featureList.append(0) # probability of being a clockwise spiral galaxy
        featureList.append(0) # probability of being an anticlockwise spiral galaxy
        featureList.append(0) # probability of being an edge-on galaxy
        featureList.append(0) # probability of being a merger

    
    #Append remaining features to featureList
    featureList.extend(calcColourDiff(ph.fiberColour).values())
    featureList.extend(calcColourDiff(ph.modelColour).values())
    featureList.extend(calcColourDiff(ph.petroColour).values())
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

def saveFeatureVectors(csvPath, outPath):
    """
    This function reads a CSV file containing object IDs, retrieves the feature vectors for each object ID using the getFeatures function, 
    and writes the feature vectors to a text file.

    Parameters:
    csvPath (str): The path to the CSV file containing the object IDs
    outPath (str): The path to the output text file

    Returns:
    None
    """
    # Open the CSV file
    with open(csvPath, 'r') as f:
        reader = csv.reader(f)

        # Open the output text file for writing
        with open(outPath, 'w') as out_file:
            for row in reader:
                object_id = int(row[0])
                # Try to get the features for the object ID
                try:
                    out_file.write(str(getFeatures(object_id, True)))
                # If the object ID is not in Galaxy Zoo, get the features for the object ID without the Galaxy Zoo classifications
                except Exception:
                    out_file.write(str(getFeatures(object_id, False)))
                    continue