import csv
import pandas as pd
import numpy as np
import sdss


def getData(obj_id):
    """
    Returns a string containing the object ID, classification, and flattened data of the given object.

    Parameters:
    obj_id (int): The ID of the object.

    Returns:
    str: A string containing the object ID, classification, and flattened data.
    """
    ph = sdss.PhotoObj(obj_id)
    data = ph.cutout_image()
    classification = ph.type
    flattened_data = ' '.join(map(str, data.flatten()))
    return f"{obj_id} {classification} {flattened_data}\n"

def writeList(csvPath='Objectlist.csv'):
    """
    Writes a text file with the object ID, classification, and flattened data of each object in the CSV file.

    Parameters:
    csvPath (str): The path to the CSV file. Defaults to 'Objectlist.csv'.
    """
    # Open the CSV file
    with open(csvPath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        # Open the output text file for writing
        with open('galaxyDataset.txt', 'w') as out_file:

            # Loop through each row in the CSV file
            for row in reader:
                # Get the object ID from the first column
                object_id = int(row[0])
                out_file.write(getData(object_id))

def getObj(obj_id):
    """
    Returns the object from the SDSS database.

    Parameters:
    obj_id (int): The ID of the object.

    Returns:
    PhotoObj: The object from the SDSS database.
    """
    return sdss.PhotoObj(obj_id)

