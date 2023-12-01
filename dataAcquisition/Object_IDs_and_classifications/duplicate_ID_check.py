# import pandas as pd

# # Load the CSV data into a pandas DataFrame
# df = pd.read_csv("dataAcquisition\Object_IDs_and_classifications\mg_new.csv")

# # Drop duplicate rows based on the first column
# unique_rows = df.drop_duplicates(subset=df.columns[0])

# # Calculate the number of duplicate rows
# num_duplicates = len(df) - len(unique_rows)

# if num_duplicates > 0:
#     # Save the DataFrame with unique rows to a new CSV file
#     unique_rows.to_csv("dataAcquisition\Object_IDs_and_classifications\mg_new.csv", index=False)
# elif num_duplicates == 0:
#     print("No duplicate rows found")

import pandas as pd
import numpy as np

# Load the CSV data into a pandas DataFrame
df = pd.read_csv("dataAcquisition\Object_IDs_and_classifications\ZooSpec\combined_50k.csv")

# Identify duplicate rows based on the first column
duplicates = df.duplicated(subset=df.columns[0])

# Find the row numbers of the duplicate values
duplicate_rows = np.where(duplicates)[0]

# Print a message if there are any duplicates
if duplicates.any():
    print(f"There are {duplicates.sum()} duplicate values in the first column.")
    # Print the row numbers of the duplicate values
    print(f"Row numbers of duplicate values in the first column: {duplicate_rows}") 

else:
    print("There are no duplicate values in the first column.")