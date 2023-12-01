import pandas as pd

# Load the CSV data into pandas DataFrames
df1 = pd.read_csv("dataAcquisition\Object_IDs_and_classifications\GZ_mergers_296.csv")
df2 = pd.read_csv("dataAcquisition\Object_IDs_and_classifications\mg_spec_420.csv")

# Concatenate both dataframes
combined = pd.concat([df1, df2])

combined_col1 = pd.concat([df1.iloc[:, 0], df2.iloc[:, 0]])

# Count the number of unique values
num_unique = combined_col1.nunique()

if num_unique > 0:
    # Drop duplicate rows based on the first column
    unique_rows = combined.drop_duplicates(subset=combined.columns[0])

    # Save the DataFrame with unique rows to a new CSV file
    unique_rows.to_csv(f'GZ_mergers_{num_unique}', index=False)

    print(f'New file with unique rows has been saved as "unique_rows.csv".')

elif num_unique == 0:
    print("No duplicate rows found.")