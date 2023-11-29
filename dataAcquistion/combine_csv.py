import pandas as pd
import glob

# Step 2: Get a list of filenames
filenames = glob.glob('dataAcquistion\Object_IDs_and_classifications\*.csv')  # replace with your actual path and file pattern

# Step 3: Read each CSV file into a DataFrame and store all DataFrames in a list
df_list = [pd.read_csv(f) for f in filenames]

# Step 4: Concatenate all DataFrames in the list into one DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# Step 5: Write the combined DataFrame to a new CSV file
combined_df.to_csv('dataAcquistion\Object_IDs_and_classifications\combined.csv', index=False)