import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    seed = 22
    # Load the CSV file into a DataFrame, skipping the first two rows
    df = pd.read_csv('dataAcquisition\Object_IDs_and_classifications\ZooSpec\combined_50k.csv')

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=seed) # set seed for random splitting

    # Split the DataFrame into training, validation, and test datasets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed)
    valid_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=seed)

    # Output first column (IDs) for the training, validation, and test datasets to CSV files
    train_df.iloc[:, 0].to_csv('dataAcquisition/Split_data_IDs/train_50k.csv', index=False)
    valid_df.iloc[:, 0].to_csv('dataAcquisition/Split_data_IDs/valid_50k.csv', index=False)
    test_df.iloc[:, 0].to_csv('dataAcquisition/Split_data_IDs/test_50k.csv', index=False)

if __name__ == '__main__':
    main()