# Milestone1/scripts/data_preprocessing.py
# (Corrected Version)

import pandas as pd

def load_data(file_path):
    """
    Loads the raw CMAPSS .txt data into a pandas DataFrame.
    """
    # Define the column names
    column_names = [
        'unit_nr', 'cycle', 'setting1', 'setting2', 'setting3',
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
        's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
        's20', 's21'
    ]
    
    # --- THIS IS THE FIX ---
    # Use sep='\s+' to handle variable whitespace between numbers
    df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
    
    # The 'df.drop' line is not needed
    
    print(f"Loaded data from {file_path}. Shape: {df.shape}")
    return df

if __name__ == '__main__':
    # This allows you to test the script directly
    test_path = '../data/train_FD001.txt'
    try:
        df = load_data(test_path)
        print("\n--- Sample Data Head ---")
        print(df.head())
        print("\n--- Data Info ---")
        df.info() # Check here for any 'non-null' counts
    except FileNotFoundError:
        print(f"Test file not found at {test_path}. Make sure the file is in the correct location.")