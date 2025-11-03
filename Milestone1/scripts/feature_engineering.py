# Milestone1/scripts/feature_engineering.py

import numpy as np

def add_rul(df):
    """
    Calculates and adds the RUL (Remaining Useful Life) column to the DataFrame.
    RUL is the number of cycles remaining until failure.
    """
    # Get the max cycle for each engine
    max_cycles = df.groupby('unit_nr')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max_cycle']
    
    # Merge max_cycle back into the original dataframe
    df = df.merge(max_cycles, on='unit_nr', how='left')
    
    # Calculate RUL
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    # Drop the temporary max_cycle column
    df = df.drop(columns=['max_cycle'])
    
    print("Added RUL column.")
    return df

def clip_rul(df, clip_value=130):
    """
    Clips the RUL values. 
    It's common practice to cap RUL, as degradation is not linear
    far from failure.
    """
    df['RUL'] = df['RUL'].clip(upper=clip_value)
    print(f"Clipped RUL at {clip_value} cycles.")
    return df

def create_sequences(df, sensor_cols, sequence_length):
    """
    Generates rolling window sequences from the time-series data.
    """
    sequences = []
    targets = []
    
    # Get unique engine IDs
    engine_ids = df['unit_nr'].unique()
    
    for engine_id in engine_ids:
        # Get data for a single engine
        engine_df = df[df['unit_nr'] == engine_id]
        
        # Get sensor data and RUL data as numpy arrays
        sensor_data = engine_df[sensor_cols].values
        rul_data = engine_df['RUL'].values
        
        # Create sequences
        # We slide a window of size 'sequence_length' over the data
        for i in range(len(engine_df) - sequence_length):
            seq = sensor_data[i : i + sequence_length]
            target = rul_data[i + sequence_length - 1] # RUL at the end of the sequence
            
            sequences.append(seq)
            targets.append(target)
            
    print(f"Generated {len(sequences)} sequences.")
    
    return np.array(sequences), np.array(targets)