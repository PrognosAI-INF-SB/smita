# Milestone1/scripts/milestone1_prep.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

# Import our custom functions
from data_preprocessing import load_data
from feature_engineering import add_rul, clip_rul, create_sequences

# --- 1. Define Constants ---
TRAIN_FILE = '../data/train_FD001.txt'
SAVE_DIR = '../../Milestone2/data/processed/' # Save processed data for Milestone 2

# These are the columns to use for training. 
# We exclude settings and sensors that are constant.
SENSOR_COLS = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 
    's13', 's14', 's15', 's17', 's20', 's21'
]
SEQUENCE_LENGTH = 50  # Number of time steps in each sequence
RUL_CLIP_VALUE = 130  # Max RUL value

def main():
    print("--- Starting Milestone 1: Data Preparation ---")
    
    # --- 2. Load Data ---
    df_train = load_data(TRAIN_FILE)
    
    # --- 3. Feature Engineering ---
    # Calculate RUL
    df_train = add_rul(df_train)
    # Clip RUL
    df_train = clip_rul(df_train, RUL_CLIP_VALUE)
    
    # --- 4. Normalize Sensor Data ---
    # We use MinMaxScaler to scale sensor values between 0 and 1
    scaler = MinMaxScaler()
    
    # Fit the scaler ONLY on the training data
    df_train[SENSOR_COLS] = scaler.fit_transform(df_train[SENSOR_COLS])
    print("Normalized sensor data using MinMaxScaler.")
    
    # --- 5. Create Sequences ---
    # This transforms the data into (samples, timesteps, features) shape
    # which is required by LSTMs/GRUs
    sequences, targets = create_sequences(df_train, SENSOR_COLS, SEQUENCE_LENGTH)
    
    # --- 6. Save Processed Data ---
    # We'll save the processed data and the scaler
    # so Milestone 2 can use them for training.
    # Make sure the target directory exists!
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    np.save(os.path.join(SAVE_DIR, 'train_sequences.npy'), sequences)
    np.save(os.path.join(SAVE_DIR, 'train_targets.npy'), targets)
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
    
    print(f"\n--- Milestone 1 Complete ---")
    print(f"Saved processed sequences to {SAVE_DIR}train_sequences.npy")
    print(f"Saved processed targets to {SAVE_DIR}train_targets.npy")
    print(f"Saved scaler to {SAVE_DIR}scaler.pkl")

if __name__ == '__main__':
    main()