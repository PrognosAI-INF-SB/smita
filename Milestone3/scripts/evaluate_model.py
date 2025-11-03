# Milestone3/scripts/evaluate_model.py

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# --- 1. Define Constants (Corrected Paths) ---
# Go up two levels (..) to PrognosAI/, then down to the correct folders
DATA_DIR = '../../Milestone1/data/'
PROCESSED_DATA_DIR = '../../Milestone2/data/processed/'
MODEL_PATH = '../../Milestone2/models/PrognosAI_LSTM.keras'

# Save plots locally within Milestone3
PLOT_SAVE_DIR = '../plots/' 
os.makedirs(PLOT_SAVE_DIR, exist_ok=True) # Ensure plot directory exists

TEST_FILE = 'test_FD001.txt'
RUL_FILE = 'RUL_FD001.txt'

# Load sensor columns and sequence length from Milestone 1
SENSOR_COLS = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 
    's13', 's14', 's15', 's17', 's20', 's21'
]
SEQUENCE_LENGTH = 50
RUL_CLIP_VALUE = 130 # From Milestone 1

# --- 2. Load Model and Scaler ---
print("Loading model and scaler...")
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl'))
except (IOError, FileNotFoundError) as e:
    print(f"Error loading files: {e}")
    print("Ensure Milestone 1 & 2 were completed successfully.")
    print(f"Looking for model at: {MODEL_PATH}")
    print(f"Looking for scaler at: {os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl')}")
    exit()

model.summary()

# --- 3. Load and Process Test Data ---
print("Loading and processing test data...")

# Load raw test data
column_names = [
    'unit_nr', 'cycle', 'setting1', 'setting2', 'setting3',
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
    's20', 's21'
]
df_test_raw = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE), sep='\s+', header=None, names=column_names)

# Load true RUL values
df_rul = pd.read_csv(os.path.join(DATA_DIR, RUL_FILE), sep='\s+', header=None, names=['true_RUL'])

# Normalize the test data sensors
# IMPORTANT: Use .transform(), NOT .fit_transform()
df_test_raw[SENSOR_COLS] = scaler.transform(df_test_raw[SENSOR_COLS])

# --- 4. Prepare Test Sequences ---
print("Preparing test sequences...")
test_sequences = []
test_targets = []
engine_ids = df_test_raw['unit_nr'].unique()

for engine_id in engine_ids:
    engine_df = df_test_raw[df_test_raw['unit_nr'] == engine_id]
    last_sequence = engine_df[SENSOR_COLS].tail(SEQUENCE_LENGTH).values
    
    if len(last_sequence) < SEQUENCE_LENGTH:
        padded_sequence = np.zeros((SEQUENCE_LENGTH, len(SENSOR_COLS)))
        padded_sequence[-len(last_sequence):] = last_sequence
        test_sequences.append(padded_sequence)
    else:
        test_sequences.append(last_sequence)
        
    true_rul_value = df_rul.loc[engine_id - 1, 'true_RUL']
    test_targets.append(min(true_rul_value, RUL_CLIP_VALUE))

X_test = np.array(test_sequences)
y_test = np.array(test_targets)

print(f"Test sequences shape: {X_test.shape}")
print(f"Test targets shape: {y_test.shape}")

# --- 5. Make Predictions ---
print("Making predictions on test data...")
y_pred = model.predict(X_test)
y_pred = y_pred.flatten()

# --- 6. Evaluate Performance ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n--- Model Evaluation Complete ---")
print(f"Test RMSE: {rmse:.4f}")

# --- 7. Plot Predictions vs. Actual ---
print("Generating predictions plot...")
plt.figure(figsize=(12, 6))
# Using markers for better visibility
plt.plot(y_test, 'o-', label='Actual RUL', markersize=5, alpha=0.7)
plt.plot(y_pred, 'x--', label='Predicted RUL', markersize=5, alpha=0.7)
plt.title(f'PrognosAI: Test Set Predictions (RMSE: {rmse:.4f})')
plt.xlabel('Engine ID')
plt.ylabel('Remaining Useful Life (Cycles)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plot_path = os.path.join(PLOT_SAVE_DIR, 'test_predictions_vs_actual.png')
plt.savefig(plot_path)
print(f"Saved prediction plot to {plot_path}")
plt.show()