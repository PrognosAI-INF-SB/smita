# Milestone4/scripts/run_alert_system.py

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# --- 1. Define Risk Thresholds ---
# These are the RUL values that trigger an alert.
# You can tune these based on your needs.
CRITICAL_LEVEL = 15  # RUL <= 15: Needs immediate maintenance
WARNING_LEVEL = 30   # RUL <= 30: Schedule maintenance soon

# --- 2. Define Constants (Paths) ---
DATA_DIR = '../../Milestone1/data/'
PROCESSED_DATA_DIR = '../../Milestone2/data/processed/'
MODEL_PATH = '../../Milestone2/models/PrognosAI_LSTM.keras'

TEST_FILE = 'test_FD001.txt'

# Sensor columns and sequence length
SENSOR_COLS = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 
    's13', 's14', 's15', 's17', 's20', 's21'
]
SEQUENCE_LENGTH = 50

def classify_risk(rul_prediction):
    """Classifies a predicted RUL into a risk category."""
    if rul_prediction <= CRITICAL_LEVEL:
        return "🔴 CRITICAL (Maintain Immediately)"
    elif rul_prediction <= WARNING_LEVEL:
        return "🟡 WARNING (Schedule Maintenance)"
    else:
        return "🟢 STABLE (No Action)"

def main():
    print("--- PrognosAI: Starting Alert System ---")
    
    # --- 3. Load Model and Scaler ---
    print("Loading model and scaler...")
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl'))
    except (IOError, FileNotFoundError) as e:
        print(f"Error loading files: {e}")
        print("Ensure Milestone 1 & 2 were completed successfully.")
        return

    # --- 4. Load and Process Test Data (Simulating new data) ---
    print("Loading and processing live data (from test set)...")
    
    column_names = [
        'unit_nr', 'cycle', 'setting1', 'setting2', 'setting3',
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
        's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
        's20', 's21'
    ]
    df_test_raw = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE), sep='\s+', header=None, names=column_names)
    df_test_raw[SENSOR_COLS] = scaler.transform(df_test_raw[SENSOR_COLS])
    
    # --- 5. Prepare Last Sequence for Each Engine ---
    test_sequences = []
    engine_ids_list = []
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
            
        engine_ids_list.append(engine_id)

    X_test = np.array(test_sequences)

    # --- 6. Make Predictions and Classify Risk ---
    print("Generating RUL predictions and risk alerts...")
    y_pred = model.predict(X_test).flatten()
    
    # Create a results DataFrame
    results = pd.DataFrame({
        'Engine_ID': engine_ids_list,
        'Predicted_RUL': np.round(y_pred, 2)
    })
    
    results['Risk_Level'] = results['Predicted_RUL'].apply(classify_risk)
    
    # Sort by RUL to see most critical engines first
    results = results.sort_values(by='Predicted_RUL')
    
    # --- 7. Display Maintenance Alert Report ---
    print("\n\n" + "="*60)
    print("      PrognosAI: DAILY MAINTENANCE ALERT REPORT")
    print("="*60)
    print(results.to_string()) # .to_string() prints all rows
    print("="*60)
    
    # Print a summary
    critical_count = (results['Risk_Level'] == "🔴 CRITICAL (Maintain Immediately)").sum()
    warning_count = (results['Risk_Level'] == "🟡 WARNING (Schedule Maintenance)").sum()
    
    print("\n--- Alert Summary ---")
    print(f"🔴 CRITICAL Engines: {critical_count}")
    print(f"🟡 WARNING Engines:  {warning_count}")
    print(f"🟢 STABLE Engines:   {100 - critical_count - warning_count}")
    print("--- System Run Complete ---")


if __name__ == '__main__':
    main()