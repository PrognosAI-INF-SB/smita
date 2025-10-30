import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ----------------------------------------------
# Load and preprocess test data
# ----------------------------------------------
print("🔹 Loading test data...")

# Path to CMAPSS test data
test_path = r"C:\Users\siddharth\Downloads\archive\CMaps\test_FD001.txt"
test_df = pd.read_csv(test_path, sep=r"\s+", header=None)

# Define column names: 5 (id + cycle + 3 op_settings) + 21 sensor columns
col_names = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1, 22)]
test_df.columns = col_names

print("🔹 Preprocessing data...")

# Select only the 21 sensor columns for model input (model trained on these)
sensor_cols = [f"s{i}" for i in range(1, 22)]
X_test = test_df[sensor_cols]

# Load correct scaler (trained with 21 features)
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
if not os.path.exists(scaler_path):
    print("⚠️ scaler.pkl not found. Please create it using the same features used during training.")
    exit()

scaler = joblib.load(scaler_path)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------
# Load trained model
# ----------------------------------------------
print("🔹 Loading model...")
model_path = r"D:\PrognosAI_Project\Milestone3\trained_model.h5"
if not os.path.exists(model_path):
    print("❌ Model file not found at:", model_path)
    exit()

model = tf.keras.models.load_model(model_path)

# ----------------------------------------------
# Predict Remaining Useful Life (RUL)
# ----------------------------------------------
print("🔹 Predicting...")
y_pred = model.predict(X_test_scaled)
print(f"✅ Prediction completed. Shape: {y_pred.shape}")

# ----------------------------------------------
# Save predictions to CSV
# ----------------------------------------------
output_df = pd.DataFrame({
    "engine_id": test_df["engine_id"],
    "cycle": test_df["cycle"],
    "Predicted_RUL": y_pred.flatten()
})

output_csv_path = os.path.join(os.path.dirname(__file__), "predicted_RUL.csv")
output_df.to_csv(output_csv_path, index=False)

print(f"✅ Predictions saved successfully at: {output_csv_path}")
