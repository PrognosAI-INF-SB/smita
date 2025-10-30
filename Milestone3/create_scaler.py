import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

print("🔹 Creating scaler with 21 sensor features...")

# Path to CMAPSS train data
train_path = r"C:\Users\siddharth\Downloads\archive\CMaps\train_FD001.txt"
train_df = pd.read_csv(train_path, sep=r"\s+", header=None)

# Define column names
col_names = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1, 22)]
train_df.columns = col_names

# Use only 21 sensor columns
sensor_cols = [f"s{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
scaler.fit(train_df[sensor_cols])

scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
joblib.dump(scaler, scaler_path)

print(f"✅ scaler.pkl successfully created with {len(sensor_cols)} features at {scaler_path}")
