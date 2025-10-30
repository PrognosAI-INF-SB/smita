import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# --------------------------
# 🔹 PATH SETTINGS
# --------------------------
BASE_PATH = r"D:\PrognosAI_Project"
TEST_DATA_PATH = os.path.join(BASE_PATH, "Milestone3", "test_FD001.txt")
SCALER_PATH = os.path.join(BASE_PATH, "Milestone3", "scaler.pkl")
MODEL_PATH = os.path.join(BASE_PATH, "Milestone3", "trained_model.h5")
OUTPUT_CSV_PATH = os.path.join(BASE_PATH, "Milestone4", "rul_predictions.csv")
OUTPUT_GRAPH_PATH = os.path.join(BASE_PATH, "Milestone4", "RUL_Evaluation_Graph.jpg")

# --------------------------
# 🔹 LOAD TEST DATA
# --------------------------
print("🔹 Loading test data...")
col_names = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1, 22)]
test_df = pd.read_csv(TEST_DATA_PATH, sep=r"\s+", header=None, names=col_names)

# Simulate actual RUL values for visualization (since CMAPSS doesn’t include them)
actual_rul = test_df.groupby("engine_id")["cycle"].max().reset_index()
actual_rul.columns = ["engine_id", "max_cycle"]
test_df = test_df.merge(actual_rul, on="engine_id")
test_df["RUL"] = test_df["max_cycle"] - test_df["cycle"]

# --------------------------
# 🔹 LOAD SCALER & MODEL
# --------------------------
print("🔹 Loading scaler and trained model...")
scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH, compile=False)

# --------------------------
# 🔹 PREPARE DATA FOR PREDICTION
# --------------------------
# ✅ FIXED: Include operation settings + sensor data
feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1, 22)]

X_test = test_df[feature_cols]
X_test_scaled = scaler.transform(X_test)
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# --------------------------
# 🔹 MAKE PREDICTIONS
# --------------------------
print("🔹 Predicting Remaining Useful Life (RUL)...")
predicted_rul = model.predict(X_test_scaled)
test_df["Predicted_RUL"] = predicted_rul.flatten()

# Save predictions to CSV
test_df[["engine_id", "cycle", "RUL", "Predicted_RUL"]].to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ Predictions saved to: {OUTPUT_CSV_PATH}")

# --------------------------
# 🔹 CALCULATE METRICS
# --------------------------
print("🔹 Calculating performance metrics...")
y_true = test_df["RUL"]
y_pred = test_df["Predicted_RUL"]

mae = mean_absolute_error(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n📊 Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# --------------------------
# 🔹 VISUALIZATION
# --------------------------
print("🔹 Generating graph...")
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_true[:200])), y_true[:200], color='blue', label='Actual RUL', alpha=0.6)
plt.scatter(range(len(y_pred[:200])), y_pred[:200], color='red', label='Predicted RUL', alpha=0.6)
plt.title("Predicted vs Actual Remaining Useful Life (RUL)")
plt.xlabel("Sample Index")
plt.ylabel("RUL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_GRAPH_PATH)
plt.show()  # 👈 Automatically show graph
plt.close()

print(f"✅ Graph saved as: {OUTPUT_GRAPH_PATH}")
print("\n🎯 Milestone 4 successfully completed!")
