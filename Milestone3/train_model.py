import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# =============================
# 🔹 Define file paths
# =============================
base_folder = r"D:\PrognosAI_Project\Milestone3"
train_data_path = os.path.join(base_folder, "train_FD001.txt")
model_save_path = os.path.join(base_folder, "trained_model.h5")
scaler_save_path = os.path.join(base_folder, "scaler.pkl")
graph_folder = os.path.join(base_folder, "train_model_graph")

# Create graph folder if not exists
os.makedirs(graph_folder, exist_ok=True)

# =============================
# 🔹 Load the dataset
# =============================
print("🔹 Loading training data...")
col_names = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv(train_data_path, sep=r"\s+", header=None, names=col_names)

# =============================
# 🔹 Data Preprocessing
# =============================
train["max_cycle"] = train.groupby("engine_id")["cycle"].transform("max")
train["RUL"] = train["max_cycle"] - train["cycle"]
train.drop("max_cycle", axis=1, inplace=True)

features = train.drop(["RUL", "engine_id", "cycle"], axis=1)
target = train["RUL"]

# =============================
# 🔹 Normalize Data
# =============================
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Split data
X_train, X_val, y_train, y_val = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

# =============================
# 🔹 Build Model
# =============================
print("🔹 Building model...")
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# =============================
# 🔹 Train Model
# =============================
print("🔹 Training model...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64, verbose=1)

# =============================
# 🔹 Save Model & Scaler
# =============================
model.save(model_save_path)
with open(scaler_save_path, 'wb') as file:
    pickle.dump(scaler, file)

print(f"✅ Model trained and saved at: {model_save_path}")
print(f"✅ Scaler saved at: {scaler_save_path}")

# =============================
# 🔹 Plot Training Graph
# =============================
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title("Training vs Validation Loss", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)

# Save graph
graph_path = os.path.join(graph_folder, "training_loss_graph.jpg")
plt.savefig(graph_path)
plt.show()

print(f"📊 Training graph saved at: {graph_path}")

