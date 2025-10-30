# 📘 Step 1: Import Libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# 📗 Step 2: Load Preprocessed Data
data = pd.read_csv(r"D:\PrognosAI_Project\Milestone1\Preprocessed_Train.csv")

# Example: separating features and target (adjust as per your dataset)
X = data.drop(columns=['RUL'])  # or whatever your target column is
y = data['RUL']

# 📘 Step 3: Normalize & Split
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 📙 Step 4: Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

# 📒 Step 5: Define Model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 📕 Step 6: Train Model — here we get the `history` variable
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# 📘 Step 7: Plot and Save Training Graph
graph_folder = r"D:\PrognosAI_Project\train_model_graph"
os.makedirs(graph_folder, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

graph_path = os.path.join(graph_folder, 'train_model_graph.jpg')
plt.savefig(graph_path, format='jpg')
plt.show()

print(f"✅ Training graph saved successfully at: {graph_path}")

# 📔 Step 8: Save Model
model.save(r"D:\PrognosAI_Project\Milestone3\trained_model.keras")

print("✅ Model trained and saved successfully!")
