# Milestone2/scripts/train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib
import os

# --- 1. Define Constants ---
DATA_DIR = '../data/processed/'
MODEL_SAVE_PATH = '../models/'
HISTORY_SAVE_PATH = '../history/'

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(HISTORY_SAVE_PATH, exist_ok=True)

# --- 2. Load Data ---
print("Loading processed data...")
try:
    sequences = np.load(os.path.join(DATA_DIR, 'train_sequences.npy'))
    targets = np.load(os.path.join(DATA_DIR, 'train_targets.npy'))
except FileNotFoundError:
    print(f"Error: Processed data not found in {DATA_DIR}")
    print("Please run Milestone 1 preparation scripts first.")
    exit()

print(f"Sequences shape: {sequences.shape}") # Should be (samples, 50, 14)
print(f"Targets shape: {targets.shape}")     # Should be (samples,)

# --- 3. Split Data (Train/Validation) ---
# We'll split the training data into a training set and a validation set
# to monitor the model's performance on unseen data during training.
X_train, X_val, y_train, y_val = train_test_split(
    sequences, targets, test_size=0.2, random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# --- 4. Define Model Architecture ---
# Get input shape from the data
# (sequence_length, num_features)
input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential([
    # First LSTM layer with dropout
    LSTM(units=50, return_sequences=True, input_shape=input_shape),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    
    # Dense layer for final regression output
    Dense(units=1) # Output is a single value (RUL)
])

# Compile the model
# We use Mean Squared Error (MSE) for regression
# and Root Mean Squared Error (RMSE) as a metric
model.compile(
    optimizer='adam', 
    loss='mean_squared_error',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

model.summary()

# --- 5. Train the Model ---
print("\nStarting model training...")

# Add an EarlyStopping callback to prevent overfitting
# This stops training if the validation loss doesn't improve
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,  # You can increase this if needed
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[callback],
    verbose=1
)

print("Training complete.")

# --- 6. Save Model and History ---
print("Saving model and training history...")

# Save the trained model
model.save(os.path.join(MODEL_SAVE_PATH, 'PrognosAI_LSTM.keras'))

# Save the history object to be used by visualize_results.py
joblib.dump(history.history, os.path.join(HISTORY_SAVE_PATH, 'training_history.pkl'))

print(f"Model saved to {MODEL_SAVE_PATH}")
print(f"History saved to {HISTORY_SAVE_PATH}")
print("\n--- Milestone 2: Training Complete ---")