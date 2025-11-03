# Milestone2/scripts/visualize_results.py

import matplotlib.pyplot as plt
import joblib
import os

# --- 1. Define Constants ---
HISTORY_PATH = '../history/training_history.pkl'
SAVE_DIR = '../plots/'

os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. Load History ---
print(f"Loading training history from {HISTORY_PATH}...")
try:
    history = joblib.load(HISTORY_PATH)
except FileNotFoundError:
    print(f"Error: History file not found at {HISTORY_PATH}")
    print("Please run train_model.py first.")
    exit()

# --- 3. Plot Training & Validation Loss (MSE) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss (MSE)')
plt.plot(history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# --- 4. Plot Training & Validation Metric (RMSE) ---
plt.subplot(1, 2, 2)
plt.plot(history['rmse'], label='Training RMSE')
plt.plot(history['val_rmse'], label='Validation RMSE')
plt.title('Model Performance (RMSE)')
plt.xlabel('Epoch')
plt.ylabel('Root Mean Squared Error')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the plot
plot_save_path = os.path.join(SAVE_DIR, 'training_performance_curves.png')
plt.savefig(plot_save_path)
print(f"Saved performance plots to {plot_save_path}")

# Display the plot
plt.show()

print("\n--- Visualization Complete ---")