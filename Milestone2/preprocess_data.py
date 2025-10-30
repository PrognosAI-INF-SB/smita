import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# ======================================
# Step 1: Define paths
# ======================================
base_dir = r"D:\PrognosAI_Project\Milestone1"
output_dir = r"D:\PrognosAI_Project\Milestone2"

train_path = os.path.join(base_dir, "train_data.csv")
test_path = os.path.join(base_dir, "test_data.csv")

# ======================================
# Step 2: Load data
# ======================================
train_data_original = pd.read_csv(train_path)
test_data_original = pd.read_csv(test_path)

# ======================================
# Step 3: Normalize sensor values
# ======================================
sensor_cols = [col for col in train_data_original.columns if col.startswith('s')]

# Copy to preserve original for graph
train_data = train_data_original.copy()
test_data = test_data_original.copy()

scaler = MinMaxScaler(feature_range=(0, 1))
train_data[sensor_cols] = scaler.fit_transform(train_data[sensor_cols])
test_data[sensor_cols] = scaler.transform(test_data[sensor_cols])

# ======================================
# Step 4: Save preprocessed data
# ======================================
os.makedirs(output_dir, exist_ok=True)
train_preprocessed_path = os.path.join(output_dir, "train_preprocessed.csv")
test_preprocessed_path = os.path.join(output_dir, "test_preprocessed.csv")

train_data.to_csv(train_preprocessed_path, index=False)
test_data.to_csv(test_preprocessed_path, index=False)

# ======================================
# Step 5: Generate before vs after graph
# ======================================
plt.figure(figsize=(14, 6))

# --- Before normalization ---
plt.subplot(1, 2, 1)
plt.plot(train_data_original[sensor_cols[:5]].mean(), marker='o', color='red')
plt.title("Before Normalization")
plt.xlabel("Sensor Index (s1 - s5)")
plt.ylabel("Raw Sensor Values")
plt.grid(True)

# --- After normalization ---
plt.subplot(1, 2, 2)
plt.plot(train_data[sensor_cols[:5]].mean(), marker='o', color='green')
plt.title("After Normalization (0–1 Scale)")
plt.xlabel("Sensor Index (s1 - s5)")
plt.ylabel("Normalized Values")
plt.grid(True)

plt.suptitle("Sensor Data Comparison: Before vs After Normalization", fontsize=14, weight='bold')

# Save and show graph
graph_path = os.path.join(output_dir, "normalization_comparison_graph.jpg")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(graph_path, format='jpg')
plt.show()

# ======================================
# Step 6: Final message
# ======================================
print("✅ Data preprocessing and comparison visualization completed successfully!")
print(f"📂 Preprocessed data saved in: {output_dir}")
print(f"🖼️ Graph saved as: {graph_path}")
