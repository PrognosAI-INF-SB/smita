import pandas as pd
from sklearn.model_selection import train_test_split

# Load your main dataset (adjust file name if different)
data_path = r"D:\PrognosAI_Project\Milestone1\train_data.csv"
data = pd.read_csv(data_path)

# Split into 80% train and 20% test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save both
train_data.to_csv(r"D:\PrognosAI_Project\Milestone1\train_data_split.csv", index=False)
test_data.to_csv(r"D:\PrognosAI_Project\Milestone1\test_data.csv", index=False)

print("✅ Data split complete:")
print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
