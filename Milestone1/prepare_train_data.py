import pandas as pd
import os

# ------------------------------
# 🔹 PATH CONFIGURATION
# ------------------------------
base_path = r"C:\Users\siddharth\Downloads\archive\CMaps"

train_file = os.path.join(base_path, "train_FD001.txt")
test_file = os.path.join(base_path, "test_FD001.txt")

# Output CSV paths
train_csv = r"D:\PrognosAI_Project\Milestone1\train_data.csv"
test_csv = r"D:\PrognosAI_Project\Milestone1\test_data.csv"

# ------------------------------
# 🔹 COLUMN NAMES (CMAPSS FD001)
# ------------------------------
col_names = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] \
            + [f"s{i}" for i in range(1, 22)]

# ------------------------------
# 🔹 LOAD AND SAVE TRAIN DATA
# ------------------------------
if os.path.exists(train_file):
    print(f"📘 Reading training data from: {train_file}")
    train = pd.read_csv(train_file, sep=r"\s+", header=None, names=col_names)
    train.to_csv(train_csv, index=False)
    print(f"✅ Train data saved at: {train_csv}")
else:
    print("❌ train_FD001.txt not found!")

# ------------------------------
# 🔹 LOAD AND SAVE TEST DATA
# ------------------------------
if os.path.exists(test_file):
    print(f"📗 Reading test data from: {test_file}")
    test = pd.read_csv(test_file, sep=r"\s+", header=None, names=col_names)
    test.to_csv(test_csv, index=False)
    print(f"✅ Test data saved at: {test_csv}")
else:
    print("❌ test_FD001.txt not found! Please check the CMaps folder.")
