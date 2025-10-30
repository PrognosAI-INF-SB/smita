import pandas as pd
import numpy as np

# Column names
col_names = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] \
            + [f"s{i}" for i in range(1, 22)]

# Load dataset
train = pd.read_csv(r"D:\PrognosAI_Project\Milestone1\train_FD001.txt",
                    sep=r"\s+", header=None, names=col_names)

print("Dataset Loaded Successfully ✅")
print(train.head())

# Compute Remaining Useful Life (RUL)
rul = train.groupby('engine_id')['cycle'].max().reset_index()
rul.columns = ['engine_id', 'max_cycle']
train = train.merge(rul, on='engine_id', how='left')
train['RUL'] = train['max_cycle'] - train['cycle']
train.drop('max_cycle', axis=1, inplace=True)

train.to_csv(r"D:\PrognosAI_Project\Milestone1\Preprocessed_Train.csv", index=False)

print("Preprocessed data saved successfully ✅")
