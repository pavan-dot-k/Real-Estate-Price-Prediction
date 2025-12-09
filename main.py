import pandas as pd
df = pd.read_csv('datasets/FInal Dataset/dataset_for_training.csv')
print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")