import numpy as np
import pandas as pd

file_name = r"C:\Users\dell\OneDrive\Desktop\Student Data.xlsx"
try:
    df_full = pd.read_excel(file_name)
except FileNotFoundError:
    print(f"Error: File not found at {file_name}")
    exit()
try:
    column_name = 'age'  
    data = df_full[column_name].dropna().values
except KeyError:
    print(f"Error: Column '{column_name}' not found in the Excel file.")
    exit()
if data.size == 0:
    print("Error: The selected column is empty or contains only missing values.")
    exit()
mean_val = np.mean(data)
std_dev = np.std(data)
z_scores_numpy = (data - mean_val) / std_dev
print(f"Original Data (first 10 values): {data[:10]}...")
print(f"Z-Scores (first 10 values):\n{z_scores_numpy[:10]}...") 

outlier_threshold = 2.0
is_outlier = np.abs(z_scores_numpy) > outlier_threshold
outliers = data[is_outlier]
print("-" * 30)
print(f"Analysis on Column: {column_name}")
print(f"Data Points Detected as Outlier (Z > {outlier_threshold}): {len(outliers)}")
print(f"Detected Outlier Values: {outliers}")
print(f"Number of Non-Outlier Values: {len(data) - len(outliers)}")