import pandas as pd
data = {
    "Weight": [50, 30, 40, 45, 38, 68],
    "Height": [None, 45, 70, 80, 50, 34],
    "Class": ["Normal", "Overweight", "Underweight", "Normal", "Overweight", "Overweight"]
}
df = pd.DataFrame(data)
numerical_cols = df.select_dtypes(include=['number'])
string_cols = df.select_dtypes(include=['object'])
numerical_list = numerical_cols.values.tolist()
string_list = string_cols['Class'].tolist()
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\nFirst five rows:")
print(df.head())
print("\nSize (total elements):", df.size)
print("\nNumber of missing values per column:")
print(df.isnull().sum())
print("\nSum of numerical columns:")
print(numerical_cols.sum())
print("\nAverage (mean) of numerical columns:")
print(numerical_cols.mean())
print("\nMin values of numerical columns:")
print(numerical_cols.min())
print("\nMax values of numerical columns:")
print(numerical_cols.max())
print("\nFeature:")
print(numerical_list)
print("\nLabel:")
print(string_list)
# Exporting data to a new CSV file
df.to_csv('output_data.csv', index=False)
print("\nData exported to 'output_data.csv'")
