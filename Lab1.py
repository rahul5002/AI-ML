import pandas as pd

file_path = r"C:\Users\dell\OneDrive\Desktop\Height_Weight_Class.xlsx"
df = pd.read_excel(file_path)
print("Dataset:\n", df)

target_row = df[df["Class"].isnull()].iloc[0]
training_df = df[df["Class"].notnull()].copy()

training_df["Dist"] = (
    (training_df["Height (CM)"] - target_row["Height (CM)"])**2 +
    (training_df["Weight (KG)"] - target_row["Weight (KG)"])**2
) ** 0.5

sorted_df = training_df.sort_values("Dist")
k_value = 5
nearest_neighbors = sorted_df.head(k_value)
predicted_class = nearest_neighbors["Class"].mode()[0]

print("\nNearest neighbors:\n", nearest_neighbors[["Height (CM)", "Weight (KG)", "Class", "Dist"]])
print("\nPredicted Class for missing row:", predicted_class)
