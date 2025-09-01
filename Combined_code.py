import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

file_path = r"C:\Users\dell\OneDrive\Desktop\Height_Weight_Class.xlsx"
df = pd.read_excel(file_path)
print("Dataset:\n", df)

target_row = df[df["Class"].isnull()].iloc[0]
training_df = df[df["Class"].notnull()].copy()

X = training_df[["Height (CM)", "Weight (KG)"]].values
y = training_df["Class"].values

k_value = 5
neigh = KNeighborsClassifier(n_neighbors=k_value)
neigh.fit(X, y)

target_features = [[target_row["Height (CM)"], target_row["Weight (KG)"]]]

predicted_class = neigh.predict(target_features)[0]
predicted_proba = neigh.predict_proba(target_features)[0]

distances, indices = neigh.kneighbors(target_features, n_neighbors=k_value)
nearest_neighbors = training_df.iloc[indices[0]][["Height (CM)", "Weight (KG)", "Class"]].copy()
nearest_neighbors["Dist"] = distances[0]

print("\nNearest neighbors:\n", nearest_neighbors[["Height (CM)", "Weight (KG)", "Class", "Dist"]])
print("\nPredicted Class for missing row:", predicted_class)
print("Prediction probabilities:", predicted_proba)