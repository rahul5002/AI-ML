import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

file_path = r"C:\Users\dell\OneDrive\Desktop\creditcard.csv"
df = pd.read_csv(file_path)
target_column = 'Class'
print(f"--- Analysis (Original Data) ---")
num_classes = df[target_column].nunique()
print(f"Number of classes: {num_classes}")
class_counts = df[target_column].value_counts()
print("\nNumber of examples per class:")
print(class_counts)
if len(class_counts) == 2 and (class_counts.iloc[0] != class_counts.iloc[1]):
    print("\nDataset is unbalanced. Balancing...")
    if class_counts.iloc[0] > class_counts.iloc[1]:
        majority_df = df[df[target_column] == class_counts.index[0]]
        minority_df = df[df[target_column] == class_counts.index[1]]
    else:
        majority_df = df[df[target_column] == class_counts.index[1]]
        minority_df = df[df[target_column] == class_counts.index[0]]
    print(f"Original shape (Majority): {majority_df.shape}")
    print(f"Original shape (Minority): {minority_df.shape}")
    minority_oversampled = resample(minority_df,
                                    replace=True,
                                    n_samples=len(majority_df),
                                    random_state=42)

    balanced_df = pd.concat([majority_df, minority_oversampled]) 
    print("\n--- Analysis (Balanced Data) ---")
    print("Balancing complete.")
    print("New class distribution:")
    print(balanced_df[target_column].value_counts())
else:
    print("\nDataset is already balanced or is not a binary classification task.")
    balanced_df = df

print("\n--- 3. Preparing Data for Modeling ---")
X = balanced_df.drop(target_column, axis=1)
y = balanced_df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("\n--- 4. Scaling Features ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")
print("\n--- 5. Training the Logistic Regression Model ---")
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.")
print("\n--- 6. Evaluating the Model ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))