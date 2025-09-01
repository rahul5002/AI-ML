import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = r"C:\Users\dell\OneDrive\Desktop\cluster_data.xlsx"
df = pd.read_excel(file_path)
print("Dataset:\n", df)

X = df.iloc[:, [0, 1]].to_numpy()#data frame to numpy array conversion

def kmeans(X, k, max_iters=100):
    n_samples = X.shape[0]
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx]
    
    for _ in range(max_iters):
        distances = np.zeros((n_samples, k))  # dist. from all points to centroid
        for i in range(k):
            distances[:, i] = np.sqrt(((X - centroids[i]) ** 2).sum(axis=1))
        
        labels = np.argmin(distances, axis=1)  # nearest centroid ko points assign ke liye
        
        old_centroids = centroids.copy()
        
        for i in range(k):
            if np.sum(labels == i) > 0:
                centroids[i] = np.mean(X[labels == i], axis=0)
        
        if np.all(old_centroids == centroids):
            break
    
    return labels, centroids

k = 3
labels, centroids = kmeans(X, k)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title('K-means Clustering Results (Manual Implementation)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()