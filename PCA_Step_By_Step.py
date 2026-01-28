import numpy as np

def pca(X, k):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    print("--- Step 1: Standardized Data ---")
    print(Z[:5])
    print("\n" + "="*50 + "\n")
    n_samples = Z.shape[0]
    covariance_matrix = (Z.T @ Z) / (n_samples - 1)
    print("--- Step 2: Covariance Matrix ---")
    print(covariance_matrix)
    print("\n" + "="*50 + "\n")
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print("--- Step 3: Eigenvalues & Eigenvectors (Unsorted) ---")
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors (as columns):\n", eigenvectors)
    print("\n" + "-"*20 + "\n")   
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    print("--- Step 4: Sorted Eigenvalues & Eigenvectors ---")
    print("Sorted Eigenvalues:\n", sorted_eigenvalues)
    print("Sorted Eigenvectors (as columns):\n", sorted_eigenvectors)
    print("\n" + "="*50 + "\n")
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = sorted_eigenvalues / total_variance
    print("--- Step 5: Variance ---")
    for i in range(len(sorted_eigenvalues)):
        ratio = explained_variance_ratio[i] if total_variance > 0 else 0
        cumulative = np.sum(explained_variance_ratio[:i+1]) if total_variance > 0 else 0
        print("PC{}: Explained Variance Ratio = {:.4f} (Cumulative = {:.4f})".format(
            i+1, 
            ratio, 
            cumulative
        ))
    print("\n" + "="*50 + "\n")
    W = sorted_eigenvectors[:, :k]
    print("--- Step 6: Projection Matrix W ---")
    print(W)
    print("\n" + "="*50 + "\n")
    X_pca = Z @ W
    print("--- Step 7: New Transformed Data ---")
    print(X_pca[:5])
    print("\n" + "="*50 + "\n")
    return X_pca

if __name__ == "__main__":
    data_x = np.array([2.0, 3.0, 4.0, 5.0])
    data_y = np.array([3.0, 4.0, 5.0, 6.0])
    X = np.hstack((data_x.reshape(-1, 1), data_y.reshape(-1, 1)))
    print("Original Data Shape:", X.shape)
    print("Original Data (first 5 rows):\n", X[:5])
    print("\n" + "="*50 + "\n")
    k = 1
    X_transformed = pca(X, k)
    print("Final Transformed Data Shape:", X_transformed.shape)
    print("Final Transformed Data:\n", X_transformed)