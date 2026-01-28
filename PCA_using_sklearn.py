import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

data = np.array([
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])
pca = PCA(n_components=2)
pca.fit(data)
pc1_axis = pca.components_[0]
pc2_axis = pca.components_[1]
mean = pca.mean_
transformed_data = pca.transform(data)
projections_pc1 = transformed_data[:, 0]
print(f"Data Mean: {mean}")
print(f"PC1 Axis (Eigenvector 1): {pc1_axis}")
print(f"PC2 Axis (Eigenvector 2): {pc2_axis}")
print(f"Explained Variance (Eigenvalues): {pca.explained_variance_}")
print(f"\nTransformed Data (Coordinates in PC space):\n{transformed_data}")
print(f"\nProjections on PC1: {projections_pc1}")
plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], alpha=0.7, s=100, label='Original Data (X)')
plt.scatter(mean[0], mean[1], color='red', marker='x', s=150, label='Data Mean')
scale_factor = 2.0 
plt.arrow(mean[0], mean[1], 
          pc1_axis[0] * np.sqrt(pca.explained_variance_[0]) * scale_factor, 
          pc1_axis[1] * np.sqrt(pca.explained_variance_[0]) * scale_factor,
          head_width=0.1, head_length=0.2, fc='green', ec='green', 
          label=f'PC1 Axis (Var: {pca.explained_variance_[0]:.2f})')
pc2_len = np.sqrt(pca.explained_variance_[1]) * scale_factor
if pc2_len < 0.1: 
    pc2_len = 0.5 
plt.arrow(mean[0], mean[1], 
          pc2_axis[0] * pc2_len, 
          pc2_axis[1] * pc2_len,
          head_width=0.1, head_length=0.2, fc='purple', ec='purple', 
          label=f'PC2 Axis (Var: {pca.explained_variance_[1]:.6f})')
projections_on_pc1_2d = transformed_data[:, 0:1] * pc1_axis + mean
plt.scatter(projections_on_pc1_2d[:, 0], projections_on_pc1_2d[:, 1], 
            s=250, facecolors='none', edgecolors='blue', 
            marker='^', label='Projection onto PC1')
plt.title('PCA of 2D Data', fontsize=16)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend()
plt.grid(True)
plt.axis('equal') 
plt.tight_layout()
plt.savefig('pca_visualization_2d.png') 
plt.figure(figsize=(10, 4))
plt.scatter(projections_pc1, np.zeros_like(projections_pc1), 
            marker='o', s=100, c=projections_pc1, cmap='viridis', 
            label='Projected Points (on PC1)')
plt.axvline(0, color='red', linestyle='--', label='Origin (Mean)')
for i, txt in enumerate(projections_pc1):
    plt.text(txt, 0.05, f'{txt:.2f}', ha='center', fontsize=9)
plt.title('1D View of Projections onto PC1', fontsize=16)
plt.xlabel('PC1 Coordinate (Distance from Mean)', fontsize=12)
plt.yticks([]) 
plt.grid(axis='x')
plt.legend()
plt.tight_layout()
plt.savefig('pca_projections_1d.png') 
plt.show()
print("\nGenerated and saved 'pca_visualization_2d.png' and 'pca_projections_1d.png'.")