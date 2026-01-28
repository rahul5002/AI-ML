import numpy as np
from sklearn.decomposition import PCA

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
print(f"PC1 Axis: {pc1_axis}")
print(f"PC2 Axis: {pc2_axis}")
transformed_data = pca.transform(data)
projections_pc1 = transformed_data[:, 0]
print(f"\nProjections on PC1: {projections_pc1}")  