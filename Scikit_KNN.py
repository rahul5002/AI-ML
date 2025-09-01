X=[[0,0],[1,1],[2,2],[3,3]]
y=[0,0,1,1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)
print(neigh.predict([[2,1.1]]))
print(neigh.predict_proba([[1,1]]))
