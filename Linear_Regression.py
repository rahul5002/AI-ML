import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([[2], [3], [5], [6], [8], [10]]) 
y = np.array([35, 45, 50, 65, 70, 75])
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (B1): {slope}")
print(f"Intercept (B0): {intercept}")
y_predicted = model.predict(X)
mse = mean_squared_error(y, y_predicted)
print(f"Mean Squared Error (MSE): {mse}")

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_predicted, color='red', linewidth=2, label='Linear Regression Line')
plt.title('Linear Regression Model')
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.legend()
plt.grid(True)
plt.show()