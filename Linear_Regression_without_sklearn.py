import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, 3, 5, 6, 8, 10], dtype=float)
y = np.array([35, 45, 50, 65, 70, 75], dtype=float)
x_mean = np.mean(x)
y_mean = np.mean(y)
numer = np.sum((x - x_mean) * (y - y_mean))
denom = np.sum((x - x_mean) ** 2)
m = numer / denom
c = y_mean - m * x_mean
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")
y_pred = m * x + c
mse = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error: {mse}")

plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, y_pred, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Model")
plt.legend()
plt.show()
