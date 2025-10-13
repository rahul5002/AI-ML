import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 60, 30, 40, 50]
plt.grid(True)  
plt.grid(color='blue', linestyle='--', linewidth=2)
plt.plot(x, y)
plt.show()