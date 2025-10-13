import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 60, 30, 40, 50]
data = [10, 20, 20, 30, 40, 40, 40, 50]
plt.hist(data, bins=5, color='orange')
plt.show()