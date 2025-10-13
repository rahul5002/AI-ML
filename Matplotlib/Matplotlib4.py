import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 60, 30, 40, 50]
plt.plot(x, y, color='green', linewidth=2, linestyle='-')
plt.xlim(0, 6)
plt.ylim(0, 60)
plt.title('Customized Plot', fontsize=14, color='blue')
plt.show()