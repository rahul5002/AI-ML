import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 60, 30, 40, 50]
plt.plot(x, y)
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Title of the Plot')
plt.legend(['Data Series'])
plt.show()