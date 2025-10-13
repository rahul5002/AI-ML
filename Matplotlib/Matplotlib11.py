import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 60, 30, 40, 50]
sizes = [25, 35, 20, 20]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(x, y)
ax[0, 0].set_title('Line Graph')
ax[0, 0].set_xlabel('X Label 1', labelpad=10, fontsize=6, color='red')
ax[0, 0].set_ylabel('Y Label 1', labelpad=10, fontsize=6, color='red')
ax[0, 0].legend("Legend",loc="upper right")
ax[0, 1].bar(x, y)
ax[0, 1].set_title('Bar Chart')
ax[1, 0].scatter(x, y)
ax[1, 0].set_title('Scatter Plot')
ax[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%')
ax[1, 1].set_title('Pie Chart')
plt.tight_layout()  # prevents overlap
plt.show()


