import matplotlib.pyplot as plt

sizes = [25, 35, 20, 20]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.show()